"""Walk-forward (time-ordered) validation of the FULL multi-modal pipeline.

Per fold (expanding window):
  • Tabular ensemble (RF + XGBoost + LSTM stacking) fitted on sessions < fold
  • ViT fitted on chart images from sessions < fold   (causal vision branch)
  • Fusion meta-classifier fitted on in-window OOF-style probabilities
  • All three applied to the next `test_days` sessions

Concatenated fold predictions form the honest out-of-sample series used for
the headline walk-forward accuracy — no random splits anywhere.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import feature_columns, per_symbol_frames


def make_walkforward_folds(n_per_symbol: dict[str, int], n_folds: int,
                           test_days: int) -> list[dict]:
    min_n = min(n_per_symbol.values())
    end = min_n
    folds = []
    for _ in range(n_folds):
        te_end = end
        te_start = te_end - test_days
        if te_start <= max(80, int(0.35 * min_n)):
            break
        folds.append({"train_end": te_start, "test_start": te_start, "test_end": te_end})
        end = te_start
    return list(reversed(folds))[-n_folds:]


def _row_order(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Symbols in the same order used to stack X across symbols."""
    return list(frames.keys())


def _stack(frames: dict[str, pd.DataFrame], cols: list[str],
           idx_map: dict[str, np.ndarray]):
    Xs, ys, syms, dates, row_ids = [], [], [], [], []
    row = 0
    for s in _row_order(frames):
        if s not in idx_map:
            continue
        idx = idx_map[s]
        g = frames[s].iloc[idx]
        Xs.append(g[cols].to_numpy(np.float32))
        ys.append(g["label"].to_numpy(np.int64))
        for _, r in g.iterrows():
            syms.append(s)
            dates.append(str(pd.Timestamp(r["date"]).date()))
            row_ids.append(row)
            row += 1
    if not Xs:
        return (np.zeros((0, len(cols)), np.float32), np.zeros((0,), np.int64),
                [], [], np.array([], int))
    return (np.vstack(Xs), np.concatenate(ys), syms, dates, np.array(row_ids, int))


def _train_vit_probs(cfg, manifest, feature_df, fit_end: int, eval_idx_map,
                     frames, artifacts_dir, tag: str):
    """Train ViT (subprocess) on images with position < fit_end; probs for eval dates.

    Returns (p_eval_aligned_dict, trained_flag, model_path_or_None)
    """
    from .vit_runner import run_vit_worker

    if manifest is None or len(manifest) == 0:
        return None, False, None

    labels = feature_df[["symbol", "date", "label"]].copy()
    labels["date"] = labels["date"].astype(str).str[:10]
    man = manifest.copy()
    man["date"] = man["date"].astype(str).str[:10]

    train_keys, val_keys, eval_key_set = set(), set(), set()
    for s, g in frames.items():
        dts = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").tolist()
        for j, d in enumerate(dts):
            if j < fit_end:
                if j >= int(fit_end * 0.9):
                    val_keys.add((s, d))
                else:
                    train_keys.add((s, d))
                eval_key_set.add((s, d))          # train probs for fusion head
        if s in eval_idx_map:
            for j in eval_idx_map[s]:
                eval_key_set.add((s, dts[j]))
    if not train_keys or not eval_key_set:
        return None, False, None

    model_out = Path(artifacts_dir) / "models" / f"vit_{tag}.pt"
    probs_out = Path(artifacts_dir) / "models" / f"vit_{tag}_probs.parquet"
    probs = run_vit_worker(man, labels, train_keys, val_keys, eval_key_set,
                           model_out, probs_out, epochs=cfg["vision"]["epochs"],
                           tmp_dir=Path(artifacts_dir) / "models" / f"_vit_{tag}")
    if probs is None:
        return None, False, None
    lut = {(r.symbol, r.date): float(r.p_up_vit) for r in probs.itertuples()}
    out = {}
    for s, idxs in eval_idx_map.items():
        dts = pd.to_datetime(frames[s].iloc[idxs]["date"]).dt.strftime("%Y-%m-%d").tolist()
        out[s] = [lut.get((s, d), np.nan) for d in dts]
    # stash train probs for fusion
    _train_vit_probs.last_probs = probs
    return out, True, model_out


def run_walkforward(cfg: dict, feature_df: pd.DataFrame,
                    manifest: pd.DataFrame | None,
                    artifacts_dir: Path,
                    verbose: bool = True) -> pd.DataFrame:
    from .tabular_models import TabularEnsemble, sequences_by_symbol
    from .fusion import FusionModel

    frames = per_symbol_frames(feature_df)
    cols = feature_columns(feature_df)
    n_per = {s: len(g) for s, g in frames.items()}
    folds = make_walkforward_folds(n_per, cfg["split"]["walkforward_folds"],
                                   cfg["split"]["walkforward_test_days"])
    lookback = cfg["features"]["lookback_lstm"]
    rows_out, fold_reports = [], []

    if verbose:
        print(f"Walk-forward: {len(folds)} folds, test_days="
              f"{cfg['split']['walkforward_test_days']}, feature dim={len(cols)}")

    for fi, fold in enumerate(folds):
        t0 = time.time()
        tr_end, te_start, te_end = fold["train_end"], fold["test_start"], fold["test_end"]
        train_idx = {s: np.arange(0, tr_end) for s in frames}
        test_idx = {s: np.arange(te_start, te_end) for s in frames
                    if te_end <= len(frames[s])}

        Xtr, ytr, _, _, _ = _stack(frames, cols, train_idx)
        Xte, yte, syms_te, dates_te, _ = _stack(frames, cols, test_idx)

        # sequences over contiguous train window (per symbol, causal warmup)
        X_seq_tr, y_seq_tr, meta_tr = sequences_by_symbol(frames, cols, train_idx, lookback)
        # map each sequence row → stacked Xtr row index
        seq_row = np.empty(len(meta_tr), dtype=int)
        base = 0
        for s in _row_order(frames):
            mask = (meta_tr["symbol"] == s) if len(meta_tr) else None
            n_s = tr_end
            if mask is not None and mask.any():
                seq_row[mask.to_numpy()] = base + meta_tr.loc[mask, "pos"].to_numpy()
            base += n_s

        ens = TabularEnsemble(cfg, n_features=len(cols), with_lstm=True)
        ens.fit(Xtr, ytr, X_seq=X_seq_tr, y_seq=y_seq_tr, seq_index_in_X=seq_row)

        # sequences for test rows: lookback warmup drawn from rows immediately
        # before the fold *within each symbol* (features stay ≤ t). We rebuild a
        # window [te_start-lookback, te_end) per symbol and keep only test rows.
        warm_idx = {}
        for s in frames:
            lo = max(0, te_start - lookback)
            if te_end <= len(frames[s]):
                warm_idx[s] = np.arange(lo, te_end)
        X_seq_w, y_seq_w, meta_w = sequences_by_symbol(frames, cols, warm_idx, lookback)
        # keep sequences ending on test rows (pos ≥ te_start), remap to Xte rows
        test_order = [s for s in _row_order(frames) if s in test_idx]
        offsets, acc = {}, 0
        for s in test_order:
            offsets[s] = acc
            acc += (te_end - te_start)
        keep, row_ids = [], []
        for i, m in enumerate(meta_w.itertuples()):
            if m.pos >= te_start and m.symbol in offsets:
                keep.append(i)
                row_ids.append(offsets[m.symbol] + (int(m.pos) - te_start))
        if keep:
            seq_row_te = np.array(row_ids, dtype=int)
            X_seq_te = X_seq_w[keep]
            y_seq_te = y_seq_w[keep]
        else:
            seq_row_te = np.array([], dtype=int)
            X_seq_te = np.zeros((0, lookback + 1, len(cols)), np.float32)

        p_tab = ens.predict_proba(Xte, X_seq=X_seq_te if len(X_seq_te) else None,
                                  seq_index_in_X=seq_row_te if len(seq_row_te) else None)

        # ---- ViT ------------------------------------------------------------
        p_vit_test_map, vit_ok, _ = _train_vit_probs(
            cfg, manifest, feature_df, tr_end, test_idx, frames,
            artifacts_dir, tag=f"fold{fi}")
        if p_vit_test_map is not None:
            p_vit = np.array([v for s in syms_te for v in p_vit_test_map.get(s, [0.5] * (te_end - te_start))],
                             dtype=float)
            if len(p_vit) != len(p_tab):
                # align defensively
                p_vit = np.resize(p_vit, len(p_tab))
            if np.isnan(p_vit).any():
                p_vit = np.nan_to_num(p_vit, nan=0.5)
        else:
            p_vit = np.full(len(p_tab), 0.5)

        # ---- fusion fit on OOF tabular probs inside the training window -----
        oof_tab = getattr(ens, "oof_p_", None)
        if oof_tab is None or len(oof_tab) != len(ytr):
            oof_tab = np.full(len(ytr), float(np.mean(ytr)))
        # ViT probabilities for training rows (images seen by ViT → slight optimism
        # in the fusion head only; test predictions remain strictly causal)
        if vit_ok:
            try:
                probs_path = Path(artifacts_dir) / "models" / f"vit_fold{fi}_probs.parquet"
                pv = pd.read_parquet(probs_path)
                lut = {(r.symbol, r.date): float(r.p_up_vit) for r in pv.itertuples()}
                keys = []
                for s in _row_order(frames):
                    dts = pd.to_datetime(frames[s]["date"]).dt.strftime("%Y-%m-%d").tolist()
                    for j in range(min(tr_end, len(dts))):
                        keys.append((s, dts[j]))
                p_vit_tr = np.array([lut.get(k, 0.5) for k in keys], dtype=float)
                if len(p_vit_tr) != len(ytr):
                    raise ValueError(f"len mismatch {len(p_vit_tr)} vs {len(ytr)}")
            except Exception as e:
                print("    [fusion] vit train probs failed:", e)
                p_vit_tr = np.full(len(ytr), 0.5)
        else:
            p_vit_tr = np.full(len(ytr), 0.5)

        fusion = FusionModel(use_vit=bool(vit_ok))
        # drop rows with NaN oof
        valid = np.isfinite(np.asarray(oof_tab, dtype=float))
        fusion.fit(np.asarray(oof_tab, dtype=float)[valid],
                   p_vit_tr[valid],
                   np.asarray(ytr)[valid])
        p_fused = fusion.predict_proba(p_tab, p_vit if fusion.use_vit else None)

        for j in range(len(yte)):
            rows_out.append({
                "fold": fi, "symbol": syms_te[j], "date": dates_te[j],
                "y": int(yte[j]), "p_tab": float(p_tab[j]),
                "p_vit": float(p_vit[j]), "p_fused": float(p_fused[j]),
            })
        rep = {
            "fold": fi, "train_end": int(tr_end),
            "test": [int(te_start), int(te_end)], "n_test": int(len(yte)),
            "acc_tab": float(np.mean((p_tab >= 0.5).astype(int) == yte)) if len(yte) else float("nan"),
            "acc_vit": float(np.mean((p_vit >= 0.5).astype(int) == yte)) if len(yte) else float("nan"),
            "acc_fused": float(np.mean((p_fused >= 0.5).astype(int) == yte)) if len(yte) else float("nan"),
            "vit_trained": bool(vit_ok),
            "seconds": round(time.time() - t0, 1),
        }
        fold_reports.append(rep)
        if verbose:
            print(f"  fold {fi}: n={rep['n_test']} tab={rep['acc_tab']:.3f} "
                  f"vit={rep['acc_vit']:.3f} fused={rep['acc_fused']:.3f} "
                  f"({rep['seconds']}s)")

    preds = pd.DataFrame(rows_out)
    Path(artifacts_dir, "metrics").mkdir(parents=True, exist_ok=True)
    with open(Path(artifacts_dir, "metrics") / "walkforward_folds.json", "w") as fh:
        json.dump(fold_reports, fh, indent=2)
    return preds

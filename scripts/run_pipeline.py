"""End-to-end OptionEdge pipeline orchestration.

Stages
------
1. ingest      — NSE index + F&O bhavcopy → prices / option-chain tables
2. features    — technicals + options microstructure (PCR, OI, IV, Greeks)
3. charts      — candlestick PNGs for the vision branch
4. holdout     — train all models on first 80% (10% calib), evaluate last 20%
                 → ablation: tabular vs ViT vs fusion  (+ threshold tuning)
5. walkforward — time-ordered refit CV across folds → headline honest accuracy
6. backtest    — trade BUY_CE/BUY_PE/NO_TRADE on the holdout with real option OHLC
7. reports     — metrics JSON, plots, SQLite, latest signal snapshot

Run:  python scripts/run_pipeline.py [--fast] [--stage NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from optionedge.config import load_config, ensure_dirs, seed_everything  # noqa: E402
from optionedge import data_ingest  # noqa: E402
from optionedge.dataset import (build_feature_table, feature_columns,  # noqa: E402
                                per_symbol_frames, chronological_splits)
from optionedge.tabular_models import TabularEnsemble, sequences_by_symbol  # noqa: E402
from optionedge.fusion import FusionModel, tune_threshold  # noqa: E402
from optionedge.evaluate import (classification_metrics, selective_metrics,  # noqa: E402
                                 benchmark_table, save_metrics,
                                 plot_equity_and_confusion)
from optionedge.signals import probabilities_to_signals  # noqa: E402
from optionedge.backtest import run_backtest  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_ingest(cfg: dict, fast: bool):
    proc = Path(cfg["data"]["processed_dir"])
    prices_p = proc / "prices.parquet"
    chain_p = proc / "fo_chain.parquet"
    if fast:
        # limit date span for smoke tests
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["data"]["start_date"] = "2024-01-01"
        cfg["data"]["end_date"] = "2025-06-30"
    if prices_p.exists() and chain_p.exists() and not fast:
        prices = pd.read_parquet(prices_p)
        chain = pd.read_parquet(chain_p)
        log(f"ingest: cached prices={prices.shape} chain={chain.shape}")
        return cfg, prices, chain
    log("ingest: reading NSE bhavcopy archives …")
    prices, chain = data_ingest.ingest(cfg)
    log(f"ingest: prices={prices.shape} chain={chain.shape} "
        f"({prices['date'].min().date()} → {prices['date'].max().date()})")
    return cfg, prices, chain


def stage_features(cfg: dict, prices, chain):
    log("features: technicals + options microstructure …")
    df = build_feature_table(cfg)
    cols = feature_columns(df)
    log(f"features: table={df.shape}, n_feature_cols={len(cols)}")
    out = Path(cfg["data"]["processed_dir"]) / "features.parquet"
    df.to_parquet(out, index=False)
    return df


def stage_charts(cfg: dict, prices):
    from optionedge.charts import generate_all_charts
    log("charts: rendering candlestick images …")
    if cfg.get("_fast"):
        prices = prices.copy()
    manifest = generate_all_charts(cfg, prices)
    log(f"charts: {len(manifest)} images")
    return manifest


def _lstm_ready_xy(frames, cols, idx_map, lookback):
    return sequences_by_symbol(frames, cols, idx_map, lookback)


def stage_holdout(cfg: dict, df: pd.DataFrame, manifest: pd.DataFrame | None):
    """Train/calibrate on first (1-calib-test), evaluate final test_fraction slice."""
    log("holdout: building split & training models …")
    frames = per_symbol_frames(df)
    cols = feature_columns(df)
    n_min = min(len(g) for g in frames.values())
    tr_i, ca_i, te_i = chronological_splits(
        n_min, cfg["split"]["test_fraction"], cfg["split"]["calib_fraction"])
    log(f"  per-symbol rows≈{n_min}: train={len(tr_i)} calib={len(ca_i)} test={len(te_i)}")

    train_idx = {s: tr_i for s in frames}
    calib_idx = {s: ca_i for s in frames}
    test_idx = {s: te_i for s in frames}

    def stack(idx_map, need_seq=False):
        Xs, ys, syms, dates = [], [], [], []
        row = 0
        for s, g in frames.items():
            sub = g.iloc[idx_map[s]]
            Xs.append(sub[cols].to_numpy(np.float32))
            ys.append(sub["label"].to_numpy(np.int64))
            for _, r in sub.iterrows():
                syms.append(s)
                dates.append(str(pd.Timestamp(r["date"]).date()))
        return np.vstack(Xs), np.concatenate(ys), syms, dates

    Xtr, ytr, _, _ = stack(train_idx)
    Xca, yca, syms_ca, dates_ca = stack(calib_idx)
    Xte, yte, syms_te, dates_te = stack(test_idx)

    lookback = cfg["features"]["lookback_lstm"]
    X_seq_tr, y_seq_tr, meta_tr = _lstm_ready_xy(frames, cols, train_idx, lookback)
    seq_row = None
    if len(meta_tr):
        base = 0
        seq_row = np.empty(len(meta_tr), dtype=int)
        for s, g in frames.items():
            mask = (meta_tr["symbol"] == s).to_numpy()
            seq_row[mask] = base + meta_tr.loc[mask, "pos"].to_numpy()
            base += len(tr_i)

    ens = TabularEnsemble(cfg, n_features=len(cols), with_lstm=True)
    ens.fit(Xtr, ytr, X_seq=X_seq_tr, y_seq=y_seq_tr, seq_index_in_X=seq_row)
    ens.save(Path(cfg["paths"]["models"]) / "tabular_ensemble.joblib")
    log("holdout: tabular ensemble trained")

    # --- vision (isolated subprocess: TF+torch share unstable in-process) ----
    p_vit_ca = p_vit_te = None
    vit_ok = False
    if manifest is not None and len(manifest):
        from optionedge.vit_runner import run_vit_worker
        train_keys, val_keys = set(), set()
        eval_keys = set()
        for s, g in frames.items():
            dts = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").tolist()
            cut = int(tr_i[-1]) + 1
            for j, d in enumerate(dts):
                if j < cut:
                    if j >= int(cut * 0.9):
                        val_keys.add((s, d))
                    else:
                        train_keys.add((s, d))
                    eval_keys.add((s, d))          # fusion-head needs train probs
                elif j in set(ca_i.tolist()):
                    eval_keys.add((s, d))
                elif j in set(te_i.tolist()):
                    eval_keys.add((s, d))
        labels = df[["symbol", "date", "label"]].copy()
        labels["date"] = labels["date"].astype(str).str[:10]
        try:
            vit_path = Path(cfg["paths"]["models"]) / "vit_final.pt"
            probs_path = Path(cfg["paths"]["models"]) / "vit_final_probs.parquet"
            log(f"holdout: training ViT on {len(train_keys)} images (subprocess) …")
            probs = run_vit_worker(manifest, labels, train_keys, val_keys, eval_keys,
                                   vit_path, probs_path, epochs=cfg["vision"]["epochs"],
                                   tmp_dir=Path(cfg["paths"]["models"]) / "_vit_holdout")
            if probs is not None:
                vit_ok = True
                log(f"holdout: ViT done — {len(probs)} probs")
                lut = {(r.symbol, r.date): float(r.p_up_vit) for r in probs.itertuples()}
                p_vit_ca = np.array([lut.get((s, d), 0.5) for s, d in zip(syms_ca, dates_ca)])
                p_vit_te = np.array([lut.get((s, d), 0.5) for s, d in zip(syms_te, dates_te)])
            else:
                log("holdout: ViT worker failed — vision branch disabled")
        except Exception as e:
            log(f"holdout: ViT failed: {e}")

    # --- probabilities ------------------------------------------------------
    p_tab_ca = ens.predict_proba(Xca, X_seq=None, seq_index_in_X=None)
    p_tab_te = ens.predict_proba(Xte, X_seq=None, seq_index_in_X=None)
    if p_vit_te is None:
        p_vit_ca = np.full(len(p_tab_ca), 0.5)
        p_vit_te = np.full(len(p_tab_te), 0.5)

    # fusion fit: OOF tabular probs on TRAIN + ViT train probs
    oof_tab = getattr(ens, "oof_p_", None)
    if oof_tab is None or len(oof_tab) != len(ytr):
        oof_tab = np.full(len(ytr), float(np.mean(ytr)))
    # ViT probabilities on train images already computed by the worker
    p_vit_tr = np.full(len(ytr), 0.5)
    if vit_ok and 'probs' in dir() and probs is not None:
        keys = []
        for s, g in frames.items():
            dts = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d").tolist()
            for j in range(len(tr_i)):
                keys.append((s, dts[j]))
        lut = {(r.symbol, r.date): float(r.p_up_vit) for r in probs.itertuples()}
        p_vit_tr = np.array([lut.get(k, 0.5) for k in keys], dtype=float)

    fusion = FusionModel(use_vit=vit_ok)
    fusion.fit(np.asarray(oof_tab, float), p_vit_tr, ytr)
    fusion.save(Path(cfg["paths"]["models"]) / "fusion.joblib")

    p_fus_ca = fusion.predict_proba(p_tab_ca, p_vit_ca if fusion.use_vit else None)
    p_fus_te = fusion.predict_proba(p_tab_te, p_vit_te if fusion.use_vit else None)

    # threshold calibration on calibration slice (fused)
    thr = tune_threshold(p_fus_ca, yca, target_coverage=0.5)
    hi = thr.get("hi", cfg["signals"]["confidence_hi"])
    lo = thr.get("lo", cfg["signals"]["confidence_lo"])
    hi = min(max(hi, 0.51), 0.75)
    lo = max(min(lo, 0.49), 0.25)
    log(f"holdout: thresholds hi={hi:.3f} lo={lo:.3f} calib_cov={thr.get('coverage')}")

    results = {
        "tabular": {**classification_metrics(yte, p_tab_te),
                    "modality": "options-chain + technicals", "validation": "holdout 20%"},
        "vision_vit": {**classification_metrics(yte, p_vit_te),
                       "modality": "candlestick ViT", "validation": "holdout 20%"},
        "fused": {**classification_metrics(yte, p_fus_te),
                  "modality": "ViT + tabular ensemble", "validation": "holdout 20%"},
        "fused_selective": selective_metrics(yte, p_fus_te, hi, lo),
        "thresholds": {"hi": float(hi), "lo": float(lo)},
    }
    preds = pd.DataFrame({
        "symbol": syms_te, "date": dates_te, "y": yte,
        "p_tab": p_tab_te, "p_vit": p_vit_te, "p_fused": p_fus_te,
    })
    for name, res in [("tabular", p_tab_te), ("vision_vit", p_vit_te),
                      ("fused", p_fus_te)]:
        log(f"holdout {name}: acc={results[name]['accuracy']:.4f} "
            f"f1={results[name]['f1']:.4f} auc={results[name].get('roc_auc', float('nan')):.4f}")

    # latest-day signals from fused model on final row of each symbol
    latest = []
    last_idx = {s: np.array([len(frames[s]) - 1]) for s in frames}
    # include final calib/test row predictions: simplest — refit predict on last rows
    # We already have test preds; last test row per symbol:
    for s in frames:
        sub = preds[preds["symbol"] == s].sort_values("date")
        if len(sub):
            r = sub.iloc[-1]
            sig = int(np.clip(np.round(probabilities_to_signals(
                np.array([r["p_fused"]]), hi, lo)[0]), -1, 1))
            from optionedge.signals import SIGNAL_MAP
            latest.append({
                "symbol": s, "date": r["date"],
                "p_up_fused": float(r["p_fused"]),
                "p_up_tabular": float(r["p_tab"]),
                "p_up_vit": float(r["p_vit"]),
                "signal": SIGNAL_MAP[sig],
                "confidence": float(max(r["p_fused"], 1 - r["p_fused"])),
                "thresholds": {"hi": float(hi), "lo": float(lo)},
            })

    out = {
        "results": results,
        "predictions": preds,
        "latest_signals": latest,
        "ensemble": ens,
        "fusion": fusion,
    }
    save_metrics(results, Path(cfg["paths"]["metrics"]) / "holdout_results.json")
    preds.to_parquet(Path(cfg["paths"]["metrics"]) / "holdout_predictions.parquet",
                     index=False)
    pd.DataFrame(latest).to_json(Path(cfg["paths"]["metrics"]) / "latest_signals.json",
                                 orient="records", indent=2)
    return out


def stage_walkforward(cfg: dict, df: pd.DataFrame, manifest, fast: bool):
    from optionedge.walkforward import run_walkforward
    if fast:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg["split"]["walkforward_folds"] = 2
        cfg["split"]["walkforward_test_days"] = 40
        cfg["vision"]["epochs"] = 3
        cfg["tabular"]["lstm"]["epochs"] = 5
        cfg["tabular"]["rf"]["n_estimators"] = 120
        cfg["tabular"]["xgb"]["n_estimators"] = 120
    log("walkforward: running time-ordered CV …")
    preds = run_walkforward(cfg, df, manifest, Path(cfg["paths"]["artifacts"]))
    metrics = {
        "tabular": {**classification_metrics(preds["y"], preds["p_tab"]),
                    "modality": "options-chain + technicals",
                    "validation": "walk-forward (refit each fold)"},
        "vision_vit": {**classification_metrics(preds["y"], preds["p_vit"]),
                       "modality": "candlestick ViT",
                       "validation": "walk-forward (refit each fold)"},
        "fused": {**classification_metrics(preds["y"], preds["p_fused"]),
                  "modality": "ViT + tabular ensemble",
                  "validation": "walk-forward (refit each fold)"},
        "fused_selective": selective_metrics(
            preds["y"], preds["p_fused"],
            cfg["signals"]["confidence_hi"], cfg["signals"]["confidence_lo"]),
    }
    save_metrics(metrics, Path(cfg["paths"]["metrics"]) / "walkforward_results.json")
    preds.to_parquet(Path(cfg["paths"]["metrics"]) / "walkforward_predictions.parquet",
                     index=False)
    for k in ("tabular", "vision_vit", "fused"):
        log(f"walkforward {k}: acc={metrics[k]['accuracy']:.4f} "
            f"f1={metrics[k]['f1']:.4f} auc={metrics[k].get('roc_auc', float('nan')):.4f}")
    return preds, metrics


def stage_backtest(cfg: dict, holdout, prices, chain):
    log("backtest: simulating signals on holdout …")
    preds = holdout["predictions"]
    hi = holdout["results"]["thresholds"]["hi"]
    lo = holdout["results"]["thresholds"]["lo"]
    sig = probabilities_to_signals(preds["p_fused"].to_numpy(), hi, lo)
    sig_df = preds.copy()
    sig_df["p_up"] = sig_df["p_fused"]
    sig_df["signal"] = sig
    sig_df["confidence"] = np.where(sig == 0, 0.5,
                                    np.where(sig == 1, preds["p_fused"],
                                             1 - preds["p_fused"]))
    # restrict prices/chain to holdout dates (need one day BEFORE first signal for entry)
    d0 = pd.to_datetime(preds["date"]).min() - pd.Timedelta(days=7)
    d1 = pd.to_datetime(preds["date"]).max() + pd.Timedelta(days=7)
    pr = prices[(prices["date"] >= d0) & (prices["date"] <= d1)].copy()
    ch = chain[(chain["date"] >= d0) & (chain["date"] <= d1)].copy()
    res = run_backtest(cfg, sig_df, pr, ch)
    stats = res["stats"]
    log(f"backtest: trades={stats.get('n_trades')} "
        f"win_rate={stats.get('win_rate', float('nan')):.3f} "
        f"total_points={stats.get('total_net_points', float('nan')):.1f}")
    if len(res["trades"]):
        res["trades"].to_csv(Path(cfg["paths"]["metrics"]) / "backtest_trades.csv",
                             index=False)
    save_metrics(stats, Path(cfg["paths"]["metrics"]) / "backtest_stats.json")
    return res


def stage_reports(cfg, holdout, wf_preds, wf_metrics, bt, prices):
    log("reports: plots + benchmark table + sqlite …")
    plots = plot_equity_and_confusion(
        wf_preds if wf_preds is not None else holdout["predictions"],
        {"trades": bt["trades"]},
        cfg["paths"]["plots"])
    # ablation table
    abl = benchmark_table({
        "Tabular holdout": holdout["results"]["tabular"],
        "ViT holdout": holdout["results"]["vision_vit"],
        "Fused holdout": holdout["results"]["fused"],
        "Tabular walk-forward": wf_metrics["tabular"],
        "ViT walk-forward": wf_metrics["vision_vit"],
        "Fused walk-forward": wf_metrics["fused"],
    })
    abl.to_csv(Path(cfg["paths"]["metrics"]) / "benchmark_table.csv", index=False)
    log(abl.to_string(index=False))

    try:
        from optionedge.db import store_all
        store_all(cfg, prices, pd.read_parquet(
            Path(cfg["data"]["processed_dir"]) / "features.parquet"),
            wf_preds, holdout["predictions"], bt["trades"],
            {"holdout_fused_acc": holdout["results"]["fused"]["accuracy"],
             "wf_fused_acc": wf_metrics["fused"]["accuracy"],
             "bt_win_rate": bt["stats"].get("win_rate")})
        log("reports: sqlite written")
    except Exception as e:
        log(f"reports: sqlite skipped ({e})")
    return plots


# ---------------------------------------------------------------------------

STAGES = ["ingest", "features", "charts", "holdout", "walkforward", "backtest", "reports"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small smoke-test config")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--skip-vit", action="store_true", help="skip vision branch")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    seed_everything(cfg["random_seed"])
    cfg["_fast"] = args.fast
    if args.fast:
        cfg["vision"]["epochs"] = min(cfg["vision"]["epochs"], 4)
        cfg["tabular"]["lstm"]["epochs"] = min(cfg["tabular"]["lstm"]["epochs"], 6)
        cfg["split"]["walkforward_folds"] = 2
        cfg["split"]["walkforward_test_days"] = 40

    want = STAGES if args.stage == "all" else [args.stage]
    state = {}

    def need(s):
        return s in want or want == ["all"]

    # Persistent local state between --stage runs
    state_file = Path(cfg["paths"]["artifacts"]) / "_pipeline_state.json"

    if "ingest" in want or args.stage == "all":
        cfg, prices, chain = stage_ingest(cfg, args.fast)
        state["_prices"], state["_chain"] = prices, chain
    else:
        proc = Path(cfg["data"]["processed_dir"])
        if (proc / "prices.parquet").exists():
            state["_prices"] = pd.read_parquet(proc / "prices.parquet")
            state["_chain"] = pd.read_parquet(proc / "fo_chain.parquet")

    if "features" in want:
        df = stage_features(cfg, state["_prices"], state["_chain"])
        state["df"] = df
    elif (Path(cfg["data"]["processed_dir"]) / "features.parquet").exists():
        state["df"] = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "features.parquet")

    if "charts" in want:
        manifest = stage_charts(cfg, state["_prices"])
        state["manifest"] = manifest
    else:
        mpath = Path(cfg["data"]["charts_dir"]) / "manifest.csv"
        state["manifest"] = pd.read_csv(mpath) if mpath.exists() else None

    if args.skip_vit:
        state["manifest"] = None

    holdout = None
    if "holdout" in want:
        holdout = stage_holdout(cfg, state["df"], state["manifest"])
        state["holdout"] = holdout
    else:
        # backtest needs holdout predictions — recompute quickly? require stage
        pass

    wf_preds = wf_metrics = None
    if "walkforward" in want:
        wf_preds, wf_metrics = stage_walkforward(cfg, state["df"],
                                                 state["manifest"], args.fast)
    else:
        wpath = Path(cfg["paths"]["metrics"]) / "walkforward_results.json"
        if wpath.exists():
            wf_metrics = json.load(open(wpath))
            pp = Path(cfg["paths"]["metrics"]) / "walkforward_predictions.parquet"
            wf_preds = pd.read_parquet(pp) if pp.exists() else None

    bt = None
    if "backtest" in want:
        if holdout is None:
            hp = Path(cfg["paths"]["metrics"]) / "holdout_predictions.parquet"
            hr = Path(cfg["paths"]["metrics"]) / "holdout_results.json"
            if hp.exists() and hr.exists():
                res = json.load(open(hr))
                holdout = {"predictions": pd.read_parquet(hp), "results": res}
                log("backtest: loaded holdout predictions from disk")
            else:
                raise SystemExit("--stage backtest requires holdout stage first "
                                 "(or run --stage all)")
        bt = stage_backtest(cfg, holdout, state["_prices"], state["_chain"])
    else:
        tpath = Path(cfg["paths"]["metrics"]) / "backtest_trades.csv"
        if tpath.exists():
            bt = {"trades": pd.read_csv(tpath),
                  "stats": json.load(open(Path(cfg["paths"]["metrics"]) / "backtest_stats.json"))}

    if "reports" in want:
        if holdout is None:
            hp = Path(cfg["paths"]["metrics"]) / "holdout_predictions.parquet"
            hr = Path(cfg["paths"]["metrics"]) / "holdout_results.json"
            if hp.exists() and hr.exists():
                holdout = {"predictions": pd.read_parquet(hp),
                           "results": json.load(open(hr))}
        if holdout is not None and wf_metrics is not None and bt is not None:
            stage_reports(cfg, holdout, wf_preds, wf_metrics, bt, state["_prices"])
        else:
            log(f"reports: skipped (holdout={holdout is not None}, "
                f"wf={wf_metrics is not None}, bt={bt is not None})")

    log("pipeline done ✔")


if __name__ == "__main__":
    main()

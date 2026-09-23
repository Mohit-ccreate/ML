"""Tabular branch: Random Forest, XGBoost, LSTM + stacking ensemble."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Individual models
# ---------------------------------------------------------------------------

def build_rf(cfg: dict) -> RandomForestClassifier:
    p = cfg["tabular"]["rf"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], class_weight="balanced_subsample",
        random_state=cfg["random_seed"], n_jobs=2)


def build_xgb(cfg: dict) -> XGBClassifier:
    p = cfg["tabular"]["xgb"]
    return XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
        reg_lambda=p["reg_lambda"], objective="binary:logistic",
        eval_metric="logloss", random_state=cfg["random_seed"],
        n_jobs=2, tree_method="hist")


def build_lstm(cfg: dict, n_features: int):
    import tensorflow as tf
    from tensorflow import keras
    p = cfg["tabular"]["lstm"]
    model = keras.Sequential([
        keras.layers.Input(shape=(cfg["features"]["lookback_lstm"], n_features)),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.LSTM(p["units"], return_sequences=False),
        keras.layers.Dropout(p["dropout"]),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(p["lr"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int,
                   positions: np.ndarray | None = None):
    """Causal sequences: rows [i-lookback, i] predict label at i.

    positions: original positional ids per row — sequences never cross
    symbol boundaries upstream (caller passes contiguous symbol blocks).
    """
    Xs, ys, idxs = [], [], []
    n = len(X)
    for i in range(lookback, n):
        Xs.append(X[i - lookback:i + 1])
        ys.append(y[i])
        idxs.append(positions[i] if positions is not None else i)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(idxs)


def sequences_by_symbol(frames: dict[str, pd.DataFrame], cols: list[str],
                         idx_map: dict[str, np.ndarray], lookback: int):
    """Build LSTM sequences per symbol then concat; returns X_seq, y, meta positions.

    meta rows align with returned sequences (symbol, date, pos).
    """
    X_seq, ys, meta = [], [], []
    for sym, g in frames.items():
        idx = idx_map[sym]
        sub = g.iloc[idx]
        X = sub[cols].to_numpy(np.float32)
        y = sub["label"].to_numpy(np.int64)
        dates = sub["date"].to_numpy()
        Xs, ys_, ps = make_sequences(X, y, lookback, positions=np.arange(len(sub)))
        X_seq.append(Xs)
        ys.append(ys_)
        for p_ in ps:
            meta.append({"symbol": sym, "pos": int(p_), "date": dates[p_]})
    if not X_seq:
        return (np.zeros((0, lookback + 1, len(cols)), np.float32),
                np.zeros((0,), np.int64), pd.DataFrame(columns=["symbol", "pos", "date"]))
    return np.concatenate(X_seq), np.concatenate(ys), pd.DataFrame(meta)


# ---------------------------------------------------------------------------
# Stacking ensemble over RF / XGB / LSTM probability outputs
# ---------------------------------------------------------------------------

class TabularEnsemble:
    """Soft-stacking of RF + XGB (+ optional LSTM) with a logistic meta-learner.

    OOF meta-features for the stacker are produced with TimeSeriesSplit inside
    the training window only (no leakage into validation/test).
    """

    def __init__(self, cfg: dict, n_features: int, with_lstm: bool = True):
        self.cfg = cfg
        self.n_features = n_features
        self.with_lstm = with_lstm
        self.rf = build_rf(cfg)
        self.xgb = build_xgb(cfg)
        self.lstm = None
        self.scaler = StandardScaler()
        self.meta: LogisticRegression | None = None
        self.feature_means_: np.ndarray | None = None
        self.lstm_ready = False

    # -- helpers ------------------------------------------------------------
    def _fit_lstm(self, X_seq_train, y_train, X_seq_val, y_val, sample_weight=None):
        p = self.cfg["tabular"]["lstm"]
        # scale using flattened stats
        flat = X_seq_train.reshape(-1, X_seq_train.shape[-1])
        self.scaler.fit(flat)
        Xtr = self.scaler.transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])) \
            .reshape(X_seq_train.shape)
        Xva = self.scaler.transform(X_seq_val.reshape(-1, X_seq_val.shape[-1])) \
            .reshape(X_seq_val.shape) if len(X_seq_val) else None
        self.lstm = build_lstm(self.cfg, self.n_features)
        sw = sample_weight
        val_data = (Xva, y_val) if Xva is not None and len(y_val) else None
        self.lstm.fit(Xtr, y_train, sample_weight=sw,
                      epochs=p["epochs"], batch_size=p["batch_size"],
                      validation_data=val_data, verbose=0,
                      shuffle=True)
        self.lstm_ready = True
        # free graph/memory before the vision branch runs (small machines)
        try:
            import tensorflow as tf, gc
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass

    def _base_probs_tabular(self, X: np.ndarray, y: np.ndarray,
                            fit: bool = True) -> np.ndarray:
        """Return nx3 probabilities [rf, xgb, lr-baseline] (fast bases)."""
        if fit:
            self.rf.fit(X, y)
            self.xgb.fit(X, y)
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        return np.column_stack([p_rf, p_xgb])

    def fit(self, X, y, X_seq=None, y_seq=None, seq_index_in_X: np.ndarray | None = None):
        """Full fit.

        X, y          : tabular rows (one walk-forward training window, ordered)
        X_seq, y_seq  : optional LSTM sequences aligned to a subset of rows;
                        seq_index_in_X maps each sequence to its row in X.
        Stores OOF base probabilities in ``self.oof_p_`` for the fusion head.
        """
        # median imputation stats (LSTM / scaler cannot ingest NaN)
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self.feature_medians_ = med
        X = np.where(np.isfinite(X), X, med[None, :])
        if X_seq is not None and len(X_seq):
            X_seq = np.asarray(X_seq, dtype=np.float64)
            X_seq = np.where(np.isfinite(X_seq),
                             X_seq, np.broadcast_to(med, X_seq.shape))
        # OOF predictions for tabular bases via TimeSeriesSplit
        n = len(y)
        oof_rf = np.full(n, np.nan)
        oof_xgb = np.full(n, np.nan)
        k = 4 if n > 400 else max(2, min(3, n // 50))
        tscv = TimeSeriesSplit(n_splits=k)
        for tr_i, va_i in tscv.split(X):
            rf = build_rf(self.cfg)
            xb = build_xgb(self.cfg)
            rf.fit(X[tr_i], y[tr_i])
            xb.fit(X[tr_i], y[tr_i])
            oof_rf[va_i] = rf.predict_proba(X[va_i])[:, 1]
            oof_xgb[va_i] = xb.predict_proba(X[va_i])[:, 1]
        # In-sample fill for first fold rows that CV never validated (use refit models later)
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        nan_mask = np.isnan(oof_rf)
        if nan_mask.any():
            oof_rf[nan_mask] = self.rf.predict_proba(X[nan_mask])[:, 1]
            oof_xgb[nan_mask] = self.xgb.predict_proba(X[nan_mask])[:, 1]

        oof_lstm = None
        if self.with_lstm and X_seq is not None and len(X_seq) > 50:
            # chronological split for LSTM val + OOF approximation:
            # fit LSTM on first 85% of sequences, predict remaining 15%;
            # rows before that get LSTM's own training-window probabilities
            # (meta-learner still never sees test rows).
            m = len(X_seq)
            cut = int(m * 0.85)
            # scale/fit
            flat = X_seq[:cut].reshape(-1, X_seq.shape[-1])
            self.scaler.fit(flat)
            Xtr = self.scaler.transform(X_seq[:cut].reshape(-1, X_seq.shape[-1])).reshape(X_seq[:cut].shape)
            Xall = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
            self.lstm = build_lstm(self.cfg, self.n_features)
            p = self.cfg["tabular"]["lstm"]
            self.lstm.fit(Xtr, y_seq[:cut], epochs=p["epochs"], batch_size=p["batch_size"],
                          validation_split=0.1, verbose=0, shuffle=True)
            p_all = self.lstm.predict(Xall, verbose=0).ravel()
            oof_lstm_rows = np.full(len(y), np.nan)
            if seq_index_in_X is not None:
                oof_lstm_rows[seq_index_in_X[:cut]] = p_all[:cut]      # in-window fit rows
                oof_lstm_rows[seq_index_in_X[cut:]] = p_all[cut:]      # held-out seq → true OOF
            else:
                oof_lstm_rows = p_all  # fallback (only when aligned 1:1)
            oof_lstm = oof_lstm_rows
            self.lstm_ready = True

        # meta design matrix
        M = [oof_rf, oof_xgb]
        if oof_lstm is not None and not np.isnan(oof_lstm).any():
            M.append(oof_lstm)
        M = np.column_stack(M)
        self.meta = LogisticRegression(C=1.0, max_iter=1000)
        self.meta.fit(M, y)
        # OOF *tabular ensemble* probability (what the fusion head consumes as p_tab)
        self.oof_p_ = self.meta.predict_proba(M)[:, 1]
        self.n_meta_features_ = M.shape[1]
        return self

    def predict_proba(self, X, X_seq=None, seq_index_in_X: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if hasattr(self, "feature_medians_") and self.feature_medians_ is not None:
            med = self.feature_medians_
            if X.shape[1] == len(med):
                X = np.where(np.isfinite(X), X, med[None, :])
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        cols = [p_rf, p_xgb]
        if self.lstm_ready and self.lstm is not None and X_seq is not None and len(X_seq):
            X_seq = np.asarray(X_seq, dtype=np.float64)
            if hasattr(self, "feature_medians_") and self.feature_medians_ is not None \
                    and X_seq.shape[-1] == len(self.feature_medians_):
                med = self.feature_medians_
                X_seq = np.where(np.isfinite(X_seq), X_seq, np.broadcast_to(med, X_seq.shape))
            Xs = self.scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
            p_lstm = self.lstm.predict(Xs, verbose=0).ravel()
            full = np.full(len(X), np.nan)
            full[seq_index_in_X] = p_lstm
            if np.isnan(full).any():
                # rows without sequence (warmup): impute with mean of available
                full[np.isnan(full)] = np.nanmean(full)
            cols.append(full)
        M = np.column_stack(cols)
        if M.shape[1] < self.n_meta_features_:
            # pad (no lstm available at predict-time) with mean column values
            while M.shape[1] < self.n_meta_features_:
                M = np.column_stack([M, np.nanmean(M, axis=1)])
        return self.meta.predict_proba(M)[:, 1]

    def base_predict_proba(self, X) -> dict[str, np.ndarray]:
        out = {"rf": self.rf.predict_proba(X)[:, 1],
               "xgb": self.xgb.predict_proba(X)[:, 1]}
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TabularEnsemble":
        return joblib.load(path)

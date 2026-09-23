"""Signal generation: Buy CE / Buy PE / No Trade."""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_MAP = {1: "BUY_CE", -1: "BUY_PE", 0: "NO_TRADE"}


def probabilities_to_signals(p_up: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Return int array in {1 (CE), -1 (PE), 0 (flat)}."""
    p_up = np.asarray(p_up, dtype=float)
    sig = np.zeros(len(p_up), dtype=int)
    sig[p_up >= hi] = 1
    sig[p_up <= lo] = -1
    return sig


def signals_frame(dates, symbols, p_up, p_tab, p_vit, hi, lo) -> pd.DataFrame:
    sig = probabilities_to_signals(p_up, hi, lo)
    return pd.DataFrame({
        "date": dates,
        "symbol": symbols,
        "p_up": p_up,
        "p_up_tabular": p_tab,
        "p_up_vit": p_vit,
        "signal": sig,
        "signal_label": [SIGNAL_MAP[s] for s in sig],
        "confidence": np.where(sig == 0, 0.5, np.where(sig == 1, p_up, 1 - p_up)),
    })

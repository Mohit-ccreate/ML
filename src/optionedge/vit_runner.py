"""Helpers to run the ViT worker in an isolated subprocess (TF/torch safety)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "scripts" / "vit_worker.py"


def run_vit_worker(manifest: pd.DataFrame, labels: pd.DataFrame,
                   train_keys: set, val_keys: set, eval_keys: set,
                   model_out: Path, probs_out: Path,
                   epochs: int | None = None,
                   tmp_dir: Path | None = None,
                   verbose: bool = True) -> pd.DataFrame | None:
    """Fit ViT in a subprocess; return probs DataFrame (symbol, date, p_up_vit)."""
    tmp_dir = Path(tmp_dir or (model_out.parent / f"_vit_tmp_{model_out.stem}"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    m_path = tmp_dir / "manifest.csv"
    l_path = tmp_dir / "labels.parquet"
    tr_path = tmp_dir / "train_keys.parquet"
    va_path = tmp_dir / "val_keys.parquet"
    ev_path = tmp_dir / "eval_keys.parquet"

    manifest.to_csv(m_path, index=False)
    labels.to_parquet(l_path, index=False)
    pd.DataFrame(sorted(train_keys), columns=["symbol", "date"]).to_parquet(tr_path, index=False)
    pd.DataFrame(sorted(val_keys), columns=["symbol", "date"]).to_parquet(va_path, index=False)
    pd.DataFrame(sorted(eval_keys), columns=["symbol", "date"]).to_parquet(ev_path, index=False)

    cmd = [sys.executable, str(WORKER),
           "--manifest", str(m_path), "--labels", str(l_path),
           "--train-keys", str(tr_path), "--val-keys", str(va_path),
           "--eval-keys", str(ev_path),
           "--model-out", str(model_out), "--probs-out", str(probs_out)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    env = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PATH": "/usr/local/bin:/usr/bin:/bin",
           "HOME": str(Path.home()),
           "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(REPO), timeout=3600)
    if verbose:
        tail = (proc.stdout or "").strip().splitlines()
        if tail:
            print("    [vit-worker]", tail[-1][:300])
        if proc.returncode != 0:
            print("    [vit-worker] FAILED rc=", proc.returncode)
            print(proc.stderr[-1500:] if proc.stderr else "")
    if proc.returncode != 0 or not probs_out.exists():
        return None
    return pd.read_parquet(probs_out)

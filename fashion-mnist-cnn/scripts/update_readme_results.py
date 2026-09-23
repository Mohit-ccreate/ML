#!/usr/bin/env python
"""Regenerate the results table in README.md from run artifacts.

Reads runs/<name>/history.json and runs/<name>/test_metrics.json (produced by
train.py and evaluate.py) and rewrites the block between the
``<!-- RESULTS:BEGIN -->`` / ``<!-- RESULTS:END -->`` markers.

    python scripts/update_readme_results.py            # cnn + mlp
    python scripts/update_readme_results.py cnn mlp resnet
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
README = HERE / "README.md"
LABELS = {"cnn": "`cnn` (SimpleCNN)", "mlp": "`mlp` (baseline)"}


def row(run: str) -> str:
    run_dir = HERE / "runs" / run
    hist_path, test_path = run_dir / "history.json", run_dir / "test_metrics.json"
    label = LABELS.get(run, f"`{run}`")
    if not hist_path.exists():
        return f"| {label} | — | — | — | — | — | — |"
    hist = json.loads(hist_path.read_text())
    best = max(hist, key=lambda h: h["val_acc"])
    minutes = sum(h["seconds"] for h in hist) / 60
    params = "—"
    ckpt_cfg = run_dir / "config.json"
    if ckpt_cfg.exists():
        params = {"cnn": "584k", "mlp": "535k"}.get(json.loads(ckpt_cfg.read_text())["model"]["name"], "—")
    test_acc = f1 = "—"
    if test_path.exists():
        tm = json.loads(test_path.read_text())
        test_acc, f1 = f"**{100 * tm['test_accuracy']:.2f} %**", f"{tm['macro_f1']:.4f}"
    return (
        f"| {label} | {params} | {len(hist)} | {100 * best['val_acc']:.2f} % | "
        f"{test_acc} | {f1} | {minutes:.0f} min |"
    )


def main(runs: list[str]) -> None:
    header = (
        "| Model | Params | Epochs | Val acc | **Test acc** | Macro F1 | Train time (2-core CPU) |\n"
        "|-------|-------:|-------:|--------:|-------------:|---------:|------------------------:|\n"
    )
    table = header + "\n".join(row(r) for r in runs)
    text = README.read_text()
    new = re.sub(
        r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
        f"<!-- RESULTS:BEGIN -->\n{table}\n<!-- RESULTS:END -->",
        text,
        flags=re.S,
    )
    README.write_text(new)
    print(table)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cnn", "mlp"])

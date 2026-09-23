# ML

Hands-on machine-learning projects, each self-contained in its own folder
with its own README, requirements and tests.

| Project | Task | Stack | Result |
|---------|------|-------|--------|
| [`fashion-mnist-cnn/`](fashion-mnist-cnn/) | Image classification (10 clothing classes, 28×28 grayscale) | PyTorch, torchvision, scikit-learn | see project README |

## Conventions

- One folder per project; reusable code lives in an importable package
  (`<project>/<pkg>/`), thin CLI scripts on top (`train.py`, `evaluate.py`, …).
- Hyper-parameters live in a YAML config; every run writes its resolved config
  and metrics next to its checkpoints.
- Datasets (`data/`) and training runs (`runs/`) are git-ignored — each project
  README explains how to fetch data and reproduce the numbers. Small plots
  worth keeping are copied to `<project>/assets/`.
- `python -m pytest` inside a project runs its test suite.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r fashion-mnist-cnn/requirements.txt
```

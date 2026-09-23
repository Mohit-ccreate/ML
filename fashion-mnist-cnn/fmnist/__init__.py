"""fmnist — a compact, end-to-end PyTorch image-classification project on Fashion-MNIST.

Modules
-------
data        : datasets, transforms, train/val/test DataLoaders
model       : SimpleCNN (main model) and an MLP baseline
engine      : train_one_epoch / evaluate loops
utils       : seeding, device selection, config + checkpoint helpers
visualize   : training curves, confusion matrix, sample grids
"""

__version__ = "0.1.0"

from .data import CLASSES, MEAN, STD, get_dataloaders  # noqa: F401
from .model import build_model, count_parameters  # noqa: F401

"""Training and evaluation helpers for the Potts model package."""

from utils.train.train_val import train_val_split, train_val_split_indices


def plot_training_curves(*args, **kwargs):
    """Lazily import and display LBFGS training curves."""
    from utils.train.diagnostics import plot_training_curves as _plot_training_curves

    return _plot_training_curves(*args, **kwargs)


__all__ = [
    "plot_training_curves",
    "train_val_split",
    "train_val_split_indices",
]

from __future__ import annotations

from typing import Sequence

import numpy as np


def train_val_split_indices(
    n_items: int,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shuffled train/validation indices for ``n_items`` rows."""
    if n_items < 2:
        raise ValueError("At least two items are required for a train/validation split.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)

    n_val = max(1, int(round(n_items * val_frac)))
    n_val = min(n_val, n_items - 1)

    return idx[n_val:], idx[:n_val]


def train_val_split(
    seqs: Sequence[str],
    weights: np.ndarray,
    val_frac: float = 0.1,
    seed: int = 0,
    return_indices: bool = False,
):
    """Split sequences and weights into train/validation subsets.

    The default return value preserves the original four-item API. Pass
    ``return_indices=True`` to also receive the train and validation indices.
    """
    if len(seqs) != len(weights):
        raise ValueError("seqs and weights must have the same length.")

    train_idx, val_idx = train_val_split_indices(len(seqs), val_frac=val_frac, seed=seed)
    train_seqs = [seqs[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    train_weights = weights[train_idx]
    val_weights = weights[val_idx]

    if return_indices:
        return train_seqs, train_weights, val_seqs, val_weights, train_idx, val_idx
    return train_seqs, train_weights, val_seqs, val_weights

"""Shared amino-acid alphabet, encoding, and decoding helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# Compatibility aliases used by older scripts.
ALPHABET = AA_ALPHABET
IDX_toAA = IDX_TO_AA
AA_toIDX = AA_TO_IDX


def encode_seqs(seqs: list[str], L: Optional[int] = None) -> torch.LongTensor:
    """
    Encode amino-acid sequences into shape ``(N, L)`` int64 tensors.

    Encoded values are indices into :data:`AA_ALPHABET`.
    """
    if len(seqs) == 0:
        raise ValueError("No sequences provided.")

    if L is None:
        L = len(seqs[0])

    for seq in seqs:
        if len(seq) != L:
            raise ValueError("All sequences must have the same length.")
        for aa in seq:
            if aa not in AA_TO_IDX:
                raise ValueError(f"Invalid amino acid {aa!r} in sequence {seq!r}.")

    arr = np.array([[AA_TO_IDX[aa] for aa in seq] for seq in seqs], dtype=np.int64)
    return torch.from_numpy(arr)


def flatten_samples(samples: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert sampled sequences to a 2D NumPy array of shape ``(M, L)``.

    Accepts arrays/tensors with shape ``(L,)``, ``(N, L)``, or any sample/chain
    prefix ending in ``L``, e.g. sampler output with shape
    ``(num_samples, batch_size, L)``.
    """
    if torch.is_tensor(samples):
        arr = samples.detach().cpu().numpy()
    else:
        arr = np.asarray(samples)

    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim < 1:
        raise ValueError("samples must have at least one dimension")
    return arr.reshape(-1, arr.shape[-1])


def decode_sequences(
    samples: np.ndarray | torch.Tensor,
    alphabet: str = AA_ALPHABET,
) -> list[str]:
    """Decode integer-encoded samples to amino-acid strings."""
    X = flatten_samples(samples)
    if not np.issubdtype(X.dtype, np.integer):
        return ["".join(row.astype(str).tolist()) for row in X]

    q = len(alphabet)
    if X.size and (X.min() < 0 or X.max() >= q):
        raise ValueError(f"encoded samples contain states outside [0, {q - 1}]")
    return ["".join(alphabet[int(idx)] for idx in row) for row in X]


__all__ = [
    "AA_ALPHABET",
    "AA_TO_IDX",
    "AA_toIDX",
    "ALPHABET",
    "IDX_TO_AA",
    "IDX_toAA",
    "decode_sequences",
    "encode_seqs",
    "flatten_samples",
]

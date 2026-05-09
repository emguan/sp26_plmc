"""Backward-compatible sequence encoding helpers.

New code should import shared alphabet and encoding utilities from
:mod:`utils.preprocessing`. This module keeps the historical NumPy-oriented
``encode_sequences`` wrapper.
"""

from __future__ import annotations

import numpy as np
import torch

from utils.preprocessing import (
    AA_ALPHABET,
    AA_TO_IDX,
    IDX_TO_AA,
    decode_sequences as _decode_sequences,
    encode_seqs,
    flatten_samples,
)

ALPHABET = AA_ALPHABET
IDX_toAA = IDX_TO_AA
AA_toIDX = AA_TO_IDX


def encode_sequences(seqs: list[str], alphabet: str = ALPHABET) -> np.ndarray:
    """Convert sequences to an integer NumPy array of shape ``(N, L)``."""
    if alphabet != AA_ALPHABET:
        aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
        return np.asarray([[aa_to_idx[aa] for aa in seq] for seq in seqs], dtype=np.int64)
    return encode_seqs(seqs).cpu().numpy()


def decode_sequences(X: np.ndarray | torch.Tensor, alphabet: str = ALPHABET) -> list[str]:
    """Convert an integer array/tensor to amino-acid strings."""
    return _decode_sequences(X, alphabet=alphabet)


__all__ = [
    "AA_ALPHABET",
    "AA_TO_IDX",
    "AA_toIDX",
    "ALPHABET",
    "IDX_TO_AA",
    "IDX_toAA",
    "decode_sequences",
    "encode_sequences",
    "encode_seqs",
    "flatten_samples",
]

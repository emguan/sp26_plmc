""" ignore this file, sanity check for emily """

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import torch

from utils.preprocessing import encode_seqs
from utils.sanity_check import ALPHABET, MI


def _encode_sequences(seqs: list[str], device):
    """Encode amino-acid sequences into an (N, L) LongTensor on ``device``."""
    return encode_seqs(seqs).to(device)


def _compute_mi_matrix_legacy(seqs: list[str], weights: list[float]) -> list[list[float]]:
    """Compatibility fallback when torch is unavailable."""
    length = len(seqs[0])
    matrix = [[0.0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        for j in range(length):
            matrix[i][j] = MI(seqs, weights, i, j)
    return matrix


def compute_mi_matrix(
    seqs: Iterable[str],
    weights: Iterable[float],
    device: str | torch.device | None = None,
    x: float = 0.3,
) -> list[list[float]]:
    """Compute the full pairwise mutual-information matrix for an MSA.

    This implementation is vectorized with torch and can run on CPU or GPU.
    """
    seq_list = list(seqs)
    weight_list = list(weights)

    if not seq_list:
        raise ValueError("No sequences were provided.")

    length = len(seq_list[0])
    if any(len(seq) != length for seq in seq_list):
        raise ValueError("All sequences must have the same length.")
    if len(weight_list) != len(seq_list):
        raise ValueError("`weights` must have the same length as `seqs`.")

    torch_device = torch.device(device) if device is not None else torch.device("cpu")

    x_tensor = _encode_sequences(seq_list, torch_device)
    w_tensor = torch.tensor(weight_list, dtype=torch.float32, device=torch_device)

    if torch.any(w_tensor < 0):
        raise ValueError("Weights must be non-negative.")

    # Sequence reweighting term w_m from utils.sanity_check, batched.
    xL = int(x * length)
    pairwise_hamming = (x_tensor[:, None, :] != x_tensor[None, :, :]).sum(dim=-1)
    close_mask = (pairwise_hamming < xL).to(torch.float32)
    neighbor_weight_sum = close_mask @ w_tensor
    wm = 1.0 / torch.clamp_min(neighbor_weight_sum, 1e-12)

    eff_weights = wm * w_tensor
    m_eff = torch.sum(eff_weights)

    # one_hot: (N, L, q) where q is alphabet size.
    q = len(ALPHABET)
    one_hot = torch.nn.functional.one_hot(x_tensor, num_classes=q).to(torch.float32)

    # Frequencies f_i(a): (L, q)
    fi = torch.einsum("nlq,n->lq", one_hot, eff_weights) / torch.clamp_min(m_eff, 1e-12)

    # Co-occurrences f_ij(a,b): (L, L, q, q)
    fij = (
        torch.einsum("nia,njb,n->ijab", one_hot, one_hot, eff_weights)
        / torch.clamp_min(m_eff, 1e-12)
    )

    denom = fi[:, None, :, None] * fi[None, :, None, :]
    positive = fij > 0
    safe_ratio = torch.where(positive, fij / torch.clamp_min(denom, 1e-12), torch.ones_like(fij))
    mi_terms = torch.where(positive, fij * torch.log(safe_ratio), torch.zeros_like(fij))

    matrix = mi_terms.sum(dim=(-1, -2))
    return matrix.detach().cpu().tolist()


def compute_mi_matrix_from_dataframe(
    data: pd.DataFrame,
    device: str | torch.device | None = None,
    x: float = 0.3,
) -> list[list[float]]:
    required = {"AA_SEQ", "AA_WEIGHT"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return compute_mi_matrix(
        data["AA_SEQ"].tolist(),
        data["AA_WEIGHT"].tolist(),
        device=device,
        x=x,
    )


def main() -> None:
    path = "data/NGS-161.00.txt"

    data = pd.read_csv(path, sep="\t")
    data = data.head(1000)

    # Optional sanity print
    print(f"Loaded {len(data)} sequences from {path}")
    print(data.head())

    # Use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Computing MI matrix on {device}...")

    mi_matrix = compute_mi_matrix_from_dataframe(
        data,
        device=device,
        x=0.3,
    )

    print("MI matrix:")
    for row in mi_matrix:
        print(" ".join(f"{v:.6f}" for v in row))


if __name__ == "__main__":
    main()
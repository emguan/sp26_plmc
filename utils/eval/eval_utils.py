"""Evaluation utilities for trained Potts models and generated sequences."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from utils.preprocessing import flatten_samples
from utils.train.train_val import train_val_split_indices


def _coerce_X_weights(
    X: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """Normalize evaluation inputs for conditional diagnostics."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.long)
    X = X.to(device=device, dtype=torch.long)
    if X.dim() != 2:
        raise ValueError(f"X must have shape (N, L), got {tuple(X.shape)}")

    if weights is None:
        w = torch.ones(X.shape[0], dtype=dtype, device=device)
    elif torch.is_tensor(weights):
        w = weights.to(device=device, dtype=dtype)
    else:
        w = torch.as_tensor(weights, dtype=dtype, device=device)

    if w.dim() != 1 or w.shape[0] != X.shape[0]:
        raise ValueError("weights must have shape (N,) matching X")
    if not torch.all(torch.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if torch.any(w < 0):
        raise ValueError("weights contain negative values")
    if torch.all(w == 0):
        raise ValueError("all weights are zero")
    return X, w


@torch.no_grad()
def conditional_logits_all_sites(
    X: np.ndarray | torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
) -> torch.Tensor:
    """Compute conditional logits for every position."""
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.long)
    X = X.to(device=h.device, dtype=torch.long)
    if X.dim() != 2:
        raise ValueError(f"X must have shape (N, L), got {tuple(X.shape)}")

    N, L = X.shape
    if h.dim() != 2:
        raise ValueError(f"h must have shape (L, q), got {tuple(h.shape)}")
    if h.shape[0] != L:
        raise ValueError(f"X length {L} does not match h length {h.shape[0]}")
    q = h.shape[1]
    if J.shape != (L, L, q, q):
        raise ValueError(f"J must have shape {(L, L, q, q)}, got {tuple(J.shape)}")

    logits = h.unsqueeze(0).expand(N, L, q).clone()
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            logits[:, i, :] += J[i, j, :, X[:, j]].T
    return logits


@torch.no_grad()
def avg_neglogp_per_site(
    X: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    h: torch.Tensor,
    J: torch.Tensor,
) -> float:
    """Weighted average negative log-pseudolikelihood per site."""
    X, w = _coerce_X_weights(X, weights, h.device, h.dtype)
    N, L = X.shape
    logits = conditional_logits_all_sites(X, h, J)
    loss_per_site = F.cross_entropy(
        logits.reshape(N * L, -1),
        X.reshape(N * L),
        reduction="none",
    ).reshape(N, L)
    return float((loss_per_site * w[:, None]).sum().div(w.sum() * L).cpu().item())


@torch.no_grad()
def conditional_perplexity(
    X: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    h: torch.Tensor,
    J: torch.Tensor,
) -> float:
    """Per-site conditional perplexity: exp(avg negative log-pseudolikelihood)."""
    return float(np.exp(avg_neglogp_per_site(X, weights, h, J)))


@torch.no_grad()
def conditional_accuracy(
    X: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor | None,
    h: torch.Tensor,
    J: torch.Tensor,
) -> float:
    """Weighted conditional amino-acid prediction accuracy over all positions."""
    X, w = _coerce_X_weights(X, weights, h.device, h.dtype)
    logits = conditional_logits_all_sites(X, h, J)
    correct = (logits.argmax(dim=-1) == X).to(dtype=h.dtype)
    return float((correct * w[:, None]).sum().div(w.sum() * X.shape[1]).cpu().item())


def mean_pairwise_hamming(
    samples: np.ndarray | torch.Tensor,
    max_pairs: int = 20000,
    seed: int = 0,
) -> float:
    """Estimate mean normalized pairwise Hamming distance in [0, 1]."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    rng = np.random.default_rng(seed)
    X = flatten_samples(samples)
    M = X.shape[0]
    if M < 2:
        return 0.0

    total_pairs = M * (M - 1) // 2
    if total_pairs <= max_pairs:
        dists = [np.mean(X[i] != X[j]) for i in range(M) for j in range(i + 1, M)]
        return float(np.mean(dists))

    idx_i = rng.integers(0, M, size=max_pairs)
    idx_j = rng.integers(0, M, size=max_pairs)
    mask = idx_i != idx_j
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.mean(X[idx_i[mask]] != X[idx_j[mask]], axis=1)))


def position_frequencies(
    X: np.ndarray | torch.Tensor,
    q: int = 20,
    weights: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Compute weighted residue frequencies at each position as an array (L, q)."""
    X_arr = flatten_samples(X).astype(np.int64, copy=False)
    if X_arr.size and (X_arr.min() < 0 or X_arr.max() >= q):
        raise ValueError(f"X contains states outside [0, {q - 1}]")

    if weights is None:
        w = np.ones(X_arr.shape[0], dtype=np.float64)
    elif torch.is_tensor(weights):
        w = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        w = np.asarray(weights, dtype=np.float64)

    if w.ndim != 1 or w.shape[0] != X_arr.shape[0]:
        raise ValueError("weights must have shape (N,) matching flattened X")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if np.any(w < 0):
        raise ValueError("weights contain negative values")
    denom = float(w.sum())
    if denom <= 0:
        raise ValueError("sum of weights must be positive")

    _, L = X_arr.shape
    freqs = np.zeros((L, q), dtype=np.float64)
    for i in range(L):
        freqs[i, :] = np.bincount(X_arr[:, i], weights=w, minlength=q)[:q]
    return freqs / denom


def position_entropy(
    X: np.ndarray | torch.Tensor,
    q: int = 20,
    weights: np.ndarray | torch.Tensor | None = None,
    base: str | int = "e",
) -> np.ndarray:
    """Position-wise entropy in nats by default, or bits with base=2."""
    freqs = position_frequencies(X, q=q, weights=weights)
    entropy = -np.sum(freqs * np.log(freqs + 1e-12), axis=1)
    if base == 2:
        entropy = entropy / np.log(2)
    elif base != "e":
        raise ValueError("base must be 'e' or 2")
    return entropy


def train_val_split_indices_from_ckpt(
    ckpt: dict[str, Any],
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load split indices from a checkpoint or recreate them from split metadata."""
    split = ckpt.get("split", {})
    if "train_idx" in split and "val_idx" in split:
        return np.asarray(split["train_idx"], dtype=np.int64), np.asarray(split["val_idx"], dtype=np.int64)

    return train_val_split_indices(
        N,
        val_frac=float(split.get("val_frac", 0.1)),
        seed=int(split.get("split_seed", 0)),
    )


@torch.no_grad()
def conditional_eval(
    X_np: np.ndarray,
    weights_np: np.ndarray | None,
    h: torch.Tensor,
    J: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    """Return average conditional NLL, perplexity, and accuracy."""
    h = h.to(device)
    J = J.to(device)
    X = torch.as_tensor(X_np, dtype=torch.long, device=device)
    weights = None if weights_np is None else torch.as_tensor(weights_np, dtype=h.dtype, device=device)
    return {
        "avg_neglogp_per_site": avg_neglogp_per_site(X, weights, h, J),
        "conditional_perplexity": conditional_perplexity(X, weights, h, J),
        "conditional_accuracy": conditional_accuracy(X, weights, h, J),
    }


def pair_frequencies(
    X: np.ndarray,
    q: int = 20,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weighted pair frequencies with shape ``(L, L, q, q)``."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, L), got {X.shape}")

    N, L = X.shape
    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != N:
        raise ValueError("weights must have shape (N,) matching X")

    out = np.zeros((L, L, q, q), dtype=np.float64)
    for i in range(L):
        xi = X[:, i]
        for j in range(L):
            idx = xi * q + X[:, j]
            out[i, j] = np.bincount(idx, weights=weights, minlength=q * q).reshape(q, q)

    denom = weights.sum()
    if denom <= 0:
        raise ValueError("sum of weights must be positive")
    return out / denom


def mutual_information_matrix(
    X: np.ndarray,
    q: int = 20,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pairwise mutual information for encoded sequences."""
    _, L = np.asarray(X).shape
    f1 = position_frequencies(X, q=q, weights=weights)
    f2 = pair_frequencies(X, q=q, weights=weights)
    mi = np.zeros((L, L), dtype=np.float64)

    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            pij = f2[i, j]
            denom = f1[i][:, None] * f1[j][None, :]
            mask = pij > 0
            mi[i, j] = np.sum(pij[mask] * np.log((pij[mask] + 1e-12) / (denom[mask] + 1e-12)))
    return mi


def upper_triangle_values(M: np.ndarray) -> np.ndarray:
    """Return values above the diagonal of a square matrix."""
    return M[np.triu_indices(M.shape[0], k=1)]


def frequency_match_metrics(
    X_ref: np.ndarray,
    X_gen: np.ndarray,
    ref_weights: np.ndarray | None = None,
    q: int = 20,
) -> dict[str, float]:
    """Compare single, pair, and MI statistics between reference and generated data."""
    f_ref = position_frequencies(X_ref, q=q, weights=ref_weights)
    f_gen = position_frequencies(X_gen, q=q)
    pf_ref = pair_frequencies(X_ref, q=q, weights=ref_weights)
    pf_gen = pair_frequencies(X_gen, q=q)
    mi_ref_ut = upper_triangle_values(mutual_information_matrix(X_ref, q=q, weights=ref_weights))
    mi_gen_ut = upper_triangle_values(mutual_information_matrix(X_gen, q=q))

    if np.std(mi_ref_ut) < 1e-12 or np.std(mi_gen_ut) < 1e-12:
        mi_corr = float("nan")
    else:
        mi_corr = float(np.corrcoef(mi_ref_ut, mi_gen_ut)[0, 1])

    train_entropy = float(position_entropy(X_ref, q=q, weights=ref_weights).mean())
    gen_entropy = float(position_entropy(X_gen, q=q).mean())
    return {
        "single_freq_mae": float(np.abs(f_ref - f_gen).mean()),
        "single_freq_max_abs": float(np.abs(f_ref - f_gen).max()),
        "pair_freq_mae": float(np.abs(pf_ref - pf_gen).mean()),
        "pair_freq_max_abs": float(np.abs(pf_ref - pf_gen).max()),
        "mi_corr_train_vs_generated": mi_corr,
        "mean_train_entropy": train_entropy,
        "mean_generated_entropy": gen_entropy,
        "entropy_ratio_gen_over_train": float(gen_entropy / max(train_entropy, 1e-12)),
    }


def unique_fraction(seqs: list[str]) -> float:
    """Return the fraction of generated sequence strings that are unique."""
    if len(seqs) == 0:
        return 0.0
    return len(set(seqs)) / len(seqs)


def nearest_train_hamming(
    X_gen: np.ndarray,
    X_train: np.ndarray,
    max_train_compare: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Nearest normalized Hamming distance from each generated row to training data."""
    rng = np.random.default_rng(seed)
    if max_train_compare is not None and X_train.shape[0] > max_train_compare:
        X_ref = X_train[rng.choice(X_train.shape[0], size=max_train_compare, replace=False)]
    else:
        X_ref = X_train

    dists = np.zeros(X_gen.shape[0], dtype=np.float64)
    for n, x in enumerate(X_gen):
        dists[n] = np.mean(X_ref != x[None, :], axis=1).min()
    return dists


def novelty_metrics(
    train_seqs: list[str],
    gen_seqs: list[str],
    X_gen: np.ndarray,
    X_train: np.ndarray,
) -> dict[str, float]:
    """Summarize exact-match novelty and nearest-training Hamming distances."""
    train_set = set(train_seqs)
    exact_matches = sum(seq in train_set for seq in gen_seqs)
    nearest = nearest_train_hamming(X_gen, X_train)
    return {
        "generated_unique_fraction": unique_fraction(gen_seqs),
        "exact_train_match_fraction": float(exact_matches / len(gen_seqs)),
        "mean_nearest_train_hamming": float(nearest.mean()),
        "median_nearest_train_hamming": float(np.median(nearest)),
        "min_nearest_train_hamming": float(nearest.min()),
    }


def duplicate_report(gen_seqs: list[str], top_k: int = 10) -> pd.DataFrame:
    """Return the top repeated generated sequences."""
    df = pd.Series(gen_seqs).value_counts().head(top_k).reset_index()
    df.columns = ["AA_SEQ", "count"]
    return df


@torch.no_grad()
def sequence_scores(
    X_np: np.ndarray,
    h: torch.Tensor,
    J: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Compute Potts scores; higher scores indicate higher model preference."""
    X = torch.as_tensor(X_np, dtype=torch.long, device=device)
    h = h.to(device)
    J = J.to(device)
    N, L = X.shape
    scores = torch.zeros(N, dtype=h.dtype, device=device)
    positions = torch.arange(L, device=device)
    scores += h[positions[None, :], X].sum(dim=1)
    for i in range(L):
        for j in range(i + 1, L):
            scores += J[i, j, X[:, i], X[:, j]]
    return scores.detach().cpu().numpy()


def random_sequences_like(n: int, L: int, q: int = 20, seed: int = 0) -> np.ndarray:
    """Generate random encoded sequences for score baselines."""
    return np.random.default_rng(seed).integers(0, q, size=(n, L), dtype=np.int64)


def score_summary(name: str, scores: np.ndarray) -> dict[str, float]:
    """Return named summary statistics for a score vector."""
    return {
        f"{name}_score_mean": float(np.mean(scores)),
        f"{name}_score_std": float(np.std(scores)),
        f"{name}_score_p05": float(np.quantile(scores, 0.05)),
        f"{name}_score_p50": float(np.quantile(scores, 0.50)),
        f"{name}_score_p95": float(np.quantile(scores, 0.95)),
    }


__all__ = [
    "avg_neglogp_per_site",
    "conditional_accuracy",
    "conditional_eval",
    "conditional_logits_all_sites",
    "conditional_perplexity",
    "duplicate_report",
    "frequency_match_metrics",
    "mean_pairwise_hamming",
    "mutual_information_matrix",
    "nearest_train_hamming",
    "novelty_metrics",
    "pair_frequencies",
    "position_entropy",
    "position_frequencies",
    "random_sequences_like",
    "score_summary",
    "sequence_scores",
    "train_val_split_indices_from_ckpt",
    "unique_fraction",
    "upper_triangle_values",
]

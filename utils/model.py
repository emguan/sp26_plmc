from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.preprocessing import (
    AA_ALPHABET,
    AA_TO_IDX,
    IDX_TO_AA,
    decode_sequences,
    encode_seqs,
    flatten_samples,
)
from utils.eval.eval_utils import (
    avg_neglogp_per_site,
    conditional_accuracy,
    conditional_logits_all_sites,
    conditional_perplexity,
    mean_pairwise_hamming,
    position_entropy,
    position_frequencies,
)


def unique_fraction(samples: np.ndarray | torch.Tensor, alphabet: str = AA_ALPHABET) -> float:
    """Fraction of generated sequences that are unique."""
    seqs = decode_sequences(samples, alphabet=alphabet)
    if len(seqs) == 0:
        return 0.0
    return len(set(seqs)) / len(seqs)


def hamming_distance(a: str | np.ndarray | torch.Tensor, b: str | np.ndarray | torch.Tensor) -> float:
    """Normalized Hamming distance between two equal-length sequences or arrays."""
    a_arr = np.asarray(list(a)) if isinstance(a, str) else np.asarray(a)
    b_arr = np.asarray(list(b)) if isinstance(b, str) else np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Inputs must have the same shape, got {a_arr.shape} and {b_arr.shape}")
    return float(np.mean(a_arr != b_arr))


def aa_frequency_by_position(
    X: np.ndarray | torch.Tensor,
    aa: str = "W",
    alphabet: str = AA_ALPHABET,
    weights: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Frequency of one amino acid at each position."""
    if len(aa) != 1 or aa not in alphabet:
        raise ValueError(f"aa must be one residue from alphabet {alphabet!r}")
    freqs = position_frequencies(X, q=len(alphabet), weights=weights)
    return freqs[:, alphabet.index(aa)]


def generation_diagnostics(
    samples: np.ndarray | torch.Tensor,
    train_X: np.ndarray | torch.Tensor | None = None,
    train_weights: np.ndarray | torch.Tensor | None = None,
    alphabet: str = AA_ALPHABET,
    max_pairs: int = 20000,
    seed: int = 0,
) -> Dict[str, object]:
    """Summarize generated sequence diversity and compare to training data when provided."""
    X_gen = flatten_samples(samples)
    out: Dict[str, object] = {
        "num_generated": int(X_gen.shape[0]),
        "L": int(X_gen.shape[1]),
        "unique_fraction": float(unique_fraction(X_gen, alphabet=alphabet)),
        "mean_pairwise_hamming": float(mean_pairwise_hamming(X_gen, max_pairs=max_pairs, seed=seed)),
        "mean_position_entropy": float(position_entropy(X_gen, q=len(alphabet)).mean()),
    }

    if "W" in alphabet:
        out["W_freq_by_position"] = aa_frequency_by_position(X_gen, aa="W", alphabet=alphabet).tolist()

    if train_X is not None:
        train_H = position_entropy(train_X, q=len(alphabet), weights=train_weights)
        gen_H = position_entropy(X_gen, q=len(alphabet))
        out.update({
            "train_mean_position_entropy": float(train_H.mean()),
            "generated_mean_position_entropy": float(gen_H.mean()),
            "entropy_ratio_gen_over_train": float(gen_H.mean() / max(float(train_H.mean()), 1e-12)),
        })
        if "W" in alphabet:
            out["train_W_freq_by_position"] = aa_frequency_by_position(
                train_X,
                aa="W",
                alphabet=alphabet,
                weights=train_weights,
            ).tolist()

    return out


def normalize_weights(weights: np.ndarray, mode: str = "mean1") -> np.ndarray:
    """
    Normalize fixed per-sequence weights before training.

    mode:
      - "none": no normalization
      - "mean1": mean weight = 1
      - "sum1": sum weights = 1
      - "sumN": sum weights = N
    """
    w = np.asarray(weights, dtype=np.float64).copy()

    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if len(w) == 0:
        raise ValueError("weights cannot be empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.all(w == 0):
        raise ValueError("all weights are zero")

    if mode == "none":
        return w

    if mode == "mean1":
        m = w.mean()
        if m <= 0:
            raise ValueError("mean weight must be positive")
        return w / m

    if mode == "sum1":
        s = w.sum()
        if s <= 0:
            raise ValueError("sum of weights must be positive")
        return w / s

    if mode == "sumN":
        s = w.sum()
        if s <= 0:
            raise ValueError("sum of weights must be positive")
        return w * (len(w) / s)

    raise ValueError(f"Unknown weight normalization mode: {mode!r}")


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish effective sample size:
      N_eff = (sum w)^2 / sum(w^2)
    """
    w = np.asarray(weights, dtype=np.float64)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)


def sanity_check_dataset(
    seqs: List[str],
    weights: np.ndarray,
    q: int = 20,
) -> Dict[str, float]:
    """
    Run dataset sanity checks and return summary stats.
    """
    if len(seqs) != len(weights):
        raise ValueError("seqs and weights must have the same length")

    if len(seqs) == 0:
        raise ValueError("No sequences provided")

    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("Sequences must all have the same length")
        for c in s:
            if c not in AA_TO_IDX:
                raise ValueError(f"Invalid amino acid {c!r} in sequence {s!r}")

    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if np.any(w < 0):
        raise ValueError("weights contain negative values")
    if np.all(w == 0):
        raise ValueError("all weights are zero")

    n_unique = len(set(seqs))
    if n_unique != len(seqs):
        print(f"[warn] Found duplicates: {len(seqs) - n_unique} repeated sequences.")

    n = len(seqs)
    neff_kish = effective_sample_size(w)
    sum_w = float(np.sum(w))
    mean_w = float(np.mean(w))
    min_w = float(np.min(w))
    max_w = float(np.max(w))

    # Rough parameter count for symmetric Potts
    n_params_h = L * q
    n_params_J = (L * (L - 1) // 2) * q * q

    if neff_kish < 5:
        print(f"[warn] Very small effective sample size: N_eff={neff_kish:.3f}")
    if neff_kish < L:
        print(f"[warn] N_eff ({neff_kish:.2f}) is smaller than sequence length L ({L}).")
    if neff_kish < 0.1 * (n_params_h + n_params_J) / q:
        print("[warn] Data are very small relative to model size; expect strong regularization needs.")

    return {
        "N": float(n),
        "L": float(L),
        "sum_w": sum_w,
        "mean_w": mean_w,
        "min_w": min_w,
        "max_w": max_w,
        "kish_neff": neff_kish,
        "n_unique": float(n_unique),
        "n_params_h": float(n_params_h),
        "n_params_J": float(n_params_J),
    }


# --- Model ---
@dataclass
class PottsPLMConfig:
    L: int
    q: int = 20
    lambda_h: float = 1e-3
    lambda_J_scale: float = 1e-2
    use_marks_lambda_J: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    grad_clip_norm: Optional[float] = 10.0


class PottsPLM(nn.Module):
    """
    Symmetric pairwise Potts model trained by weighted site-factored pseudolikelihood.
    """

    def __init__(self, cfg: PottsPLMConfig):
        super().__init__()
        self.cfg = cfg
        L, q = cfg.L, cfg.q

        self.h = nn.Parameter(torch.zeros((L, q), device=cfg.device, dtype=cfg.dtype))
        J_init = 0.01 * torch.randn((L, L, q, q), device=cfg.device, dtype=cfg.dtype)
        self.J_raw = nn.Parameter(J_init)

    def _symmetrized_J(self) -> torch.Tensor:
        J = 0.5 * (self.J_raw + self.J_raw.transpose(0, 1).transpose(2, 3))
        diag_idx = torch.arange(self.cfg.L, device=J.device)
        J[diag_idx, diag_idx, :, :] = 0.0
        return J

    def lambda_J_value(self) -> float:
        cfg = self.cfg
        if cfg.use_marks_lambda_J:
            return float(cfg.lambda_J_scale * cfg.q * (cfg.L - 1))
        return float(cfg.lambda_J_scale)

    @torch.no_grad()
    def conditional_logits_all_sites(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Compute per-site conditional logits under the current symmetrized model."""
        return conditional_logits_all_sites(X, self.h, self._symmetrized_J())

    @torch.no_grad()
    def avg_neglogp_per_site(
        self,
        X: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        """Weighted average negative log-pseudolikelihood per site."""
        return avg_neglogp_per_site(X, weights, self.h, self._symmetrized_J())

    @torch.no_grad()
    def conditional_perplexity(
        self,
        X: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        """Per-site conditional perplexity on an encoded evaluation set."""
        return conditional_perplexity(X, weights, self.h, self._symmetrized_J())

    @torch.no_grad()
    def conditional_accuracy(
        self,
        X: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        """Weighted conditional amino-acid prediction accuracy."""
        return conditional_accuracy(X, weights, self.h, self._symmetrized_J())

    @torch.no_grad()
    def evaluation_summary(
        self,
        X: np.ndarray | torch.Tensor,
        weights: np.ndarray | torch.Tensor | None = None,
    ) -> Dict[str, float]:
        """Return all model-level conditional diagnostics for an encoded evaluation set."""
        return {
            "avg_neglogp_per_site": self.avg_neglogp_per_site(X, weights),
            "conditional_perplexity": self.conditional_perplexity(X, weights),
            "conditional_accuracy": self.conditional_accuracy(X, weights),
        }

    def pseudolikelihood_nll(
        self,
        X: torch.LongTensor,
        w: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        X = X.to(cfg.device)
        w = w.to(cfg.device).to(cfg.dtype)

        if X.dim() != 2 or X.shape[1] != cfg.L:
            raise ValueError(f"X must have shape (N, {cfg.L}). Got {tuple(X.shape)}.")
        if w.dim() != 1 or w.shape[0] != X.shape[0]:
            raise ValueError("w must have shape (N,) matching X.")
        if not torch.all(torch.isfinite(w)):
            raise ValueError("Weights contain non-finite values.")
        if torch.any(w < 0):
            raise ValueError("Weights contain negative values.")
        if torch.all(w == 0):
            raise ValueError("All weights are zero.")

        N, L = X.shape
        q = cfg.q
        J = self._symmetrized_J()

        total_neglogp = torch.zeros((), device=cfg.device, dtype=cfg.dtype)

        for i in range(L):
            logits = self.h[i].unsqueeze(0).expand(N, q).clone()

            for j in range(L):
                if j == i:
                    continue
                xj = X[:, j]
                logits = logits + J[i, j][:, xj].transpose(0, 1)

            logp = F.log_softmax(logits, dim=-1)
            xi = X[:, i]
            ll_i = logp[torch.arange(N, device=cfg.device), xi]
            total_neglogp = total_neglogp - torch.sum(w * ll_i)

        lambda_h = float(cfg.lambda_h)
        lambda_J = self.lambda_J_value()

        reg_h = lambda_h * torch.sum(self.h * self.h)
        reg_J = lambda_J * torch.sum(J * J)
        loss = total_neglogp + reg_h + reg_J

        if not return_parts:
            return loss

        with torch.no_grad():
            h_norm = torch.norm(self.h).item()
            J_norm = torch.norm(J).item()
            mean_abs_h = torch.mean(torch.abs(self.h)).item()
            mean_abs_J = torch.mean(torch.abs(J)).item()

        parts = {
            "neglogp": float(total_neglogp.detach().cpu().item()),
            "reg_h": float(reg_h.detach().cpu().item()),
            "reg_J": float(reg_J.detach().cpu().item()),
            "lambda_h": lambda_h,
            "lambda_J": lambda_J,
            "sum_w": float(w.sum().detach().cpu().item()),
            "h_norm": float(h_norm),
            "J_norm": float(J_norm),
            "mean_abs_h": float(mean_abs_h),
            "mean_abs_J": float(mean_abs_J),
        }
        return loss, parts

    @torch.no_grad()
    def delta_E(self, wt: str, mut: str) -> float:
        cfg = self.cfg
        if len(wt) != cfg.L or len(mut) != cfg.L:
            raise ValueError(f"wt/mut must have length {cfg.L}.")

        wt_idx = torch.tensor([AA_TO_IDX[c] for c in wt], device=cfg.device)
        mut_idx = torch.tensor([AA_TO_IDX[c] for c in mut], device=cfg.device)
        J = self._symmetrized_J()

        def energy(x_idx: torch.Tensor) -> torch.Tensor:
            E = torch.zeros((), device=cfg.device, dtype=cfg.dtype)
            E = E + torch.sum(self.h[torch.arange(cfg.L, device=cfg.device), x_idx])
            for i in range(cfg.L):
                for j in range(i + 1, cfg.L):
                    E = E + J[i, j, x_idx[i], x_idx[j]]
            return E

        return float((energy(mut_idx) - energy(wt_idx)).detach().cpu().item())


def fit_potts_plm_lbfgs(
    seqs: List[str],
    weights: np.ndarray,
    lambda_h: float = 1e-3,
    lambda_J_scale: float = 1e-2,
    use_marks_lambda_J: bool = False,
    weight_norm: str = "mean1",
    weight_cap: Optional[float] = 1.0,
    max_iter: int = 200,
    device: str = "cpu",
) -> PottsPLM:
    """
    Fit Potts PLM using L-BFGS with safer defaults for small weighted datasets.
    """
    capped_weights = cap_weights(weights, max_weight=weight_cap)
    raw_stats = sanity_check_dataset(seqs, capped_weights)
    print(f"Dataset stats (after weight_cap={weight_cap!r}, pre-normalization):", raw_stats)
    
    normed_weights = normalize_weights(capped_weights, mode=weight_norm)
    normed_stats = sanity_check_dataset(seqs, normed_weights)
    print(f"Dataset stats (after weight_norm={weight_norm!r}):", normed_stats)

    L = len(seqs[0])
    X = encode_seqs(seqs, L=L).to(device)
    w = torch.tensor(normed_weights, dtype=torch.float32, device=device)

    cfg = PottsPLMConfig(
        L=L,
        q=20,
        lambda_h=lambda_h,
        lambda_J_scale=lambda_J_scale,
        use_marks_lambda_J=use_marks_lambda_J,
        device=device,
        dtype=torch.float32,
        grad_clip_norm=10.0,
    )
    model = PottsPLM(cfg).to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = model.pseudolikelihood_nll(X, w, return_parts=False)
        loss.backward()

        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)

        return loss

    optimizer.step(closure)

    loss, parts = model.pseudolikelihood_nll(X, w, return_parts=True)
    print("Final loss parts:", parts)
    print("Final conditional evaluation:", model.evaluation_summary(X, w))

    # A few interpretable diagnostics
    neglogp = parts["neglogp"]
    reg_h = parts["reg_h"]
    reg_J = parts["reg_J"]

    if reg_h > neglogp:
        print("[warn] reg_h > neglogp: fields may be over-regularized.")
    if reg_J > neglogp:
        print("[warn] reg_J > neglogp: couplings may be over-regularized.")
    if parts["mean_abs_J"] < 1e-3:
        print("[warn] mean_abs_J is very small: couplings may be collapsing toward zero.")
    if parts["mean_abs_h"] < 1e-3:
        print("[warn] mean_abs_h is very small: fields may be collapsing toward zero.")

    return model

def cap_weights(weights: np.ndarray, max_weight: Optional[float] = 1.0) -> np.ndarray:
    """
    Cap fixed per-sequence weights before training.

    Use ``max_weight=None`` to disable capping. Capping is applied before
    weight normalization so extreme counts cannot dominate the fit.
    """
    w = np.asarray(weights, dtype=np.float64).copy()

    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if len(w) == 0:
        raise ValueError("weights cannot be empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if np.all(w == 0):
        raise ValueError("all weights are zero")

    if max_weight is None:
        return w
    if max_weight <= 0:
        raise ValueError("max_weight must be positive or None")

    return np.minimum(w, float(max_weight))


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({
        "AA_SEQ": ["AESFVEWDSW", "AESFVEWDAW"],
        "AA_CNT": [1, 3],
        "AA_WEIGHT": [1.0, 0.7],
        "NT_UNIQ": [1, 1],
    })

    # Fixed pre-training sample weights
    df["W_TRAIN"] = df["AA_WEIGHT"].astype(float)

    model = fit_potts_plm_lbfgs(
        seqs=df["AA_SEQ"].tolist(),
        weights=df["W_TRAIN"].to_numpy(),
        lambda_h=1e-3,
        lambda_J_scale=1e-2,
        use_marks_lambda_J=False,
        weight_norm="mean1",
        weight_cap=10,
        device="cpu",
        max_iter=150,
    )

    print("ΔE example:", model.delta_E("AESFVEWDSW", "AESFVEWDAW"))
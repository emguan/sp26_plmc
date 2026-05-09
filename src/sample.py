from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch


from utils.preprocessing import AA_ALPHABET, decode_sequences, encode_seqs
from utils.model import (
    cap_weights,
    normalize_weights,
)
from utils.sample.sampling_loop import BatchedFastDCAGibbsSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample sequences from a trained Potts model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input TSV path. Used for sampling diagnostics.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Trained model checkpoint.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        help="Output TSV path for sampled sequences.",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temp", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--burn-in-frac", type=float, default=0.1)
    parser.add_argument("--sweeps-per-sample", type=int, default=1)
    parser.add_argument("--thinning", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[SAMPLE 1/4] Loading model...")
    ckpt = torch.load(args.model, map_location=args.device)

    h = ckpt["h"].to(args.device)
    J = ckpt["J"].to(args.device)

    weighting = ckpt.get("weighting", {})
    weight_norm = weighting.get("weight_norm", "mean1")
    weight_cap = weighting.get("weight_cap", 1.0)

    print("[SAMPLE 2/4] Loading input dataset for diagnostics...")
    data = pd.read_csv(args.input, sep="\t")

    required = {"AA_SEQ", "AA_WEIGHT"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    seqs = data["AA_SEQ"].astype(str).tolist()

    weights = data["AA_WEIGHT"].astype(float)
    if "AA_CNT" in data.columns:
        weights = weights * data["AA_CNT"].astype(float)

    raw_weights = weights.to_numpy(copy=True)

    cap_value = None if weight_cap <= 0 else weight_cap
    full_eval_weights = cap_weights(raw_weights, max_weight=cap_value)
    full_eval_weights = normalize_weights(full_eval_weights, mode=weight_norm)

    full_X = encode_seqs(seqs, L=h.shape[0]).to(args.device)

    print("[SAMPLE 3/4] Running Gibbs sampling...")
    with torch.no_grad():
        sampler = BatchedFastDCAGibbsSampler(
            h=h.detach(),
            J=J.detach(),
            temp=args.temp,
        )

        sampled, sample_diagnostics = sampler.sample(
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            burn_in_frac=args.burn_in_frac,
            sweeps_per_sample=args.sweeps_per_sample,
            thinning=args.thinning,
            random_order=True,
            store_samples=True,
            return_last_state=False,
            return_diagnostics=True,
            train_X=full_X,
            train_weights=full_eval_weights,
            alphabet=AA_ALPHABET,
        )

    print("[INFO] Sampling diagnostics:", sample_diagnostics)

    if sampled is None:
        raise RuntimeError("Sampler returned no samples.")

    print("[SAMPLE 4/4] Writing sampled sequences...")
    sampled_cpu = sampled.cpu()
    decoded = decode_sequences(sampled_cpu, alphabet=AA_ALPHABET)

    rows = []
    n_chains = sampled_cpu.shape[1]
    for flat_idx, aa_seq in enumerate(decoded):
        rows.append(
            {
                "sample_idx": flat_idx // n_chains,
                "chain_idx": flat_idx % n_chains,
                "AA_SEQ": aa_seq,
            }
        )

    sampled_df = pd.DataFrame(rows)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(args.samples_output, sep="\t", index=False)

    print("[DONE] Sampling complete.")
    print(f"[INFO] Samples saved to: {args.samples_output}")
    print(f"[INFO] Samples shape: {tuple(sampled.shape)}  # (num_samples, batch_size, L)")


if __name__ == "__main__":
    main()
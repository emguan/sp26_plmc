from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from utils.preprocessing import encode_seqs

from utils.model import (
    fit_potts_plm_lbfgs,
    cap_weights,
    normalize_weights,
)
from utils.train.train_val import train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Potts model from an MSA TSV file."
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input TSV path. Must include AA_SEQ and AA_WEIGHT columns. AA_CNT is optional.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Output path for serialized model tensors.",
    )

    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--max-iter", type=int, default=200, help="LBFGS max_iter for training.")

    parser.add_argument("--lambda-h", type=float, default=0.01, help="L2 regularization strength for h.")
    parser.add_argument(
        "--lambda-J-scale",
        type=float,
        default=1.0,
        help="L2 regularization scale for J.",
    )

    parser.add_argument(
        "--weight-norm",
        type=str,
        default="mean1",
        choices=["none", "mean1", "sum1", "sumN"],
        help="Weight normalization mode.",
    )
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=10.0,
        help="Maximum training weight before normalization. Use <=0 to disable capping.",
    )

    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of sequences held out for validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for train/validation split.",
    )

    return parser.parse_args()


def process_weights(
    weights: np.ndarray,
    weight_cap: float,
    weight_norm: str,
) -> np.ndarray:
    cap_value = None if weight_cap <= 0 else weight_cap
    out = cap_weights(weights, max_weight=cap_value)
    out = normalize_weights(out, mode=weight_norm)
    return out


def main() -> None:
    args = parse_args()

    print("[TRAIN 1/4] Loading input dataset...")
    data = pd.read_csv(args.input, sep="\t")

    required = {"AA_SEQ", "AA_WEIGHT"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    seqs = data["AA_SEQ"].astype(str).tolist()
    if len(seqs) == 0:
        raise ValueError("Input dataset has no sequences.")

    print("[TRAIN 2/4] Preparing weights and splitting data...")
    weights = data["AA_WEIGHT"].astype(float)
    if "AA_CNT" in data.columns:
        weights = weights * data["AA_CNT"].astype(float)

    raw_weights = weights.to_numpy(copy=True)

    train_seqs, train_weights_raw, val_seqs, val_weights_raw, train_idx, val_idx = train_val_split(
        seqs=seqs,
        weights=raw_weights,
        val_frac=args.val_frac,
        seed=args.split_seed,
        return_indices=True,
    )

    print(f"[INFO] Full dataset N={len(seqs)}, L={len(seqs[0])}")
    print(f"[INFO] Train N={len(train_seqs)}, Val N={len(val_seqs)}")

    print("[TRAIN 3/4] Fitting Potts model...")
    weight_cap_arg = None if args.weight_cap <= 0 else args.weight_cap

    model = fit_potts_plm_lbfgs(
        seqs=train_seqs,
        weights=train_weights_raw,
        lambda_h=args.lambda_h,
        lambda_J_scale=args.lambda_J_scale,
        use_marks_lambda_J=False,
        weight_norm=args.weight_norm,
        weight_cap=weight_cap_arg,
        max_iter=args.max_iter,
        device=args.device,
    )

    print("[TRAIN 4/4] Computing diagnostics and saving model...")

    train_X = encode_seqs(train_seqs, L=model.cfg.L).to(args.device)
    train_eval_weights = process_weights(train_weights_raw, args.weight_cap, args.weight_norm)
    train_eval = model.evaluation_summary(
        train_X,
        torch.as_tensor(train_eval_weights, dtype=torch.float32, device=args.device),
    )

    val_X = encode_seqs(val_seqs, L=model.cfg.L).to(args.device)
    val_eval_weights = process_weights(val_weights_raw, args.weight_cap, args.weight_norm)
    val_eval = model.evaluation_summary(
        val_X,
        torch.as_tensor(val_eval_weights, dtype=torch.float32, device=args.device),
    )

    full_X = encode_seqs(seqs, L=model.cfg.L).to(args.device)
    full_eval_weights = process_weights(raw_weights, args.weight_cap, args.weight_norm)
    full_eval = model.evaluation_summary(
        full_X,
        torch.as_tensor(full_eval_weights, dtype=torch.float32, device=args.device),
    )

    print("[INFO] Train conditional diagnostics:", train_eval)
    print("[INFO] Val conditional diagnostics:", val_eval)
    print("[INFO] Full-data conditional diagnostics:", full_eval)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.save(
            {
                "h": model.h.detach().cpu(),
                "J": model._symmetrized_J().detach().cpu(),
                "config": vars(model.cfg),
                "train_diagnostics": train_eval,
                "val_diagnostics": val_eval,
                "full_diagnostics": full_eval,
                "split": {
                    "val_frac": args.val_frac,
                    "split_seed": args.split_seed,
                    "train_N": len(train_seqs),
                    "val_N": len(val_seqs),
                    "full_N": len(seqs),
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                },
                "weighting": {
                    "weight_norm": args.weight_norm,
                    "weight_cap": args.weight_cap,
                },
            },
            args.model_output,
        )

    print("[DONE] Training complete.")
    print(f"[INFO] Model saved to: {args.model_output}")


if __name__ == "__main__":
    main()
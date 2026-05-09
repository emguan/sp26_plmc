from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from utils.preprocessing import AA_ALPHABET, encode_seqs
from utils.model import (
    cap_weights,
    normalize_weights,
)
from utils.eval.eval_utils import (
    train_val_split_indices_from_ckpt,
    conditional_eval,
    frequency_match_metrics,
    novelty_metrics,
    mean_pairwise_hamming,
    unique_fraction,
    duplicate_report,
    random_sequences_like,
    sequence_scores,
    score_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Potts model and sampled sequences."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input TSV dataset path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Trained model checkpoint path.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Generated samples file. Supports .tsv or .pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Machine-readable TSV report path.",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def decode_tensor_sequences(
    X: torch.Tensor,
    alphabet: str = AA_ALPHABET,
) -> list[str]:
    """
    Decode encoded tensor of shape (N, L) or (num_samples, batch_size, L).
    """
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])
    elif X.ndim != 2:
        raise ValueError(f"Expected samples tensor with ndim 2 or 3, got shape {tuple(X.shape)}")

    X = X.cpu()

    seqs = []
    for row in X:
        seqs.append("".join(alphabet[int(i)] for i in row.tolist()))

    return seqs


def load_generated_sequences(path: Path) -> list[str]:
    """
    Load generated sequences from either:
      - TSV file with AA_SEQ column
      - PT file containing tensor or dict with key 'samples'
    """
    if path.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "samples" not in ckpt:
                raise ValueError("Sample .pt dict must contain key 'samples'.")
            samples = ckpt["samples"]
            alphabet = ckpt.get("alphabet", AA_ALPHABET)
        else:
            samples = ckpt
            alphabet = AA_ALPHABET

        if not isinstance(samples, torch.Tensor):
            samples = torch.as_tensor(samples)

        return decode_tensor_sequences(samples, alphabet=alphabet)

    sample_df = pd.read_csv(path, sep="\t")

    if "AA_SEQ" not in sample_df.columns:
        raise ValueError("Sample file must contain AA_SEQ column.")

    return sample_df["AA_SEQ"].astype(str).tolist()


def write_human_readable_report(
    output_path: Path,
    *,
    all_seqs: list[str],
    train_seqs: list[str],
    val_seqs: list[str],
    gen_seqs: list[str],
    L: int,
    train_cond: dict,
    val_cond: dict,
    full_cond: dict,
    freq_metrics: dict,
    nov_metrics: dict,
    div_metrics: dict,
    score_metrics: dict,
    dup_df: pd.DataFrame,
) -> Path:
    """
    Write a human-readable text report next to the TSV report.
    """
    human_output = output_path.with_suffix(".txt")

    with human_output.open("w", encoding="utf-8") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n\n")

        f.write("Dataset sizes\n")
        f.write("-------------\n")
        f.write(f"Full N: {len(all_seqs)}\n")
        f.write(f"Train N: {len(train_seqs)}\n")
        f.write(f"Val N: {len(val_seqs)}\n")
        f.write(f"Generated N: {len(gen_seqs)}\n")
        f.write(f"L: {L}\n\n")

        f.write("Conditional fit diagnostics\n")
        f.write("---------------------------\n")
        f.write(f"Train NLL/site: {train_cond['avg_neglogp_per_site']:.6f}\n")
        f.write(f"Val NLL/site: {val_cond['avg_neglogp_per_site']:.6f}\n")
        f.write(f"Full NLL/site: {full_cond['avg_neglogp_per_site']:.6f}\n\n")

        f.write(f"Train perplexity: {train_cond['conditional_perplexity']:.6f}\n")
        f.write(f"Val perplexity: {val_cond['conditional_perplexity']:.6f}\n")
        f.write(f"Full perplexity: {full_cond['conditional_perplexity']:.6f}\n\n")

        f.write(f"Train accuracy: {train_cond['conditional_accuracy']:.6f}\n")
        f.write(f"Val accuracy: {val_cond['conditional_accuracy']:.6f}\n")
        f.write(f"Full accuracy: {full_cond['conditional_accuracy']:.6f}\n\n")

        f.write("Generated sequence diagnostics\n")
        f.write("------------------------------\n")
        f.write(f"Generated unique fraction: {div_metrics['generated_unique_fraction']:.6f}\n")
        f.write(
            f"Generated mean pairwise Hamming: "
            f"{div_metrics['generated_mean_pairwise_hamming']:.6f}\n\n"
        )

        f.write("Frequency match metrics\n")
        f.write("-----------------------\n")
        for k, v in freq_metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

        f.write("\nNovelty metrics\n")
        f.write("---------------\n")
        for k, v in nov_metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

        f.write("\nScore separation diagnostics\n")
        f.write("----------------------------\n")
        for k, v in score_metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")

        f.write("\nTop duplicate generated sequences\n")
        f.write("---------------------------------\n")
        if len(dup_df) == 0:
            f.write("No duplicate sequences found.\n")
        else:
            f.write(dup_df.to_string(index=False))
            f.write("\n")

    return human_output


def main() -> None:
    args = parse_args()

    print("[EVAL 1/6] Loading model...")
    ckpt = torch.load(args.model, map_location=args.device)

    h = ckpt["h"].to(args.device)
    J = ckpt["J"].to(args.device)

    weighting = ckpt.get("weighting", {})
    weight_norm = weighting.get("weight_norm", "mean1")
    weight_cap = weighting.get("weight_cap", 1.0)

    print("[EVAL 2/6] Loading input dataset...")
    data = pd.read_csv(args.input, sep="\t")

    required = {"AA_SEQ", "AA_WEIGHT"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    all_seqs = data["AA_SEQ"].astype(str).tolist()

    weights = data["AA_WEIGHT"].astype(float)
    if "AA_CNT" in data.columns:
        weights = weights * data["AA_CNT"].astype(float)

    raw_weights = weights.to_numpy(copy=True)

    cap_value = None if weight_cap <= 0 else weight_cap
    weights_full = cap_weights(raw_weights, max_weight=cap_value)
    weights_full = normalize_weights(weights_full, mode=weight_norm)

    X_full_torch = encode_seqs(all_seqs, L=h.shape[0])
    X_full = X_full_torch.cpu().numpy()

    train_idx, val_idx = train_val_split_indices_from_ckpt(
        ckpt,
        N=len(all_seqs),
    )

    X_train = X_full[train_idx]
    X_val = X_full[val_idx]
    w_train = weights_full[train_idx]
    w_val = weights_full[val_idx]

    train_seqs = [all_seqs[i] for i in train_idx]
    val_seqs = [all_seqs[i] for i in val_idx]

    print("[EVAL 3/6] Loading sampled sequences...")
    gen_seqs = load_generated_sequences(args.samples)
    X_gen_torch = encode_seqs(gen_seqs, L=h.shape[0])
    X_gen = X_gen_torch.cpu().numpy()

    print(
        f"[INFO] Full N={len(all_seqs)}, "
        f"Train N={len(train_seqs)}, "
        f"Val N={len(val_seqs)}, "
        f"Gen N={len(gen_seqs)}"
    )

    print("[EVAL 4/6] Conditional fit diagnostics...")
    train_cond = conditional_eval(X_train, w_train, h, J, device=args.device)
    val_cond = conditional_eval(X_val, w_val, h, J, device=args.device)
    full_cond = conditional_eval(X_full, weights_full, h, J, device=args.device)

    print("[INFO] Train conditional:", train_cond)
    print("[INFO] Val conditional:", val_cond)
    print("[INFO] Full conditional:", full_cond)

    print("[EVAL 5/6] Generated sequence diagnostics...")
    freq_metrics = frequency_match_metrics(
        X_ref=X_full,
        X_gen=X_gen,
        ref_weights=weights_full,
        q=len(AA_ALPHABET),
    )

    nov_metrics = novelty_metrics(
        train_seqs=all_seqs,
        gen_seqs=gen_seqs,
        X_gen=X_gen,
        X_train=X_full,
    )

    div_metrics = {
        "generated_unique_fraction": unique_fraction(gen_seqs),
        "generated_mean_pairwise_hamming": mean_pairwise_hamming(
            X_gen,
            max_pairs=args.max_pairs,
            seed=args.seed,
        ),
    }

    dup_df = duplicate_report(gen_seqs, top_k=10)

    print("[INFO] Frequency metrics:", freq_metrics)
    print("[INFO] Novelty metrics:", nov_metrics)
    print("[INFO] Diversity metrics:", div_metrics)
    print("[INFO] Top duplicates:")
    print(dup_df.to_string(index=False))

    print("[EVAL 6/6] Score separation diagnostics...")
    X_random = random_sequences_like(
        n=X_gen.shape[0],
        L=X_gen.shape[1],
        q=len(AA_ALPHABET),
        seed=args.seed,
    )

    train_scores = sequence_scores(X_train, h, J, device=args.device)
    val_scores = sequence_scores(X_val, h, J, device=args.device)
    gen_scores = sequence_scores(X_gen, h, J, device=args.device)
    random_scores = sequence_scores(X_random, h, J, device=args.device)

    score_metrics = {}
    score_metrics.update(score_summary("train", train_scores))
    score_metrics.update(score_summary("val", val_scores))
    score_metrics.update(score_summary("generated", gen_scores))
    score_metrics.update(score_summary("random", random_scores))

    print("[INFO] Score metrics:", score_metrics)

    report = {
        "full_N": len(all_seqs),
        "train_N": len(train_seqs),
        "val_N": len(val_seqs),
        "generated_N": len(gen_seqs),
        "L": X_full.shape[1],

        "train_avg_neglogp_per_site": train_cond["avg_neglogp_per_site"],
        "val_avg_neglogp_per_site": val_cond["avg_neglogp_per_site"],
        "full_avg_neglogp_per_site": full_cond["avg_neglogp_per_site"],

        "train_conditional_perplexity": train_cond["conditional_perplexity"],
        "val_conditional_perplexity": val_cond["conditional_perplexity"],
        "full_conditional_perplexity": full_cond["conditional_perplexity"],

        "train_conditional_accuracy": train_cond["conditional_accuracy"],
        "val_conditional_accuracy": val_cond["conditional_accuracy"],
        "full_conditional_accuracy": full_cond["conditional_accuracy"],

        **freq_metrics,
        **nov_metrics,
        **div_metrics,
        **score_metrics,
    }

    print("[EVAL DONE] Writing reports...")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame([report])
    report_df.to_csv(args.output, sep="\t", index=False)

    duplicates_output = args.output.with_name(args.output.stem + "_duplicates.tsv")
    dup_df.to_csv(duplicates_output, sep="\t", index=False)

    human_output = write_human_readable_report(
        args.output,
        all_seqs=all_seqs,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        gen_seqs=gen_seqs,
        L=X_full.shape[1],
        train_cond=train_cond,
        val_cond=val_cond,
        full_cond=full_cond,
        freq_metrics=freq_metrics,
        nov_metrics=nov_metrics,
        div_metrics=div_metrics,
        score_metrics=score_metrics,
        dup_df=dup_df,
    )

    print("[DONE] Evaluation complete.")
    print(f"[INFO] TSV report saved to: {args.output}")
    print(f"[INFO] Human-readable report saved to: {human_output}")
    print(f"[INFO] Duplicate report saved to: {duplicates_output}")


if __name__ == "__main__":
    main()
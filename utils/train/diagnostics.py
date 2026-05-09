from __future__ import annotations

from utils.preprocessing import AA_ALPHABET, decode_sequences, flatten_samples
from utils.eval.eval_utils import (
    avg_neglogp_per_site,
    conditional_accuracy,
    conditional_logits_all_sites,
    conditional_perplexity,
    mean_pairwise_hamming,
    position_entropy,
    position_frequencies,
)
from utils.model import (
    aa_frequency_by_position,
    generation_diagnostics,
    hamming_distance,
    unique_fraction,
)

ALPHABET = AA_ALPHABET


def plot_training_curves(history):
    """
    Display LBFGS training/validation diagnostic curves.

    ``history`` may contain the LBFGS-step metrics emitted by
    :func:`utils.model.fit_potts_plm_lbfgs`:
    ``train_avg_neglogp_per_site``, ``val_avg_neglogp_per_site``,
    ``train_accuracy``, and ``val_accuracy``. For compatibility with older
    callers, ``train_loss``/``val_loss`` and ``train_acc``/``val_acc`` are
    also accepted.
    """
    import matplotlib.pyplot as plt

    steps = history.get("step")

    def x_values(key: str):
        values = history[key]
        if steps is not None and len(steps) == len(values):
            return steps
        return range(len(values))

    loss_train_key = (
        "train_avg_neglogp_per_site"
        if "train_avg_neglogp_per_site" in history
        else "train_loss"
    )
    loss_val_key = (
        "val_avg_neglogp_per_site"
        if "val_avg_neglogp_per_site" in history
        else "val_loss"
    )
    if loss_train_key in history and loss_val_key in history:
        plt.figure()
        plt.plot(x_values(loss_train_key), history[loss_train_key], label="train")
        plt.plot(x_values(loss_val_key), history[loss_val_key], label="validation")
        plt.xlabel("LBFGS optimization step")
        plt.ylabel("average negative log-pseudolikelihood per site")
        plt.legend()
        plt.title("Negative log-pseudolikelihood per site")
        plt.show()

    acc_train_key = "train_accuracy" if "train_accuracy" in history else "train_acc"
    acc_val_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    if acc_train_key in history and acc_val_key in history:
        plt.figure()
        plt.plot(x_values(acc_train_key), history[acc_train_key], label="train")
        plt.plot(x_values(acc_val_key), history[acc_val_key], label="validation")
        plt.xlabel("LBFGS optimization step")
        plt.ylabel("conditional amino acid accuracy")
        plt.legend()
        plt.title("Conditional prediction accuracy")
        plt.show()

    for metric in (
        "train_avg_neglogp_per_site",
        "val_avg_neglogp_per_site",
        "train_accuracy",
        "val_accuracy",
    ):
        if metric not in history:
            continue
        plt.figure()
        plt.plot(x_values(metric), history[metric])
        plt.xlabel("LBFGS optimization step")
        plt.ylabel(metric)
        plt.title(metric)
        plt.show()


__all__ = [
    "AA_ALPHABET",
    "ALPHABET",
    "aa_frequency_by_position",
    "avg_neglogp_per_site",
    "conditional_accuracy",
    "conditional_logits_all_sites",
    "conditional_perplexity",
    "decode_sequences",
    "flatten_samples",
    "generation_diagnostics",
    "hamming_distance",
    "mean_pairwise_hamming",
    "plot_training_curves",
    "position_entropy",
    "position_frequencies",
    "unique_fraction",
]

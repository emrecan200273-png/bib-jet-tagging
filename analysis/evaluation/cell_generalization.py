"""Check train, validation and test losses for the same saved model."""

from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test")


def _binary_labels(values, context):
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{context} must contain a nonempty 1D array of binary labels")
    return values.astype(np.int8, copy=False)


def _history_arrays(history, selected_epoch):
    if not history:
        raise ValueError("Expected at least one epoch in the training history")
    epochs = np.array([row.get("epoch", i) for i, row in enumerate(history, start=1)])
    if epochs.dtype.kind not in "iu" or not np.all(epochs > 0) or np.any(epochs[1:] <= epochs[:-1]):
        raise ValueError("Training history epochs must be positive, increasing integers")
    if selected_epoch not in epochs:
        raise ValueError("The selected epoch is not in the training history")
    train = np.array([row["train_logloss"] for row in history], dtype=np.float64)
    validation = np.array([row["validation_logloss"] for row in history], dtype=np.float64)
    if not np.isfinite(train).all() or not np.isfinite(validation).all():
        raise ValueError("Training history losses must be finite")
    return epochs, train, validation


def summarize_generalization(labels, logits_by_split, train_reference_labels, history, selected_epoch):
    """Compare all splits with dropout off and weights from the training sample.

    Logits must follow the label order. Loss is divided by each split's total
    weight; AUC is unweighted. The training history uses a separate loss scale.
    """
    from sklearn.metrics import roc_auc_score

    reference = _binary_labels(train_reference_labels, "Training reference")
    reference_counts = np.bincount(reference, minlength=2)
    if (reference_counts == 0).any():
        raise ValueError("Training reference must contain both classes")
    class_weights = len(reference) / (2 * reference_counts)

    missing = [split for split in SPLITS if split not in labels or split not in logits_by_split]
    if missing:
        raise ValueError(f"Missing labels or logits for splits: {', '.join(missing)}")
    metrics = {}
    for split in SPLITS:
        y = _binary_labels(labels[split], split)
        logits = np.asarray(logits_by_split[split], dtype=np.float64)
        if logits.shape != y.shape or not np.isfinite(logits).all():
            raise ValueError(f"{split} logits must be finite and have the same shape as the labels")
        weights = class_weights[y]
        # This form stays stable even for very large positive or negative logits.
        losses = np.logaddexp(0, np.where(y == 1, -logits, logits))
        weight_sum = weights.sum()
        n_llp, n_unpaired = np.bincount(y, minlength=2)
        metrics[split] = {
            "n_jets": len(y),
            "n_llp": int(n_llp),
            "n_unpaired": int(n_unpaired),
            "weight_sum": float(weight_sum),
            "weighted_logloss": float(np.dot(losses, weights / weight_sum)),
            "roc_auc": float(roc_auc_score(y, logits)) if n_llp and n_unpaired else None,
        }
    gaps = {}
    for name in ("weighted_logloss", "roc_auc"):
        train, validation = (metrics[split][name] for split in ("train", "validation"))
        gaps[name] = validation - train if train is not None and validation is not None else None

    epochs, _, val_history = _history_arrays(history, selected_epoch)
    best = int(np.flatnonzero(epochs == selected_epoch)[0])
    after = val_history[best + 1:]
    return {
        "splits": metrics,
        "validation_minus_train": gaps,
        "class_weights": {"LLP": float(class_weights[0]), "unpaired": float(class_weights[1])},
        "reference_class_counts": {"LLP": int(reference_counts[0]), "unpaired": int(reference_counts[1])},
        "selected_epoch": int(selected_epoch),
        "validation_trajectory": {
            "epochs_run": len(epochs),
            "selected_epoch_validation_logloss": float(val_history[best]),
            "final_validation_logloss": float(val_history[-1]),
            "final_minus_selected_validation_logloss": float(val_history[-1] - val_history[best]),
            "epochs_after_selected": len(after),
            "highest_validation_logloss_after_selected": float(after.max()) if len(after) else None,
        },
    }


def plot_learning_curves(histories, path):
    """Plot the optimization and validation history without using test scores."""
    import matplotlib.pyplot as plt

    if not histories:
        raise ValueError("Expected a training history for at least one model")
    curves = [
        (name, info["selected_epoch"], _history_arrays(info["history"], info["selected_epoch"]))
        for name, info in histories.items()
    ]
    fig, axes = plt.subplots(
        1, len(curves), figsize=(3.8 * len(curves), 3.35), sharey=True, squeeze=False,
    )
    try:
        for ax, (name, selected, (epochs, train, validation)) in zip(axes[0], curves):
            ax.plot(epochs, train, "--", lw=1.5, color="#596779", label="Train during updates")
            ax.plot(epochs, validation, lw=2, color="#1765ab", label="Validation in eval mode")
            ax.axvline(selected, ls=":", lw=1.5, color="#9a4f22", label="Selected epoch")
            ax.scatter(selected, validation[epochs == selected], s=26, color="#9a4f22", zorder=3)
            ax.set(title=name, xlabel="Epoch")
            ax.grid(alpha=0.18)
        axes[0, 0].set_ylabel("Weighted log loss in fit history")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=min(3, len(names)), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.095, 1, 1))
        path = Path(path)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        return path
    finally:
        plt.close(fig)

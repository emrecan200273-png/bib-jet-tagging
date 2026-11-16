"""Evaluation for the BIB tagging feasibility study.

Labels are 1 = unpaired (BIB enriched), 0 = truth-matched LLP simulation.
Unpaired labels are not verified BIB truth for every jet. A score at or above the
threshold is tagged as BIB.  All rates below describe this selected sample,
not ATLAS event rates or performance on collision data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DELAYED_TIME_NS = 3.0
NEAR_DELAYED_MAX_NS = 10.0


def jet_time_regions(frame):
    """Return aligned masks for all jets, delayed jets, and the 3–10 ns region."""
    time = frame["jet_time_ATLAS"]
    delayed = time.gt(DELAYED_TIME_NS).to_numpy(dtype=bool, na_value=False)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "delayed": delayed,
        "3_to_10_ns": delayed & time.le(NEAR_DELAYED_MAX_NS).to_numpy(dtype=bool, na_value=False),
    }


def _inputs(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and scores must be nonempty, equally sized 1D arrays.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 (LLP) or 1 (BIB).")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must all be finite.")
    return y.astype(np.int8), scores


def choose_threshold(y, scores, target_llp_keep=0.95):
    """Choose the smallest validation cut with the requested LLP keep fraction.

    Keep means ``score < threshold``, so ties matter.  The cut is one floating
    point step above the kth LLP score, with k = ceil(target * n_LLP).  This
    gives at least the requested empirical keep fraction and maximizes BIB
    tagging subject to that constraint.  A tied group can make keep exceed
    the target.  Use validation data only; do not select a cut on test data.
    """
    y, scores = _inputs(y, scores)
    if not np.isfinite(target_llp_keep) or not 0 < target_llp_keep <= 1:
        raise ValueError("target_llp_keep must be in (0, 1].")
    llp = scores[y == 0]
    if not len(llp):
        raise ValueError("Threshold selection needs at least one LLP jet.")
    k = min(len(llp), int(np.ceil(target_llp_keep * len(llp))))
    # Avoid an extra jet from a multiplication such as 0.07 * 100 rounding up.
    while k > 1 and (k - 1) / len(llp) >= target_llp_keep:
        k -= 1
    selected = np.partition(llp, k - 1)[k - 1]
    threshold = float(np.nextafter(selected, np.inf))
    if not np.isfinite(threshold):
        raise ValueError("The LLP scores are too large to form a finite threshold.")
    return threshold


def weighted_cut(frame, scores, weights, target):
    """Choose a validation cut by retained LLP weight instead of jet count."""
    signal = frame["label"].eq(0).to_numpy() & (weights > 0)
    if not signal.any():
        raise ValueError("Common kinematic validation needs LLP jets")
    values, mass = scores[signal], weights[signal]
    ordered = np.argsort(values)
    cumulative = np.cumsum(mass[ordered])
    edge = np.searchsorted(cumulative, target * cumulative[-1])
    return float(np.nextafter(values[ordered[edge]], np.inf))


def rates(frame, scores, threshold, weights=None):
    """Summarize cuts and AUC on finite scores, with optional kinematic weights."""
    weights = np.ones(len(frame)) if weights is None else weights
    keep = (weights > 0) & np.isfinite(scores)
    labels, scores, weights = frame["label"].to_numpy()[keep], scores[keep], weights[keep]
    tagged = scores >= threshold
    result = {"auc": (float(roc_auc_score(labels, scores, sample_weight=weights))
                      if len(np.unique(labels)) == 2 else None)}
    for key, label, success in [("llp_kept", 0, ~tagged), ("unpaired_tagged", 1, tagged)]:
        selected = labels == label
        result[key] = {
            "total": int(selected.sum()),
            "count": int(success[selected].sum()),
            "rate": float(np.average(success[selected], weights=weights[selected])) if selected.any() else None,
        }
    return result


def _event_bootstrap(y, tagged, groups, repeats, seed):
    """Resample events; omit boundary intervals that would imply certainty.

    If every observed class jet succeeds or every jet fails, the empirical
    bootstrap cannot produce an unseen outcome. Returning [0, 0] or [1, 1]
    would therefore hide the uncertainty, especially for small subgroups.
    """
    if repeats == 0:
        return None, None
    codes, unique_groups = pd.factorize(groups, sort=False)
    n_events = len(unique_groups)
    totals = np.column_stack([
        np.bincount(codes, weights=(y == label), minlength=n_events)
        for label in (0, 1)
    ])
    successes = np.column_stack([
        np.bincount(codes, weights=((y == 0) & ~tagged), minlength=n_events),
        np.bincount(codes, weights=((y == 1) & tagged), minlength=n_events),
    ])
    rng = np.random.default_rng(seed)
    rates = np.full((repeats, 2), np.nan)
    for i in range(repeats):
        selected_events = rng.integers(0, n_events, size=n_events)
        denominator = totals[selected_events].sum(axis=0)
        numerator = successes[selected_events].sum(axis=0)
        np.divide(numerator, denominator, out=rates[i], where=denominator > 0)
    intervals = []
    observed_n = totals.sum(axis=0)
    observed_success = successes.sum(axis=0)
    for column in range(2):
        finite = rates[:, column][np.isfinite(rates[:, column])]
        boundary = (observed_n[column] == 0 or observed_success[column] == 0
                    or observed_success[column] == observed_n[column])
        intervals.append(
            [float(x) for x in np.quantile(finite, [0.025, 0.975])]
            if len(finite) and not boundary else None
        )
    # Caller reports BIB tagging before LLP retention.
    return intervals[1], intervals[0]


def evaluate_scores(frame, scores, threshold, *, bootstrap_repeats=200, seed=42):
    """Return JSON-safe jet-level metrics and event-bootstrap rate intervals.

    ``event_group`` must identify the same physical event across all its jets
    and be unique across unrelated source events.  Percentile 95% intervals
    resample whole events (not independent jets).  They describe test-sample
    sampling variation at a fixed validation threshold; they do not include
    model training, threshold selection, label purity or data/MC uncertainty.
    Intervals are omitted (None) for observed rates of exactly 0 or 1 because
    empirical bootstrap resampling cannot bound outcomes absent from the data.
    Average precision depends on the observed BIB fraction reported here.
    """
    required = {"label", "event_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation table lacks columns: {sorted(missing)}")
    if frame["event_group"].isna().any():
        raise ValueError("event_group cannot contain missing values.")
    if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, (int, np.integer)) or bootstrap_repeats < 0:
        raise ValueError("bootstrap_repeats must be a nonnegative integer.")
    if not np.isfinite(threshold):
        raise ValueError("The threshold must be finite.")
    y, scores = _inputs(frame["label"], scores)
    tagged = scores >= threshold
    n_bib = int(np.sum(y == 1))
    n_llp = int(np.sum(y == 0))
    tp = int(np.sum((y == 1) & tagged))
    fn = n_bib - tp
    fp = int(np.sum((y == 0) & tagged))
    tn = n_llp - fp
    bib_ci, llp_ci = _event_bootstrap(
        y, tagged, frame["event_group"].to_numpy(), bootstrap_repeats, seed,
    )
    return {
        "n_jets": int(len(y)),
        "n_events": int(frame["event_group"].nunique()),
        "class_counts": {"BIB": n_bib, "LLP": n_llp},
        "observed_bib_fraction": float(n_bib / len(y)),
        "roc_auc": float(roc_auc_score(y, scores)) if n_bib and n_llp else None,
        "average_precision": float(average_precision_score(y, scores)) if n_bib and n_llp else None,
        "threshold": float(threshold),
        "bib_tag_fraction": float(tp / n_bib) if n_bib else None,
        "llp_keep_fraction": float(tn / n_llp) if n_llp else None,
        "bib_tag_fraction_ci95": bib_ci,
        "llp_keep_fraction_ci95": llp_ci,
        "confusion": {
            "BIB_tagged_BIB": tp, "BIB_kept_as_LLP": fn,
            "LLP_tagged_BIB": fp, "LLP_kept_as_LLP": tn,
        },
        "bootstrap": {
            "unit": "event_group", "repeats": int(bootstrap_repeats),
            "seed": int(seed), "interval": "95% percentile; fixed threshold",
            "boundary_policy": "Boundary intervals omitted; bootstrap cannot bound unseen outcomes.",
        },
    }


def _save(fig, output_dir, name):
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [path]


def _edges(values, *, count=10):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, count + 1)))
    return edges if len(edges) > 1 else None


def make_plots(test_frame, score_sets, threshold, histories, importance, output_dir):
    """Save PNG figures; return their absolute ``Path`` objects.

    ``score_sets`` and ``threshold`` map display names to arrays and fixed
    validation cuts. ``histories`` maps names to XGBoost ``evals_result()``.
    ``importance`` may be None or a table with feature/importance columns and
    an optional model column. The kinematic panels show unweighted test-jet
    fractions, with bins of fewer than 20 class jets omitted.
    """
    if not score_sets:
        raise ValueError("At least one score set is required.")
    y = test_frame["label"].to_numpy()
    scores = {name: _inputs(y, values)[1] for name, values in score_sets.items()}
    if set(scores) - set(threshold):
        raise ValueError("Every score set needs a fixed validation threshold.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    style = {"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
             "axes.spines.right": False, "legend.fontsize": 8}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        for name, values in scores.items():
            if len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, values)
            auc = roc_auc_score(y, values)
            line, = axes[0].plot(fpr, tpr, label=f"{name}: AUC {auc:.3f}")
            tagged = values >= threshold[name]
            axes[0].scatter(np.mean(tagged[y == 0]), np.mean(tagged[y == 1]),
                            color=line.get_color(), s=30, zorder=3)
            precision, recall, _ = precision_recall_curve(y, values)
            ap = average_precision_score(y, values)
            axes[1].plot(recall, precision, label=f"{name}: AP {ap:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        axes[0].set(xlabel="LLP fraction tagged as BIB", ylabel="BIB fraction tagged",
                    title="Test ROC (dots: validation working points)", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].axhline(np.mean(y == 1), color="0.6", linestyle="--",
                        label=f"Observed BIB fraction: {np.mean(y == 1):.3f}")
        axes[1].set(xlabel="BIB recall (fraction tagged)", ylabel="BIB precision among tagged jets",
                    title="Test precision–recall", xlim=(0, 1), ylim=(0, 1.02))
        for ax in axes:
            ax.legend(loc="best")
            ax.grid(alpha=0.2)
        fig.suptitle("Acceptance-limited unpaired vs truth-matched LLP jets; unpaired ≠ pure BIB truth")
        fig.tight_layout()
        paths.extend(_save(fig, output_dir, "test_discrimination"))

        fig, axes = plt.subplots(1, len(scores), figsize=(5.2 * len(scores), 4), squeeze=False)
        for ax, (name, values) in zip(axes.flat, scores.items()):
            bounds = (min(0.0, float(values.min())), max(1.0, float(values.max())))
            bins = np.linspace(*bounds, 41)
            for label, display, color in ((0, "Truth-matched LLP simulation", "#2563a6"), (1, "Unpaired (BIB label)", "#c66b12")):
                selected = values[y == label]
                if len(selected):
                    ax.hist(selected, bins=bins, weights=np.full(len(selected), 1 / len(selected)),
                            histtype="step", linewidth=1.7, color=color, label=display)
            ax.axvline(threshold[name], color="0.25", linestyle="--", label="Validation cut")
            ax.set(xlabel="Model score (higher = more BIB-like)", ylabel="Fraction of class jets per bin",
                   title=name)
            ax.legend(loc="best")
            ax.grid(alpha=0.2)
        fig.suptitle("Test scores; each class normalized separately")
        fig.tight_layout()
        paths.extend(_save(fig, output_dir, "test_scores"))

        columns = [("jet_pt_GeV", r"Jet $p_T$ [GeV]"), ("jet_eta", r"Jet $\eta$"),
                   ("jet_time_ATLAS", "ATLAS jet time [ns]")]
        available = [(column, title) for column, title in columns if column in test_frame]
        if available:
            fig, axes = plt.subplots(2, len(available), figsize=(5 * len(available), 7), squeeze=False)
            for column_index, (column, title) in enumerate(available):
                x = test_frame[column].to_numpy(dtype=float)
                bins = _edges(x)
                if bins is None:
                    for row in range(2):
                        axes[row, column_index].text(0.5, 0.5, "Insufficient variation", ha="center", transform=axes[row, column_index].transAxes)
                    continue
                codes = np.searchsorted(bins, x, side="right") - 1
                codes[x == bins[-1]] = len(bins) - 2
                centers = (bins[:-1] + bins[1:]) / 2
                for name, values in scores.items():
                    tagged = values >= threshold[name]
                    for row, label in enumerate((1, 0)):
                        mask = (y == label) & np.isfinite(x)
                        denominator = np.bincount(codes[mask], minlength=len(centers))
                        success = tagged if label == 1 else ~tagged
                        numerator = np.bincount(codes[mask], weights=success[mask], minlength=len(centers))
                        rate = np.full(len(centers), np.nan)
                        np.divide(numerator, denominator, out=rate, where=denominator >= 20)
                        axes[row, column_index].plot(centers, rate, "o-", label=name, markersize=4)
                for row in range(2):
                    ax = axes[row, column_index]
                    ax.set(xlabel=title, ylabel="BIB tag fraction" if row == 0 else "LLP keep fraction", ylim=(0, 1.04))
                    if column == "jet_pt_GeV" and np.all(bins > 0):
                        ax.set_xscale("log")
                        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                        ax.xaxis.set_minor_formatter(NullFormatter())
                    ax.grid(alpha=0.2)
                axes[0, column_index].legend(loc="best")
            fig.suptitle("Test working-point dependence; equal-occupancy bins, class count ≥ 20")
            fig.tight_layout()
            paths.extend(_save(fig, output_dir, "test_kinematic_dependence"))

        nonempty_histories = {name: history for name, history in (histories or {}).items() if history}
        if nonempty_histories:
            fig, axes = plt.subplots(1, len(nonempty_histories), figsize=(5.2 * len(nonempty_histories), 4), squeeze=False)
            for ax, (name, history) in zip(axes.flat, nonempty_histories.items()):
                for dataset, metrics in history.items():
                    if "logloss" not in metrics:
                        continue
                    display = {"validation_0": "Training", "validation_1": "Validation"}.get(dataset, dataset)
                    ax.plot(np.arange(1, len(metrics["logloss"]) + 1), metrics["logloss"], label=display)
                ax.set(xlabel="Boosting round", ylabel="Binary log loss", title=name)
                ax.legend(loc="best")
                ax.grid(alpha=0.2)
            fig.suptitle("Training and validation loss; test data excluded from fitting")
            fig.tight_layout()
            paths.extend(_save(fig, output_dir, "training_loss"))

        if importance is not None and len(importance):
            if not {"feature", "importance"}.issubset(importance.columns):
                raise ValueError("Importance table needs feature and importance columns.")
            groups = list(importance.groupby("model", sort=False)) if "model" in importance else [("XGBoost", importance)]
            fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.8), squeeze=False)
            for ax, (name, table) in zip(axes.flat, groups):
                table = table.nlargest(12, "importance").sort_values("importance")
                ax.barh(table["feature"], table["importance"], color="#2563a6")
                ax.set(xlabel="XGBoost importance (gain)", title=name)
                ax.grid(axis="x", alpha=0.2)
            fig.suptitle("Model split importance; predictive associations do not establish physics causes")
            fig.tight_layout()
            paths.extend(_save(fig, output_dir, "feature_importance"))
    return paths

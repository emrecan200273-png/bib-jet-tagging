"""Compare the saved baseline with XGBoost trained on zero-vertex unpaired events."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from train_xgboost import (
    MODEL_FEATURES, balanced_weights, fit_booster, predict_model, score_saved_model, write_json,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgb_evaluation import DELAYED_TIME_NS, choose_threshold, evaluate_scores
from xgb_matching import ABS_ETA_EDGES, BIN_COLUMNS, PT_EDGES_GEV, add_kinematic_bins, supported_kinematic_bins


MODEL_NAMES = ("baseline", "zero_vertex")
DISPLAY_NAMES = {"baseline": "Original training", "zero_vertex": "Zero-vertex training"}
SPLITS = ("train", "validation", "test")
JET_ID = ["event_group", "jet_index"]


def load_jets(jet_path, vertex_path):
    jets = pd.read_csv(jet_path)
    vertices = pd.read_csv(
        vertex_path, usecols=["runNumber", "eventNumber", "vertex_count", "status"],
        dtype={"runNumber": "int64", "eventNumber": "int64", "vertex_count": "Int64"},
    )
    if (not vertices["status"].eq("matched").fillna(False).all()
            or not vertices["vertex_count"].ge(0).fillna(False).all()):
        raise ValueError("The vertex mapping has unmatched events or invalid counts")
    keys = "unpaired:" + vertices["runNumber"].astype(str) + ":" + vertices["eventNumber"].astype(str)
    if not keys.is_unique:
        raise ValueError("The vertex mapping repeats a run/event key")
    if jets[["event_group", "jet_index", "split", "label", "sample"]].isna().any().any():
        raise ValueError("The baseline jet table has missing identities, splits or labels")
    if jets.duplicated(JET_ID).any() or jets.groupby("event_group")["split"].nunique().max() != 1:
        raise ValueError("The baseline jet table has duplicate jets or overlapping event splits")
    if not jets["sample"].map({"LLP": 0, "unpaired": 1}).eq(jets["label"]).all():
        raise ValueError("Expected truth-matched LLP=0 and unpaired=1")
    unpaired = jets["label"].eq(1)
    counts = pd.Series(vertices["vertex_count"].array, index=keys)
    jets["vertex_count"] = jets["event_group"].where(unpaired).map(counts)
    if jets.loc[unpaired, "vertex_count"].isna().any():
        raise ValueError("The vertex mapping does not cover every unpaired jet's event")
    selected = jets.loc[jets["label"].eq(0) | jets["vertex_count"].eq(0).fillna(False)].copy()
    for frame in (jets, selected):
        if set(frame["split"]) != set(SPLITS) or not frame.groupby("split")["label"].nunique().eq(2).all():
            raise ValueError("Each original split needs both classes before and after the vertex cut")
    return jets, selected


def population(frame):
    counts = (frame.assign(delayed=frame["jet_time_ATLAS"].gt(DELAYED_TIME_NS))
              .groupby(["split", "sample"], observed=True)
              .agg(jets=("label", "size"), events=("event_group", "nunique"), delayed_jets=("delayed", "sum")))
    return counts.reset_index().to_dict(orient="records")


def paired_interval(frame, scores, thresholds, label, seed, repeats=2000, model_names=MODEL_NAMES):
    """Bootstrap the change in jet fraction by resampling the same events for both models."""
    mask = frame["label"].to_numpy() == label
    if not mask.any():
        return None
    success = [scores[name][mask] >= thresholds[name] for name in model_names]
    difference = success[1].astype(int) - success[0].astype(int)
    if label == 0:
        difference = -difference
    codes, events = pd.factorize(frame.loc[mask, "event_group"], sort=False)
    totals = np.bincount(codes)
    gains = np.bincount(codes, weights=difference)
    interval = None
    if gains.any():
        rng = np.random.default_rng(seed)
        draws = np.empty(repeats)
        for i in range(repeats):
            sample = rng.integers(0, len(events), len(events))
            draws[i] = gains[sample].sum() / totals[sample].sum()
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
    return {"fraction_change": float(difference.mean()), "jet_count_change": int(difference.sum()),
            "improved_decisions": int((difference == 1).sum()), "worsened_decisions": int((difference == -1).sum()),
            "ci95": interval, "events": len(events)}


def compare_region(frame, scores, thresholds, seed):
    metrics = {name: evaluate_scores(frame, scores[name], thresholds[name], seed=seed, bootstrap_repeats=2000)
               for name in MODEL_NAMES}
    delta = {key: paired_interval(frame, scores, thresholds, label, seed)
             for label, key in [(0, "llp_kept"), (1, "unpaired_tagged")]}
    return {"models": metrics, "paired_change": delta}


def kinematic_comparison(train, predictions, model_names=MODEL_NAMES, *, region_masks=None):
    """Compare test rates within shared pT/|eta| bins at the main validation cuts."""
    supported = supported_kinematic_bins(add_kinematic_bins(train))
    binned = add_kinematic_bins(predictions)
    if region_masks is None:
        region_masks = {"all": np.ones(len(binned), dtype=bool),
                        "delayed": binned["jet_time_ATLAS"].gt(DELAYED_TIME_NS)}
    records, pooled = [], {}
    for region, mask in region_masks.items():
        frame = binned.loc[mask]
        rows = []
        for (pt, eta), group in frame.groupby(BIN_COLUMNS, sort=True):
            if (pt, eta) not in supported:
                continue
            llp = group["label"].eq(0)
            llp_count, unpaired_count = int(llp.sum()), int((~llp).sum())
            row = {"region": region, "pt_low_GeV": PT_EDGES_GEV[pt], "pt_high_GeV": PT_EDGES_GEV[pt + 1],
                   "abs_eta_low": ABS_ETA_EDGES[eta], "abs_eta_high": ABS_ETA_EDGES[eta + 1],
                   "llp_jets": llp_count, "unpaired_jets": unpaired_count,
                   "bin_weight": min(llp_count, unpaired_count)}
            for name in model_names:
                tagged = group[f"{name}_tagged"]
                row[f"{name}_llp_kept"] = int((llp & ~tagged).sum())
                row[f"{name}_unpaired_tagged"] = int((~llp & tagged).sum())
            rows.append(row)
        records.extend(rows)
        shared = [row for row in rows if row["bin_weight"]]
        total_weight = sum(row["bin_weight"] for row in shared)
        pooled[region] = {
            "shared_bins": len(shared), "bin_weight_sum": total_weight,
            "llp_jets_in_shared_bins": sum(row["llp_jets"] for row in shared),
            "unpaired_jets_in_shared_bins": sum(row["unpaired_jets"] for row in shared),
            "rates": {f"{name}_{metric}_fraction": (
                sum(row["bin_weight"] * row[f"{name}_{metric}"] / row[denominator] for row in shared) / total_weight
                if total_weight else None)
                for name in model_names
                for metric, denominator in [("llp_kept", "llp_jets"), ("unpaired_tagged", "unpaired_jets")]},
        }
    return pd.DataFrame(records), pooled


def make_comparison_plot(results, path, display_names=DISPLAY_NAMES, title="XGBoost on the same test jets"):
    n_models = len(display_names)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.5 + 1.1 * (n_models - 2)),
                             sharex=True, layout="constrained")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    colors = ("#315987", "#11736d") if n_models == 2 else ("#315987", "#855477", "#11736d", "#9c5a26")
    main = results.loc[results["region"].isin(["all", "delayed"])]
    minimum = min(55, max(0, 10 * np.floor(10 * main[["fraction", "ci95_low"]].min().min()) - 10))
    for row, (region, region_label) in enumerate([("all", "All selected jets"), ("delayed", "ATLAS time > 3 ns")]):
        for column, (metric, metric_label) in enumerate([("unpaired_tagged", "Unpaired jets tagged"),
                                                          ("llp_kept", "LLP jets kept")]):
            ax = axes[row, column]
            subset = results.loc[results["region"].eq(region) & results["metric"].eq(metric)]
            for y, (result, color) in enumerate(zip(subset.itertuples(index=False), colors)):
                if pd.notna(result.ci95_low):
                    ax.hlines(y, 100 * result.ci95_low, 100 * result.ci95_high, color=color, linewidth=2)
                ax.scatter(100 * result.fraction, y, color=color, s=55, zorder=3)
                ax.text(0.015, y, f"{result.successes:,}/{result.total:,}", transform=ax.get_yaxis_transform(), va="center", color=color)
            ax.set(title=f"{region_label} · {metric_label}", xlim=(minimum, 101), ylim=(n_models - .5, -.5),
                   yticks=range(n_models), yticklabels=list(display_names.values()) if column == 0 else [""] * n_models)
            ax.grid(axis="x", alpha=.2)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if row == 1:
                ax.set_xlabel("Percent of jets")
    fig.supxlabel("Thresholds target 95% LLP retention on full validation. Lines show 95% event-bootstrap intervals.\n"
                  "At 0% and 100%, resampling repeats the observed rate and the interval is omitted.", fontsize=9)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def run_study(baseline_dir, vertex_path, output_dir):
    baseline_dir, vertex_path, output_dir = map(lambda path: Path(path).resolve(), (baseline_dir, vertex_path, output_dir))
    if output_dir == baseline_dir or baseline_dir in output_dir.parents:
        raise ValueError("Use a separate output directory for this study")
    baseline_metrics = json.loads((baseline_dir / "metrics.json").read_text())
    schema = json.loads((baseline_dir / "feature_schema.json").read_text())
    features, seed = schema["features"], baseline_metrics["seed"]
    target = baseline_metrics["target_llp_keep_on_validation"]
    if features != MODEL_FEATURES["XGBoost"]:
        raise ValueError("Saved feature list differs from the baseline study")
    jet_path = baseline_dir / "training_jets.csv.gz"
    original, clean = load_jets(jet_path, vertex_path)
    train, validation, test = (clean.loc[clean["split"].eq(split)].copy() for split in SPLITS)

    train_weights = balanced_weights(train["label"], train["label"])
    val_weights = balanced_weights(validation["label"], train["label"])
    print(f"Training on {len(train):,} jets, including {train['label'].sum():,} zero-vertex unpaired jets", flush=True)
    model = fit_booster(train, validation, features, baseline_metrics["training_parameters"],
                        baseline_metrics["early_stopping"]["rounds"], train_weights, val_weights,
                        include_train_eval=True)

    def scores_for(frame):
        baseline = score_saved_model(frame, baseline_dir)
        if not baseline["in_selection"].all():
            raise ValueError("The saved baseline does not accept the selected evaluation jets")
        return {"baseline": baseline["bib_score"].to_numpy(), "zero_vertex": predict_model(model, frame, features)}

    val_scores = scores_for(validation)
    thresholds = {name: choose_threshold(validation["label"], score, target) for name, score in val_scores.items()}
    if thresholds["baseline"] != schema["threshold"]:
        raise ValueError("The baseline's saved validation threshold could not be reproduced")
    val_delayed = validation["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy()
    delayed_thresholds = {name: choose_threshold(validation.loc[val_delayed, "label"], score[val_delayed], target)
                          for name, score in val_scores.items()}
    test_scores = scores_for(test)
    test_delayed = test["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy()
    delayed_scores = {name: score[test_delayed] for name, score in test_scores.items()}
    comparisons = {"all": compare_region(test, test_scores, thresholds, seed),
                   "delayed": compare_region(test.loc[test_delayed], delayed_scores, thresholds, seed),
                   "delayed_at_delayed_validation_target": compare_region(test.loc[test_delayed], delayed_scores, delayed_thresholds, seed)}

    test_columns = ["sample", "label", "event_group", "source_file", "row_index", "jet_index",
                    "jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta", "vertex_count"]
    predictions = test[test_columns].copy()
    for name, score in test_scores.items():
        predictions[f"{name}_score"] = score
        predictions[f"{name}_tagged"] = score >= thresholds[name]
    bins, kinematic = kinematic_comparison(train, predictions)
    results = []
    for region, comparison in comparisons.items():
        for name, values in comparison["models"].items():
            for metric, count, denominator, rate in [
                ("unpaired_tagged", "BIB_tagged_BIB", "BIB", "bib_tag_fraction"),
                ("llp_kept", "LLP_kept_as_LLP", "LLP", "llp_keep_fraction"),
            ]:
                low, high = values[f"{rate}_ci95"] or (None, None)
                results.append({"region": region, "model": name, "metric": metric,
                                "successes": values["confusion"][count], "total": values["class_counts"][denominator],
                                "fraction": values[rate], "ci95_low": low, "ci95_high": high})
    results = pd.DataFrame(results)
    inputs = [jet_path, vertex_path, baseline_dir / "xgboost.ubj", baseline_dir / "metrics.json",
              baseline_dir / "feature_schema.json"]
    fingerprints = {}
    for path in inputs:
        with path.open("rb") as handle:
            fingerprints[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "inputs_sha256": fingerprints, "seed": seed, "features": features,
        "training_parameters": baseline_metrics["training_parameters"], "early_stopping": baseline_metrics["early_stopping"],
        "training_and_validation_selection": "Unpaired vertex_count == 0 for the new model; original LLP selection and event splits retained",
        "baseline_training_population": population(original), "clean_training_population": population(clean),
        "target_llp_keep_on_validation": target, "delayed_time_cut_ns_exclusive": DELAYED_TIME_NS,
        "thresholds_from_all_validation_jets": thresholds, "thresholds_from_delayed_validation_jets": delayed_thresholds,
        "validation_at_main_thresholds": {name: evaluate_scores(validation, score, thresholds[name], bootstrap_repeats=0)
                                          for name, score in val_scores.items()},
        "comparisons_on_same_test_jets": comparisons, "clean_model_trees": model.get_booster().num_boosted_rounds(),
        "kinematic_comparison": {"method": "Same fixed baseline bins; train-supported by at least 3 jets per class; "
                                            "pooled test fractions weight each common bin by its smaller class count; "
                                            "uses the main validation thresholds for both models", "regions": kinematic},
        "limitations": ["Unpaired is a working BIB label, including at zero reconstructed vertices.",
                        "Previously inspected test split; this is an exploratory comparison of one run and one signal sample.",
                        "Paired intervals resample whole test events with thresholds fixed; no training or data/MC uncertainty."],
        "versions": {name: version(name) for name in ("xgboost", "pandas", "numpy", "scikit-learn")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "xgboost.ubj")
    write_json(output_dir / "feature_schema.json", {**schema, "threshold": thresholds["zero_vertex"],
                                                     "training_unpaired_vertex_count": 0})
    reloaded = score_saved_model(test, output_dir)
    if not np.array_equal(reloaded["bib_score"].to_numpy(), test_scores["zero_vertex"]):
        raise AssertionError("The saved clean model did not reproduce its test scores")
    predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False)
    bins.to_csv(output_dir / "kinematic_bins.csv", index=False)
    results.to_csv(output_dir / "results.csv", index=False)
    make_comparison_plot(results, output_dir / "comparison.png")
    write_json(output_dir / "metrics.json", report)
    return results


def main():
    outputs = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=outputs / "xgboost_baseline")
    parser.add_argument("--vertex-mapping", type=Path, default=outputs / "unpaired_vertex_match/event_mapping.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=outputs / "xgboost_vertex_comparison")
    args = parser.parse_args()
    results = run_study(args.baseline_dir, args.vertex_mapping, args.output_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

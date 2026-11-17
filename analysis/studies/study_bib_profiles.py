"""Compare cell timing with a constant time and with both BIB beam directions."""

import argparse
import hashlib
import json
from pathlib import Path

from analysis import PROJECT_ROOT

import numpy as np
import pandas as pd
import uproot

from analysis.data.cell_jet_data import EVENTS_PER_CHUNK, INVALID_CELL, _source_path, read_jet_cell_spans
from analysis.data.data_paths import resolve_data_paths
from analysis.training.train_xgboost_vertex import load_jets
from analysis.evaluation.xgb_evaluation import choose_threshold, jet_time_regions, rates, weighted_cut
from analysis.evaluation.xgb_matching import add_kinematic_bins, overlap_weights, supported_kinematic_bins


C_MM_PER_NS = 299.792458
MIN_CELLS = 4
TARGET = 0.95
SCORES = ("stored_emil", "equal_weights_both_directions", "resolution_weights_both_directions")
CELL_FIELDS = (
    "cell_time", "cell_x", "cell_y", "cell_z", "cell_time_sigma", "cell_e",
    "cell_dEta_jet", "cell_dPhi_jet",
)
BRANCHES = (
    "runNumber", "eventNumber", "jet_pt", "jet_eta", "jet_n_clusters",
    "cluster_n_cells", *CELL_FIELDS,
)
UPSTREAM = {
    "repository": "https://gitlab.cern.ch/ezaya/athjetconstituentsalg-public",
    "inspected_commit": "a7c625c364893bbc90248fbb98a5251c1a54fdeb",
    "source": "AthJetConstituents/src/JetTimingFunctions.cxx:215–259",
}


def _digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _score(flat_variance, bib_variance):
    if not (np.isfinite(flat_variance) and np.isfinite(bib_variance)
            and flat_variance > 0 and bib_variance > 0):
        return np.nan
    return flat_variance / (flat_variance + bib_variance)


def profile_scores(cells):
    """Return the equal-weight score, resolution-weighted score, and Emil replay."""
    time, x, y, z, time_sigma, energy, deta, dphi = cells.T
    spatial_timing = np.column_stack((time, x, y, z))
    usable = np.isfinite(spatial_timing).all(axis=1) & (spatial_timing != INVALID_CELL).all(axis=1)
    model_fields = np.column_stack((energy, deta, dphi))
    model_cells = usable & (energy > 0) & np.isfinite(model_fields).all(axis=1)
    model_cells &= (model_fields != INVALID_CELL).all(axis=1)
    timed_cells = usable & np.isfinite(time_sigma) & (time_sigma > 0)
    distance = np.hypot(np.hypot(x, y), z)
    beam_time = np.column_stack((z - distance, -z - distance)) / C_MM_PER_NS

    def compare(selection, weights):
        times = time[selection]
        mean_time = np.average(times, weights=weights)
        constant_variance = np.average((times - mean_time) ** 2, weights=weights)
        residual = times[:, None] - beam_time[selection]
        # Fit a common time offset before comparing the two timing shapes.
        centered = residual - np.average(residual, axis=0, weights=weights)
        beam_variances = np.average(centered**2, axis=0, weights=weights)
        mean_z = np.average(z[selection], weights=weights)
        time_z_covariance = np.sum(weights * (times - mean_time) * (z[selection] - mean_z))
        positive_direction = time_z_covariance > 0 or (time_z_covariance == 0 and mean_z >= 0)
        return constant_variance, beam_variances, 0 if positive_direction else 1

    equal_weight_score, resolution_weight_score, emil_replay = np.nan, np.nan, np.nan
    if model_cells.sum() >= MIN_CELLS:
        constant_variance, beam_variances, _ = compare(model_cells, np.ones(model_cells.sum()))
        equal_weight_score = _score(constant_variance, beam_variances.min())
    if timed_cells.sum() >= MIN_CELLS:
        weights = 1 / time_sigma[timed_cells] ** 2
        constant_variance, beam_variances, direction = compare(timed_cells, weights)
        resolution_weight_score = _score(constant_variance, beam_variances.min())
        emil_replay = _score(constant_variance, beam_variances[direction])
    return equal_weight_score, resolution_weight_score, emil_replay


def _attach_scores(jets, data_root, manifest):
    expected_sizes = {item["path"]: item["size"] for item in manifest["files"]}
    scores = np.full((len(jets), 3), np.nan)
    visited = np.zeros(len(jets), dtype=bool)
    cells_read = 0
    for relative, group in jets.groupby("source_file", sort=False):
        source = _source_path(data_root, relative)
        if source.stat().st_size != expected_sizes.get(relative):
            raise ValueError(f"ROOT file missing from manifest or size changed: {relative}")
        with uproot.open(source, array_cache=None) as handle:
            tree = handle["JetContent"]
            if set(BRANCHES) - set(tree.keys()):
                raise ValueError(f"Missing BIB profile branches: {relative}")
            for _, subset in group.groupby(group["row_index"] // EVENTS_PER_CHUNK, sort=True):
                cells, spans = read_jet_cell_spans(tree, subset, relative, cell_fields=CELL_FIELDS)
                for index, (start, stop) in zip(subset.index, spans, strict=True):
                    scores[index] = profile_scores(cells[start:stop])
                    visited[index] = True
                    cells_read += stop - start
    if not visited.all():
        raise ValueError("At least one selected jet was not mapped to its cells")
    return scores, cells_read


def _evaluate(frame, name, cut, weights):
    summary = rates(frame, frame[name].to_numpy(), cut, weights)
    result = {"roc_auc": summary["auc"]}
    for sample in ("llp_kept", "unpaired_tagged"):
        result[f"{sample}_total"] = summary[sample]["total"]
        result[f"{sample}_successes"] = summary[sample]["count"]
        result[f"{sample}_fraction"] = summary[sample]["rate"]
    return result


def run_study(project_root):
    root = Path(project_root).resolve()
    outputs = root / "analysis/outputs"
    jet_table = outputs / "xgboost_baseline/training_jets.csv.gz"
    vertex_map = outputs / "unpaired_vertex_match/event_mapping.csv.gz"
    data_root, manifest_path = resolve_data_paths(root)
    manifest = json.loads(manifest_path.read_text())
    _, jets = load_jets(jet_table, vertex_map)
    jets = jets.reset_index(drop=True)
    values, cells_read = _attach_scores(jets, data_root.resolve(), manifest)
    jets["stored_emil"] = jets["jet_BIB_score"]
    for i, name in enumerate(SCORES[1:]):
        jets[name] = values[:, i]
    replay, stored = values[:, 2], jets["stored_emil"].to_numpy()
    finite_replay, finite_stored = np.isfinite(replay), np.isfinite(stored)
    both_valid = finite_replay & finite_stored
    residual = np.abs(replay[both_valid] - stored[both_valid])
    if not both_valid.any():
        raise ValueError("No stored BIB profile scores can be replayed")
    jets["common_finite"] = np.isfinite(jets[list(SCORES)].to_numpy()).all(axis=1)
    jets = add_kinematic_bins(jets)
    train = jets.loc[jets["split"].eq("train")]
    support = supported_kinematic_bins(train)

    coverage = []
    for split, group in jets.groupby("split", sort=False):
        for region, keep in jet_time_regions(group).items():
            for sample, by_class in group.loc[keep].groupby("sample", sort=False):
                coverage.append({
                    "split": split, "region": region, "sample": sample, "jets": len(by_class),
                    "common_finite": int(by_class["common_finite"].sum()),
                    **{name: int(np.isfinite(by_class[name]).sum()) for name in SCORES},
                })
    validation = jets.loc[jets["split"].eq("validation") & jets["common_finite"] & jet_time_regions(jets)["delayed"]]
    test = jets.loc[jets["split"].eq("test") & jets["common_finite"]]
    test_regions = {name: test.loc[keep] for name, keep in jet_time_regions(test).items()}
    results = []
    cuts = {}
    for mode in ("raw", "shared_kinematics"):
        shared = mode == "shared_kinematics"
        weights = overlap_weights(validation, support) if shared else np.ones(len(validation))
        test_weights = {region: overlap_weights(subset, support) if shared else np.ones(len(subset))
                        for region, subset in test_regions.items()}
        cuts[mode] = {}
        for model in SCORES:
            score = validation[model].to_numpy()
            cut = (weighted_cut(validation, score, weights, TARGET) if shared
                   else choose_threshold(validation["label"], score, TARGET))
            cuts[mode][model] = {"value": cut, **_evaluate(validation, model, cut, weights)}
            for region, subset in test_regions.items():
                results.append({"region": region, "population": mode, "score": model,
                                "delayed_validation_threshold": cut, **_evaluate(subset, model, cut, test_weights[region])})

    report = {
        "scope": ("Exploratory jet-level comparison using the locked XGBoost event splits, LLP simulation, "
                  "and zero-vertex unpaired data as the working BIB reference. No classifier is trained."),
        "method": ("For each jet use g±=(±z−hypot(R,z))/c and weighted centered time variance. "
                   "Score is Var(t)/(Var(t)+min Var(t−g±)). The fitted per-jet offset is a weighted mean, "
                   "so a common timing shift does not affect this shape score. "
                   "The stored score uses the weighted time/z covariance to pick a direction."),
        "equal_weights": "Positive-energy cells valid for neural inputs; all cells have weight one.",
        "resolution_weights": "Cells with valid geometry, times and positive time uncertainty; weight 1/σ².",
        "minimum_timed_cells": MIN_CELLS,
        "selection": ("Same finite-score jets for all methods. ATLAS stored jet time > 3 ns defines delayed. "
                      "The 3–10 ns range excludes 3 ns and includes 10 ns."),
        "shared_kinematics": ("Bins supported by at least three jets per class in training. "
                              "Each comparison split/region gives the classes equal total weight within a bin. "
                              "The shared-kinematics cut uses the delayed validation weights."),
        "target_llp_keep_on_delayed_validation": TARGET,
        "calibration": cuts,
        "stored_score_replay": {
            "finite_stored": int(finite_stored.sum()), "finite_replay": int(finite_replay.sum()),
            "common": int(both_valid.sum()), "different_validity": int((finite_stored != finite_replay).sum()),
            "max_absolute_error": float(residual.max()),
            "differences_gt_1e_7": int((residual > 1e-7).sum()),
        },
        "coverage": coverage,
        "test_results": results,
        "provenance": {
            "inputs_sha256": {str(path): _digest(path) for path in (jet_table, vertex_map, manifest_path)},
            "script_sha256": _digest(Path(__file__).resolve()),
            "root_sources": jets["source_file"].nunique(),
            "root_validation": ("File sizes verified against the source manifest; jet identities and "
                                "nesting validated on read. ROOT file contents are not rehashed by this study."),
            "mapped_jets": len(jets), "stored_cells_read": int(cells_read), "stored_score_source": UPSTREAM,
        },
    }
    output_dir = outputs / "bib_profile_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(coverage).to_csv(output_dir / "coverage.csv", index=False)
    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(f"Saved profile comparisons in {run_study(args.project_root)}")

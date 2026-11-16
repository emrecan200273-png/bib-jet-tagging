"""Compare floating-offset BIB timing templates without training a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
from sklearn.metrics import roc_auc_score

from cell_jet_data import EVENTS_PER_CHUNK, INVALID_CELL, _source_path, read_jet_cell_spans
from data_paths import resolve_data_paths
from train_xgboost_vertex import load_jets
from xgb_evaluation import choose_threshold, jet_time_regions, weighted_cut
from xgb_matching import add_kinematic_bins, overlap_weights, supported_kinematic_bins


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
    """Return equal-weight both-direction, resolution-weighted both-direction, and Emil replay."""
    t, x, y, z, sigma, energy, deta, dphi = cells.T
    spatial_timing = np.column_stack((t, x, y, z))
    usable = np.isfinite(spatial_timing).all(axis=1) & (spatial_timing != INVALID_CELL).all(axis=1)
    nn_fields = np.column_stack((energy, deta, dphi))
    nn_cells = usable & (energy > 0) & np.isfinite(nn_fields).all(axis=1)
    nn_cells &= (nn_fields != INVALID_CELL).all(axis=1)
    timed = usable & np.isfinite(sigma) & (sigma > 0)
    distance = np.hypot(np.hypot(x, y), z)
    beam_time = np.column_stack((z - distance, -z - distance)) / C_MM_PER_NS

    def compare(selection, weights):
        times = t[selection]
        mean = np.average(times, weights=weights)
        approximate_flat = np.average((times - mean) ** 2, weights=weights)
        residual = times[:, None] - beam_time[selection]
        centered = residual - np.average(residual, axis=0, weights=weights)
        bib = np.average(centered**2, axis=0, weights=weights)
        mean_z = np.average(z[selection], weights=weights)
        cov_tz = np.sum(weights * (times - mean) * (z[selection] - mean_z))
        return approximate_flat, bib, 0 if cov_tz > 0 or (cov_tz == 0 and mean_z >= 0) else 1

    equal, both, replay = np.nan, np.nan, np.nan
    if nn_cells.sum() >= MIN_CELLS:
        flat, bib, _ = compare(nn_cells, np.ones(nn_cells.sum()))
        equal = _score(flat, bib.min())
    if timed.sum() >= MIN_CELLS:
        flat, bib, direction = compare(timed, 1 / sigma[timed] ** 2)
        both, replay = _score(flat, bib.min()), _score(flat, bib[direction])
    return equal, both, replay


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
    present = weights > 0
    weights = weights[present]
    labels = frame["label"].to_numpy()[present]
    score = frame[name].to_numpy()[present]
    tags = score >= cut
    result = {"roc_auc": float(roc_auc_score(labels, score, sample_weight=weights))
              if len(np.unique(labels)) == 2 else None}
    for sample, label, decisions in (("llp_kept", 0, ~tags), ("unpaired_tagged", 1, tags)):
        sel = labels == label
        result[f"{sample}_total"] = int(sel.sum())
        result[f"{sample}_successes"] = int(decisions[sel].sum())
        result[f"{sample}_fraction"] = float(np.average(decisions[sel], weights=weights[sel])) if sel.any() else None
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
        weights = overlap_weights(validation, support) if mode == "shared_kinematics" else np.ones(len(validation))
        cuts[mode] = {}
        for model in SCORES:
            score = validation[model].to_numpy()
            cut = (weighted_cut(validation, score, weights, TARGET) if mode == "shared_kinematics"
                   else choose_threshold(validation["label"], score, TARGET))
            cuts[mode][model] = {"value": cut, **_evaluate(validation, model, cut, weights)}
            for region, subset in test_regions.items():
                test_weights = overlap_weights(subset, support) if mode == "shared_kinematics" else np.ones(len(subset))
                results.append({"region": region, "population": mode, "score": model,
                                "delayed_validation_threshold": cut, **_evaluate(subset, model, cut, test_weights)})

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
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f"Saved profile comparisons in {run_study(args.project_root)}")

"""Test a lower pT cut, more calorimeter types, and timing-only inputs."""

import argparse
import json
from pathlib import Path

from analysis import PROJECT_ROOT

import numpy as np
import pandas as pd

from analysis.training.train_xgboost import balanced_weights, fit_booster, predict_model, score_saved_model, write_json
from analysis.training.train_xgboost_vertex import JET_ID, load_jets, paired_interval, population
from analysis.data.xgb_data import PHYSICS_FEATURES, prepare_training_data, preserve_splits, tile_fraction_mask
from analysis.evaluation.xgb_evaluation import choose_threshold, jet_time_regions, rates, weighted_cut
from analysis.evaluation.xgb_matching import add_kinematic_bins, overlap_weights, supported_kinematic_bins


def run_study(project_root):
    root = Path(project_root).resolve()
    output = root / "analysis/outputs"
    source = output / "xgboost_baseline"
    old_model = output / "xgboost_vertex_comparison"
    vertex_file = output / "unpaired_vertex_match/event_mapping.csv.gz"
    study = output / "jet_acceptance_study"
    extraction = study / "extraction"
    settings = json.loads((source / "metrics.json").read_text())
    seed, target = settings["seed"], settings["target_llp_keep_on_validation"]
    print("Reading verified files at pT >= 15 GeV with Tile endpoints allowed", flush=True)
    prepare_training_data(root, extraction, seed=seed, min_pt_GeV=15., require_mixed_calorimeters=False)
    old, old_clean = load_jets(source / "training_jets.csv.gz", vertex_file)
    _, expanded = load_jets(extraction / "training_jets.csv.gz", vertex_file)
    expanded, split_audit = preserve_splits(expanded, old, seed)
    expanded["mixed"] = tile_fraction_mask(expanded["jet_TileEnergyFrac"], require_mixed=True)
    expanded["original"] = expanded["mixed"] & expanded["jet_pt_GeV"].ge(30)
    checked = [*JET_ID, "split", "label", "source_file", "row_index"]
    repeated_jets = expanded.loc[expanded["original"], checked].sort_values(JET_ID).reset_index(drop=True)
    original_jets = old_clean[checked].sort_values(JET_ID).reset_index(drop=True)
    if not repeated_jets.equals(original_jets):
        raise ValueError("Expanded extraction did not reproduce the original selected jets and splits")
    expanded = add_kinematic_bins(expanded)
    same_region = expanded.loc[expanded["original"]]
    train, val, test = (expanded.loc[expanded["split"].eq(split)] for split in ("train", "validation", "test"))
    support = supported_kinematic_bins(same_region.loc[same_region["split"].eq("train")])
    val_shared = overlap_weights(val.loc[val["original"]], support)
    fixed_val = val["original"].to_numpy()
    fixed_test = test["original"].to_numpy()
    time_regions = jet_time_regions(test)
    original_regions = {name: fixed_test & mask for name, mask in time_regions.items()}
    extended_test = {f"{name}_pilot_test": mask for name, mask in original_regions.items()}
    extended_test.update(all_expanded_test=time_regions["all"], delayed_expanded_test=time_regions["delayed"])
    features = list(PHYSICS_FEATURES)
    time_features = ["jet_time_ATLAS", "jet_time_spread"]
    selections = [("15_gev_mixed", "mixed", features, False), ("15_gev_all_calorimeters", None, features, False),
                  ("15_gev_all_calorimeters_no_tile_fraction", None, [f for f in features if f != "jet_TileEnergyFrac"], False),
                  ("common_kinematics_timing_only", "original", time_features, True)]
    scores = {"reference": (score_saved_model(val, old_model)["bib_score"].to_numpy(),
                            score_saved_model(test, old_model)["bib_score"].to_numpy())}
    details = {}
    for name, region, inputs, reweight in selections:
        group_train, group_val = (frame.loc[frame[region]] if region else frame for frame in (train, val))
        if reweight:
            train_w, val_w = (overlap_weights(frame, support) for frame in (group_train, group_val))
            group_train, group_val = group_train.loc[train_w > 0], group_val.loc[val_w > 0]
            train_w, val_w = train_w[train_w > 0], val_w[val_w > 0]
        else:
            train_w, val_w = (balanced_weights(frame["label"], group_train["label"]) for frame in (group_train, group_val))
        print(f"Training {name} on {len(group_train):,} jets", flush=True)
        model = fit_booster(group_train, group_val, inputs, settings["training_parameters"],
                            settings["early_stopping"]["rounds"], train_w, val_w)
        details[name] = {"features": inputs, "training_jets": len(group_train), "validation_jets": len(group_val),
                         "trees": model.get_booster().num_boosted_rounds()}
        validation_scores = predict_model(model, val, inputs)
        test_scores = predict_model(model, test, inputs)
        if region:
            validation_scores[~val[region].to_numpy()] = np.nan
            test_scores[~test[region].to_numpy()] = np.nan
        scores[name] = validation_scores, test_scores

    fixed_val_data = val.loc[fixed_val]
    cuts = {name: choose_threshold(fixed_val_data["label"], val_s[fixed_val], target) for name, (val_s, _) in scores.items()}
    stored = json.loads((old_model / "feature_schema.json").read_text())["threshold"]
    if cuts["reference"] != stored:
        raise ValueError("The fixed validation data did not reproduce the saved reference threshold")
    # Preserve the shared-bin population and set a separate validation working point for it.
    shared_cuts = {name: weighted_cut(fixed_val_data, val_s[fixed_val], val_shared, target)
                   for name, (val_s, _) in scores.items()}
    report = {"splits": split_audit, "training": details, "target_llp_keep": target,
              "fixed_validation_cut": cuts, "common_kinematics_validation_cut": shared_cuts,
              "comparisons": {}, "common_kinematics": {}, "paired_fixed_test": {}}
    original_test = test.loc[fixed_test]
    original_scores = {name: score[fixed_test] for name, (_, score) in scores.items()}
    for name, (_, score) in scores.items():
        report["comparisons"][name] = {key: rates(test.loc[mask], score[mask], cuts[name])
                                       for key, mask in extended_test.items()}
        report["common_kinematics"][name] = {}
        for key in ("all", "delayed"):
            mask = original_regions[key]
            sample = test.loc[mask]
            weights = overlap_weights(sample, support)
            report["common_kinematics"][name][key] = rates(sample, score[mask], shared_cuts[name], weights)
        if name != "reference":
            paired_results = {}
            for region, mask in original_regions.items():
                # The masks above span all expanded jets; this table has only the original jets.
                original_mask = mask[fixed_test]
                region_jets = original_test.loc[original_mask]
                region_scores = {model: values[original_mask] for model, values in original_scores.items()}
                paired_results[region] = {
                    metric: paired_interval(region_jets, region_scores, cuts, label, seed,
                                            model_names=("reference", name))
                    for metric, label in [("llp_kept", 0), ("unpaired_tagged", 1)]
                }
            report["paired_fixed_test"][name] = paired_results
    report["selected_jet_counts"] = {
        name: population(frame)
        for name, frame in [("30_gev_mixed", same_region), ("15_gev_mixed", expanded.loc[expanded["mixed"]]),
                            ("15_gev_all_calorimeters", expanded)]
    }
    write_json(study / "metrics.json", report)
    print(f"Saved {study / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    run_study(parser.parse_args().project_root)

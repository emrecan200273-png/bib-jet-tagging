"""Test jet selection and feature shortcuts while preserving existing event splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_xgboost import balanced_weights, fit_booster, predict_model, score_saved_model, write_json
from train_xgboost_vertex import JET_ID, load_jets, paired_interval, population
from xgb_data import PHYSICS_FEATURES, prepare_training_data, preserve_splits, tile_fraction_mask
from xgb_evaluation import choose_threshold, jet_time_regions, rates, weighted_cut
from xgb_matching import add_kinematic_bins, overlap_weights, supported_kinematic_bins


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
    frozen = expanded.loc[expanded["original"], checked]
    if not frozen.sort_values(JET_ID).reset_index(drop=True).equals(old_clean[checked].sort_values(JET_ID).reset_index(drop=True)):
        raise ValueError("Expanded extraction did not reproduce the original selected jets and splits")
    expanded = add_kinematic_bins(expanded)
    same_region = expanded.loc[expanded["original"]]
    train, val, test = (expanded.loc[expanded["split"].eq(split)] for split in ("train", "validation", "test"))
    support = supported_kinematic_bins(same_region.loc[same_region["split"].eq("train")])
    val_shared = overlap_weights(val.loc[val["original"]], support)
    fixed_val = val["original"].to_numpy()
    fixed_test = test["original"].to_numpy()
    time_regions = jet_time_regions(test)
    delayed, near_delayed = (time_regions[name] for name in ("delayed", "3_to_10_ns"))
    extended_test = {"all_pilot_test": fixed_test, "delayed_pilot_test": fixed_test & delayed,
                     "3_to_10_ns_pilot_test": fixed_test & near_delayed, "all_expanded_test": time_regions["all"],
                     "delayed_expanded_test": delayed}
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
        detail = {"features": inputs, "training_jets": len(group_train), "validation_jets": len(group_val),
                  "trees": model.get_booster().num_boosted_rounds()}
        details[name] = detail
        val_s, test_s = (predict_model(model, frame, inputs) for frame in (val, test))
        if region:
            val_s[~val[region].to_numpy()] = np.nan
            test_s[~test[region].to_numpy()] = np.nan
        scores[name] = val_s, test_s

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
        for key, mask in [("all", fixed_test), ("delayed", fixed_test & delayed)]:
            sample = test.loc[mask]
            weights = overlap_weights(sample, support)
            report["common_kinematics"][name][key] = rates(sample, score[mask], shared_cuts[name], weights)
        if name != "reference":
            report["paired_fixed_test"][name] = {
                key: {metric: paired_interval(original_test.loc[mask], {n: s[mask] for n, s in original_scores.items()},
                                              cuts, label, seed, model_names=("reference", name))
                      for metric, label in [("llp_kept", 0), ("unpaired_tagged", 1)]}
                for key, mask in [("all", np.ones(len(original_test), dtype=bool)),
                                  ("delayed", delayed[fixed_test]), ("3_to_10_ns", near_delayed[fixed_test])]
            }
    report["selected_jet_counts"] = {
        name: population(frame)
        for name, frame in [("30_gev_mixed", same_region), ("15_gev_mixed", expanded.loc[expanded["mixed"]]),
                            ("15_gev_all_calorimeters", expanded)]
    }
    write_json(study / "metrics.json", report)
    print(f"Saved {study / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    run_study(parser.parse_args().project_root)

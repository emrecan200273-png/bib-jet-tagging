"""Train from one unpaired run and test on the other runs with stored jet times."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache/matplotlib"))

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from data_paths import resolve_data_paths
from train_xgboost import balanced_weights, fit_booster, predict_model, write_json
from xgb_data import assign_event_splits, preserve_splits, sanitize_values
from xgb_evaluation import DELAYED_TIME_NS, choose_threshold, evaluate_scores, jet_time_regions, rates, weighted_cut
from xgb_matching import add_kinematic_bins, overlap_weights, supported_kinematic_bins


TRAIN_RUN = 484258
TRIGGER = "passedTrigUnpairedIso"
BRANCHES = ("runNumber", "eventNumber", "lumiBlockNumber", "nPrimaryVertices", TRIGGER,
            "jet_pt", "jet_eta", "jet_time", "jet_phi")
FEATURES = {"time_only": ["jet_time_ATLAS"], "kinematics_only": ["jet_pt_GeV", "jet_abs_eta"],
            "time_and_kinematics": ["jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta"]}


def accumulate_counts(counter, values):
    keys, counts = np.unique(values, return_counts=True)
    counter.update({int(key): int(count) for key, count in zip(keys, counts, strict=True)})


def read_sidecar(directory):
    files = sorted(path for path in directory.glob("*.root") if path.is_file() and not path.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No ROOT files in {directory}")
    raw_rows, trigger_rows, wrapped_ids = Counter(), Counter(), Counter()
    parts, file_audit, event_parts = [], [], []
    for number, path in enumerate(files, 1):
        with uproot.open(path) as file:
            tree = file["BIB"]
            file_audit.append({"name": path.name, "size_bytes": path.stat().st_size, "rows": tree.num_entries})
            for chunk in tree.iterate(BRANCHES, step_size=100_000, library="ak"):
                accumulate_counts(raw_rows, ak.to_numpy(chunk.runNumber))
                selected = chunk[(chunk.nPrimaryVertices == 0) & (chunk[TRIGGER] == 1)]
                run = ak.to_numpy(selected.runNumber)
                raw_event = ak.to_numpy(selected.eventNumber)
                lumi = ak.to_numpy(selected.lumiBlockNumber)
                if raw_event.dtype != np.dtype("int32") or np.any(run <= 0) or np.any(lumi < 0):
                    raise ValueError(f"Unexpected event-ID schema in {path.name}")
                # The sidecar stores event IDs in signed 32-bit fields. Preserve
                # all the stored bits, including rows with the sign bit set.
                event = raw_event.astype(np.uint32)
                accumulate_counts(trigger_rows, run)
                accumulate_counts(wrapped_ids, run[raw_event < 0])
                group = [f"unpaired:{r}:{e}" for r, e in zip(run, event, strict=True)]
                event_parts.append(pd.DataFrame({"event_group": group, "lumi": lumi}))
                lengths = ak.to_numpy(ak.num(selected.jet_pt, axis=1))
                fields = {}
                for branch in ("jet_pt", "jet_eta", "jet_time", "jet_phi"):
                    if not np.array_equal(lengths, ak.to_numpy(ak.num(selected[branch], axis=1))):
                        raise ValueError(f"Mismatched {branch} vector lengths in {path.name}")
                    fields[branch] = sanitize_values(ak.to_numpy(ak.flatten(selected[branch], axis=1)))
                pt, eta, time = fields["jet_pt"] / 1000, fields["jet_eta"], fields["jet_time"]
                keep = np.isfinite(pt) & np.isfinite(eta) & np.isfinite(time) & (pt >= 30) & (np.abs(eta) < 1.7)
                if keep.any():
                    parts.append(pd.DataFrame({"run": np.repeat(run, lengths)[keep],
                                               "signed_negative_id": np.repeat(raw_event < 0, lengths)[keep],
                                               "event_group": np.repeat(np.asarray(group, dtype=str), lengths)[keep],
                                               "jet_index": ak.to_numpy(ak.flatten(ak.local_index(selected.jet_pt, axis=1)))[keep],
                                               "jet_pt_GeV": pt[keep], "jet_abs_eta": np.abs(eta[keep]),
                                               "jet_time_ATLAS": time[keep], "eta": eta[keep], "phi": fields["jet_phi"][keep]}))
        if number in (1, len(files)) or number % 25 == 0:
            print(f"Scanned {number}/{len(files)} sidecar files", flush=True)
    events = pd.concat(event_parts, ignore_index=True)
    groups, jets = ["event_group"], ["event_group", "jet_index"]
    if events.drop_duplicates().duplicated(groups).any():
        raise ValueError("A sidecar run/event key maps to multiple luminosity blocks")
    jets_raw = pd.concat(parts, ignore_index=True)
    unique_jets = jets_raw.drop_duplicates()
    if unique_jets.duplicated(jets).any():
        raise ValueError("Conflicting jet values for a repeated run/event/jet key")
    frame = unique_jets.assign(sample="unpaired", label=1)
    audit = {"files": file_audit, "raw_rows_by_run": dict(raw_rows), "selected_event_rows_by_run": dict(trigger_rows),
             "selected_events_with_signed_negative_number_by_run": dict(wrapped_ids),
             "repeated_selected_event_rows": int(events.duplicated(groups).sum()),
             "repeated_selected_jet_rows": len(jets_raw) - len(frame)}
    return frame, audit


def plot_run_comparison(results, path, target):
    names = {"time_only": "Only jet time", "kinematics_only": "Only pT and |eta|", "time_and_kinematics": "Jet time, pT and |eta|"}
    colors = ("#18776f", "#96633d", "#355c99")
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4.7), layout="constrained")
    for ax, (region, title) in zip(axes, [("all", "All times"), ("delayed", "Time > 3 ns"), ("3_to_10_ns", "3 < time ≤ 10 ns")], strict=True):
        calibration = "all" if region == "all" else "delayed"
        group = results.loc[results["region"].eq(region) & results["run"].ne(TRAIN_RUN)
                            & results["validation_region"].eq(calibration)]
        for (name, title_name), color in zip(names.items(), colors, strict=True):
            rows = group.loc[group["model"].eq(name)]
            ax.plot(rows["run"].astype(str), rows["unpaired_tag_fraction"] * 100, marker="o", color=color, label=title_name)
        ax.set_title(title)
        ax.set_xlabel("All-time validation cut" if region == "all" else "Delayed-validation cut", fontsize=9)
        ax.set_ylim(-2, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis="x", rotation=65)
        ax.grid(axis="y", alpha=.24)
    axes[0].set_ylabel("Unpaired jets tagged (%)")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=.96)
    held_out = results.loc[results["run"].ne(TRAIN_RUN), "run"].nunique()
    fig.suptitle(f"Models trained on run {TRAIN_RUN} and tested on {held_out} other runs", fontweight="bold")
    fig.supxlabel(f"Unpaired run · validation cuts target at least {target:.0%} LLP retention in the indicated region", fontsize=9)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def run_study(project_root):
    root = Path(project_root).resolve()
    outputs = root / "analysis/outputs"
    baseline = outputs / "xgboost_baseline"
    prior = pd.read_csv(baseline / "training_jets.csv.gz")
    settings = json.loads((baseline / "metrics.json").read_text())
    seed, target = settings["seed"], settings["target_llp_keep_on_validation"]
    data_root, _ = resolve_data_paths(root)
    directory = data_root / "non_collision/unpaired/user.ezaya.UNPAIRED_nPV_nocells.EXT0"
    unpaired, audit = read_sidecar(directory)
    signal = prior.loc[prior["label"].eq(0)]
    columns = ["event_group", "sample", "label", "jet_pt_GeV", "jet_abs_eta", "jet_time_ATLAS"]
    reference = pd.concat([signal[columns], unpaired.loc[unpaired["run"].eq(TRAIN_RUN), columns]], ignore_index=True)
    reference, split_audit = preserve_splits(assign_event_splits(reference, seed), prior, seed)
    train, validation, test = (reference.loc[reference["split"].eq(split)] for split in ("train", "validation", "test"))
    signal_test = test.loc[test["label"].eq(0), columns]
    if set(signal_test["event_group"]) != set(signal.loc[signal["split"].eq("test"), "event_group"]):
        raise ValueError("The LLP test events changed")
    train_w, val_w = (balanced_weights(frame["label"], train["label"]) for frame in (train, validation))
    support = supported_kinematic_bins(add_kinematic_bins(train))
    delayed_val = validation["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy()
    delayed_val_frame = add_kinematic_bins(validation.loc[delayed_val])
    shared_val_w = overlap_weights(delayed_val_frame, support)
    models, summary = {}, {}
    thresholds = {region: {} for region in ("all", "delayed", "common_delayed")}
    for name, features in FEATURES.items():
        model = fit_booster(train, validation, features, settings["training_parameters"],
                            settings["early_stopping"]["rounds"], train_w, val_w)
        models[name] = model
        val_score = predict_model(model, validation, features)
        thresholds["all"][name] = choose_threshold(validation["label"], val_score, target)
        thresholds["delayed"][name] = choose_threshold(delayed_val_frame["label"], val_score[delayed_val], target)
        thresholds["common_delayed"][name] = weighted_cut(delayed_val_frame, val_score[delayed_val], shared_val_w, target)
        summary[name] = {"features": features, "trees": model.get_booster().num_boosted_rounds(),
                         "validation_llp_keep": {region: rates(frame, scores, thresholds[region][name], weights)["llp_kept"]
                                                 for region, frame, scores, weights in [
                                                     ("all", validation, val_score, None),
                                                     ("delayed", delayed_val_frame, val_score[delayed_val], None),
                                                     ("common_delayed", delayed_val_frame, val_score[delayed_val], shared_val_w),
                                                 ]}}

    per_run = {TRAIN_RUN: test[columns]}
    for run, frame in unpaired.loc[unpaired["run"].ne(TRAIN_RUN)].groupby("run", sort=True):
        per_run[int(run)] = pd.concat([signal_test.assign(signed_negative_id=False),
                                      frame[[*columns, "signed_negative_id"]]], ignore_index=True)
    metrics, results, common, id_check = {}, [], [], []
    for run, frame in sorted(per_run.items()):
        scores = {name: predict_model(models[name], frame, features) for name, features in FEATURES.items()}
        metrics[run] = {}
        regions = jet_time_regions(frame)
        regions["early"] = frame["jet_time_ATLAS"].lt(0).to_numpy()
        for region in ("all", "early", "delayed", "3_to_10_ns"):
            mask = regions[region]
            sample = frame.loc[mask]
            if sample.empty:
                continue
            calibrations = ("all", "delayed") if region in ("delayed", "3_to_10_ns") else ("all",)
            metrics[run][region] = {calibration: {} for calibration in calibrations}
            for calibration in calibrations:
                for name, score in scores.items():
                    threshold = thresholds[calibration][name]
                    stat = evaluate_scores(sample, score[mask], threshold, bootstrap_repeats=300, seed=seed)
                    metrics[run][region][calibration][name] = stat
                    results.append({"run": run, "region": region, "validation_region": calibration, "model": name,
                                    "unpaired_test_jets": stat["class_counts"]["BIB"],
                                    "unpaired_tagged": stat["confusion"]["BIB_tagged_BIB"], "unpaired_tag_fraction": stat["bib_tag_fraction"],
                                    "llp_test_jets": stat["class_counts"]["LLP"], "llp_kept": stat["confusion"]["LLP_kept_as_LLP"],
                                    "llp_keep_fraction": stat["llp_keep_fraction"], "auc": stat["roc_auc"]})
                    if run != TRAIN_RUN and calibration == "delayed":
                        nonnegative = mask & frame["label"].eq(1).to_numpy() & ~frame["signed_negative_id"].to_numpy()
                        id_check.append({"run": run, "region": region, "model": name, "unpaired_jets": int(nonnegative.sum()),
                                         "unpaired_tagged": int((score[nonnegative] >= threshold).sum())})
            if region == "delayed":
                weights = overlap_weights(add_kinematic_bins(sample), support)
                for name, score in scores.items():
                    stat = rates(sample, score[mask], thresholds["common_delayed"][name], weights)
                    common.append({"run": run, "model": name, "effective_weight_per_class": float(weights.sum() / 2),
                                   "unpaired_jets": stat["unpaired_tagged"]["total"], "unpaired_tag_fraction": stat["unpaired_tagged"]["rate"],
                                   "llp_jets": stat["llp_kept"]["total"], "llp_keep_fraction": stat["llp_kept"]["rate"], "auc": stat["auc"]})
    output = outputs / "unpaired_run_study"
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(output / "results.csv", index=False)
    pd.DataFrame(common).to_csv(output / "common_kinematics.csv", index=False)
    plot_run_comparison(results, output / "cross_run.png", target)
    write_json(output / "metrics.json", {
        "selection": {"vertex_count": 0, "trigger_equals_one": TRIGGER, "min_pt_GeV": 30, "max_abs_eta_exclusive": 1.7,
                      "sidecar_time_branch_in_ns": "jet_time", "signal_time_branch_in_ns": "jet_time_ATLAS"},
        "training_runs": [TRAIN_RUN], "split_audit": split_audit, "audit": audit,
        "reference_split_counts": reference.groupby(["split", "sample"]).agg(jets=("label", "size"), events=("event_group", "nunique"))
        .reset_index().to_dict(orient="records"), "thresholds_from_validation": thresholds,
        "target_llp_retention": target, "models": summary, "run_metrics": metrics, "common_kinematics": common,
        "sensitivity_excluding_negative_stored_event_ids": id_check,
        "common_kinematics_method": "Fixed pT/eta bins supported by training. In delayed validation and in each delayed test, "
                                     "weight both classes to the smaller count inside each bin, then summarize weighted rates. "
                                     "The common threshold is selected only with delayed validation data.",
        "scope": "BIB-enriched sidecar sample. No cell information or comparable Tile fraction in the sidecar. "
                 "The simulation retains truth matching and its original jet selection. All runs use the same held-out LLP jets. "
                 "Run 484258 uses its held-out events; other unpaired runs are excluded from all training and threshold selection.",
    })
    print(f"Saved {output / 'results.csv'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    run_study(parser.parse_args().project_root)

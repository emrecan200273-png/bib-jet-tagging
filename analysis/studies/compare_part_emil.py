"""Apply frozen ParT and Emil BIB cuts to the same jets with valid scores."""

import argparse
import json
from pathlib import Path

from analysis import PROJECT_ROOT

# Import the neural trainer first so its PyTorch setup precedes XGBoost imports.
from analysis.training.train_cell_models import apply_scaling, predict_logits, torch
from analysis.data.cell_jet_data import CELL_FEATURES, extract_cells
from analysis.models.cell_models import restore_model
from analysis.data.data_paths import resolve_data_paths
from analysis.training.train_xgboost import write_json
from analysis.training.train_xgboost_vertex import kinematic_comparison, load_jets, paired_interval
from analysis.evaluation.xgb_evaluation import evaluate_scores, jet_time_regions
import numpy as np
import pandas as pd


IDENTITY = ["event_group", "source_file", "row_index", "jet_index"]
MODELS = ("emil", "part")


def prepare_part(jets, data_root, checkpoint, threads):
    if checkpoint["architecture"] != "pairwise_transformer" or checkpoint["max_cells"] is not None:
        raise ValueError("Expected the saved all-cell ParT checkpoint")
    if checkpoint["cell_features"] != list(CELL_FEATURES):
        raise ValueError("The saved model and current cell reader use different features")
    features = checkpoint["jet_features"]
    jet_values = jets[features].to_numpy(np.float32)
    if not np.isfinite(jet_values).all():
        raise ValueError("The jet sample has nonfinite ParT summary features")
    cells, mask, audit = extract_cells(jets, data_root, None)
    scalars = apply_scaling(jet_values, cells, mask, checkpoint["scaling"])
    torch.set_num_threads(threads)
    model = restore_model(checkpoint["architecture"], scalars.shape[-1], cells.shape[-1],
                          checkpoint["state_dict"], **checkpoint.get("model_options", {}))
    inputs = tuple(torch.from_numpy(array) for array in (scalars, cells, mask))
    return model, inputs, audit


def score_part(jets, data_root, checkpoint, settings):
    model, inputs, audit = prepare_part(jets, data_root, checkpoint, settings["threads"])
    lengths = inputs[2].sum(1)
    usable = torch.nonzero(lengths, as_tuple=True)[0]
    probabilities = np.full(len(jets), np.nan, dtype=np.float64)
    if len(usable):
        scores = predict_logits(model, inputs, usable, settings["batch_size"], lengths)
        probabilities[usable.numpy()] = scores.sigmoid().numpy()
    return probabilities, audit


def check_part_replay(jets, reference, replayed):
    if not jets[IDENTITY].reset_index(drop=True).equals(reference[IDENTITY]):
        raise ValueError("The jet identities differ from the saved ParT test sample")
    scores = reference["pairwise_transformer_score"].to_numpy()
    if not np.allclose(replayed, scores, rtol=0, atol=2e-7):
        raise ValueError("Could not reproduce the saved ParT test scores")
    return scores


def run_comparison(root, cell_dir=None, vertex_dir=None, run=None, *, vertex_event_int32_bits=False):
    root = Path(root).resolve()
    outputs = root / "analysis/outputs"
    stored_part = outputs / "pairwise_cell_comparison"
    stored_profiles = outputs / "bib_profile_study/metrics.json"
    part_report = json.loads((stored_part / "metrics.json").read_text())
    profile_report = json.loads(stored_profiles.read_text())
    checkpoint = torch.load(stored_part / "neural_model.pt", weights_only=True)
    cuts = {"part": part_report["thresholds_from_delayed_validation_jets"]["pairwise_transformer"],
            "emil": profile_report["calibration"]["raw"]["stored_emil"]["value"]}
    if checkpoint["delayed_threshold"] != cuts["part"]:
        raise ValueError("ParT checkpoint and validation report have different delayed cuts")
    _, old = load_jets(outputs / "xgboost_baseline/training_jets.csv.gz",
                       outputs / "unpaired_vertex_match/event_mapping.csv.gz")
    control = old.loc[old["split"].eq("test")].reset_index(drop=True)
    data_root, _ = resolve_data_paths(root)
    reference_predictions = pd.read_csv(stored_part / "test_predictions.csv.gz")
    control_scores, extraction = score_part(control, data_root, checkpoint, part_report["settings"])
    reference_scores = check_part_replay(control, reference_predictions, control_scores)
    control = control.assign(part=reference_scores, emil=control["jet_BIB_score"])
    sample = control
    input_audit = None
    if cell_dir is not None:
        from analysis.data.external_tagging_jets import load_external_jets

        if old["event_group"].str.startswith(f"unpaired:{run}:").any():
            raise ValueError(f"Run {run} is part of the original dataset; use the original test mode")
        new, input_audit = load_external_jets(cell_dir, vertex_dir, run,
                                             vertex_event_int32_bits=vertex_event_int32_bits)
        if new.empty:
            raise ValueError(f"Run {run} has no eligible cell jets in these files ({input_audit['status']})")
        values, extraction = score_part(new, Path(cell_dir).resolve(), checkpoint, part_report["settings"])
        new = new.assign(part=values, emil=new["jet_BIB_score"])
        if not np.isfinite(new[list(MODELS)].to_numpy()).all(axis=1).any():
            raise ValueError(f"Run {run} has no jets with both a valid ParT score and a valid stored BIB score")
        sample = pd.concat([control.loc[control["sample"].eq("LLP")], new], ignore_index=True)

    sample = sample.assign(common_scores=np.isfinite(sample[list(MODELS)].to_numpy()).all(axis=1))
    if sample.duplicated(["event_group", "jet_index"]).any():
        raise ValueError("The comparison contains duplicate jets")
    decisions = [f"{name}_tagged" for name in MODELS]
    for name, decision in zip(MODELS, decisions, strict=True):
        sample[decision] = sample[name].ge(cuts[name]).astype("boolean").where(sample["common_scores"])
    valid = sample.loc[sample["common_scores"]]
    regions = jet_time_regions(valid)
    results = {}
    for region, mask in regions.items():
        subset = valid.loc[mask]
        if subset.empty:
            continue
        scores = {name: subset[name].to_numpy() for name in MODELS}
        results[region] = {"models": {name: evaluate_scores(subset, score, cuts[name], seed=42, bootstrap_repeats=2000)
                                      for name, score in scores.items()},
                           "part_minus_emil": {key: paired_interval(subset, scores, cuts, label, 42, model_names=MODELS)
                                               for key, label in [("llp_kept", 0), ("unpaired_tagged", 1)]}}
    bins, shared_rates = kinematic_comparison(old.loc[old["split"].eq("train")], valid, MODELS, region_masks=regions)
    stage = f"run_{run}" if cell_dir is not None else "control_484258"
    output = outputs / "part_emil_same_jets" / stage
    output.mkdir(parents=True, exist_ok=True)
    fields = [*IDENTITY, "sample", "label", "jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta",
              "common_scores", *MODELS, *decisions]
    sample[fields].to_csv(output / "scores.csv.gz", index=False)
    bins.to_csv(output / "kinematic_bins.csv", index=False)
    write_json(output / "metrics.json", {
        "sample": stage, "delayed_validation_cuts": cuts,
        "scope": "Same jets with finite scores. Fixed cuts from the original delayed validation. "
                 "The held-out LLP simulation is shared across the control and any external runs.",
        "coverage": sample.groupby("sample").agg(total=("label", "size"), common=("common_scores", "sum"))
        .reset_index().to_dict(orient="records"),
        "cell_extraction": extraction["overall"], "external_input": input_audit, "comparisons": results,
        "shared_kinematics": {
            "method": "Retain original training-supported pT and |eta| bins populated by both classes in the test region. "
                      "Weight each bin by the smaller test class count. Keep the original validation cuts.",
            "regions": shared_rates,
        },
    })
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cell-dir", type=Path, help="JetContent ROOT files for the external run")
    parser.add_argument("--vertex-dir", type=Path, help="BIB ROOT files with nPrimaryVertices")
    parser.add_argument("--run", type=int, help="Requested run number for the external sample")
    parser.add_argument("--vertex-event-int32-bits", action="store_true",
                        help="Read the sidecar's signed int32 eventNumber bits as an unsigned ID")
    args = parser.parse_args()
    external = [value is not None for value in (args.cell_dir, args.vertex_dir, args.run)]
    if (any(external) or args.vertex_event_int32_bits) and not all(external):
        parser.error("Provide --cell-dir, --vertex-dir and --run together to score an external sample")
    run_comparison(args.project_root, args.cell_dir, args.vertex_dir, args.run,
                   vertex_event_int32_bits=args.vertex_event_int32_bits)

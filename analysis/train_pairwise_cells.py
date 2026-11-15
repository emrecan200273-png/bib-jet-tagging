"""Train and compare uncapped neural models on the same calorimeter-cell data."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import train_cell_models as pilot
from cell_generalization import plot_learning_curves, summarize_generalization
from cell_jet_data import CELL_FEATURES, extract_cells
from cell_models import build_model
from data_paths import resolve_data_paths
from train_xgboost import write_json
from train_xgboost_vertex import kinematic_comparison, load_jets, make_comparison_plot, paired_interval, population
from xgb_evaluation import choose_threshold
import numpy as np
import pandas as pd
import torch


REFERENCE = "transformer_all"
PAIRWISE = "pairwise_transformer"
PARTICLE_NET = "particle_net_lite"
DEEP = "deep_cell_transformer"
BIB = "bib_cell_transformer"
DISPLAY_NAMES = {REFERENCE: "Transformer", PAIRWISE: "ParT-inspired", PARTICLE_NET: "ParticleNet-Lite",
                 DEEP: "Two-block Transformer", BIB: "Two-block Transformer + BIB profile"}
IDENTITY = ["event_group", "source_file", "row_index", "jet_index"]
PARTICLE_NET_BATCH_CELLS = 8192


def fingerprint(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_comparison(project_root, threads, architecture):
    root = Path(project_root).resolve()
    outputs = root / "analysis/outputs"
    graph = architecture == PARTICLE_NET
    deep = architecture == DEEP
    if deep:
        references, trained = (PAIRWISE,), (DEEP, BIB)
        previous_dir = outputs / "pairwise_cell_comparison"
        output_dir = outputs / "deep_cell_transformer_comparison"
    elif graph:
        references, trained = (REFERENCE, PAIRWISE), (PARTICLE_NET,)
        previous_dir = outputs / "pairwise_cell_comparison"
        output_dir = outputs / "particle_net_lite_comparison"
    else:
        references, trained = (REFERENCE,), (PAIRWISE,)
        previous_dir = outputs / "cell_model_full_comparison"
        output_dir = outputs / "pairwise_cell_comparison"
    names = {name: DISPLAY_NAMES[name] for name in (*references, *trained)}
    previous_report = previous_dir / "metrics.json"
    previous_predictions = previous_dir / "test_predictions.csv.gz"
    jet_table = outputs / "xgboost_baseline/training_jets.csv.gz"
    vertex_map = outputs / "unpaired_vertex_match/event_mapping.csv.gz"
    data_root, manifest = resolve_data_paths(root)
    previous = json.loads(previous_report.read_text())
    settings = previous["settings"]
    if settings["max_cells"] is not None or previous["cell_features"] != list(CELL_FEATURES):
        raise ValueError("Expected a saved uncapped comparison with matching cell inputs")
    seed, target = settings["seed"], previous["target_llp_keep_on_validation"]
    batch_size = settings["batch_size"]
    hashes = {str(path): fingerprint(path) for path in
              (jet_table, vertex_map, manifest, previous_report, previous_predictions)}
    for path in (jet_table, vertex_map, manifest):
        if hashes[str(path)] != previous["inputs_sha256"].get(str(path.resolve())):
            raise ValueError(f"Input differs from the saved comparison: {path}")

    _, jets = load_jets(jet_table, vertex_map)
    jets = jets.reset_index(drop=True)
    indices = {split: torch.from_numpy(np.flatnonzero(jets["split"].eq(split))) for split in pilot.SPLITS}
    val_frame, test_frame = (jets.iloc[indices[split].numpy()] for split in ("validation", "test"))
    reference_test = pd.read_csv(previous_predictions)
    if not test_frame[IDENTITY].reset_index(drop=True).equals(reference_test[IDENTITY]):
        raise ValueError("The test jets differ from the saved comparison")
    if not test_frame["label"].reset_index(drop=True).equals(reference_test["label"]):
        raise ValueError("The test labels differ from the saved comparison")

    torch.set_num_threads(threads)
    print(f"Extracting all cells for {len(jets):,} jets", flush=True)
    cells, mask, audit = extract_cells(jets, data_root, None)
    if (audit["overall"]["truncated_jets"] or audit["overall"]["retained_cells"]
            != previous["cell_extraction"]["overall"]["retained_cells"]):
        raise ValueError("The extracted cell count differs from the saved comparison")
    print(f"Mapped {audit['mapped_jets']:,} jets and {audit['overall']['retained_cells']:,} cells", flush=True)
    jet_features = previous["jet_features"]
    scaling = previous["scaling_fit_on_train"]
    scalars = pilot.apply_scaling(jets[jet_features].to_numpy(dtype=np.float32), cells, mask, scaling)
    inputs = tuple(torch.from_numpy(array) for array in (scalars, cells, mask))
    lengths = inputs[2].sum(dim=1)
    labels = torch.from_numpy(jets["label"].to_numpy(dtype=np.float32))
    model_dims = (scalars.shape[-1], cells.shape[-1])

    if deep:
        model_options = {key: scaling[key] for key in ("cell_center", "cell_scale")}
    else:
        model_options = {"angular_scale": scaling["cell_scale"][4:6]} if graph else {}
    max_batch_cells = PARTICLE_NET_BATCH_CELLS if graph else None
    test_scores = {name: reference_test[f"{name}_score"].to_numpy() for name in references}
    delayed_val = val_frame["jet_time_ATLAS"].gt(pilot.DELAYED_TIME_NS).to_numpy()
    cuts = {name: previous["thresholds_from_all_validation_jets"][name] for name in references}
    delayed_cuts = {name: previous["thresholds_from_delayed_validation_jets"][name] for name in references}
    for name in references:
        if not np.array_equal(test_scores[name] >= cuts[name], reference_test[f"{name}_tagged"].to_numpy()):
            raise ValueError(f"The saved predictions and threshold disagree for {name}")
    histories = {name: previous["training"][name] for name in references}
    generalization = {name: previous["generalization"][name] for name in references}
    split_labels = {split: labels[index].numpy() for split, index in indices.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in trained:
        model, history = pilot.fit_network(
            name, inputs, labels, indices["train"], indices["validation"], seed=seed,
            max_epochs=settings["max_epochs"], batch_size=batch_size, patience=settings["early_stop_patience"], lengths=lengths,
            model_options=model_options, max_batch_cells=max_batch_cells,
        )
        model_path = output_dir / (f"{name}.pt" if deep else "neural_model.pt")
        saved = {"architecture": name, "model_options": model_options, "state_dict": model.state_dict(),
                 "jet_features": jet_features, "cell_features": list(CELL_FEATURES), "scaling": scaling, "max_cells": None}
        torch.save(saved, model_path)
        logits = {split: pilot.predict_logits(model, inputs, index, batch_size, lengths, max_batch_cells)
                  for split, index in indices.items()}
        validation_scores = logits["validation"].sigmoid().numpy().astype(np.float64)
        test_scores[name] = logits["test"].sigmoid().numpy().astype(np.float64)
        cuts[name] = choose_threshold(val_frame["label"], validation_scores, target)
        delayed_cuts[name] = choose_threshold(val_frame.loc[delayed_val, "label"], validation_scores[delayed_val], target)
        histories[name] = history
        generalization[name] = summarize_generalization(
            split_labels, {split: values.numpy() for split, values in logits.items()}, split_labels["train"],
            history["history"], history["selected_epoch"],
        )
        saved.update(threshold=cuts[name], delayed_threshold=delayed_cuts[name])
        torch.save(saved, model_path)
        loaded = torch.load(model_path, weights_only=True)
        restored = build_model(loaded["architecture"], *model_dims, **loaded["model_options"])
        restored.load_state_dict(loaded["state_dict"])
        restored_scores = pilot.predict_logits(restored, inputs, indices["test"], batch_size, lengths, max_batch_cells).sigmoid().numpy()
        if not np.array_equal(restored_scores, test_scores[name]):
            raise AssertionError(f"Reloaded model predictions differ for {name}")

    comparisons, results = pilot.compare_test_scores(test_frame, test_scores, cuts, delayed_cuts, seed)
    delayed_test = test_frame["jet_time_ATLAS"].gt(pilot.DELAYED_TIME_NS).to_numpy()
    paired = {}
    for region, subset, thresholds in [("all", np.ones(len(test_frame), dtype=bool), cuts),
                                       ("delayed", delayed_test, cuts),
                                       ("delayed_at_delayed_validation_target", delayed_test, delayed_cuts)]:
        scores = {name: score[subset] for name, score in test_scores.items()}
        pairs = [(reference, name) for reference in references for name in trained]
        if deep:
            pairs.append((DEEP, BIB))
        differences = {(reference, name): {metric: paired_interval(
            test_frame.loc[subset], scores, thresholds, label, seed, model_names=(reference, name),
        ) for label, metric in [(0, "llp_kept"), (1, "unpaired_tagged")]} for reference, name in pairs}
        if deep:
            paired[region] = {f"{name}_minus_{reference}": stats for (reference, name), stats in differences.items()}
        elif graph:
            paired[region] = {reference: differences[(reference, PARTICLE_NET)] for reference in references}
        else:
            paired[region] = differences[(REFERENCE, PAIRWISE)]
    fields = [*IDENTITY, "sample", "label", "vertex_count", "jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta"]
    predictions = test_frame[fields].copy()
    for name, score in test_scores.items():
        predictions[f"{name}_score"] = score
        predictions[f"{name}_tagged"] = score >= cuts[name]
    bins, kinematic = kinematic_comparison(jets.iloc[indices["train"].numpy()], predictions, model_names=names)
    report = {
        "inputs_sha256": hashes, "population": population(jets), "cell_extraction": audit,
        "jet_features": jet_features, "cell_features": list(CELL_FEATURES),
        "scaling_fit_on_train": scaling, "settings": {**settings, "threads": threads,
                                                       "max_cells_per_microbatch": max_batch_cells,
                                                       "model_options": model_options},
        "torch_version": version("torch"), "target_llp_keep_on_validation": target, "trained_architectures": list(trained),
        "thresholds_from_all_validation_jets": cuts, "thresholds_from_delayed_validation_jets": delayed_cuts,
        "training": histories, "generalization": generalization, "test_comparisons": comparisons,
        "paired_test_changes": paired, "kinematic_comparison": {"regions": kinematic},
        "scope": "Exploratory comparison on the reused test split, one unpaired run and one LLP simulation. "
                 "Unpaired is a working BIB label. The neural architectures are adapted to calorimeter cells.",
    }
    predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False)
    bins.to_csv(output_dir / "kinematic_bins.csv", index=False)
    results.to_csv(output_dir / "results.csv", index=False)
    make_comparison_plot(results, output_dir / "comparison.png", names, "Cell models on the same test jets")
    plot_learning_curves({names[name]: histories[name] for name in names}, output_dir / "learning_curves.png")
    write_json(output_dir / "metrics.json", report)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--threads", type=int, default=4)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--particle-net-lite", action="store_true", help="Train ParticleNet-Lite and include both saved Transformers")
    models.add_argument("--deep-transformers", action="store_true", help="Compare two-block Transformers with and without the BIB profile")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    architecture = DEEP if args.deep_transformers else PARTICLE_NET if args.particle_net_lite else PAIRWISE
    print(run_comparison(args.project_root, args.threads, architecture).to_string(index=False))

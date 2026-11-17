"""Train calorimeter-cell models and compare scores on the same saved test split."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from analysis import PROJECT_ROOT

import analysis.training.train_cell_models as pilot
from analysis.evaluation.cell_generalization import plot_learning_curves, summarize_generalization
from analysis.data.cell_jet_data import CELL_FEATURES, extract_cells
from analysis.models.cell_models import restore_model
from analysis.data.data_paths import resolve_data_paths
from analysis.training.train_xgboost import write_json
from analysis.training.train_xgboost_vertex import kinematic_comparison, load_jets, make_comparison_plot, paired_interval, population
from analysis.evaluation.xgb_evaluation import jet_time_regions
import numpy as np
import pandas as pd
import torch


REFERENCE = "transformer_all"
PAIRWISE = "pairwise_transformer"
PARTICLE_NET = "particle_net_lite"
DEEP = "deep_cell_transformer"
BIB = "bib_cell_transformer"
SET = "set_transformer"
CLUSTER = "cluster_transformer"
DISPLAY_NAMES = {REFERENCE: "Transformer", PAIRWISE: "ParT-inspired", PARTICLE_NET: "ParticleNet-Lite",
                 DEEP: "Two-block Transformer", BIB: "Two-block Transformer + BIB profile",
                 SET: "Set Transformer", CLUSTER: "Cluster Transformer"}
# Each choice lists its output folder, saved models to compare against, and models to train.
COMPARISONS = {
    PAIRWISE: ("pairwise_cell_comparison", (REFERENCE,), (PAIRWISE,)),
    PARTICLE_NET: ("particle_net_lite_comparison", (REFERENCE, PAIRWISE), (PARTICLE_NET,)),
    DEEP: ("deep_cell_transformer_comparison", (PAIRWISE,), (DEEP, BIB)),
    SET: ("set_transformer_comparison", (PAIRWISE,), (SET,)),
    CLUSTER: ("cluster_transformer_comparison", (PAIRWISE,), (CLUSTER,)),
}
PARTICLE_NET_BATCH_CELLS = 8192


def _paired_changes(test, test_scores, cuts, delayed_cuts, seed, references, trained):
    """Compare tag rates on the same jets, with intervals sampled by event."""
    pairs = [(reference, name) for reference in references for name in trained]
    if trained == (DEEP, BIB):
        pairs.append((DEEP, BIB))
    labels = {
        (reference if trained == (PARTICLE_NET,) else f"{name}_minus_{reference}"): (reference, name)
        for reference, name in pairs
    }
    changes = {}
    for region, subset, thresholds in pilot.evaluation_regions(test, cuts, delayed_cuts):
        frame = test.loc[subset]
        scores = {name: score[subset] for name, score in test_scores.items()}
        differences = {key: {
            "llp_kept": paired_interval(frame, scores, thresholds, 0, seed, model_names=pair),
            "unpaired_tagged": paired_interval(frame, scores, thresholds, 1, seed, model_names=pair),
        } for key, pair in labels.items()}
        # Keep the direct metric keys for reports that compare just two models.
        changes[region] = next(iter(differences.values())) if len(pairs) == 1 else differences
    return changes


def run_comparison(project_root, threads, architecture, output_override=None):
    """Train a cell model using the saved jet split, scaling and reference scores."""
    root = Path(project_root).resolve()
    outputs = root / "analysis/outputs"
    use_neighbors = architecture == PARTICLE_NET
    use_timing_geometry = architecture == DEEP
    report_delayed_kinematics = architecture in (SET, CLUSTER)
    folder, references, trained = COMPARISONS[architecture]
    reference_folder = "cell_model_full_comparison" if architecture == PAIRWISE else COMPARISONS[PAIRWISE][0]
    reference_dir = outputs / reference_folder
    output_dir = outputs / folder if output_override is None else Path(output_override).resolve()
    names = {name: DISPLAY_NAMES[name] for name in (*references, *trained)}
    reference_metrics_path = reference_dir / "metrics.json"
    reference_predictions_path = reference_dir / "test_predictions.csv.gz"
    jet_table = outputs / "xgboost_baseline/training_jets.csv.gz"
    vertex_map = outputs / "unpaired_vertex_match/event_mapping.csv.gz"
    data_root, manifest = resolve_data_paths(root)
    if output_dir.is_relative_to(data_root.resolve()) or output_dir.is_relative_to(reference_dir):
        raise ValueError("Write the comparison outside the input data and reference outputs")
    reference_metrics = json.loads(reference_metrics_path.read_text())
    settings = reference_metrics["settings"]
    if settings["max_cells"] is not None or reference_metrics["cell_features"] != list(CELL_FEATURES):
        raise ValueError("Expected a saved uncapped comparison with matching cell inputs")
    seed, target = settings["seed"], reference_metrics["target_llp_keep_on_validation"]
    batch_size = settings["batch_size"]
    hashes = pilot.input_hashes((jet_table, vertex_map, manifest, reference_metrics_path, reference_predictions_path))
    for path in (jet_table, vertex_map, manifest):
        if hashes[str(path)] != reference_metrics["inputs_sha256"].get(str(path.resolve())):
            raise ValueError(f"Input differs from the saved comparison: {path}")

    # Load LLP and zero-vertex unpaired jets with their existing event split.
    _, jets = load_jets(jet_table, vertex_map)
    jets = jets.reset_index(drop=True)
    indices = pilot.split_indices(jets)
    validation_frame, test_frame = (jets.iloc[indices[split].numpy()] for split in ("validation", "test"))
    reference_predictions = pd.read_csv(reference_predictions_path)
    identity = pilot.JET_IDENTITY
    if not test_frame[identity].reset_index(drop=True).equals(reference_predictions[identity]):
        raise ValueError("The test jets differ from the saved comparison")
    if not test_frame["label"].reset_index(drop=True).equals(reference_predictions["label"]):
        raise ValueError("The test labels differ from the saved comparison")

    torch.set_num_threads(threads)
    print(f"Extracting all cells for {len(jets):,} jets", flush=True)
    cells, mask, audit, *cluster_ids = extract_cells(
        jets, data_root, None, with_cluster_ids=architecture == CLUSTER,
    )
    if (audit["overall"]["truncated_jets"] or audit["overall"]["retained_cells"]
            != reference_metrics["cell_extraction"]["overall"]["retained_cells"]):
        raise ValueError("The extracted cell count differs from the saved comparison")
    print(f"Mapped {audit['mapped_jets']:,} jets and {audit['overall']['retained_cells']:,} cells", flush=True)
    jet_features = reference_metrics["jet_features"]
    scaling = reference_metrics["scaling_fit_on_train"]
    scaled_jets = pilot.apply_scaling(jets[jet_features].to_numpy(dtype=np.float32), cells, mask, scaling)
    cell_features = list(CELL_FEATURES)
    if cluster_ids:
        cells = np.concatenate((cells, cluster_ids.pop()[..., None]), axis=-1, dtype=np.float32)
        cell_features.append("cluster_index")
    inputs = tuple(torch.from_numpy(array) for array in (scaled_jets, cells, mask))
    lengths = inputs[2].sum(dim=1)
    labels = torch.from_numpy(jets["label"].to_numpy(dtype=np.float32))
    model_dims = (scaled_jets.shape[-1], cells.shape[-1])

    if use_timing_geometry:
        model_options = {key: scaling[key] for key in ("cell_center", "cell_scale")}
    elif use_neighbors:
        model_options = {"angular_scale": scaling["cell_scale"][4:6]}
    else:
        model_options = {}
    max_batch_cells = PARTICLE_NET_BATCH_CELLS if use_neighbors else None
    test_scores = {name: reference_predictions[f"{name}_score"].to_numpy() for name in references}
    cuts = {name: reference_metrics["thresholds_from_all_validation_jets"][name] for name in references}
    delayed_cuts = {name: reference_metrics["thresholds_from_delayed_validation_jets"][name] for name in references}
    for name in references:
        if not np.array_equal(test_scores[name] >= cuts[name], reference_predictions[f"{name}_tagged"].to_numpy()):
            raise ValueError(f"The saved predictions and threshold disagree for {name}")
    histories = {name: reference_metrics["training"][name] for name in references}
    generalization = {name: reference_metrics["generalization"][name] for name in references}
    split_labels = {split: labels[index].numpy() for split, index in indices.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in trained:
        # Fit on training jets and select the epoch on validation jets.
        model, history = pilot.fit_network(
            name, inputs, labels, indices["train"], indices["validation"], seed=seed,
            max_epochs=settings["max_epochs"], batch_size=batch_size, patience=settings["early_stop_patience"], lengths=lengths,
            model_options=model_options, max_batch_cells=max_batch_cells,
        )
        model_path = output_dir / (f"{name}.pt" if use_timing_geometry else "neural_model.pt")
        saved = {"architecture": name, "model_options": model_options, "state_dict": model.state_dict(),
                 "jet_features": jet_features, "cell_features": cell_features, "scaling": scaling, "max_cells": None}
        logits = {split: pilot.predict_logits(model, inputs, index, batch_size, lengths, max_batch_cells)
                  for split, index in indices.items()}
        validation_scores = logits["validation"].sigmoid().numpy().astype(np.float64)
        test_scores[name] = logits["test"].sigmoid().numpy().astype(np.float64)
        # Choose both tag thresholds on validation jets before evaluating test tags.
        cuts[name], delayed_cuts[name] = pilot.validation_cuts(validation_frame, validation_scores, target)
        histories[name] = history
        generalization[name] = summarize_generalization(
            split_labels, {split: values.numpy() for split, values in logits.items()}, split_labels["train"],
            history["history"], history["selected_epoch"],
        )
        saved.update(threshold=cuts[name], delayed_threshold=delayed_cuts[name])
        torch.save(saved, model_path)
        loaded = torch.load(model_path, weights_only=True)
        restored = restore_model(loaded["architecture"], *model_dims, loaded["state_dict"], **loaded["model_options"])
        restored_scores = pilot.predict_logits(restored, inputs, indices["test"], batch_size, lengths, max_batch_cells).sigmoid().numpy()
        if not np.array_equal(restored_scores, test_scores[name]):
            raise AssertionError(f"Reloaded model predictions differ for {name}")

    comparisons, results = pilot.compare_test_scores(test_frame, test_scores, cuts, delayed_cuts, seed)
    paired = _paired_changes(test_frame, test_scores, cuts, delayed_cuts, seed, references, trained)
    # Write one row per test jet, with a score and tag from each model.
    fields = [*identity, "sample", "label", "vertex_count", "jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta"]
    predictions = test_frame[fields].copy()
    for name, score in test_scores.items():
        predictions[f"{name}_score"] = score
        predictions[f"{name}_tagged"] = score >= cuts[name]
    train_frame = jets.iloc[indices["train"].numpy()]
    bins, kinematic = kinematic_comparison(train_frame, predictions, model_names=names)
    tables = {"test_predictions.csv.gz": predictions, "kinematic_bins.csv": bins, "results.csv": results}
    report = {
        "inputs_sha256": hashes, "population": population(jets), "cell_extraction": audit,
        "jet_features": jet_features, "cell_features": cell_features,
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
    if report_delayed_kinematics:
        delayed_predictions = predictions.assign(**{
            f"{name}_tagged": score >= delayed_cuts[name] for name, score in test_scores.items()
        })
        regions = jet_time_regions(test_frame)
        delayed_masks = {key: regions[key] for key in ("delayed", "3_to_10_ns")}
        delayed_bins, delayed_kinematic = kinematic_comparison(
            train_frame, delayed_predictions, model_names=names, region_masks=delayed_masks,
        )
        tables["kinematic_bins_delayed_cut.csv"] = delayed_bins
        report["kinematic_comparison_at_delayed_validation_cut"] = delayed_kinematic
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    make_comparison_plot(results, output_dir / "comparison.png", names, "Cell models on the same test jets")
    plot_learning_curves({names[name]: histories[name] for name in names}, output_dir / "learning_curves.png")
    write_json(output_dir / "metrics.json", report)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, help="Write this run to a separate output folder")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--particle-net-lite", dest="architecture", action="store_const", const=PARTICLE_NET,
                        help="Train ParticleNet-Lite and include both saved Transformers")
    models.add_argument("--deep-transformers", dest="architecture", action="store_const", const=DEEP,
                        help="Compare two-block Transformers with and without the BIB profile")
    models.add_argument("--set-transformer", dest="architecture", action="store_const", const=SET,
                        help="Train a Set Transformer with induced attention and compare against saved ParT")
    models.add_argument("--cluster-transformer", dest="architecture", action="store_const", const=CLUSTER,
                        help="Train a Transformer that pools cells within clusters and compare against saved ParT")
    parser.set_defaults(architecture=PAIRWISE)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    print(run_comparison(args.project_root, args.threads, args.architecture, args.output_dir).to_string(index=False))

"""Train a jet MLP, Deep Sets and a small Transformer on the same selected jets."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

from analysis import PROJECT_ROOT

# Use a single OpenMP thread by default in the mixed PyTorch/XGBoost process.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load PyTorch's native runtime before importing the XGBoost helpers.
import torch
from torch.nn import functional as F

from analysis.training.train_xgboost import balanced_weights, score_saved_model, write_json
from analysis.training.train_xgboost_vertex import (
    DELAYED_TIME_NS, SPLITS, kinematic_comparison, load_jets, make_comparison_plot, population,
)
import numpy as np
import pandas as pd

from analysis.data.cell_jet_data import CELL_FEATURES, extract_cells
from analysis.evaluation.cell_generalization import plot_learning_curves, summarize_generalization
from analysis.models.cell_models import build_model, restore_model
from analysis.data.data_paths import resolve_data_paths
from analysis.evaluation.xgb_evaluation import choose_threshold, comparison_rows, evaluate_scores


DISPLAY_NAMES = {"xgboost": "XGBoost", "jet_mlp": "Jet MLP", "deep_sets": "Deep Sets", "transformer": "Transformer"}
FULL_CELL_NAMES = {"deep_sets_64": "Deep Sets · 64", "deep_sets_all": "Deep Sets · all",
                   "transformer_64": "Transformer · 64", "transformer_all": "Transformer · all"}
MAX_ATTENTION_PAIRS = 2_000_000
JET_IDENTITY = ["event_group", "source_file", "row_index", "jet_index"]


def split_indices(jets):
    """Read the existing split assignments without reshuffling events."""
    return {split: torch.from_numpy(np.flatnonzero(jets["split"].eq(split))) for split in SPLITS}


def input_hashes(paths):
    hashes = {}
    for path in paths:
        with path.open("rb") as handle:
            hashes[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    return hashes


def validation_cuts(frame, scores, target):
    """Choose score cuts that keep the target fraction of validation LLP jets."""
    delayed = frame["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy()
    full_cut = choose_threshold(frame["label"], scores, target)
    delayed_cut = choose_threshold(frame.loc[delayed, "label"], scores[delayed], target)
    return full_cut, delayed_cut


def apply_scaling(jets, cells, mask, scaling):
    """Scale cells in place, then reset padded positions."""
    np.subtract(cells, np.asarray(scaling["cell_center"], dtype=np.float32), out=cells)
    np.divide(cells, np.asarray(scaling["cell_scale"], dtype=np.float32), out=cells)
    cells[~mask] = 0
    if not np.isfinite(cells).all():
        raise ValueError("Nonfinite cell inputs after scaling")
    jet_center = np.asarray(scaling["jet_center"], dtype=np.float32)
    jet_scale = np.asarray(scaling["jet_scale"], dtype=np.float32)
    return (jets - jet_center) / jet_scale


def standardize(jets, cells, mask, train):
    """Fit both scalers on training jets and their valid cells only."""
    if not np.isfinite(jets).all():
        raise ValueError("The jet-only and cell models need finite summary features")
    rows, columns = np.nonzero(mask[train])
    cell_train = cells[train[rows], columns]
    if not len(cell_train) or not np.isfinite(cell_train).all():
        raise ValueError("Need finite training cells to fit their scaler")

    def scaler(values):
        center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale == 0] = 1
        return center, scale

    jet_center, jet_scale = scaler(jets[train])
    cell_center, cell_scale = scaler(cell_train)
    scaling = {"jet_center": jet_center.tolist(), "jet_scale": jet_scale.tolist(),
               "cell_center": cell_center.tolist(), "cell_scale": cell_scale.tolist()}
    return apply_scaling(jets, cells, mask, scaling), cells, scaling


def model_batches(inputs, indices, lengths, max_batch_cells=None):
    """Group similar cell counts to limit memory, keeping every valid cell."""
    if lengths is None:
        yield slice(None), tuple(tensor[indices] for tensor in inputs)
        return
    counts = lengths[indices]
    order = counts.argsort(descending=True, stable=True)
    start = 0
    while start < len(order):
        width = int(counts[order[start]])
        capacity = max(1, MAX_ATTENTION_PAIRS // (width + 1) ** 2)
        if max_batch_cells is not None:
            capacity = max(1, min(capacity, max_batch_cells // max(width, 1)))
        positions = order[start:start + capacity]
        rows = indices[positions]
        yield positions, (inputs[0][rows], inputs[1][rows, :width], inputs[2][rows, :width])
        start += len(positions)


@torch.inference_mode()
def predict_logits(model, inputs, indices, batch_size, lengths=None, max_batch_cells=None):
    model.eval()
    chunks = []
    for batch in indices.split(batch_size):
        logits = torch.empty(len(batch))
        for positions, tensors in model_batches(inputs, batch, lengths, max_batch_cells):
            logits[positions] = model(*tensors)
        chunks.append(logits)
    logits = torch.cat(chunks)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite network output")
    return logits


def evaluate_model(model, inputs, indices, batch_size, labels=None, weights=None, lengths=None, max_batch_cells=None):
    """Return BIB scores, or weighted loss when labels are supplied."""
    logits = predict_logits(model, inputs, indices, batch_size, lengths, max_batch_cells)
    if labels is not None:
        return float(F.binary_cross_entropy_with_logits(logits, labels[indices], weight=weights))
    return logits.sigmoid().numpy().astype(np.float64)


def fit_network(name, inputs, labels, train, validation, *, seed, max_epochs, batch_size, patience, lengths=None,
                model_options=None, max_batch_cells=None):
    """Train until validation stops improving, then restore the best epoch."""
    torch.manual_seed(seed)
    model = build_model(name, inputs[0].shape[-1], inputs[1].shape[-1], **(model_options or {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_weights = torch.from_numpy(balanced_weights(labels[train], labels[train]).astype(np.float32))
    validation_weights = torch.from_numpy(balanced_weights(labels[validation], labels[train]).astype(np.float32))
    best_loss, best_epoch, state = float("inf"), 0, None
    history = []
    print(f"Training {DISPLAY_NAMES.get(name, name.replace('_', ' ').capitalize())}", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.
        for order in torch.randperm(len(train)).split(batch_size):
            batch = train[order]
            optimizer.zero_grad(set_to_none=True)
            # Accumulate smaller pieces before one update for the whole batch.
            for positions, tensors in model_batches(inputs, batch, lengths, max_batch_cells):
                logits = model(*tensors)
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels[batch[positions]], weight=train_weights[order[positions]], reduction="sum",
                ) / len(batch)
                loss.backward()
                train_loss += float(loss.detach()) * len(batch)
            optimizer.step()
        validation_loss = evaluate_model(model, inputs, validation, batch_size, labels, validation_weights, lengths, max_batch_cells)
        history.append({"epoch": epoch, "train_logloss": train_loss / len(train), "validation_logloss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, state = validation_loss, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch}: validation logloss {validation_loss:.4f}", flush=True)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(state)
    print(f"  selected epoch {best_epoch} using validation logloss", flush=True)
    return model, {"selected_epoch": best_epoch, "history": history,
                   "trainable_parameters": sum(p.numel() for p in model.parameters())}


def replay_capped_models(pilot, jets, jet_values, cells, mask, indices, identity):
    report = json.loads((pilot / "metrics.json").read_text())
    checkpoint = torch.load(pilot / "neural_models.pt", weights_only=True)
    if checkpoint["max_cells"] != 64 or report["jet_features"] != checkpoint["jet_features"]:
        raise ValueError("Expected the saved 64-cell pilot")
    if checkpoint["cell_features"] != list(CELL_FEATURES):
        raise ValueError("The 64-cell pilot used a different cell representation")
    width = checkpoint["max_cells"]
    capped_cells, capped_mask = cells[:, :width].copy(), mask[:, :width].copy()
    if capped_cells.shape[1] < width:
        pad = width - capped_cells.shape[1]
        capped_cells = np.pad(capped_cells, ((0, 0), (0, pad), (0, 0)))
        capped_mask = np.pad(capped_mask, ((0, 0), (0, pad)))
    capped_jets = apply_scaling(jet_values, capped_cells, capped_mask, checkpoint["scaling"])
    inputs = tuple(torch.from_numpy(array) for array in (capped_jets, capped_cells, capped_mask))
    saved_test = pd.read_csv(pilot / "test_predictions.csv.gz")
    if not jets.iloc[indices["test"].numpy()][identity].reset_index(drop=True).equals(saved_test[identity]):
        raise ValueError("The 64-cell pilot used different test jets")
    logits, histories, test_scores = {}, {}, {}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(report["settings"]["threads"])
        for architecture in ("deep_sets", "transformer"):
            name = f"{architecture}_64"
            model = restore_model(
                architecture, capped_jets.shape[-1], capped_cells.shape[-1], checkpoint["state_dicts"][architecture],
            )
            logits[name] = {split: predict_logits(model, inputs, index, report["settings"]["batch_size"])
                            for split, index in indices.items()}
            test_scores[name] = saved_test[f"{architecture}_score"].to_numpy()
            if not np.allclose(logits[name]["test"].sigmoid().numpy(), test_scores[name], rtol=0, atol=2e-7):
                raise ValueError(f"Replayed {architecture} scores differ from the saved 64-cell pilot")
            histories[name] = report["training"][architecture]
    finally:
        torch.set_num_threads(threads)
    return report, logits, histories, test_scores


def evaluation_regions(test_frame, thresholds, delayed_thresholds):
    delayed = test_frame["jet_time_ATLAS"].gt(DELAYED_TIME_NS).to_numpy()
    return (("all", np.ones(len(test_frame), dtype=bool), thresholds),
            ("delayed", delayed, thresholds),
            ("delayed_at_delayed_validation_target", delayed, delayed_thresholds))


def compare_test_scores(test_frame, test_scores, thresholds, delayed_thresholds, seed):
    comparisons, rows = {}, []
    for region, subset, cuts in evaluation_regions(test_frame, thresholds, delayed_thresholds):
        frame = test_frame.loc[subset]
        comparisons[region] = {}
        for name in thresholds:
            metrics = evaluate_scores(frame, test_scores[name][subset], cuts[name],
                                      seed=seed, bootstrap_repeats=2000)
            comparisons[region][name] = metrics
            rows.extend(comparison_rows(region, name, metrics))
    return comparisons, pd.DataFrame(rows)


def run_pilot(project_root, max_cells, max_epochs, batch_size, threads):
    torch.set_num_threads(threads)
    root = Path(project_root).resolve()
    outputs = root / "analysis/outputs"
    baseline = outputs / "xgboost_baseline"
    vertex_xgb_dir = outputs / "xgboost_vertex_comparison"
    vertex_path = outputs / "unpaired_vertex_match/event_mapping.csv.gz"
    all_cells = max_cells is None
    capped_pilot = outputs / "cell_model_pilot"
    output_dir = outputs / "cell_model_full_comparison" if all_cells else capped_pilot
    display_names = FULL_CELL_NAMES if all_cells else DISPLAY_NAMES
    data_root, manifest = resolve_data_paths(root)
    schema = json.loads((vertex_xgb_dir / "feature_schema.json").read_text())
    xgb_metrics = json.loads((vertex_xgb_dir / "metrics.json").read_text())
    seed, target = xgb_metrics["seed"], xgb_metrics["target_llp_keep_on_validation"]
    _, jets = load_jets(baseline / "training_jets.csv.gz", vertex_path)
    jets = jets.reset_index(drop=True)
    indices = split_indices(jets)
    train = indices["train"].numpy()
    validation_frame, test_frame = (jets.iloc[indices[split].numpy()] for split in ("validation", "test"))
    identity = JET_IDENTITY
    xgb_predictions = pd.read_csv(vertex_xgb_dir / "test_predictions.csv.gz")
    if not test_frame[identity].reset_index(drop=True).equals(xgb_predictions[identity]):
        raise ValueError("The test jets differ from the vertex study")
    print(f"Extracting cells for {len(jets):,} jets from {jets['source_file'].nunique()} files", flush=True)
    cells, mask, audit = extract_cells(jets, data_root, max_cells)
    print(f"Mapped {audit['mapped_jets']:,} jets and retained {audit['overall']['retained_cells']:,} cells", flush=True)
    jet_values = jets[schema["features"]].to_numpy(dtype=np.float32)
    models, histories, validation_scores, test_scores, logits = {}, {}, {}, {}, {}
    architectures = {name: name for name in ("jet_mlp", "deep_sets", "transformer")}
    if all_cells:
        if audit["overall"]["valid_cells"] != audit["overall"]["retained_cells"] or audit["overall"]["truncated_jets"]:
            raise ValueError("The uncapped input discarded valid cells")
        capped_report, logits, histories, test_scores = replay_capped_models(capped_pilot, jets, jet_values, cells, mask, indices, identity)
        if any(capped_report["settings"][key] != value for key, value in [("seed", seed), ("max_epochs", max_epochs), ("batch_size", batch_size)]):
            raise ValueError("Use the seed, maximum epochs and batch size from the 64-cell pilot")
        validation_scores = {name: scores["validation"].sigmoid().numpy().astype(np.float64) for name, scores in logits.items()}
        architectures = {f"{name}_all": name for name in ("deep_sets", "transformer")}
    scaled_jets, cells, scaling = standardize(jet_values, cells, mask, train)
    inputs = tuple(torch.from_numpy(array) for array in (scaled_jets, cells, mask))
    lengths = inputs[2].sum(dim=1) if all_cells else None
    labels = torch.from_numpy(jets["label"].to_numpy(dtype=np.float32))
    for name, architecture in architectures.items():
        model, histories[name] = fit_network(architecture, inputs, labels, indices["train"], indices["validation"],
                                             seed=seed, max_epochs=max_epochs, batch_size=batch_size, patience=5, lengths=lengths)
        models[name] = model.state_dict()
        eval_splits = SPLITS if all_cells else ("validation", "test")
        logits[name] = {split: predict_logits(model, inputs, indices[split], batch_size, lengths) for split in eval_splits}
        validation_scores[name] = logits[name]["validation"].sigmoid().numpy().astype(np.float64)
        test_scores[name] = logits[name]["test"].sigmoid().numpy().astype(np.float64)
    if not all_cells:
        for frame, score_set in [(validation_frame, validation_scores), (test_frame, test_scores)]:
            xgb = score_saved_model(frame, vertex_xgb_dir)
            if not xgb["in_selection"].all():
                raise ValueError("The saved zero-vertex XGBoost does not accept every pilot evaluation jet")
            score_set["xgboost"] = xgb["bib_score"].to_numpy()
        if not np.array_equal(test_scores["xgboost"], xgb_predictions["zero_vertex_score"].to_numpy()):
            raise ValueError("The saved XGBoost scores differ from the vertex study")

    thresholds, delayed_thresholds = {}, {}
    for name in display_names:
        thresholds[name], delayed_thresholds[name] = validation_cuts(validation_frame, validation_scores[name], target)
    if all_cells:
        for architecture in ("deep_sets", "transformer"):
            name = f"{architecture}_64"
            for current, key in [(thresholds, "thresholds_from_all_validation_jets"),
                                 (delayed_thresholds, "thresholds_from_delayed_validation_jets")]:
                saved = capped_report[key][architecture]
                if not np.isclose(current[name], saved, rtol=0, atol=2e-7):
                    raise ValueError(f"The {architecture} 64-cell validation cut could not be reproduced")
                current[name] = saved
    elif thresholds["xgboost"] != schema["threshold"]:
        raise ValueError("The saved XGBoost working point could not be reproduced")
    comparisons, results = compare_test_scores(test_frame, test_scores, thresholds, delayed_thresholds, seed)
    columns = [*identity, "sample", "label", "vertex_count", "jet_time_ATLAS", "jet_pt_GeV", "jet_abs_eta"]
    predictions = test_frame[columns].copy()
    for name in display_names:
        predictions[f"{name}_score"] = test_scores[name]
        predictions[f"{name}_tagged"] = test_scores[name] >= thresholds[name]
    bins, kinematic = kinematic_comparison(jets.iloc[train], predictions, model_names=display_names)
    sources = [baseline / "training_jets.csv.gz", vertex_path, vertex_xgb_dir / "feature_schema.json",
               vertex_xgb_dir / "test_predictions.csv.gz", manifest]
    if all_cells:
        sources += [capped_pilot / name for name in ("neural_models.pt", "test_predictions.csv.gz", "metrics.json")]
    else:
        sources.append(vertex_xgb_dir / "xgboost.ubj")
    report = {
        "population": population(jets), "cell_extraction": audit, "inputs_sha256": input_hashes(sources),
        "jet_features": schema["features"], "cell_features": list(CELL_FEATURES), "scaling_fit_on_train": scaling,
        "settings": {"seed": seed, "max_cells": max_cells, "max_epochs": max_epochs, "batch_size": batch_size,
                     "threads": threads, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "early_stop_patience": 5,
                     "max_attention_pairs_per_microbatch": MAX_ATTENTION_PAIRS if all_cells else None},
        "torch_version": version("torch"), "target_llp_keep_on_validation": target,
        "delayed_time_cut_ns_exclusive": DELAYED_TIME_NS,
        "thresholds_from_all_validation_jets": thresholds, "thresholds_from_delayed_validation_jets": delayed_thresholds,
        "validation_llp_keep": {name: float((scores[validation_frame["label"].eq(0).to_numpy()] < thresholds[name]).mean())
                                for name, scores in validation_scores.items()},
        "training": histories, "test_comparisons": comparisons,
        "kinematic_comparison": {"method": "Fixed pT and |eta| bins with at least three jets per class in training. "
                                           "Pooled test rates weight shared bins by their smaller class count. "
                                           "Every model uses its main validation threshold.", "regions": kinematic},
        "scope": "Exploratory comparison on the previously used test split, one unpaired run and one LLP simulation. "
                 "Unpaired is a working BIB label. The lightweight Transformer uses masked self-attention on cells.",
    }
    if all_cells:
        split_labels = {split: labels[index].numpy() for split, index in indices.items()}
        report["generalization"] = {name: summarize_generalization(
            split_labels, {split: values.numpy() for split, values in logits[name].items()},
            split_labels["train"], histories[name]["history"], histories[name]["selected_epoch"],
        ) for name in display_names}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dicts": models, "jet_features": schema["features"], "cell_features": list(CELL_FEATURES),
                  "scaling": scaling, "max_cells": max_cells, "thresholds": thresholds, "architectures": architectures}
    path = output_dir / "neural_models.pt"
    torch.save(checkpoint, path)
    saved = torch.load(path, weights_only=True)
    for name, state in saved["state_dicts"].items():
        restored = restore_model(architectures[name], scaled_jets.shape[-1], cells.shape[-1], state)
        if not np.array_equal(evaluate_model(restored, inputs, indices["test"], batch_size, lengths=lengths), test_scores[name]):
            raise AssertionError(f"Reloaded {name} predictions differ")
    predictions.to_csv(output_dir / "test_predictions.csv.gz", index=False)
    bins.to_csv(output_dir / "kinematic_bins.csv", index=False)
    results.to_csv(output_dir / "results.csv", index=False)
    make_comparison_plot(results, output_dir / "comparison.png", display_names, "Cell models on the same test jets")
    if all_cells:
        plot_learning_curves({display_names[name]: histories[name] for name in display_names}, output_dir / "learning_curves.png")
    write_json(output_dir / "metrics.json", report)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    cell_count = parser.add_mutually_exclusive_group()
    cell_count.add_argument("--max-cells", type=int, default=64)
    cell_count.add_argument("--all-cells", action="store_true", help="Compare an uncapped run to the saved 64-cell pilot")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.max_cells, args.max_epochs, args.batch_size, args.threads) < 1:
        parser.error("Cell cap, epochs, batch size and threads must be positive")
    max_cells = None if args.all_cells else args.max_cells
    print(run_pilot(args.project_root, max_cells, args.max_epochs, args.batch_size, args.threads).to_string(index=False))


if __name__ == "__main__":
    main()

"""Check cell-count and cell-time shortcuts against the saved ParT model."""

import argparse
import json
from pathlib import Path

from train_cell_models import predict_logits, torch
from cell_jet_data import CELL_FEATURES
from compare_part_emil import check_part_replay, prepare_part
from data_paths import resolve_data_paths
from train_xgboost import write_json
from train_xgboost_vertex import load_jets, paired_interval
from xgb_evaluation import choose_threshold, evaluate_scores, jet_time_regions
import numpy as np
import pandas as pd


def run_diagnostics(root):
    root = Path(root).resolve()
    outputs = root / "analysis/outputs"
    saved = outputs / "pairwise_cell_comparison"
    checkpoint = torch.load(saved / "neural_model.pt", weights_only=True)
    report = json.loads((saved / "metrics.json").read_text())
    settings = report["settings"]
    seed, batch_size = settings["seed"], settings["batch_size"]
    data_root, _ = resolve_data_paths(root)
    _, clean = load_jets(outputs / "xgboost_baseline/training_jets.csv.gz",
                        outputs / "unpaired_vertex_match/event_mapping.csv.gz")
    jets = clean.loc[clean["split"].isin(["validation", "test"])].reset_index(drop=True)
    model, inputs, audit = prepare_part(jets, data_root, checkpoint, settings["threads"])
    indices = {split: np.flatnonzero(jets["split"].eq(split)) for split in ("validation", "test")}
    val, test = (jets.iloc[indices[split]] for split in indices)
    test_idx = torch.from_numpy(indices["test"])
    lengths = inputs[2].sum(1)
    counts = lengths.numpy()

    def score():
        return predict_logits(model, inputs, test_idx, batch_size, lengths).sigmoid().numpy().astype(np.float64)

    reference = pd.read_csv(saved / "test_predictions.csv.gz")
    original_scores = check_part_replay(test, reference, score())

    rng = np.random.default_rng(seed)
    times = inputs[1].numpy()[:, :, CELL_FEATURES.index("cell_time_ns")]
    shuffled = {}
    original_times = times.copy()
    for repeat in range(3):
        for index in indices["test"]:
            length = counts[index]
            times[index, :length] = rng.permutation(original_times[index, :length])
        shuffled[f"shuffled_times_{repeat + 1}"] = score()

    multiplicity_score = -counts.astype(float)
    count_cut = choose_threshold(val["label"], multiplicity_score[indices["validation"]],
                                 report["target_llp_keep_on_validation"])
    scores = {"part": original_scores, **shuffled,
              "negative_cell_count": multiplicity_score[indices["test"]]}
    cuts = {name: count_cut if name == "negative_cell_count" else checkpoint["threshold"] for name in scores}
    comparisons, differences = {}, {}
    for region, keep in jet_time_regions(test).items():
        subset = test.loc[keep]
        local = {name: values[keep] for name, values in scores.items()}
        comparisons[region] = {name: evaluate_scores(subset, values, cuts[name], seed=seed) for name, values in local.items()}
        differences[region] = {name: {metric: paired_interval(subset, local, cuts, label, seed, model_names=("part", name))
                                       for label, metric in [(0, "llp_kept"), (1, "unpaired_tagged")]}
                               for name in shuffled}
    output = outputs / "cell_input_study"
    output.mkdir(exist_ok=True)
    write_json(output / "metrics.json", {
        "extraction": audit["overall"], "cuts": cuts,
        "scores": "Reuse the stored ParT cut after time permutation. The count-only cut is chosen on validation.",
        "interpretation": "Within-jet time permutation is a stress test of time association with other cell features.",
        "median_cell_count_test": {name: float(np.median(counts[indices["test"]][test["sample"].eq(name)]))
                                   for name in ("LLP", "unpaired")},
        "comparisons": comparisons, "paired_permutation_changes": differences,
    })
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    run_diagnostics(parser.parse_args().project_root)

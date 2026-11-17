"""A secondary, conditional comparison with equal pT/eta-bin class counts.

This reduces obvious sample-spectrum differences. It is not a replacement for
independent-run or data/MC validation, and finite bins leave residual differences.
"""
import numpy as np
import pandas as pd


PT_EDGES_GEV = np.geomspace(30., 3000., 22)
ABS_ETA_EDGES = np.linspace(0., 1.7, 10)
SPLITS = ("train", "validation", "test")
MIN_TRAIN_JETS_PER_CLASS_PER_BIN = 3
MIN_HELD_OUT_JETS_PER_CLASS = 200
BIN_COLUMNS = ["pt_bin", "eta_bin"]


def _bin_indices(values, edges):
    """Histogram-style bins: left closed, with the final upper edge included."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    bins = np.full(len(values), -1, dtype=int)
    bins[valid] = np.minimum(
        np.searchsorted(edges, values[valid], side="right") - 1, len(edges) - 2,
    )
    return bins


def add_kinematic_bins(frame):
    """Return a copy with the fixed pT and |eta| bin IDs; -1 is out of range."""
    return frame.assign(
        pt_bin=_bin_indices(frame["jet_pt_GeV"], PT_EDGES_GEV),
        eta_bin=_bin_indices(frame["jet_abs_eta"], ABS_ETA_EDGES),
    )


def supported_kinematic_bins(binned_train):
    """Find bins with enough jets of both classes in the training split."""
    in_range = binned_train[BIN_COLUMNS].ge(0).all(axis=1)
    counts = (binned_train.loc[in_range].groupby([*BIN_COLUMNS, "label"], observed=True).size()
              .unstack("label", fill_value=0).reindex(columns=[0, 1], fill_value=0))
    supported = counts.min(axis=1).ge(MIN_TRAIN_JETS_PER_CLASS_PER_BIN)
    return {(int(pt), int(eta)) for pt, eta in counts.index[supported]}


def overlap_weights(frame, support):
    """Match the two class totals within each supported bin, without subsampling."""
    weights = np.zeros(len(frame))
    for key, group in frame.reset_index(drop=True).groupby(BIN_COLUMNS, sort=False):
        if key not in support:
            continue
        counts = group["label"].value_counts()
        if len(counts) == 2:
            weights[group.index] = counts.min() / group["label"].map(counts)
    return weights


def match_kinematics(frame, *, seed=42):
    """Return a matched copy and audit while preserving original event splits.

    Training alone defines supported bins (at least three jets of each class).
    Within each supported bin and each existing split, independently draw the
    smaller class count from each class without replacement. Neither model
    scores nor timing/shower features are read to select rows.
    """
    required = {
        "label", "split", "event_group", "source_file", "jet_index",
        "jet_pt_GeV", "jet_abs_eta",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Kinematic matching lacks columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        # Nonfinite kinematics may be excluded below, but identities must exist.
        identity = list(required - {"jet_pt_GeV", "jet_abs_eta"})
        if frame.empty or frame[identity].isna().any().any():
            raise ValueError("Matching requires nonempty data and complete identities/labels/splits")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Matching requires both labels, LLP=0 and unpaired=1")
    if set(frame["split"].unique()) != set(SPLITS):
        raise ValueError("Matching requires the existing train, validation and test splits")
    if frame.groupby("event_group")["split"].nunique().max() != 1:
        raise ValueError("An event group appears in more than one original split")
    if frame.groupby("event_group")["label"].nunique().max() != 1:
        raise ValueError("An event group has conflicting class labels")
    if frame.duplicated(["event_group", "jet_index"]).any():
        raise ValueError("Matching input repeats the same physical jet")

    # Sorting before sampling makes selections stable under input row reordering.
    sort_keys = ["event_group", "jet_index", "source_file"]
    if "row_index" in frame.columns:
        sort_keys.append("row_index")
    work = add_kinematic_bins(frame.sort_values(sort_keys, kind="stable").reset_index(drop=True))
    in_range = work[BIN_COLUMNS].ge(0).all(axis=1).to_numpy()
    supported = supported_kinematic_bins(work.loc[work["split"].eq("train")])
    if not supported:
        raise ValueError("No training bin contains at least three jets of each class")
    work_bins = list(zip(work["pt_bin"], work["eta_bin"]))
    supported_mask = np.fromiter((key in supported for key in work_bins), dtype=bool, count=len(work))
    # Positions in each group follow the stable row order used for sampling.
    by_bin = work.groupby(["split", *BIN_COLUMNS, "label"], observed=True, sort=False).indices
    # Separate reproducible random streams keep one split's draw independent of
    # the numbers or row order in the other splits.
    streams = np.random.SeedSequence(seed).spawn(len(SPLITS))
    selected_indices, bin_records, split_records = [], [], []
    for split, stream in zip(SPLITS, streams):
        rng = np.random.default_rng(stream)
        is_split = work["split"].eq(split).to_numpy()
        selected_count = 0
        for pt_bin, eta_bin in sorted(supported):
            classes = [by_bin.get((split, pt_bin, eta_bin, label), ()) for label in (0, 1)]
            pairs = min(map(len, classes))
            if pairs:
                for positions in classes:
                    selected_indices.extend(rng.choice(positions, size=pairs, replace=False).tolist())
            selected_count += pairs
            bin_records.append({
                "split": split, "pt_bin": pt_bin, "eta_bin": eta_bin,
                "pt_low_GeV": float(PT_EDGES_GEV[pt_bin]),
                "pt_high_GeV": float(PT_EDGES_GEV[pt_bin + 1]),
                "abs_eta_low": float(ABS_ETA_EDGES[eta_bin]),
                "abs_eta_high": float(ABS_ETA_EDGES[eta_bin + 1]),
                "input_LLP": len(classes[0]), "input_unpaired": len(classes[1]),
                "selected_LLP": pairs, "selected_unpaired": pairs, "pairs": pairs,
            })
        for label, name in [(0, "LLP"), (1, "unpaired")]:
            rows = is_split & work["label"].eq(label).to_numpy()
            supported_count = int(np.count_nonzero(rows & supported_mask))
            split_records.append({
                "split": split, "sample": name, "label": label,
                "input_jets": int(rows.sum()),
                "excluded_out_of_range": int(np.count_nonzero(rows & ~in_range)),
                "excluded_unsupported": int(np.count_nonzero(rows & in_range & ~supported_mask)),
                "supported_input_jets": supported_count,
                "excluded_class_excess": supported_count - selected_count,
                "selected_jets": selected_count,
            })
        if split != "train" and selected_count < MIN_HELD_OUT_JETS_PER_CLASS:
            raise ValueError(
                f"Kinematic matching leaves too few {split} jets: {dict.fromkeys((0, 1), selected_count)}; "
                f"need at least {MIN_HELD_OUT_JETS_PER_CLASS} per class"
            )

    matched = work.loc[sorted(selected_indices)].reset_index(drop=True)
    if matched.groupby("event_group")["split"].nunique().max() != 1:
        raise ValueError("Matching introduced event leakage between splits")
    matched_counts = (
        matched.groupby(["split", "pt_bin", "eta_bin", "label"], observed=True).size()
        .unstack("label", fill_value=0).reindex(columns=[0, 1], fill_value=0)
    )
    if not matched_counts[0].eq(matched_counts[1]).all():
        raise ValueError("Matching did not produce equal class counts in every split/bin")
    event_summary = (
        matched.groupby(["split", "label"], observed=True)
        .agg(jets=("event_group", "size"), events=("event_group", "nunique"))
        .reset_index().to_dict(orient="records")
    )
    audit = {
        "purpose": "Exploratory conditional comparison after pT/absEta-bin balancing",
        "seed": int(seed), "pt_edges_GeV": PT_EDGES_GEV.tolist(),
        "abs_eta_edges": ABS_ETA_EDGES.tolist(),
        "edge_convention": "Left-closed bins; final upper edge included; nonfinite/outside excluded",
        "support_defined_on": "train only",
        "minimum_training_jets_each_class_per_bin": MIN_TRAIN_JETS_PER_CLASS_PER_BIN,
        "supported_bins": [{"pt_bin": pt, "eta_bin": eta} for pt, eta in sorted(supported)],
        "matching": "Within each original split/bin, sample equal class counts without replacement",
        "selection_variables": ["jet_pt_GeV", "jet_abs_eta", "label", "split"],
        "scores_and_timing_used_for_selection": False,
        "original_splits_preserved": True, "overlapping_event_groups": 0,
        "total_input_jets": len(frame), "total_selected_jets": len(matched),
        "pairs": len(matched) // 2, "split_summary": split_records,
        "event_summary": event_summary, "bin_counts": bin_records,
        "limitations": [
            "Finite bins leave within-bin pT/eta differences; evaluate a kinematics-only control again.",
            "Held-out labels are used for fixed bin balancing, so this is a conditional diagnostic sample.",
            "Jet sampling changes multiplicity; event groups remain separate but jets are not independent.",
            "Equal sample counts are not physical class priors, collision purity or detector-wide validation.",
            "A different run, signal mass/lifetime and data/MC checks are still needed.",
        ],
    }
    return matched, audit

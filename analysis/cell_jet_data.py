"""Read the calorimeter cells of jets already selected in the saved jet table."""

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot


CELL_FEATURES = (
    "cell_time_ns", "log_relative_energy", "cell_R_m", "cell_z_m",
    "cell_dEta_jet", "cell_dPhi_jet",
)
CELL_BRANCHES = ("cell_e", "cell_time", "cell_x", "cell_y", "cell_z", "cell_dEta_jet", "cell_dPhi_jet")
JET_CELL_KEYS = (
    "runNumber", "eventNumber", "jet_pt", "jet_eta", "jet_n_clusters",
    "cluster_n_cells",
)
BRANCHES = (*JET_CELL_KEYS, *CELL_BRANCHES)
REQUIRED_COLUMNS = (
    "source_file", "row_index", "jet_index", "event_group", "split", "label", "jet_eta",
)
CLASS_SAMPLES = {0: "LLP", 1: "unpaired"}
INVALID_CELL = -999999.0
EVENTS_PER_CHUNK = 128


def _int_vector(values, name, context):
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and (values.dtype.kind not in "iu" or np.any(values < 0))):
        raise ValueError(f"Expected nonnegative integer {name}: {context}")
    return values


def _flatten(values):
    return ak.to_numpy(ak.flatten(values, axis=1), allow_missing=False)


def _offsets(counts):
    return np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))


def _source_path(data_root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a relative source_file path: {relative!r}")
    source = (data_root / relative).resolve()
    if not source.is_relative_to(data_root) or not source.is_file():
        raise ValueError(f"Cannot map source_file to a ROOT file inside the data folder: {relative}")
    return source


def _read_rows(tree, fields, rows, relative):
    """Read the sorted unique indices from one bounded range."""
    start, stop = int(rows[0]), int(rows[-1]) + 1
    if start < 0 or stop > tree.num_entries:
        raise ValueError(f"Saved row_index is outside JetContent in {relative}")
    arrays = tree.arrays(fields, entry_start=start, entry_stop=stop, library="ak")[rows - start]
    return arrays, f"{relative}, rows {start}–{stop - 1}"


def _verify_event_keys(arrays, subset, event_index, context):
    run = _int_vector(ak.to_numpy(arrays["runNumber"], allow_missing=False), "runNumber", context)
    event = _int_vector(ak.to_numpy(arrays["eventNumber"], allow_missing=False), "eventNumber", context)
    event_ids = [f"{int(r)}:{int(e)}" for r, e in zip(run, event, strict=True)]
    expected_ids = [f"{CLASS_SAMPLES[label]}:{event_ids[i]}" for label, i in zip(subset["label"], event_index)]
    if not np.array_equal(subset["event_group"].to_numpy(), expected_ids):
        raise ValueError(f"Saved run/event identity disagrees with the ROOT row: {context}")


def _jet_cell_offsets(arrays, cell_fields, context, *, with_cluster_ids=False):
    def lengths(field):
        return ak.to_numpy(ak.num(arrays[field], axis=1), allow_missing=False)

    event_n_jets = lengths("jet_pt")
    event_n_clusters = lengths("cluster_n_cells")
    event_n_cells = lengths(cell_fields[0])
    for fields, expected in ((("jet_eta", "jet_n_clusters"), event_n_jets), (cell_fields, event_n_cells)):
        for field in fields:
            if not np.array_equal(lengths(field), expected):
                raise ValueError(f"Unaligned {field} vectors: {context}")

    jet_n_clusters = _int_vector(_flatten(arrays["jet_n_clusters"]), "jet_n_clusters", context)
    cluster_n_cells = _int_vector(_flatten(arrays["cluster_n_cells"]), "cluster_n_cells", context)
    event_jet_offsets = _offsets(event_n_jets)
    event_cluster_offsets = _offsets(event_n_clusters)
    jet_cluster_offsets = _offsets(jet_n_clusters)
    cluster_cell_offsets = _offsets(cluster_n_cells)
    if not np.array_equal(jet_cluster_offsets[event_jet_offsets], event_cluster_offsets):
        raise ValueError(f"sum(jet_n_clusters) does not match the cluster count: {context}")
    if not np.array_equal(cluster_cell_offsets[event_cluster_offsets], _offsets(event_n_cells)):
        raise ValueError(f"sum(cluster_n_cells) does not match the cell count: {context}")
    layout = (event_n_jets, event_jet_offsets, cluster_cell_offsets[jet_cluster_offsets])
    if with_cluster_ids:
        cluster_ids = np.repeat(np.arange(len(cluster_n_cells), dtype=np.int32), cluster_n_cells)
        return (*layout, cluster_ids)
    return layout


def _selected_jet_indices(arrays, subset, event_index, event_n_jets, event_jet_offsets, context):
    jet_indices = _int_vector(subset["jet_index"].to_numpy(), "jet_index", context)
    if np.any(jet_indices >= event_n_jets[event_index]):
        raise ValueError(f"Saved jet_index is outside the ROOT jet vector: {context}")
    jet_indices = event_jet_offsets[event_index] + jet_indices.astype(np.int64, copy=False)
    comparisons = [("jet_eta", "jet_eta", 1.0)]
    if "jet_pt_GeV" in subset.columns:
        comparisons.append(("jet_pt_GeV", "jet_pt", 0.001))
    for saved, field, scale in comparisons:
        root_values = _flatten(arrays[field])[jet_indices] * scale
        if (not np.isfinite(root_values).all() or np.any(root_values == INVALID_CELL * scale)
                or not np.allclose(subset[saved].to_numpy(dtype=float), root_values, rtol=1e-9, atol=1e-8)):
            raise ValueError(f"Saved {saved} disagrees with the ROOT jet at jet_index: {context}")
    return jet_indices


def read_jet_cell_spans(tree, subset, relative, *, cell_fields=CELL_BRANCHES, with_cluster_ids=False):
    """Return cell fields and jet spans, plus flat cluster IDs if requested."""
    rows, event_index = np.unique(subset["row_index"].to_numpy(), return_inverse=True)
    arrays, context = _read_rows(tree, (*JET_CELL_KEYS, *cell_fields), rows, relative)
    _verify_event_keys(arrays, subset, event_index, context)
    event_n_jets, event_jet_offsets, offsets, *cluster_ids = _jet_cell_offsets(
        arrays, cell_fields, context, with_cluster_ids=with_cluster_ids,
    )
    jets = _selected_jet_indices(arrays, subset, event_index, event_n_jets, event_jet_offsets, context)
    fields = np.column_stack([_flatten(arrays[field]) for field in cell_fields])
    return fields, np.column_stack((offsets[jets], offsets[jets + 1])), *cluster_ids


def _read_chunk(tree, subset, relative, max_cells, store_features, stored, valid, energy_kept, with_cluster_ids):
    flat_cells, spans, *cluster_ids = read_jet_cell_spans(
        tree, subset, relative, with_cluster_ids=with_cluster_ids,
    )
    for i, (lo, hi) in zip(subset.index, spans, strict=True):
        row = flat_cells[lo:hi]
        stored[i] = len(row)
        keep = np.isfinite(row).all(axis=1) & (row != INVALID_CELL).all(axis=1) & (row[:, 0] > 0)
        row = row[keep]
        valid[i] = len(row)
        if not len(row):
            continue

        energies = row[:, 0].astype(np.float64, copy=False)
        total_energy = energies.sum()
        if not np.isfinite(total_energy):
            raise ValueError(f"Nonfinite energy sum for jet table row {i}: {relative}")
        top = np.argsort(-energies, kind="stable")[:max_cells]
        e, t, x, y, z, d_eta, d_phi = row[top].T
        e_fraction = e / total_energy
        features = np.column_stack((t, np.log1p(100.0 * e_fraction), np.hypot(x, y) / 1000.0,
                                    z / 1000.0, d_eta, d_phi)).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"Nonfinite cell features for jet table row {i}: {relative}")
        local_ids = None
        if with_cluster_ids:
            _, local_ids = np.unique(cluster_ids[0][lo:hi][keep][top], return_inverse=True)
            local_ids = local_ids.astype(np.int32, copy=False)
        store_features(i, features, local_ids)
        energy_kept[i] = e_fraction.sum()


def _quantiles(values):
    if not len(values):
        return None
    points = (0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)
    return {f"p{int(p * 100):02d}": float(v) for p, v in zip(points, np.quantile(values, points), strict=True)}


def extract_cells(jets, data_root, max_cells, *, with_cluster_ids=False):
    """Return aligned (float32 cells, bool valid mask, JSON-ready audit).

    A True mask entry is an actual cell, False is padding. Missing or invalid
    cell measurements are skipped; a jet without usable cells stays in its
    original position with an all-False mask. A mismatch to the saved jet
    identity or a broken jet/cluster/cell count aborts the extraction. Pass
    None to keep all valid cells and pad only to the cohort's observed maximum.
    With cluster IDs, the fourth result labels retained cells with consecutive
    IDs in original cluster order; padding is -1.
    """
    if max_cells is not None:
        if (not isinstance(max_cells, (int, np.integer)) or isinstance(max_cells, (bool, np.bool_))
                or max_cells <= 0):
            raise ValueError("max_cells must be a positive integer or None")
        max_cells = int(max_cells)
    missing = set(REQUIRED_COLUMNS) - set(jets.columns)
    if missing:
        raise ValueError(f"Missing required columns in the saved jet table: {sorted(missing)}")
    frame = jets.reset_index(drop=True)
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("The saved jet table contains missing mapping fields")
    if len(frame):
        for key in ("row_index", "jet_index", "label"):
            _int_vector(frame[key].to_numpy(), key, "saved jet table")
        if (not frame["label"].isin(CLASS_SAMPLES).all()
                or not frame["split"].isin(["train", "validation", "test"]).all()):
            raise ValueError("Expected labels LLP=0/unpaired=1 and the original train/validation/test splits")
        if (frame.duplicated(["event_group", "jet_index"]).any()
                or frame.duplicated(["source_file", "row_index", "jet_index"]).any()):
            raise ValueError("The saved jet table repeats a jet identity")
        if frame.groupby("event_group", sort=False)["split"].nunique().gt(1).any():
            raise ValueError("The same event appears in multiple splits")

    data_root = Path(data_root).expanduser().resolve()

    def allocate(width):
        shape = (len(frame), width)
        return (np.zeros((*shape, len(CELL_FEATURES)), dtype=np.float32), np.zeros(shape, dtype=bool),
                np.full(shape, -1, dtype=np.int32) if with_cluster_ids else None)

    def store_padded(i, features, local_ids):
        n = len(features)
        cells[i, :n] = features
        mask[i, :n] = True
        if with_cluster_ids:
            cluster_ids[i, :n] = local_ids

    def store_ragged(i, features, local_ids):
        ragged[i] = features, local_ids

    if max_cells is None:
        # Discover the padding width without rereading the ROOT files.
        ragged = [None] * len(frame)
        store_features = store_ragged
    else:
        cells, mask, cluster_ids = allocate(max_cells)
        store_features = store_padded
    stored = np.full(len(frame), -1, dtype=np.int64)
    valid = np.zeros(len(frame), dtype=np.int64)
    energy_kept = np.full(len(frame), np.nan, dtype=float)
    files_read = 0
    events_read = 0
    chunks_read = 0
    for relative, group in frame.groupby("source_file", sort=False):
        source = _source_path(data_root, relative)
        with uproot.open(source, array_cache=None) as handle:
            if "JetContent" not in handle:
                raise ValueError(f"Missing JetContent in {relative}")
            tree = handle["JetContent"]
            missing = set(BRANCHES) - set(tree.keys())
            if missing:
                raise ValueError(f"Missing cell/jet branches in {relative}: {sorted(missing)}")
            if group["row_index"].max() >= tree.num_entries:
                raise ValueError(f"Saved row_index is outside JetContent in {relative}")
            for _, subset in group.groupby(group["row_index"] // EVENTS_PER_CHUNK, sort=True):
                _read_chunk(tree, subset, relative, max_cells, store_features, stored, valid, energy_kept,
                            with_cluster_ids)
                events_read += int(subset["row_index"].nunique())
                chunks_read += 1
        files_read += 1

    if np.any(stored < 0):
        raise ValueError(f"Failed to map {int((stored < 0).sum())} jets to the ROOT cell vectors")

    observed_max = int(valid.max(initial=0))
    padded_width = observed_max if max_cells is None else max_cells
    if max_cells is None:
        cells, mask, cluster_ids = allocate(padded_width)
        for i, data in enumerate(ragged):
            if data is not None:
                store_padded(i, *data)
                ragged[i] = None

    retained = mask.sum(axis=1)
    expected_retained = valid if max_cells is None else np.minimum(valid, max_cells)
    if not np.array_equal(retained, expected_retained):
        raise RuntimeError("Retained cell counts disagree with the requested extraction mode")

    def summarize(indices):
        valid_cells = valid[indices]
        retained_cells = retained[indices]
        truncated = retained_cells < valid_cells
        fraction = energy_kept[indices]
        n = len(indices)
        return {
            "jets": n,
            "jets_without_stored_cells": int((stored[indices] == 0).sum()),
            "jets_without_valid_cells": int((valid_cells == 0).sum()),
            "truncated_jets": int(truncated.sum()),
            "truncated_jet_fraction": float(truncated.mean()) if n else None,
            "stored_cells": int(stored[indices].sum()),
            "valid_cells": int(valid_cells.sum()),
            "invalid_cells": int((stored[indices] - valid_cells).sum()),
            "retained_cells": int(retained_cells.sum()),
            "discarded_by_cap_cells": int((valid_cells - retained_cells).sum()),
            "stored_cell_count_quantiles": _quantiles(stored[indices]),
            "valid_cell_count_quantiles": _quantiles(valid_cells),
            "retained_cell_count_quantiles": _quantiles(retained_cells),
            "retained_energy_fraction_quantiles": _quantiles(fraction[np.isfinite(fraction)]),
        }

    audit = {
        "features": list(CELL_FEATURES),
        "feature_definitions": {
            "cell_time_ns": "Stored cell_time in ns, unchanged",
            "log_relative_energy": "log1p(100 * cell_e / sum(cell_e of all valid jet cells before truncation))",
            "cell_R_m": "sqrt(cell_x^2 + cell_y^2), converted from mm to m",
            "cell_z_m": "Stored cell_z, converted from mm to m",
            "cell_dEta_jet": "Stored cell-to-jet pseudorapidity difference",
            "cell_dPhi_jet": "Stored cell-to-jet azimuthal difference, radians",
        },
        "max_cells": max_cells,
        "padded_width": padded_width,
        "observed_max_valid_cells": observed_max,
        "selection": (("Keep all valid cells, sorted by descending energy" if max_cells is None
                       else "Keep highest-energy valid cells")
                      + "; ties retain stored order; True mask means valid"),
        "validity": "All seven raw fields finite and different from -999999; cell_e > 0",
        "association": "Verify run/event ID and jet eta (and pT when available); cumulative jet/cluster/cell offsets validated per selected event",
        "files_read": files_read,
        "events_read": events_read,
        "chunks_read": chunks_read,
        "mapped_jets": len(frame),
        "unmapped_jets": 0,
        "overall": summarize(frame.index.to_numpy()),
        "by_split_and_label": [
            {"split": split, "label": int(label), "sample": CLASS_SAMPLES[label], **summarize(group.index.to_numpy())}
            for (split, label), group in frame.groupby(["split", "label"], sort=True)
        ],
    }
    if with_cluster_ids:
        if not np.array_equal(cluster_ids >= 0, mask):
            raise RuntimeError("Cluster IDs do not match retained cells")
        return cells, mask, audit, cluster_ids
    return cells, mask, audit

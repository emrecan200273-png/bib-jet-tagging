"""Load jets from a separate unpaired cell production for frozen tagging tests."""

from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from cell_jet_data import CELL_BRANCHES, EVENTS_PER_CHUNK, _read_rows
from match_unpaired_vertices import match, scalar_array
from xgb_data import JET_BRANCHES, PHYSICS_FEATURES, _numeric_vector, physical_invalid_mask, sanitize_values


CELL_KEYS = ("runNumber", "eventNumber", "lumiBlock")
VERTEX_KEYS = ("runNumber", "eventNumber", "lumiBlockNumber")
MIN_PT_GEV = 30.0
MAX_ABS_ETA = 1.7
JET_COLUMNS = {
    "label": "int64", "sample": "object", "split": "object", "event_group": "object",
    "source_file": "object", "row_index": "int64", "jet_index": "int64",
    "jet_pt_GeV": "float64", "jet_eta": "float64", "jet_abs_eta": "float64",
    **dict.fromkeys(PHYSICS_FEATURES, "float64"), "jet_BIB_score": "float64",
}
CUT_STAGES = (
    "all_jets_in_zero_vertex_events", "finite_pt_eta_time", "pt_cut", "eta_cut",
    "mixed_calorimeters", "finite_standard_features",
)


def _jet_chunks(tree, selected, source, cutflow):
    """Read the small ranges covering the chosen rows, in source order."""
    for _, rows in groupby(selected, key=lambda item: item[0] // EVENTS_PER_CHUNK):
        bucket = list(rows)
        row_numbers = np.fromiter((entry for entry, _ in bucket), dtype=np.int64)
        arrays, ctx = _read_rows(tree, (*CELL_KEYS, *JET_BRANCHES), row_numbers, source)
        ids = [scalar_array(arrays[key], key, ctx, integer=True) for key in CELL_KEYS]
        for (_, expected), actual in zip(bucket, zip(*ids), strict=True):
            if any(np.ma.is_masked(value) for value in actual) or tuple(map(int, actual)) != expected:
                raise ValueError(f"ROOT event key differs from the vertex join: {ctx}")

        jet_counts = ak.to_numpy(ak.num(arrays["jet_pt"], axis=1), allow_missing=False)
        n_jets = int(jet_counts.sum())
        clean = {}
        for field in JET_BRANCHES:
            values = sanitize_values(_numeric_vector(arrays, field, jet_counts, ctx))
            values[physical_invalid_mask(field, values)] = np.nan
            clean[field] = values

        pt = clean["jet_pt"] / 1000.0
        eta = clean["jet_eta"]
        abs_eta = np.abs(eta)
        tile = clean["jet_TileEnergyFrac"]
        cutflow[CUT_STAGES[0]] += n_jets
        keep = np.ones(n_jets, dtype=bool)
        for step, condition in zip(CUT_STAGES[1:], (
            np.isfinite(pt) & np.isfinite(eta) & np.isfinite(clean["jet_time_ATLAS"]),
            pt >= MIN_PT_GEV,
            abs_eta < MAX_ABS_ETA,
            (tile > 0.0) & (tile < 1.0),
            np.logical_and.reduce([np.isfinite(clean[field]) for field in PHYSICS_FEATURES]),
        ), strict=True):
            keep &= condition
            cutflow[step] += int(keep.sum())

        if keep.any():
            event_groups = [f"unpaired:{run}:{event}" for _, (run, event, _) in bucket]
            offsets = np.repeat(np.cumsum(jet_counts) - jet_counts, jet_counts)
            yield pd.DataFrame({
                "label": 1, "sample": "unpaired", "split": "test",
                "event_group": np.repeat(event_groups, jet_counts)[keep],
                "source_file": source, "row_index": np.repeat(row_numbers, jet_counts)[keep],
                "jet_index": (np.arange(n_jets) - offsets)[keep],
                "jet_pt_GeV": pt[keep], "jet_eta": eta[keep], "jet_abs_eta": abs_eta[keep],
                **{field: clean[field][keep] for field in PHYSICS_FEATURES},
                "jet_BIB_score": clean["jet_BIB_score"][keep],
            })


def _select_run(records, run):
    run_status = Counter(dict.fromkeys(("matched_zero", "matched_nonzero", "unmatched", "invalid_vertex_count",
                                       "invalid_cell_key"), 0))
    selected = defaultdict(list)
    seen_events = set()
    for source, entry, r, event, lumi, vertices, status, _, _ in records:
        if r != run:
            continue
        if status == "matched":
            status = "matched_zero" if vertices == 0 else "matched_nonzero"
        run_status[status] += 1
        if event is not None:
            if event in seen_events:
                raise ValueError(f"Run {run} repeats event {event}; the run:event identity is ambiguous")
            seen_events.add(event)
        if status == "matched_zero":
            selected[source].append((entry, (r, event, lumi)))
    return selected, run_status


def load_external_jets(cell_dir, vertex_dir, run, *, vertex_event_int32_bits=False):
    """Return the selected zero-vertex jets and audit; missing BIB scores stay NaN."""
    if not isinstance(run, (int, np.integer)) or isinstance(run, (bool, np.bool_)) or run <= 0:
        raise ValueError("run must be a positive integer run number")
    run = int(run)
    cell_dir, vertex_dir = (Path(path).expanduser().resolve() for path in (cell_dir, vertex_dir))
    join, records = match(
        cell_dir, vertex_dir, key_fields=CELL_KEYS, vertex_keys=VERTEX_KEYS,
        cell_tree="JetContent", vertex_tree="BIB", vertex_branch="nPrimaryVertices", matched_only=True,
        vertex_event_int32_bits=vertex_event_int32_bits,
    )
    selected, run_status = _select_run(records, run)
    del records

    parts = []
    cutflow = Counter(dict.fromkeys(CUT_STAGES, 0))
    needed = set((*CELL_KEYS, *JET_BRANCHES, "cluster_n_cells", *CELL_BRANCHES))
    for source, entries in sorted(selected.items()):
        path = (cell_dir / source).resolve()
        if not path.is_relative_to(cell_dir):
            raise ValueError(f"Source file escapes the supplied cell directory: {source}")
        with uproot.open(path, array_cache=None) as handle:
            tree = handle["JetContent"]
            missing = needed - set(tree.keys())
            if missing:
                raise ValueError(f"Missing required jet/cell branches in {source}: {sorted(missing)}")
            parts.extend(_jet_chunks(tree, entries, source, cutflow))

    frame = (pd.concat(parts, ignore_index=True) if parts else
             pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in JET_COLUMNS.items()}))
    if len(frame) != cutflow["finite_standard_features"] or frame.duplicated(["event_group", "jet_index"]).any():
        raise RuntimeError("Output jet counts or identities disagree with the extraction")
    n_events = sum(run_status.values())
    if not n_events:
        status = "run_not_found"
    elif not run_status["matched_zero"]:
        status = "no_zero_vertex_events"
    elif frame.empty:
        status = "no_accepted_jets"
    else:
        status = "ok"
    audit = {
        "status": status, "run": run, "vertex_join": join,
        "run_cell_events": n_events, "run_event_status": dict(run_status),
        "selected_files_read": len(selected), "jet_cutflow": dict(cutflow),
        "accepted_jets": len(frame), "accepted_events": int(frame["event_group"].nunique()),
        "accepted_jets_without_valid_stored_bib_score": int(frame["jet_BIB_score"].isna().sum()),
        "acceptance": {
            "min_jet_pt_GeV_inclusive": MIN_PT_GEV, "max_abs_eta_exclusive": MAX_ABS_ETA,
            "tile_fraction": "0 < jet_TileEnergyFrac < 1", "finite_jet_features": list(PHYSICS_FEATURES),
        },
    }
    return frame, audit

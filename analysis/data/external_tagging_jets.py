"""Load jets from a separate unpaired cell production for frozen tagging tests."""

from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from analysis.data.cell_jet_data import CELL_BRANCHES, EVENTS_PER_CHUNK, _read_rows
from analysis.data.match_unpaired_vertices import match, scalar_array
from analysis.data.xgb_data import (
    JET_BRANCHES, PHYSICS_FEATURES, apply_jet_cuts, read_jet_fields, selected_jet_frame,
)


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
    for _, block in groupby(selected, key=lambda item: item[0] // EVENTS_PER_CHUNK):
        entries_and_keys = list(block)
        row_numbers = np.fromiter((entry for entry, _ in entries_and_keys), dtype=np.int64)
        arrays, context = _read_rows(tree, (*CELL_KEYS, *JET_BRANCHES), row_numbers, source)
        id_columns = [scalar_array(arrays[key], key, context, integer=True) for key in CELL_KEYS]
        root_event_keys = zip(*id_columns)
        for (_, joined_key), root_key in zip(entries_and_keys, root_event_keys, strict=True):
            missing_id = any(np.ma.is_masked(value) for value in root_key)
            if missing_id or tuple(map(int, root_key)) != joined_key:
                raise ValueError(f"ROOT event key differs from the vertex join: {context}")

        jet_counts = ak.to_numpy(ak.num(arrays["jet_pt"], axis=1), allow_missing=False)
        n_jets = int(jet_counts.sum())
        clean = read_jet_fields(arrays, jet_counts, context)
        cutflow[CUT_STAGES[0]] += n_jets
        keep = np.ones(n_jets, dtype=bool)
        pt = apply_jet_cuts(clean, keep, cutflow, min_pt_GeV=MIN_PT_GEV, max_abs_eta=MAX_ABS_ETA, require_mixed=True)
        for field in PHYSICS_FEATURES:
            keep &= np.isfinite(clean[field])
        cutflow["finite_standard_features"] += int(keep.sum())

        if keep.any():
            event_groups = [f"unpaired:{run}:{event}" for _, (run, event, _) in entries_and_keys]
            yield selected_jet_frame(clean, pt, keep, jet_counts, row_numbers, event_groups, source,
                                     sample="unpaired", label=1, split="test")


def _select_run(records, run):
    statuses = ("matched_zero", "matched_nonzero", "unmatched", "invalid_vertex_count", "invalid_cell_key")
    run_status = Counter(dict.fromkeys(statuses, 0))
    selected = defaultdict(list)
    seen_events = set()
    for source, entry, run_number, event, lumi, vertices, status, _, _ in records:
        if run_number != run:
            continue
        if status == "matched":
            status = "matched_zero" if vertices == 0 else "matched_nonzero"
        run_status[status] += 1
        if event is not None:
            if event in seen_events:
                raise ValueError(f"Run {run} repeats event {event}; the run:event identity is ambiguous")
            seen_events.add(event)
        if status == "matched_zero":
            selected[source].append((entry, (run_number, event, lumi)))
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

    if parts:
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in JET_COLUMNS.items()})
    wrong_jet_count = len(frame) != cutflow["finite_standard_features"]
    if wrong_jet_count or frame.duplicated(["event_group", "jet_index"]).any():
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

"""Build the first XGBoost table from verified, read-only ROOT files.

The target is a working sample label: unpaired = BIB, matched LLP = signal.
Event identifiers and the existing BIB tagger are retained for checks only.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import awkward as ak
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import uproot

from analysis.data.data_paths import resolve_data_paths


PHYSICS_FEATURES = [
    "jet_time_ATLAS", "jet_time_spread", "jet_time_sigma",
    "jet_n_clusters", "jet_TileEnergyFrac",
]
KINEMATIC_FEATURES = ["jet_pt_GeV", "jet_abs_eta"]
DATA_GROUPS = {
    "collision/LLP/N1N2_rpv_1400_10ns": ("LLP", 0),
    "non_collision/unpaired/physics_Background_NTuple_unsplit_v2": ("unpaired", 1),
}
ID_BRANCHES = ["runNumber", "eventNumber"]
JET_BRANCHES = ["jet_pt", "jet_eta", *PHYSICS_FEATURES, "jet_BIB_score"]
INVALID_VALUE = -999999.0
CUTFLOW_STAGES = [
    "all_jets", "truth_selection", "finite_pt_eta_time",
    "pt_cut", "eta_cut", "mixed_calorimeters",
]


def sanitize_values(values):
    """Replace the producer's sentinel, masked values and infinities with NaN."""
    clean = np.ma.asarray(values, dtype=np.float64).filled(np.nan).copy()
    clean[(clean == INVALID_VALUE) | ~np.isfinite(clean)] = np.nan
    return clean


def physical_invalid_mask(field, values):
    """Shared training/inference checks for nonphysical stored jet values."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    invalid = np.zeros(len(values), dtype=bool)
    if field == "jet_time_sigma":
        invalid = finite & (values <= 0)
    elif field == "jet_time_spread":
        invalid = finite & (values < 0)
    elif field == "jet_n_clusters":
        invalid[finite] = (values[finite] < 1) | (values[finite] != np.floor(values[finite]))
    elif field == "jet_BIB_score":
        invalid = finite & ((values < 0) | (values > 1))
    return invalid


def tile_fraction_mask(values, *, require_mixed):
    """Select both calorimeters, or allow pure Tile/LAr in the acceptance study."""
    if require_mixed:
        return (values > 0) & (values < 1)
    return (values >= 0) & (values <= 1)


def assign_event_splits(frame, seed=42):
    """Return a copy with stratified 60/20/20 train/validation/test event splits.

    All jets from one event stay together. Sorting event keys also makes the
    assignment independent of the order of input files or jets.
    """
    required = {"event_group", "label"}
    if required - set(frame.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame[["event_group", "label"]].isna().any().any():
        raise ValueError("Splitting needs nonempty data and complete event groups/labels")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Splitting requires both labels: LLP=0 and unpaired=1")
    if frame.groupby("event_group")["label"].nunique().max() != 1:
        raise ValueError("An event group has more than one label")
    events = (
        frame[["event_group", "label"]].drop_duplicates()
        .sort_values("event_group").reset_index(drop=True)
    )
    try:
        train_validation, test = train_test_split(
            events, test_size=0.2, random_state=seed, stratify=events["label"],
        )
        train, validation = train_test_split(
            train_validation, test_size=0.25, random_state=seed,
            stratify=train_validation["label"],
        )
    except ValueError as exc:
        raise ValueError("Too few independent events for stratified 60/20/20 splitting") from exc
    groups = {"train": train, "validation": validation, "test": test}
    assignment = {}
    for split, subset in groups.items():
        if set(subset["label"].unique()) != {0, 1}:
            raise ValueError(f"The {split} split does not contain both classes")
        for key in subset["event_group"]:
            if key in assignment:
                raise ValueError(f"Event appears in more than one split: {key}")
            assignment[key] = split
    result = frame.copy()
    result["split"] = result["event_group"].map(assignment)
    if result["split"].isna().any():
        raise ValueError("Some events have not been assigned to a split")
    if result.groupby("event_group")["split"].nunique().max() != 1:
        raise ValueError("Event leakage between splits")
    return result


def preserve_splits(expanded, old, seed):
    """Keep known event splits and assign newly eligible events by a stable hash."""
    assignment = old.drop_duplicates("event_group").set_index("event_group")["split"]
    broad = expanded.copy()
    prior = broad["event_group"].map(assignment)
    changes = broad.loc[prior.notna() & broad["split"].ne(prior), "event_group"].nunique()

    def new_split(key):
        digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big") / 2**64
        return "train" if value < .6 else "validation" if value < .8 else "test"

    new = {key: new_split(key) for key in broad.loc[prior.isna(), "event_group"].unique()}
    broad["split"] = prior.fillna(broad["event_group"].map(new))
    return broad, {"new_events": len(new), "old_events_that_a_fresh_split_would_move": int(changes)}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _numeric_vector(arrays, field, jet_counts, context):
    try:
        lengths = ak.to_numpy(ak.num(arrays[field], axis=1))
    except Exception as exc:
        raise ValueError(f"Expected a per-event jet vector for {field}: {context}") from exc
    if not np.array_equal(lengths, jet_counts):
        raise ValueError(f"Jet vectors have different lengths for {field}: {context}")
    return ak.to_numpy(ak.flatten(arrays[field], axis=1), allow_missing=True)


def read_jet_fields(arrays, jet_counts, context, *, sanitation=None, sample=None):
    """Check vector sizes and replace invalid jet values with NaN."""
    clean = {}
    for field in JET_BRANCHES:
        raw = _numeric_vector(arrays, field, jet_counts, context)
        values = sanitize_values(raw)
        invalid = physical_invalid_mask(field, values)
        values[invalid] = np.nan
        if sanitation is not None:
            raw_floats = np.ma.asarray(raw, dtype=np.float64).filled(np.nan)
            sanitation.setdefault((sample, field), Counter()).update({
                "values": len(values),
                "sentinel": int((raw_floats == INVALID_VALUE).sum()),
                "nonfinite": int((~np.isfinite(raw_floats)).sum()),
                "physical_invalid": int(invalid.sum()),
                "missing_after_sanitizing": int(np.isnan(values).sum()),
            })
        clean[field] = values
    return clean


def apply_jet_cuts(clean, keep, flow, *, min_pt_GeV, max_abs_eta, require_mixed):
    """Update the mask and counts after each standard acceptance cut."""
    pt = clean["jet_pt"] / 1000.0
    eta = clean["jet_eta"]
    choices = (
        ("finite_pt_eta_time", np.isfinite(pt) & np.isfinite(eta) & np.isfinite(clean["jet_time_ATLAS"])),
        ("pt_cut", pt >= min_pt_GeV),
        ("eta_cut", np.abs(eta) < max_abs_eta),
        ("mixed_calorimeters" if require_mixed else "valid_tile_fraction",
         tile_fraction_mask(clean["jet_TileEnergyFrac"], require_mixed=require_mixed)),
    )
    for stage, condition in choices:
        keep &= condition
        flow[stage] += int(keep.sum())
    return pt


def selected_jet_frame(clean, pt, keep, counts, rows, keys, source, *, sample, label, split=None):
    """Keep the original ROOT row and within-event jet index."""
    jet_index = np.arange(len(pt)) - np.repeat(np.cumsum(counts) - counts, counts)
    return pd.DataFrame({
        "label": label, "sample": sample,
        **({"split": split} if split is not None else {}),
        "event_group": np.repeat(keys, counts)[keep], "source_file": source,
        "row_index": np.repeat(rows, counts)[keep], "jet_index": jet_index[keep],
        "jet_pt_GeV": pt[keep], "jet_eta": clean["jet_eta"][keep], "jet_abs_eta": np.abs(clean["jet_eta"][keep]),
        **{field: clean[field][keep] for field in (*PHYSICS_FEATURES, "jet_BIB_score")},
    })


def prepare_training_data(
    project_root, output_dir, *, seed=42, min_pt_GeV=30., max_abs_eta=1.7,
    verify_hashes=True, require_mixed_calorimeters=True,
):
    """Extract all events in the two selected groups, apply cuts, and save a table.

    Returns ``(training_frame, audit_dict)``. Only jet and event-ID branches are
    read, in bounded chunks; cell data and raw files are never changed.
    """
    if not np.isfinite(min_pt_GeV) or min_pt_GeV < 0:
        raise ValueError("min_pt_GeV must be finite and nonnegative")
    if not np.isfinite(max_abs_eta) or max_abs_eta <= 0:
        raise ValueError("max_abs_eta must be finite and positive")
    calorimeter_stage = "mixed_calorimeters" if require_mixed_calorimeters else "valid_tile_fraction"
    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    data_root, manifest_path = resolve_data_paths(project_root)
    data_root, manifest_path = data_root.resolve(), manifest_path.resolve()
    if output_dir.is_relative_to(data_root):
        raise ValueError("Write training outputs outside the raw data folder")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = [
        item for item in manifest["files"]
        if str(Path(item["path"]).parent) in DATA_GROUPS
        and Path(item["path"]).suffix == ".root"
    ]
    if len({item["path"] for item in records}) != len(records):
        raise ValueError("The verified manifest repeats a selected source file")
    found_groups = {str(Path(item["path"]).parent) for item in records}
    if found_groups != set(DATA_GROUPS):
        raise ValueError(f"Selected data groups missing: {sorted(set(DATA_GROUPS) - found_groups)}")

    parts, file_audits, sanitation = [], [], {}
    seen_events = {sample: set() for sample, _ in DATA_GROUPS.values()}
    cutflow = {sample: Counter() for sample, _ in DATA_GROUPS.values()}
    full_events = Counter()
    for number, record in enumerate(sorted(records, key=lambda item: item["path"]), 1):
        relative = record["path"]
        group = str(Path(relative).parent)
        sample, label = DATA_GROUPS[group]
        source = (data_root / relative).resolve()
        if not source.is_relative_to(data_root):
            raise ValueError(f"Source path escapes the data folder: {relative}")
        stat = source.stat()
        if stat.st_size != record["size"]:
            raise ValueError(f"Source size differs from verified manifest: {relative}")
        actual_hash = _sha256(source) if verify_hashes else None
        if verify_hashes and actual_hash != record["sha256"]:
            raise ValueError(f"Source SHA256 differs from verified manifest: {relative}")
        wanted = [*ID_BRANCHES, *JET_BRANCHES]
        if sample == "LLP":
            wanted.append("jet_is_llp_matched")
        file_events, file_jets, file_selected = 0, 0, 0
        with uproot.open(source) as handle:
            if "JetContent" not in handle:
                raise ValueError(f"Missing JetContent tree: {relative}")
            tree = handle["JetContent"]
            missing = set(wanted) - set(tree.keys())
            if missing:
                raise ValueError(f"Missing required branches in {relative}: {sorted(missing)}")
            for arrays in tree.iterate(wanted, step_size=10_000, library="ak"):
                context = f"{relative}, entries starting at {file_events}"
                try:
                    jet_counts = ak.to_numpy(ak.num(arrays["jet_pt"], axis=1))
                except Exception as exc:
                    raise ValueError(f"Invalid jet_pt vectors: {context}") from exc
                n_jets = int(jet_counts.sum())
                run = ak.to_numpy(arrays["runNumber"])
                event = ak.to_numpy(arrays["eventNumber"])
                for field, ids in [("runNumber", run), ("eventNumber", event)]:
                    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
                        raise ValueError(f"Invalid {field}: {context}")
                event_keys = [f"{sample}:{int(r)}:{int(e)}" for r, e in zip(run, event)]
                for key in event_keys:
                    if key in seen_events[sample]:
                        raise ValueError(f"Duplicate physical event row: {key} in {relative}")
                    seen_events[sample].add(key)
                full_events[sample] += len(arrays)
                clean = read_jet_fields(arrays, jet_counts, context, sanitation=sanitation, sample=sample)
                if sample == "LLP":
                    truth = sanitize_values(
                        _numeric_vector(arrays, "jet_is_llp_matched", jet_counts, context)
                    )
                    if np.any(~np.isin(truth, [0., 1.])):
                        raise ValueError(f"jet_is_llp_matched must be 0 or 1: {context}")
                    keep = truth == 1
                else:
                    keep = np.ones(n_jets, dtype=bool)
                flow = cutflow[sample]
                flow["all_jets"] += n_jets
                flow["truth_selection"] += int(keep.sum())
                pt = apply_jet_cuts(clean, keep, flow, min_pt_GeV=min_pt_GeV, max_abs_eta=max_abs_eta,
                                    require_mixed=require_mixed_calorimeters)
                if keep.any():
                    rows = np.arange(file_events, file_events + len(arrays))
                    parts.append(selected_jet_frame(clean, pt, keep, jet_counts, rows, event_keys, relative,
                                                    sample=sample, label=label))
                file_events += len(arrays)
                file_jets += n_jets
                file_selected += int(keep.sum())
            if file_events != tree.num_entries:
                raise ValueError(f"Not all ROOT events were read: {relative}")
            file_audits.append({
                "relative_path": relative, "sample": sample,
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "expected_sha256": record["sha256"], "actual_sha256": actual_hash,
                "hash_verified": bool(verify_hashes), "entries": int(tree.num_entries),
                "jets": file_jets, "selected_jets": file_selected,
                "branch_count": len(tree.keys()),
            })
        if number == 1 or number % 50 == 0 or number == len(records):
            print(f"Read {number}/{len(records)} verified source files", flush=True)
    if not parts:
        raise ValueError("No jets remain after the requested selection")
    frame = assign_event_splits(pd.concat(parts, ignore_index=True), seed=seed)
    summary = (
        frame.groupby(["split", "sample", "label"], observed=True)
        .agg(jets=("event_group", "size"), events=("event_group", "nunique"))
        .reset_index()
    )
    flow_records = [
        {"sample": sample, "label": label, "stage": stage, "jets": int(cutflow[sample][stage])}
        for sample, label in DATA_GROUPS.values() for stage in [*CUTFLOW_STAGES[:-1], calorimeter_stage]
    ]
    sanitation_records = [
        {"sample": sample, "field": field, **dict(counts)}
        for (sample, field), counts in sorted(sanitation.items())
    ]
    groups = [
        {
            "group": group, "sample": sample, "label": label,
            "files": sum(item["sample"] == sample for item in file_audits),
            "events": int(full_events[sample]),
            "unique_events": len(seen_events[sample]), "duplicate_event_rows": 0,
            "jets_before_selection": int(cutflow[sample]["all_jets"]),
            "selected_jets": int((frame["sample"] == sample).sum()),
            "selected_events": int(frame.loc[frame["sample"] == sample, "event_group"].nunique()),
        }
        for group, (sample, label) in DATA_GROUPS.items()
    ]
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verify_hashes": bool(verify_hashes), "seed": int(seed),
        "labels": {"0": "truth-matched LLP jet", "1": "unpaired jet (working BIB label)"},
        "label_caveats": [
            "Unpaired may contain collision contamination; the label is a sample-level assumption.",
            "Only truth-matched LLP jets are selected, rather than every jet in an LLP event.",
            "The samples are one LLP mass/lifetime point and one unpaired run.",
        ],
        "selection": {
            "min_pt_GeV_inclusive": float(min_pt_GeV),
            "max_abs_eta_exclusive": float(max_abs_eta),
            "finite_required": ["jet_pt_GeV", "jet_eta", "jet_time_ATLAS"],
            "mixed_calorimeters": ("0 < jet_TileEnergyFrac < 1, applied to both classes" if require_mixed_calorimeters
                                    else "0 <= jet_TileEnergyFrac <= 1, applied to both classes"),
            "mixed_calorimeter_reason": (
                "Common mixed-energy support because LLP truth matching requires LAr and Tile cells; energy fractions are not a full reproduction of the matching criterion."
                if require_mixed_calorimeters else "Acceptance study includes jets with a single calorimeter's jet energy."
            ),
            "llp_truth_match": "jet_is_llp_matched == 1",
            "time_definition": "Stored jet_time_ATLAS, unchanged; time unit ns",
            "energy_conversion": "jet_pt_GeV = stored jet_pt / 1000 (MeV to GeV)",
            "other_primary_missing": "NaN retained for native XGBoost missing-value handling",
        },
        "physics_features": PHYSICS_FEATURES,
        "kinematic_features": KINEMATIC_FEATURES,
        "comparator": "jet_BIB_score; higher is more BIB-like; never a primary input",
        "split": {
            "unit": "event", "group_key": "sample:runNumber:eventNumber",
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "stratification": "event label", "overlapping_event_groups": 0,
            "procedure": "Sort unique event keys; split test20%, then validation25% of remaining, same seed",
        },
        "groups": groups, "cutflow": flow_records,
        "sanitation": sanitation_records,
        "selected_missing": {field: int(frame[field].isna().sum()) for field in [*PHYSICS_FEATURES, *KINEMATIC_FEATURES, "jet_BIB_score"]},
        "split_summary": summary.to_dict(orient="records"),
        "files": file_audits,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["uproot", "awkward", "numpy", "pandas", "scikit-learn"]},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "training_jets.csv.gz", index=False,
                 compression={"method": "gzip", "mtime": 0})
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame(flow_records).to_csv(output_dir / "cutflow.csv", index=False)
    _json_write(output_dir / "data_audit.json", audit)
    return frame, audit

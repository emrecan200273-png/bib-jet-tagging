"""Read a small, fixed sample of jets and cells from the ROOT files."""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from data_paths import resolve_data_paths


GROUPS = {
    "collision/LLP/N1N2_rpv_1400_10ns": ("LLP sample", 1000),
    "non_collision/unpaired/physics_Background_NTuple_unsplit_v2": ("Unpaired sample", 6),
}
IDS = ["runNumber", "lumiBlock", "eventNumber", "BCID"]
JET_FIELDS = ["jet_e", "jet_pt", "jet_eta", "jet_phi", "jet_time_ATLAS",
              "jet_time_spread", "jet_time_sigma", "jet_n_clusters",
              "jet_TileEnergyFrac", "jet_BIB_score", "jet_chi2ndf_SM", "jet_chi2ndf_BIB"]
CELL_FIELDS = ["cell_x", "cell_y", "cell_z", "cell_e", "cell_time",
               "cell_time_sigma", "cell_quality"]
CLUSTER_FIELDS = ["cluster_e", "cluster_n_cells"]
VECTOR_FIELDS = JET_FIELDS + CLUSTER_FIELDS + CELL_FIELDS
CANDIDATE_FEATURES = ["jet_pt", "jet_eta", "jet_time_ATLAS", "jet_time_spread",
                      "jet_time_sigma", "jet_n_clusters", "jet_TileEnergyFrac"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_inventory(project_root, output_dir):
    """Check file sizes and list the branches in the selected samples."""
    project_root, output_dir = Path(project_root), Path(output_dir)
    data_root, manifest_path = resolve_data_paths(project_root)
    manifest = json.loads(manifest_path.read_text())
    records = [r for r in manifest["files"] if str(Path(r["path"]).parent) in GROUPS]
    required = set(IDS + VECTOR_FIELDS)
    inventory = []
    for record in records:
        relative = record["path"]
        path = data_root / relative
        stat = path.stat()
        if stat.st_size != record["size"]:
            raise ValueError(f"Local file size differs from verified snapshot: {relative}")
        with uproot.open(path) as handle:
            tree = handle["JetContent"]
            names = sorted(tree.keys())
            missing = required.difference(names)
            if missing:
                raise ValueError(f"Missing required branches in {relative}: {sorted(missing)}")
            inventory.append({
                "group": str(Path(relative).parent), "relative_path": relative,
                "file_size": stat.st_size, "local_mtime_ns": stat.st_mtime_ns,
                "expected_sha256": record["sha256"], "num_entries": int(tree.num_entries),
                "tree_name": "JetContent", "branch_count": len(names), "branch_names": names,
            })
    grouped = []
    for group, (sample, per_file) in GROUPS.items():
        items = [r for r in inventory if r["group"] == group]
        if not items:
            raise ValueError(f"No files in selected group: {group}")
        schemas = Counter(tuple(r["branch_names"]) for r in items)
        grouped.append({"sample": sample, "group": group, "files": len(items),
                        "total_entries": sum(r["num_entries"] for r in items),
                        "min_entries": min(r["num_entries"] for r in items),
                        "max_entries": max(r["num_entries"] for r in items),
                        "schema_variants": len(schemas), "pilot_rows_per_file": per_file,
                        "planned_pilot_rows": sum(min(per_file, r["num_entries"]) for r in items)})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(data_root), "manifest": str(manifest_path),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "scope": "Metadata for both selected groups; not the remaining DataForEmre folders",
              "checksum_scope": "Expected source hashes retained; previous full download validation is separate. This run checks file sizes, not fresh file hashes.",
              "group_summary": grouped, "files": inventory}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "file_inventory.json", result)
    return result


def extract_pilot(project_root, inventory, output_dir):
    """Read leading rows from every file; validate vector lengths and counts."""
    project_root, output_dir = Path(project_root), Path(output_dir)
    data_root, _ = resolve_data_paths(project_root)
    jets, rows, example_cells, examples = [], [], [], []
    example_sources = set()
    for file in sorted(inventory["files"], key=lambda f: (f["group"], f["relative_path"])):
        sample, per_file = GROUPS[file["group"]]
        relative = file["relative_path"]
        wanted = IDS + VECTOR_FIELDS
        has_truth_match = "jet_is_llp_matched" in file["branch_names"]
        if has_truth_match:
            wanted = wanted + ["jet_is_llp_matched"]
        stop = min(per_file, file["num_entries"])
        with uproot.open(data_root / relative) as handle:
            arrays = handle["JetContent"].arrays(wanted, entry_start=0, entry_stop=stop, library="ak")
        if len(arrays) != stop:
            raise ValueError(f"Unexpected number of rows read: {relative}")
        for index, record in enumerate(arrays):
            identity = {k: str(int(record[k])) for k in IDS}
            base = {"sample": sample, "source_file": relative, "row_index": index, **identity}
            vectors = {k: np.asarray(ak.to_list(record[k])) for k in VECTOR_FIELDS}
            njet = len(vectors["jet_pt"])
            ncluster = len(vectors["cluster_n_cells"])
            ncell = len(vectors["cell_time"])
            jet_ok = all(len(vectors[k]) == njet for k in JET_FIELDS)
            cluster_ok = all(len(vectors[k]) == ncluster for k in CLUSTER_FIELDS)
            cell_ok = all(len(vectors[k]) == ncell for k in CELL_FIELDS)
            if has_truth_match:
                matched = np.asarray(ak.to_list(record["jet_is_llp_matched"]))
                jet_ok = jet_ok and len(matched) == njet
            if not (jet_ok and cluster_ok and cell_ok):
                raise ValueError(f"Unaligned vectors: {relative}, row {index}")
            jet_counts = vectors["jet_n_clusters"]
            cluster_counts = vectors["cluster_n_cells"]
            count_values_ok = all(
                np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts)))
                for counts in (jet_counts, cluster_counts)
            )
            sum_jet_clusters = int(jet_counts.sum()) if count_values_ok else None
            sum_cluster_cells = int(cluster_counts.sum()) if count_values_ok else None
            rows.append({**base, "jet_count": njet, "cluster_count": ncluster, "cell_count": ncell,
                         "jet_vectors_aligned": jet_ok, "cluster_vectors_aligned": cluster_ok,
                         "cell_vectors_aligned": cell_ok, "nonnegative_integer_counts": count_values_ok,
                         "sum_jet_n_clusters": sum_jet_clusters, "sum_cluster_n_cells": sum_cluster_cells,
                         "jet_cluster_count_matches": sum_jet_clusters == ncluster,
                         "cluster_cell_count_matches": sum_cluster_cells == ncell})
            for jet_index in range(njet):
                jet = {**base, "jet_index": jet_index}
                jet.update({field: vectors[field][jet_index].item() for field in JET_FIELDS})
                jet["jet_is_llp_matched"] = int(matched[jet_index]) if has_truth_match else pd.NA
                jets.append(jet)
            # Use the first row with cells, without looking at its timing.
            if sample not in example_sources and ncell:
                example_sources.add(sample)
                finite_pair = np.isfinite(vectors["cell_z"]) & np.isfinite(vectors["cell_time"])
                examples.append({**base, "jet_count": njet, "cluster_count": ncluster,
                                 "cell_count": ncell, "finite_time_z_pairs": int(finite_pair.sum()),
                                 "selection": "First row with stored cells in sorted file order; no timing selection",
                                 "association_status": "All cells in row; no cell-to-jet ownership inferred"})
                for cell_index in range(ncell):
                    cell = {**base, "cell_index": cell_index, "jet_count": njet}
                    cell.update({k: vectors[k][cell_index].item() for k in CELL_FIELDS})
                    example_cells.append(cell)
    if len(example_sources) != len(GROUPS):
        raise ValueError("Could not decode a nonempty cell example from both samples")
    jets, rows, cells = pd.DataFrame(jets), pd.DataFrame(rows), pd.DataFrame(example_cells)
    for frame in (jets, rows, cells):
        for field in IDS:
            frame[field] = frame[field].astype("string")
    jets["jet_is_llp_matched"] = jets["jet_is_llp_matched"].astype("Int64")
    identity_fields = ["sample", "runNumber", "lumiBlock", "eventNumber"]
    rows["repeated_key_in_pilot"] = rows.duplicated(identity_fields, keep=False)
    rows.to_csv(output_dir / "pilot_rows.csv", index=False)
    jets.to_csv(output_dir / "pilot_jets.csv", index=False)
    cells.to_csv(output_dir / "example_cells.csv", index=False)
    write_json(output_dir / "decoded_examples.json", examples)
    return jets, rows, cells, examples


def summarize_pilot(jets, rows, cells, examples, inventory, output_dir):
    checks = ["jet_vectors_aligned", "cluster_vectors_aligned", "cell_vectors_aligned",
              "nonnegative_integer_counts", "jet_cluster_count_matches", "cluster_cell_count_matches"]
    groups = []
    quality = []
    for sample, group_rows in rows.groupby("sample", sort=False):
        group_jets = jets[jets["sample"] == sample]
        keys = ["runNumber", "lumiBlock", "eventNumber"]
        counts = group_rows.groupby(keys, dropna=False).size()
        multi = counts[counts > 1]
        matched = group_jets["jet_is_llp_matched"].dropna()
        groups.append({"sample": sample, "files_read": int(group_rows["source_file"].nunique()),
                       "rows_read": len(group_rows), "jets": len(group_jets),
                       "stored_cells_in_read_rows": int(group_rows["cell_count"].sum()),
                       "empty_jet_rows": int((group_rows["jet_count"] == 0).sum()),
                       "rows_with_multiple_jets": int((group_rows["jet_count"] > 1).sum()),
                       "distinct_recorded_event_keys": int(len(counts)),
                       "repeated_event_keys": int(len(multi)), "rows_in_repeated_event_keys": int(multi.sum()),
                       "truth_match_flag_available_jets": int(len(matched)),
                       "truth_match_flag_equals_one": int((matched == 1).sum()),
                       **{k + "_failures": int((~group_rows[k]).sum()) for k in checks}})
        for field in JET_FIELDS:
            values = group_jets[field].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            quantiles = np.quantile(finite, [0, .01, .5, .99, 1]).tolist() if len(finite) else [None] * 5
            quality.append({"sample": sample, "branch": field, "values": len(values),
                            "nonfinite": int((~np.isfinite(values)).sum()),
                            "minus_999999": int((values == -999999).sum()),
                            "negative_finite": int((finite < 0).sum()), "zero": int((finite == 0).sum()),
                            **dict(zip(["min", "p01", "median", "p99", "max"], quantiles))})
    event_sets = [set(map(tuple, rows[rows["sample"] == label][["runNumber", "lumiBlock", "eventNumber"]].to_numpy()))
                  for label, _ in GROUPS.values()]
    cross_keys = sorted(event_sets[0] & event_sets[1])
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive unweighted pilot, leading rows of every file in two selected groups; no physics labels or model training",
        "selection": {g: {"plot_label": label, "leading_rows_per_file": cap} for g, (label, cap) in GROUPS.items()},
        "groups": groups, "source_group_summary": inventory["group_summary"],
        "examples": examples, "jet_value_quality": quality,
        "same_recorded_event_keys_across_sources": [list(k) for k in cross_keys],
        "event_key_caveat": "Recorded run/lumi/event key only; MC namespace, production overlap and true physical-event identity need confirmation. No rows deduplicated.",
        "row_interpretation": "Rows can contain several jets and aligned cell arrays; event-like grouping is observed, producer-defined semantics not yet confirmed.",
        "association_status": "Counts and vector lengths checked. Ordering and cell-to-jet ownership unconfirmed; no mapping reconstructed.",
        "units": "All values retained in stored units. Energy/position/time units and time-of-flight corrections require production definitions.",
        "weighting": "Every stored jet contributes once. No event weights, luminosity normalization, quality/trigger cuts or pT/eta matching applied.",
        "truth": "LLP sample includes all reconstructed jets; jet_is_llp_matched retained only as a diagnostic, without assuming its semantics. Unpaired is not a pure-BIB truth label.",
        "feature_roles": {"candidate_reconstructed_inputs_after_definition_checks": CANDIDATE_FEATURES,
                          "reference_only": ["jet_BIB_score", "jet_chi2ndf_SM", "jet_chi2ndf_BIB"],
                          "identity_and_grouping_only": IDS + ["sample", "source_file", "row_index", "jet_index"],
                          "truth_diagnostic_only": ["jet_is_llp_matched"]},
        "versions": {name: version(name) for name in ("uproot", "awkward", "numpy", "pandas", "matplotlib", "nbformat", "nbclient")},
        "manifest_sha256": inventory["manifest_sha256"],
        "source_checksums": "Expected source hashes in file_inventory.json; not recomputed during this notebook run",
    }
    pd.DataFrame(quality).to_csv(Path(output_dir) / "jet_value_quality.csv", index=False)
    write_json(Path(output_dir) / "pilot_summary.json", result)
    return result


def meeting_notes(summary, output_dir):
    totals = ", ".join(f"{g['sample']}: {g['total_entries']:,} tree entries" for g in summary["source_group_summary"])
    executed_date = datetime.fromisoformat(summary["created_utc"]).astimezone(ZoneInfo("Europe/Stockholm")).strftime("%d %B %Y")
    lines = ["# First data look — meeting notes", "", f"Executed on the downloaded data, {executed_date}.", "",
             "## What was checked", "",
             totals + ". The pilot reads leading rows from every file, retaining stored units and all jets.", "",
             "| Sample | Files read | Rows read | Jets | Repeated recorded event keys |",
             "|---|---:|---:|---:|---:|"]
    for group in summary["groups"]:
        lines.append(f"| {group['sample']} | {group['files_read']} | {group['rows_read']} | {group['jets']} | {group['repeated_event_keys']} |")
    failures = sum(v for group in summary["groups"] for k, v in group.items() if k.endswith("_failures"))
    lines += ["", f"Vector lengths and jet/cluster/cell count sums were checked in every pilot row: **{failures} failed checks**. These checks do not establish the producer's cell-to-jet ordering.", "",
              "The three figures compare kinematics, reconstructed jet time/spread, and all stored cells in one deterministic row from each source. Relative source counts are not physical rates. No training, class labels or rejection performance are claimed.", "",
              "## Concrete findings", ""]
    for group in summary["groups"]:
        if group["truth_match_flag_available_jets"]:
            lines.append(f"- {group['sample']}: only {group['truth_match_flag_equals_one']:,} of {group['truth_match_flag_available_jets']:,} jets have `jet_is_llp_matched == 1`. The flag definition needs confirmation; all jets were retained in the comparisons.")
    for value in summary["jet_value_quality"]:
        if value["branch"] == "jet_BIB_score":
            lines.append(f"- {value['sample']}: {value['minus_999999']:,} `jet_BIB_score` values equal finite `-999999`. This looks like a sentinel and needs its documented meaning; a finite-value filter would retain it.")
    for example in summary["examples"]:
        lines.append(f"- The timing-blind {example['sample']} cell example has {example['cell_count']:,} cells across a row containing {example['jet_count']} jets. It is a decoded example, not a representative or truth-labeled pattern.")
    lines += ["- The jet kinematic distributions differ between the sources. Compare selections and MC/data production before interpreting separation.", "",
              "## Questions for Emil", "",
              "1. Which selection defines BIB in the unpaired sample, and what contamination remains? Which LLP truth-matching flag/value should define signal jets?",
              "2. What are the energy, position and time units? What ToF correction, calibration and invalid-value conventions were applied?",
              "3. How are cells linked to clusters and jets? Does ordered slicing using `jet_n_clusters` and `cluster_n_cells` reproduce the production code?",
              "4. What triggers, energy/quality thresholds and weights were used? How should the different MC/data productions and overlapping sample versions be compared?",
              "5. What defines a unique physical event across files/productions, and what does the direction/range of `jet_BIB_score` mean?", "",
              "## Next decision", "",
              "After those definitions are checked, construct a comparable jet-level feature table and an event-grouped train/validation/test split. Compare a small XGBoost baseline with the existing tagger, while keeping truth fields, identifiers and existing tagger outputs out of the initial model inputs.", ""]
    path = Path(output_dir) / "meeting_notes.md"
    path.write_text("\n".join(lines))
    return path

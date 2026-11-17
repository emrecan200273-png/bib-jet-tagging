"""Join a vertex-count ntuple to cell events without changing the ROOT files."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


EVENT_KEYS = ("runNumber", "eventNumber")
VERTEX_NAMES = ("nPrimaryVertices", "n_vrtx", "nPV")
TREE_TYPES = {"TTree", "ROOT::RNTuple"}


@dataclass(frozen=True)
class Row:
    file: str
    entry: int
    key: tuple[int | None, ...]
    value: int | float | None = None


def root_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    files = sorted(
        path for path in directory.rglob("*.root")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )
    if not files:
        raise ValueError(f"No ROOT files in {directory}")
    return files


def select_tree(handle, path, key_fields, vertex_fields=(), preferred=None, explicit=None):
    trees = {
        name: handle[name]
        for name, kind in handle.classnames(recursive=True, cycle=False).items()
        if kind in TREE_TYPES
    }
    required = set(key_fields)
    eligible = []
    for name, tree in trees.items():
        if explicit and name != explicit:
            continue
        branches = set(tree.keys())
        if not required <= branches:
            continue
        if vertex_fields:
            eligible.extend((name, tree, branch) for branch in vertex_fields if branch in branches)
        else:
            eligible.append((name, tree, None))
    if preferred and not explicit:
        eligible = [item for item in eligible if item[0] == preferred] or eligible
    if len(eligible) == 1:
        return eligible[0]
    if eligible:
        choices = ", ".join(f"{name}:{field}" if field else name for name, _, field in eligible)
        raise ValueError(f"{path}: ambiguous eligible trees/branches ({choices}); specify the tree or branch")
    branches = "; ".join(f"{name}: {', '.join(tree.keys())}" for name, tree in trees.items())
    requirement = f" and one of {', '.join(vertex_fields)}" if vertex_fields else ""
    raise ValueError(
        f"{path}: no eligible tree with {', '.join(key_fields)}{requirement}. "
        f"Available branches: {branches or '(no event trees)'}"
    )


def scalar_array(values, field, path, *, integer):
    try:
        array = ak.to_numpy(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be a scalar branch") from exc
    allowed = "iu" if integer else "iuf"
    if array.ndim != 1 or array.dtype.kind not in allowed:
        kind = "integer" if integer else "numeric"
        raise ValueError(f"{path}: {field} must be a scalar {kind} branch; got {array.dtype}")
    return array


def exact_id(value):
    if np.ma.is_masked(value):
        return None
    value = int(value)
    return value if value >= 0 else None


def raw_count(value):
    if np.ma.is_masked(value):
        return None
    value = value.item()
    return None if isinstance(value, float) and math.isnan(value) else value


def valid_count(value):
    if value is None or value < 0:
        return None
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer() and value < 2**53):
        return None
    return int(value)


def read_chunks(tree, path, key_fields, vertex_field):
    fields = [*key_fields, *([vertex_field] if vertex_field else [])]
    offset = 0
    for chunk in tree.iterate(fields, step_size="32 MB", library="ak", how=dict):
        ids = [scalar_array(chunk[field], field, path, integer=True) for field in key_fields]
        values = (
            scalar_array(chunk[vertex_field], vertex_field, path, integer=False)
            if vertex_field
            else None
        )
        yield offset, ids, values
        offset += len(ids[0])


def read_file(tree, path, relative_path, key_fields, vertex_field):
    for offset, ids, values in read_chunks(tree, path, key_fields, vertex_field):
        for index, event_ids in enumerate(zip(*ids)):
            key = tuple(exact_id(value) for value in event_ids)
            value = raw_count(values[index]) if values is not None else None
            yield Row(relative_path, offset + index, key, value)


def _nonnegative(values):
    data = np.ma.getdata(values)
    return data, ~np.ma.getmaskarray(values) & (data >= 0)


def read_matched_vertices(tree, path, relative_path, key_fields, vertex_field, target_keys, counts,
                          *, signed_event_column=None):
    """Count every raw row; materialize only rows that match a local cell key."""
    for offset, ids, values in read_chunks(tree, path, key_fields, vertex_field):
        if signed_event_column is not None:
            events = ids[signed_event_column]
            if events.dtype != np.dtype("int32"):
                raise ValueError(f"{path}: eventNumber must be signed int32 in bit-pattern mode")
            negative_events = np.ma.filled(events < 0, False)
            counts["reinterpreted_negative_vertex_event_rows"] += int(negative_events.sum())
            ids[signed_event_column] = events.astype(np.uint32)
        size = len(values)
        counts["vertex_rows"] += size
        data, valid_values = _nonnegative(values)
        if data.dtype.kind == "f":
            valid_values &= np.isfinite(data) & (data < 2**53) & (data == np.floor(data))
        counts["invalid_vertex_count_rows"] += size - int(np.count_nonzero(valid_values))

        keys = np.empty(size, dtype=target_keys.dtype)
        valid_ids = np.ones(size, dtype=bool)
        for name, column in zip(keys.dtype.names, ids):
            keys[name], valid = _nonnegative(column)
            valid_ids &= valid
        valid_rows = np.flatnonzero(valid_ids)
        counts["invalid_vertex_key_rows"] += size - len(valid_rows)
        if len(target_keys):
            positions = np.searchsorted(target_keys, keys[valid_rows])
            inside = positions < len(target_keys)
            rows = valid_rows[inside]
            matching_rows = rows[target_keys[positions[inside]] == keys[rows]]
        else:
            matching_rows = valid_rows[:0]
        counts["matched_vertex_rows"] += len(matching_rows)
        counts["vertex_only_rows"] += len(valid_rows) - len(matching_rows)
        if signed_event_column is not None:
            counts["matched_reinterpreted_negative_vertex_event_rows"] += int(negative_events[matching_rows].sum())
        for index in matching_rows:
            yield Row(
                relative_path, offset + int(index),
                tuple(int(column[index]) for column in ids), raw_count(values[index]),
            )


def _index_cells(files, directory, key_fields, tree_name, counts):
    tree_counts = Counter()
    rows, by_key = [], {}
    for path in files:
        with uproot.open(path) as handle:
            name, tree, _ = select_tree(handle, path, key_fields, preferred="JetContent", explicit=tree_name)
            tree_counts[name] += 1
            for row in read_file(tree, path, str(path.relative_to(directory)), key_fields, None):
                counts["cell_rows"] += 1
                rows.append(row)
                if None in row.key:
                    counts["invalid_cell_key_rows"] += 1
                    continue
                previous = by_key.get(row.key)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate cell event {row.key}: {previous.file}, entry {previous.entry} "
                        f"and {row.file}, entry {row.entry}; the match is not one-to-one"
                    )
                by_key[row.key] = row
    return rows, by_key, tree_counts


def _index_vertices(files, directory, key_fields, tree_name, branch, target_keys, counts, signed_event_column):
    matched_only = target_keys is not None
    tree_counts, vertices = Counter(), {}
    for path in files:
        with uproot.open(path) as handle:
            candidates = (branch,) if branch else VERTEX_NAMES
            name, tree, branch = select_tree(handle, path, key_fields, candidates, explicit=tree_name)
            if branch in key_fields:
                raise ValueError(f"{path}: the vertex-count branch must be separate from the event key")
            tree_counts[name] += 1
            source = str(path.relative_to(directory))
            if matched_only:
                rows = read_matched_vertices(tree, path, source, key_fields, branch, target_keys, counts,
                                             signed_event_column=signed_event_column)
            else:
                rows = read_file(tree, path, source, key_fields, branch)
            for row in rows:
                if not matched_only:
                    counts["vertex_rows"] += 1
                    counts["invalid_vertex_count_rows"] += valid_count(row.value) is None
                    if None in row.key:
                        counts["invalid_vertex_key_rows"] += 1
                        continue
                previous = vertices.get(row.key)
                if previous is None:
                    vertices[row.key] = row
                elif previous.value != row.value:
                    raise ValueError(
                        f"Conflicting vertex counts for event {row.key}: "
                        f"{previous.value!r} ({previous.file}, entry {previous.entry}) and "
                        f"{row.value!r} ({row.file}, entry {row.entry})"
                    )
                else:
                    counts["duplicate_vertex_rows"] += 1
    return vertices, tree_counts, branch


def _join_events(cell_rows, vertices, counts):
    records = []
    for row in cell_rows:
        sidecar = vertices.get(row.key)
        count = None
        if None in row.key:
            status = "invalid_cell_key"
        elif sidecar is None:
            counts["unmatched_cell_events"] += 1
            status = "unmatched"
        else:
            counts["key_matched_events"] += 1
            count = valid_count(sidecar.value)
            if count is None:
                counts["matched_invalid_vertex_count"] += 1
                status = "invalid_vertex_count"
            else:
                counts["matched_valid_vertex_count"] += 1
                counts["matched_zero_vertex_count"] += count == 0
                status = "matched"
        records.append((
            row.file, row.entry, *row.key, count, status,
            sidecar.file if sidecar else None, sidecar.entry if sidecar else None,
        ))
    return records


def match(
    cell_dir,
    vertex_dir,
    *,
    key_fields=EVENT_KEYS,
    vertex_keys=None,
    vertex_branch=None,
    cell_tree=None,
    vertex_tree=None,
    matched_only=False,
    vertex_event_int32_bits=False,
):
    cell_dir, vertex_dir = Path(cell_dir).resolve(), Path(vertex_dir).resolve()
    if vertex_dir.is_relative_to(cell_dir) or cell_dir.is_relative_to(vertex_dir):
        raise ValueError("Use separate cell and vertex input directories")
    key_fields = tuple(key_fields)
    vertex_keys = tuple(vertex_keys) if vertex_keys is not None else key_fields
    if (
        not key_fields
        or len(key_fields) != len(vertex_keys)
        or len(set(key_fields)) != len(key_fields)
        or len(set(vertex_keys)) != len(vertex_keys)
    ):
        raise ValueError("Cell and vertex keys must list the same number of distinct fields")
    signed_event_column = None
    if vertex_event_int32_bits:
        if not matched_only or "eventNumber" not in vertex_keys:
            raise ValueError("Vertex event bit-pattern mode requires matched-only mode and an eventNumber key")
        signed_event_column = vertex_keys.index("eventNumber")
    cell_files, vertex_files = root_files(cell_dir), root_files(vertex_dir)
    counts = Counter(dict.fromkeys((
        "cell_rows", "vertex_rows", "invalid_cell_key_rows", "invalid_vertex_key_rows",
        "invalid_vertex_count_rows", "duplicate_vertex_rows", "key_matched_events",
        "matched_valid_vertex_count", "matched_zero_vertex_count", "matched_invalid_vertex_count",
        "unmatched_cell_events", "vertex_only_event_keys", "vertex_only_valid_counts",
    ), 0))
    cell_rows, cells, cell_trees = _index_cells(cell_files, cell_dir, key_fields, cell_tree, counts)
    target_keys = None
    if matched_only:
        # Keep every bit of each uint64 event key.
        key_dtype = np.dtype([(f"k{i}", np.uint64) for i in range(len(key_fields))])
        target_keys = np.array(sorted(cells), dtype=key_dtype)
        counts.update({"matched_vertex_rows": 0, "vertex_only_rows": 0})
    if vertex_event_int32_bits:
        counts.update({"reinterpreted_negative_vertex_event_rows": 0, "matched_reinterpreted_negative_vertex_event_rows": 0})
    vertices, vertex_trees, vertex_branch = _index_vertices(
        vertex_files, vertex_dir, vertex_keys, vertex_tree, vertex_branch, target_keys, counts, signed_event_column,
    )
    records = _join_events(cell_rows, vertices, counts)

    if not matched_only:
        vertex_only = vertices.keys() - cells.keys()
        counts["vertex_only_event_keys"] = len(vertex_only)
        counts["vertex_only_valid_counts"] = sum(valid_count(vertices[key].value) is not None for key in vertex_only)
    report = {
        "cell_dir": str(cell_dir),
        "vertex_dir": str(vertex_dir),
        "cell_files": len(cell_files),
        "vertex_files": len(vertex_files),
        "cell_trees": dict(cell_trees),
        "vertex_trees": dict(vertex_trees),
        "cell_key_fields": list(key_fields),
        "vertex_key_fields": list(vertex_keys),
        "vertex_count_field": vertex_branch,
        "unique_cell_event_keys": len(cells),
        "unique_vertex_event_keys": len(vertices),
        **counts,
    }
    if matched_only:
        report.update(
            unique_vertex_event_keys=None,
            vertex_only_event_keys=None,
            vertex_only_valid_counts=None,
            unique_matched_vertex_event_keys=len(vertices),
            vertex_duplicate_check_scope="matched_cell_keys",
        )
    if vertex_event_int32_bits:
        report["vertex_event_encoding"] = "signed int32 bit pattern interpreted as uint32; full cell key unchanged"
    return report, records


def write_outputs(output_dir, report, records):
    output_dir = Path(output_dir).resolve()
    for key in ("cell_dir", "vertex_dir"):
        source = Path(report[key])
        if output_dir.is_relative_to(source):
            raise ValueError("Keep the output directory separate from the ROOT inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "event_mapping.csv.gz", "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow((
            "cell_file", "cell_entry", *report["cell_key_fields"], "vertex_count",
            "status", "vertex_file", "vertex_entry",
        ))
        writer.writerows(records)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("vertex_dir", type=Path)
    parser.add_argument("--key-branches", nargs="+", default=EVENT_KEYS)
    parser.add_argument("--vertex-key-branches", nargs="+")
    parser.add_argument("--vertex-branch", help="Vertex count branch (auto: nPrimaryVertices, n_vrtx or nPV)")
    parser.add_argument("--cell-tree")
    parser.add_argument("--vertex-tree")
    parser.add_argument(
        "--matched-only", action="store_true",
        help="Scan all vertex rows but retain and deduplicate only keys present in the cell files",
    )
    parser.add_argument("--vertex-event-int32-bits", action="store_true",
                        help="Interpret the sidecar's signed int32 eventNumber bits as unsigned; keep cell IDs exact")
    parser.add_argument("--output-dir", type=Path, help="Save summary.json and event_mapping.csv.gz")
    args = parser.parse_args(argv)
    try:
        report, records = match(
            args.cell_dir, args.vertex_dir, key_fields=args.key_branches,
            vertex_keys=args.vertex_key_branches, vertex_branch=args.vertex_branch,
            cell_tree=args.cell_tree, vertex_tree=args.vertex_tree, matched_only=args.matched_only,
            vertex_event_int32_bits=args.vertex_event_int32_bits,
        )
        if args.output_dir:
            write_outputs(args.output_dir, report, records)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Inspect one ROOT file per folder without reading event values."""
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

from analysis.data.data_paths import resolve_data_paths


def load_file_list(project_root):
    """Read the saved manifest for this project or the older CERN layout."""
    project = Path(project_root)
    if (project / "analysis/__init__.py").is_file():
        mode = "project"
        source, manifest_path = resolve_data_paths(project)
        output = project / "analysis/outputs/data_inventory/schema_inventory.json"
    else:
        mode = "CERN"
        manifest_path = project / "DataForEmre_transfer_20261029/verified_manifest.json"
        source = None
        output = project / "outputs/data_inventory/schema_inventory.json"

    manifest = json.loads(manifest_path.read_text())
    if source is None:
        source = Path(manifest["source"])
    if not source.is_dir():
        raise FileNotFoundError(f"Data folder is not accessible: {source}")
    if not any(importlib.util.find_spec(name) for name in ("uproot", "ROOT")):
        raise ModuleNotFoundError(
            "No ROOT reader found. Choose the 'PPP first look' kernel "
            "or analysis/.venv/bin/python."
        )
    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe file-list path: {relative}")
    return manifest, source, manifest_path, output, mode


def folder_summary(manifest):
    """Count direct files and choose the smallest nonempty ROOT file per folder."""
    grouped = defaultdict(list)
    for record in manifest["files"]:
        grouped[str(PurePosixPath(record["path"]).parent)].append(record)
    directories = sorted(set(manifest.get("directories", [])) | set(grouped))
    rows = [{"Folder": directory, "Files": len(grouped[directory]),
             "GB": sum(record["size"] for record in grouped[directory]) / 1e9}
            for directory in directories]
    samples = []
    for directory, records in sorted(grouped.items()):
        eligible = [r for r in records if r["path"].endswith(".root") and r["size"] > 0]
        if eligible:
            samples.append((directory, min(eligible, key=lambda r: (r["size"], r["path"]))))
    return rows, samples


def inspect_uproot(path):
    import uproot
    result = {"backend": "uproot", "version": uproot.__version__, "objects": []}
    with uproot.open(str(path)) as handle:
        for key, classname in handle.classnames(recursive=True).items():
            item = {"key": key, "class": classname}
            if classname.startswith("TTree") or "RNTuple" in classname:
                try:
                    tree = handle[key]
                    item["entries"] = int(tree.num_entries)
                    item["branches"] = dict(tree.typenames())
                except Exception as exc:
                    item["metadata_error"] = f"{type(exc).__name__}: {exc}"
            result["objects"].append(item)
    return result

def inspect_pyroot(path):
    import ROOT
    result = {"backend": "ROOT", "version": ROOT.gROOT.GetVersion(), "objects": []}
    handle = ROOT.TFile.Open(str(path), "READ")
    if not handle or handle.IsZombie():
        raise OSError("ROOT could not open a healthy file")
    def visit(directory, prefix=""):
        for key in directory.GetListOfKeys():
            name = key.GetName()
            classname = key.GetClassName()
            item = {"key": prefix + name + ";" + str(key.GetCycle()), "class": classname}
            if classname.startswith("TDirectory"):
                result["objects"].append(item)
                visit(directory.Get(name), prefix + name + "/")
                continue
            if classname.startswith("TTree"):
                try:
                    tree = directory.Get(name)
                    item["entries"] = int(tree.GetEntries())
                    item["branches"] = {}
                    def branches(items, parent=""):
                        for branch in items:
                            branch_name = parent + branch.GetName()
                            item["branches"][branch_name] = {
                                "class": branch.GetClassName(),
                                "leaves": {leaf.GetName(): leaf.GetTypeName()
                                           for leaf in branch.GetListOfLeaves()},
                            }
                            if branch.GetListOfBranches().GetEntries():
                                branches(branch.GetListOfBranches(), branch_name + "/")
                    branches(tree.GetListOfBranches())
                except Exception as exc:
                    item["metadata_error"] = f"{type(exc).__name__}: {exc}"
            elif "RNTuple" in classname:
                item["metadata_error"] = "RNTuple key found; this PyROOT fallback only inventories TTree branches"
            result["objects"].append(item)
    try:
        visit(handle)
    finally:
        handle.Close()
    return result

def inspect_samples(project, source, manifest_path, manifest, samples, mode):
    """Keep full metadata and access errors for each selected file."""
    inventory = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(project)) if manifest_path.is_relative_to(project) else str(manifest_path),
        "manifest_created_utc": manifest.get("created_utc"),
        "source": str(source),
        "original_cern_source": manifest.get("source"),
        "mode": mode,
        "mtime_check": "Compared with source record" if mode == "CERN" else "Not compared: downloaded copies have local timestamps",
        "scope": "ROOT metadata only; one file per populated source folder; no event decoding",
        "python": sys.version,
        "samples": [],
    }
    for directory, record in samples:
        path = source / record["path"]
        sample = {"group": directory, "path": record["path"],
                  "recorded_size": record["size"], "recorded_mtime_ns": record.get("mtime_ns"),
                  "recorded_sha256": record.get("sha256"), "attempt_errors": []}
        try:
            stat = path.stat()
            sample.update(live_size=stat.st_size, live_mtime_ns=stat.st_mtime_ns,
                          size_matches_snapshot=(stat.st_size == record["size"]))
            if mode == "CERN" and record.get("mtime_ns") is not None:
                sample["mtime_matches_snapshot"] = stat.st_mtime_ns == record["mtime_ns"]
            for reader in (inspect_uproot, inspect_pyroot):
                try:
                    sample["schema"] = reader(path)
                    partial = any("metadata_error" in item for item in sample["schema"]["objects"])
                    sample["status"] = "partial_metadata" if partial else "ok_metadata"
                    break
                except Exception as exc:
                    sample["attempt_errors"].append({"reader": reader.__name__,
                        "error": f"{type(exc).__name__}: {exc}"})
            else:
                sample["status"] = "failed"
        except Exception as exc:
            sample["status"] = "failed"
            sample["access_error"] = f"{type(exc).__name__}: {exc}"
        inventory["samples"].append(sample)
    return inventory


def save_inventory(inventory, output):
    """Save the complete report, then flag any unreadable or changed samples."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    problems = [sample for sample in inventory["samples"]
                if sample["status"] != "ok_metadata"
                or sample.get("size_matches_snapshot") is False]
    for sample in inventory["samples"]:
        if sample.get("mtime_matches_snapshot") is False:
            print("Source modification date changed:", sample["group"])
    if problems:
        for sample in problems:
            print(sample["group"], json.dumps(sample, indent=2))
        raise RuntimeError(f"Some files need attention. See {output}.")

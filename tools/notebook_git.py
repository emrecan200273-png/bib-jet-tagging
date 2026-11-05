"""Keep notebook source in Git while preserving executed local notebooks."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


def source_only(raw):
    notebook = json.loads(raw)
    notebook.get("metadata", {}).pop("first_look_run", None)
    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook["cells"]:
        cell.get("metadata", {}).pop("execution", None)
        cell.get("metadata", {}).pop("trusted", None)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()


def check_index():
    names = subprocess.check_output([
        "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
    ]).decode().split("\0")
    errors = []
    for name in filter(None, names):
        path = PurePosixPath(name)
        parts = path.parts
        excluded = (
            parts[0] in {"DataForEmre", "reference"}
            or parts[:2] in {("analysis", "outputs"), ("analysis", "provenance")}
            or name.endswith((".root", ".csv", ".parquet", ".tar", ".zip", ".npy", ".npz"))
            or ".root." in name
            or path.name == ".env"
            or (path.name.startswith(".env.") and path.name != ".env.example")
        )
        if excluded:
            errors.append(f"Local data or configuration is staged: {name}")
            continue
        raw = subprocess.check_output(["git", "show", f":{name}"])
        if len(raw) >= 50 * 1024 * 1024:
            errors.append(f"Unexpected large file is staged: {name}")
        if name.endswith(".ipynb"):
            if raw != source_only(raw):
                errors.append(f"Notebook outputs are staged; install the filter and add again: {name}")
            notebook = json.loads(raw)
            if any(cell.get("attachments") for cell in notebook["cells"]):
                errors.append(f"Review notebook attachments before publishing: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Staged source checked: no research data or notebook outputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check-index", action="store_true")
    args = parser.parse_args()
    if args.install:
        root = Path(__file__).resolve().parents[1]
        for key, value in {
            "filter.notebook.clean": "python3 tools/notebook_git.py",
            "filter.notebook.required": "true",
            "core.hooksPath": ".githooks",
        }.items():
            subprocess.run(["git", "-C", str(root), "config", "--local", key, value], check=True)
        (root / ".githooks" / "pre-commit").chmod(0o755)
        print("Installed notebook output filter and local commit check.")
    elif args.check_index:
        check_index()
    else:
        sys.stdout.buffer.write(source_only(sys.stdin.buffer.read()))


if __name__ == "__main__":
    main()

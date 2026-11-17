"""Find the local data and verified file list."""
import os
from pathlib import Path


def resolve_data_paths(project_root):
    project_root = Path(project_root)

    def path_from_env(name, default):
        value = Path(os.environ.get(name, str(default))).expanduser()
        return value if value.is_absolute() else project_root / value

    data_root = path_from_env("PPP_DATA_ROOT", project_root / "DataForEmre")
    manifest = path_from_env(
        "PPP_DATA_MANIFEST",
        project_root / "analysis/provenance/DataForEmre_verified_manifest_20261029.json",
    )
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {data_root}. Set PPP_DATA_ROOT to your "
            "DataForEmre folder. See README.md."
        )
    if not manifest.is_file():
        raise FileNotFoundError(
            f"File list not found: {manifest}. Set PPP_DATA_MANIFEST to your "
            "verified manifest. See README.md."
        )
    return data_root, manifest

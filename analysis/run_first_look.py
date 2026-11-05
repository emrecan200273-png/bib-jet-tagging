"""Execute a project notebook using the project-local Jupyter kernel."""
from datetime import datetime, timezone
from pathlib import Path
import os
import time
import argparse

ANALYSIS = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for key, directory in {"IPYTHONDIR": "ipython", "MPLCONFIGDIR": "matplotlib",
                       "JUPYTER_RUNTIME_DIR": "jupyter"}.items():
    cache = ANALYSIS / ".cache" / directory
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(cache)

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("notebook", nargs="?", default="02_first_data_look.ipynb",
                    choices=["01_remote_data_inventory.ipynb", "02_first_data_look.ipynb"])
args = parser.parse_args()
path = ANALYSIS / args.notebook
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=240, kernel_name="ppp-first-look",
                        resources={"metadata": {"path": str(ANALYSIS)}})
started = time.monotonic()
try:
    client.execute()
finally:
    notebook.metadata["first_look_run"] = {
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    nbformat.write(notebook, path)
print("Executed:", path)
print("Seconds:", round(time.monotonic() - started, 1))
print("Code execution counts:", [c.execution_count for c in notebook.cells if c.cell_type == "code"])
print("Errors:", sum(o.output_type == "error" for c in notebook.cells for o in c.get("outputs", [])))

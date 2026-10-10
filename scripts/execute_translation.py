from pathlib import Path
import argparse
import json
import os
import platform
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument(
        "--inprocess", action="store_true", help="Execute plain Python cells without a kernel socket"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = (
        [root / name for name in args.notebooks]
        if args.notebooks
        else sorted((root / "notebooks").glob("1[45]_*.ipynb"))
    )
    records = []
    for path in paths:
        started = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if args.inprocess:
            from inprocess import execute_inprocess

            execute_inprocess(notebook, root)
        else:
            NotebookClient(
                notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(root)}}
            ).execute()
        nbformat.write(notebook, path)
        records.append(
            {
                "notebook": path.name,
                "seconds": round(time.monotonic() - started, 2),
                "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
                "status": "passed",
            }
        )
        print(json.dumps(records[-1]), flush=True)
    (root / "results").mkdir(exist_ok=True)
    report = {
        "python": platform.python_version(),
        "engine": "inprocess" if args.inprocess else "jupyter",
        "mode": "full" if os.environ.get("PROJECT_FULL") == "1" else "short",
        "notebooks": records,
    }
    (root / "results" / "translation_execution_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

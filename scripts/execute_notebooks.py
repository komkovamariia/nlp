"""Run selected notebooks and save outputs only after a successful execution."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", help="Paths relative to the repository root")
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--inprocess", action="store_true", help="Run plain Python cells without a kernel")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per cell in Jupyter mode")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_path = root / "results" / "notebook_execution_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else {"notebooks": {}}
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "gensim",
                 "nltk", "transformers", "datasets", "sentence-transformers",
                 "matplotlib", "nbformat", "nbclient", "ipykernel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    failed = False
    for name in args.notebooks:
        path = (root / name).resolve()
        relative = str(path.relative_to(root))
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        record = {
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": packages,
            "engine": "inprocess" if args.inprocess else "jupyter",
            "code_sha256": hashlib.sha256(json.dumps(code).encode()).hexdigest(),
            "code_cells": len(code),
        }
        started = time.monotonic()
        try:
            if args.inprocess:
                from inprocess import execute_inprocess

                execute_inprocess(notebook, root)
            else:
                NotebookClient(notebook, timeout=args.timeout, kernel_name=args.kernel,
                               resources={"metadata": {"path": str(root)}}).execute()
        except Exception as error:
            failed = True
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        else:
            nbformat.write(notebook, path)
            record["status"] = "passed"
        record["seconds"] = round(time.monotonic() - started, 2)
        report["notebooks"][relative] = record
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{relative}: {record['status']} ({record['seconds']} s)", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


def execute_inprocess(notebook, root: Path):
    namespace = {"__name__": "__main__"}
    original_directory = Path.cwd()
    original_show = plt.show
    os.chdir(root)
    try:
        for number, cell in enumerate(notebook.cells, 1):
            if cell.cell_type != "code":
                continue
            outputs = []
            stdout, stderr = io.StringIO(), io.StringIO()

            def flush_streams():
                for name, buffer in [("stdout", stdout), ("stderr", stderr)]:
                    text = buffer.getvalue()
                    if text:
                        outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
                        buffer.seek(0)
                        buffer.truncate(0)

            def show(*args, **kwargs):
                flush_streams()
                for identifier in plt.get_fignums():
                    figure = plt.figure(identifier)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                    outputs.append(
                        nbformat.v4.new_output(
                            "display_data",
                            data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
                        )
                    )
                plt.close("all")

            plt.show = show
            with redirect_stdout(stdout), redirect_stderr(stderr):
                tree = ast.parse(cell.source)
                filename = f"{root.name}:cell_{number}"
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    final_expression = tree.body.pop()
                    exec(compile(tree, filename, "exec"), namespace)
                    value = eval(compile(ast.Expression(final_expression.value), filename, "eval"), namespace)
                    if value is not None:
                        flush_streams()
                        outputs.append(
                            nbformat.v4.new_output(
                                "execute_result", execution_count=number, data={"text/plain": repr(value)}
                            )
                        )
                else:
                    exec(compile(tree, filename, "exec"), namespace)
            flush_streams()
            cell.outputs = outputs
            cell.execution_count = number
        notebook.metadata["execution"] = {
            "engine": "inprocess",
            "plots": "matplotlib Agg",
            "python_cells": "executed in order",
        }
    finally:
        plt.show = original_show
        plt.close("all")
        os.chdir(original_directory)
    return notebook

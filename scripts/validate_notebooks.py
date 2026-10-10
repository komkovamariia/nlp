"""Validate notebook structure and compile Python cells without running model downloads."""
import ast
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1] / 'notebooks'
count = 0
for path in sorted(root.glob('*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'{path.name}:cell{index}')
            count += 1
    print('OK', path.name)
print(f'Validated {count} Python code cells')

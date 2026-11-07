import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('HW3_MMDA.ipynb')
notebook = nbformat.read(path, as_version=4)
print('Starting notebook kernel', flush=True)
client = NotebookClient(notebook, timeout=900, startup_timeout=60, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}},
    on_cell_start=lambda cell, cell_index: print(f'Cell {cell_index}', flush=True))
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Notebook executed successfully')

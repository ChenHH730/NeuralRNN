"""Extract full source of given cells from a notebook. Usage:
  python extract_cells.py <notebook.ipynb> <cell_idx> [cell_idx ...]
"""
import json
import sys

nb = sys.argv[1]
data = json.load(open(nb, encoding="utf-8"))
print(f"== {nb} ({len(data['cells'])} cells)")
for arg in sys.argv[2:]:
    i = int(arg)
    c = data["cells"][i]
    print(f"--- cell {i} ({c['cell_type']}, id={c.get('id')}):")
    print("".join(c["source"]))

"""
FlyWire 3D Graph — Robust Coordinate Parser (any digits, variable spacing)
- Parses coordinates like "[523768  95008  81080]" safely using regex
- Logs malformed rows to bad_coords_log.csv (row index + raw value)
- Uses ORIGINAL coordinates (no scaling)
- Plotly 3D scatter, no edges
"""

import re
import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import plotly.graph_objects as go

# --------------------------- Paths ---------------------------
connectivity_path = r"C:\Users\sysco\Downloads\FlyWire Raw Data\connections_princeton.csv\connections_princeton.csv"
coords_path       = r"C:\Users\sysco\Downloads\FlyWire Raw Data\coordinates.csv\coordinates.csv"

# --------------------------- Load Data ---------------------------
print("Loading connectivity CSV...")
connectivity = pd.read_csv(connectivity_path)
print(f"✅ Loaded {len(connectivity):,} edges")

print("Loading coordinates CSV...")
coords_df = pd.read_csv(coords_path)
print(f"✅ Loaded {len(coords_df):,} coordinate entries")

# --------------------------- Parse Coordinates (regex) ---------------------------
print("\nParsing [x y z] coordinates (regex-based, tolerant to any spacing/digits)...")

TRIPLE_INT_IN_BRACKETS = re.compile(r"\[?\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s*\]?")
bad_entries = []

def parse_position(pos_str, row_idx=None):
    s = str(pos_str)
    m = TRIPLE_INT_IN_BRACKETS.search(s)
    if not m:
        bad_entries.append((row_idx, pos_str))
        return [np.nan, np.nan, np.nan]
    try:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    except Exception as e:
        bad_entries.append((row_idx, pos_str))
        return [np.nan, np.nan, np.nan]

coords_df["parsed"] = [
    parse_position(coords_df.iloc[i, 1], row_idx=i) for i in range(len(coords_df))
]

# --- Show and save malformed rows BEFORE any filtering ---
if bad_entries:
    bad_df = pd.DataFrame(bad_entries, columns=["RowIndex", "RawValue"])
    bad_df.to_csv("bad_coords_log.csv", index=False)
    print(f"\n⚠️  Found {len(bad_entries):,} malformed coordinate rows — saved to bad_coords_log.csv")
    print("🔍 Showing 10 examples:")
    print(bad_df.head(10).to_string(index=False))
else:
    print("✅  No malformed coordinate rows detected during parsing.")

# Continue as before
valid_mask = ~coords_df["parsed"].apply(lambda v: any(np.isnan(v)))
coords_valid = coords_df.loc[valid_mask].copy()
coords_dict = {
    coords_valid.iloc[i, 0]: coords_valid["parsed"].iloc[i]
    for i in range(len(coords_valid))
}
print(f"\n✅ Parsed {len(coords_dict):,} valid coordinates out of {len(coords_df):,}")

# --- Diagnose lines that produced NaN triples (i.e. never matched regex) ---
bad_mask = coords_df["parsed"].apply(lambda v: any(np.isnan(v)))
bad_subset = coords_df.loc[bad_mask, [coords_df.columns[0], coords_df.columns[1]]]

print(f"\n⚠️  {len(bad_subset):,} coordinate rows did not match the regex.")
print("🔍  Showing 10 examples of their raw text:")
print(bad_subset.head(10).to_string(index=False))

bad_subset.to_csv("bad_coords_log.csv", index=False)
print("\n💾  Saved all unmatched coordinate rows to bad_coords_log.csv")

# --------------------------- Plotly 3D Scatter (No Scaling, No Edges) ---------------------------
print("\nPreparing 3D neuron scatter for Plotly (original coordinates)...")

# Convert coordinate dictionary to arrays
ids = list(coords_dict.keys())
coords = np.array(list(coords_dict.values()))

x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

# Create 3D scatter trace for neurons
node_trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="markers",
    marker=dict(
        size=2,
        color="cyan",
        opacity=0.8
    ),
    text=[f"Neuron {nid}" for nid in ids],
    hoverinfo="text"
)

# --------------------------- Build Figure ---------------------------
fig = go.Figure(data=[node_trace])
fig.update_layout(
    title="FlyWire 3D Neuron Positions (Original Coordinates, No Edges)",
    showlegend=False,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="black"
    ),
    paper_bgcolor="black",
    margin=dict(l=0, r=0, b=0, t=40)
)

# Save interactive HTML (still very lightweight)
output_path = "FlywireGraph_Plotly3D_originalCoords.html"
fig.write_html(output_path)
print(f"✅ Saved 3D scatter (no edges, original coordinates) to {output_path}")
print("👉 Open it in Edge or Chrome — it will load instantly, even for 140k neurons.")

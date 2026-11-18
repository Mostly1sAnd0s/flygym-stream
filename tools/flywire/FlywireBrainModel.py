"""BANC weighted breadth-first spread visualization (not a biophysical simulation).

Place in the project root or Data folder. Dependencies:
python -m pip install numpy pandas pyarrow networkx plotly tqdm
"""
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from tqdm import tqdm

MAX_NODES = 10000
DECAY_FACTOR = 0.9
EDGE_STYLE_BINS = 32


def data_folder():
    here = Path(__file__).resolve().parent
    for folder in (here / "Data", here, here.parent / "Data"):
        if (folder / "banc_888_meta.feather").is_file():
            return folder
    raise FileNotFoundError("Put this script in the project root or Data folder containing banc_888_meta.feather.")


def column(df, *names, required=True):
    lookup = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for name in names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"Missing column matching {names}. Actual columns: {list(df.columns)}")
    return None


def neuron_id(value):
    # Never convert large root IDs through float: that loses precision.
    value = str(value).strip()
    if re.fullmatch(r"\d+", value):
        return str(int(value))
    if re.fullmatch(r"\d+\.0+", value):
        return str(int(value.split('.')[0]))
    return None


def position(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = value
    else:
        parts = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(value))
    if len(parts) != 3:
        return None
    try:
        result = np.asarray(parts, dtype=float)
        return result if np.isfinite(result).all() else None
    except (TypeError, ValueError):
        return None


def estimate_missing_positions(G, coords):
    """Fill missing graph positions in synchronous layers from anchored neighbors.

    Incoming and outgoing raw synapse counts contribute. Reciprocal connections
    add their counts; self-loops do not provide an independent position.
    Known positions never change. Each layer uses only previous-layer positions.
    """
    missing = set(G).difference(coords)
    estimated = {}
    frontier = set(G).intersection(coords)
    round_number = 0
    while missing and frontier:
        sums, totals = {}, {}
        for neighbor in frontier:
            point = coords[neighbor]
            links = list(G.out_edges(neighbor, data=True)) + list(G.in_edges(neighbor, data=True))
            for u, v, data in links:
                target = v if u == neighbor else u
                if target not in missing or target == neighbor:
                    continue
                weight = data["raw_weight"]
                if target not in sums:
                    sums[target] = np.zeros(3, dtype=float)
                    totals[target] = 0.0
                sums[target] += weight * point
                totals[target] += weight
        round_number += 1
        updates = {}
        for target in sums:
            # Include ALL currently positioned neighbors, not just the new frontier.
            total = 0.0
            weighted = np.zeros(3, dtype=float)
            links = list(G.out_edges(target, data=True)) + list(G.in_edges(target, data=True))
            for u, v, data in links:
                neighbor = v if u == target else u
                if neighbor == target or neighbor not in coords:
                    continue
                weight = data["raw_weight"]
                weighted += weight * coords[neighbor]
                total += weight
            if total > 0:
                updates[target] = weighted / total
        coords.update(updates)
        estimated.update({rid: round_number for rid in updates})
        missing.difference_update(updates)
        frontier = set(updates)
        print(f"Position estimation round {round_number}: {len(updates):,} neurons.")
    print(f"Estimated {len(estimated):,} positions; {len(missing):,} remain unanchored.")
    return estimated


def load_data(folder):
    print("Loading BANC metadata and nanometer coordinates...")
    meta = pd.read_feather(folder / "banc_888_meta.feather")
    id_col = column(meta, "banc_888_id", "root_id", "Root ID")
    pos_col = column(meta, "root_position_nm")
    coords = {}
    for rid, point in zip(meta[id_col], meta[pos_col]):
        rid, point = neuron_id(rid), position(point)
        if rid is not None and point is not None:
            coords[rid] = point
    if not coords:
        raise ValueError("No valid root_position_nm triplets found.")
    print(f"Loaded {len(meta):,} metadata rows; {len(coords):,} valid positions.")

    print("Loading connectivity CSV...")
    edges = pd.read_csv(folder / "connections_princeton.csv", dtype=str)
    src = column(edges, "pre_root_id", "pre", "source", "source_id", "pre_id", "From", "Presynaptic ID")
    dst = column(edges, "post_root_id", "post", "target", "target_id", "post_id", "To", "Postsynaptic ID")
    weight = column(edges, "syn_count", "synapse_count", "count", "weight", "Synapses", "Number of Synapses")
    print(f"Using columns: source={src}, target={dst}, synapses={weight}")
    clean = pd.DataFrame({"src": edges[src].map(neuron_id), "dst": edges[dst].map(neuron_id),
                          "raw_weight": pd.to_numeric(edges[weight], errors="coerce")})
    valid = clean.src.notna() & clean.dst.notna() & np.isfinite(clean.raw_weight) & (clean.raw_weight > 0)
    print(f"Ignoring {(~valid).sum():,} invalid/nonpositive connection rows.")
    clean = clean.loc[valid].groupby(["src", "dst"], sort=False, as_index=False).raw_weight.sum()
    del edges
    graph_ids = set(clean.src) | set(clean.dst)
    overlap = graph_ids.intersection(coords)
    fraction = len(overlap) / max(1, len(graph_ids))
    print(f"Coordinate overlap: {len(overlap):,}/{len(graph_ids):,} graph neurons ({fraction:.1%}).")
    if not overlap:
        raise ValueError("Connections and BANC positions share no IDs. Download BANC v888 connections; old FAFB connections cannot be combined with BANC.")
    if fraction < 0.9:
        print("WARNING: Low ID overlap. Check dataset/version before interpreting estimated positions.")
    totals = clean.groupby("src").raw_weight.transform("sum")
    clean["weight"] = clean.raw_weight / totals
    print("Building directed weighted graph...")
    G = nx.DiGraph()
    for u, v, raw, norm in tqdm(clean.itertuples(index=False, name=None), total=len(clean), desc="Adding edges"):
        G.add_edge(u, v, raw_weight=float(raw), weight=float(norm))
    print(f"Graph: {G.number_of_nodes():,} neurons, {G.number_of_edges():,} connections.")

    labels = {}
    attributes = folder / "neurons.csv"
    if attributes.exists():
        attrs = pd.read_csv(attributes, dtype=str)
        ac = column(attrs, "Root ID", "root_id", "banc_888_id")
        fields = [column(attrs, name, required=False) for name in
                  ("Primary Cell Type", "cell_type", "Verified NT type", "Predicted NT type", "Super Class", "Soma side")]
        fields = list(dict.fromkeys(c for c in fields if c is not None))
        for row in attrs[[ac] + fields].itertuples(index=False, name=None):
            rid = neuron_id(row[0])
            if rid is not None:
                labels[rid] = "<br>".join(f"{c}: {v}" for c, v in zip(fields, row[1:]) if pd.notna(v) and str(v).strip())
        print(f"Loaded attributes for {len(labels):,} neurons.")
    estimated = estimate_missing_positions(G, coords)
    for rid in coords:
        status = (f"Estimated position (round {estimated[rid]})" if rid in estimated
                  else "Original reference position")
        labels[rid] = labels.get(rid, "") + "<br>" + status
    return G, coords, labels


def ask_integer(prompt, default, minimum):
    while True:
        value = input(prompt).strip()
        if not value:
            return default
        try:
            result = int(value)
            if result >= minimum:
                return result
        except ValueError:
            pass
        print(f"Enter an integer >= {minimum}, or leave blank.")


def choose_start(G, coords):
    manual = input("Enter neuron ID to activate (blank = auto-selection): ").strip()
    if manual:
        rid = neuron_id(manual)
        if rid not in G:
            raise ValueError(f"Neuron {manual} not found in graph.")
        return rid
    candidates = [n for n, d in G.out_degree() if 8 <= d <= 15 and n in coords]
    if not candidates:
        candidates = [n for n, d in G.out_degree() if 5 <= d <= 25 and n in coords]
    if not candidates:
        candidates = [n for n in G if n in coords]
    start = min(candidates, key=lambda n: abs(G.out_degree(n) - 10))
    print(f"Auto-selected {start}: {G.out_degree(start)} outgoing connections.")
    return start


def propagate(G, start, top_n, max_depth):
    visited, activation = {start}, {start: 1.0}
    frontier, plotted, strongest = [(start, 1.0)], [], []
    levels = 0
    for depth in tqdm(range(max_depth), desc="Depth levels"):
        next_frontier = []
        for neuron, strength in frontier:
            outgoing = sorted(G.out_edges(neuron, data=True), key=lambda e: e[2]["weight"], reverse=True)
            if top_n is not None:
                outgoing = outgoing[:top_n]
            if not outgoing:
                continue
            strongest.append((outgoing[0][0], outgoing[0][1]))
            for u, v, data in outgoing:
                spread = strength * data["weight"] * DECAY_FACTOR
                if spread <= 0:
                    continue
                plotted.append((u, v, spread))
                if v not in visited and len(visited) < MAX_NODES:
                    visited.add(v)
                    activation[v] = spread
                    next_frontier.append((v, spread))
        levels = depth + 1
        frontier = next_frontier
        if not frontier or len(visited) >= MAX_NODES:
            break
    print(f"Reached {len(visited):,} neurons; recorded {len(plotted):,} edges in {levels} levels.")
    for rid in sorted(visited)[:5]:
        print(f"Sample neuron {rid}: {G.out_degree(rid)} outgoing connections")
    return visited, activation, plotted, strongest, levels


def visualize(G, coords, labels, start, visited, activation, edges, strongest, levels, output):
    points = np.array(list(coords.values()))
    traces = [go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
              marker=dict(size=1.1, color="rgba(0,255,255,0.15)"), hoverinfo="skip", name="All neurons")]
    active_ids = sorted(visited.intersection(coords))
    print(f"Drawing {len(active_ids):,} activated neurons; {len(visited) - len(active_ids):,} lack positions.")
    if active_ids:
        active = np.array([coords[rid] for rid in active_ids])
        text = [f"ID: {rid}<br>First-arrival strength: {activation[rid]:.4g}<br>{labels.get(rid, '')}" for rid in active_ids]
        traces.append(go.Scatter3d(x=active[:, 0], y=active[:, 1], z=active[:, 2], mode="markers",
                      marker=dict(size=2, color="rgb(0,180,255)", opacity=0.85), text=text,
                      hovertemplate="%{text}<extra></extra>", name="Activated"))
    if start in coords:
        sx, sy, sz = coords[start]
        traces.append(go.Scatter3d(x=[sx], y=[sy], z=[sz], mode="markers",
                      marker=dict(size=7, color="white"), text=[start], name="Start"))
    # Opacity shows raw synapse counts; propagation still uses normalized counts.
    drawable = [(u, v) for u, v, _ in edges if u in coords and v in coords]
    counts = [G[u][v]["raw_weight"] for u, v in drawable]
    low = math.log10(min(counts)) if counts else 0.0
    high = math.log10(max(counts)) if counts else 1.0
    scale_high = high if high > low else low + 1.0
    buckets = [[] for _ in range(EDGE_STYLE_BINS)]
    strong_buckets = [[] for _ in range(EDGE_STYLE_BINS)]
    strong_set = set(strongest)
    for u, v in drawable:
        value = math.log10(G[u][v]["raw_weight"])
        strength = (value - low) / (scale_high - low)
        index = min(EDGE_STYLE_BINS - 1, int(strength * EDGE_STYLE_BINS))
        buckets[index].append((u, v))
        if (u, v) in strong_set:
            strong_buckets[index].append((u, v))
    def line_coordinates(pairs):
        xs, ys, zs = [], [], []
        for u, v in pairs:
            a, b = coords[u], coords[v]
            xs.extend((a[0], b[0], None)); ys.extend((a[1], b[1], None)); zs.extend((a[2], b[2], None))
        return xs, ys, zs
    for i, pairs in enumerate(buckets):
        if not pairs:
            continue
        strength = (i + 0.5) / EDGE_STYLE_BINS
        color = "rgb(0,180,255)"
        opacity = 0.08 + 0.87 * strength
        xs, ys, zs = line_coordinates(pairs)
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                      line=dict(width=1.5, color=color),
                      opacity=opacity, name="Synapse count"))
        if strong_buckets[i]:
            xs, ys, zs = line_coordinates(strong_buckets[i])
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                          line=dict(width=3, color=color), opacity=opacity,
                          name="Strongest outgoing edges"))
    fig = go.Figure(traces)
    fig.update_layout(title=f"BANC Weighted Spread (depth {levels}, start {start})<br><sup>Opacity: synapse count (log scale); thicker lines: strongest outgoing edges</sup>", showlegend=False,
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 aspectmode="data", bgcolor="black"), paper_bgcolor="black",
                      font=dict(color="white"), margin=dict(l=0, r=0, b=0, t=65))
    fig.write_html(str(output), include_plotlyjs=True)
    print(f"Saved interactive 3D visualization: {output}")


def main():
    folder = data_folder()
    G, coords, labels = load_data(folder)
    start = choose_start(G, coords)
    top_n = ask_integer("Number of strongest outgoing edges per neuron (blank = all): ", None, 1)
    depth = ask_integer("Maximum propagation depth (blank = 100): ", 100, 1)
    result = propagate(G, start, top_n, depth)
    output_folder = folder.parent / "Outputs"
    output_folder.mkdir(exist_ok=True)
    visualize(G, coords, labels, start, *result, output_folder / f"BANC_Plotly3D_spread_{start}.html")


if __name__ == "__main__":
    main()
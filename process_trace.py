"""Visualize a handwritten trace.json (see trace.json) as a DAG.

Each entry in "nodes" is a COMM op with a "deps" list of node ids it depends
on; edges are drawn dep -> node. Nodes are colored by "subtype" and laid out
in layers by longest-path distance from a root, so the DAG reads top-to-bottom
in execution order.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import networkx as nx

from trace_io import load_trace

SUBTYPE_COLORS = {
    "ALL_GATHER": "#6fa8dc",
    "ALL_REDUCE": "#93c47d",
    "REDUCE_SCATTER": "#f6b26b",
    "ALL_TO_ALL": "#c27ba0",
    "SEND": "#ffd966",
    "RECV": "#e69138",
    "BARRIER": "#999999",
    "PRE_LAYER": "#f1c232",
    "POST_LAYER": "#f1c232",
}
DEFAULT_COLOR = "#cccccc"


def layered_layout(graph: nx.DiGraph) -> dict:
    layers = list(nx.topological_generations(graph))
    for depth, layer_nodes in enumerate(layers):
        for n in layer_nodes:
            graph.nodes[n]["_layer"] = depth
    return nx.multipartite_layout(graph, subset_key="_layer", align="horizontal")


def draw_trace(graph: nx.DiGraph, output_path: str | None = None, show: bool = False) -> None:
    pos = layered_layout(graph)
    # multipartite_layout with align="horizontal" places layer 0 at the top when
    # we flip y, so later layers read top-to-bottom.
    pos = {n: (x, -y) for n, (x, y) in pos.items()}

    colors = [SUBTYPE_COLORS.get(d.get("subtype"), DEFAULT_COLOR) for _, d in graph.nodes(data=True)]
    labels = {n: f"{n}: {d.get('subtype')}\n{d.get('name', '')}" for n, d in graph.nodes(data=True)}

    plt.figure(figsize=(max(8, graph.number_of_nodes() * 1.2), 8))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1800)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=15, node_size=1800)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=12, label=subtype)
        for subtype, color in SUBTYPE_COLORS.items()
        if any(d.get("subtype") == subtype for _, d in graph.nodes(data=True))
    ]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved trace DAG to {output_path}")
    if show:
        plt.show()
    plt.close()

def _dot_source(graph: nx.DiGraph) -> str:
    """Shared by write_trace_to_dot (rendering) and _dot_layout_positions (layout-only)."""
    lines = ["digraph G {", "  rankdir=TB;", "  node [shape=box, style=filled, fontsize=9];"]
    for n, d in graph.nodes(data=True):
        color = SUBTYPE_COLORS.get(d.get("subtype"), DEFAULT_COLOR)
        label = f"{n}: {d.get('subtype')}\\n{d.get('name', '')}".replace('"', '\\"')
        lines.append(f'  "{n}" [label="{label}", fillcolor="{color}"];')
    for u, v in graph.edges():
        lines.append(f'  "{u}" -> "{v}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_layout_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float, float, float]]:
    """Runs graphviz 'dot' purely as a layout engine (no rendering) via 'dot -Tplain',
    to get real crossing-minimized (x, y, width, height) per node. Returns {} if the
    'dot' command isn't available. dot's plain output is in inches with y increasing
    upward; we scale to points (72/inch) and flip y so the root ends up at the top,
    matching GraphML/yEd's y-increases-downward convention.
    """
    if shutil.which("dot") is None:
        return {}

    fd, dot_path = tempfile.mkstemp(suffix=".dot")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(_dot_source(graph))
        result = subprocess.run(["dot", "-Tplain", dot_path], capture_output=True, text=True, check=True)
    finally:
        os.remove(dot_path)

    SCALE = 72.0
    raw = {}
    max_y = 0.0
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            name, x, y, w, h = parts[1], float(parts[2]) * SCALE, float(parts[3]) * SCALE, \
                float(parts[4]) * SCALE, float(parts[5]) * SCALE
            raw[name] = (x, y, w, h)
            max_y = max(max_y, y)
    return {name: (x, max_y - y, w, h) for name, (x, y, w, h) in raw.items()}


def write_trace_to_graphml(graph: nx.DiGraph, output_path: str) -> None:
    """Writes yEd-ready GraphML: real node labels/colors/positions via yFiles' 'y:'
    namespace (computed from graphviz dot's layout), plus the raw data attributes
    (type/subtype/name/comm_group/size/deps) for inspection in yEd's Properties view.
    Opening this in yEd shows a laid-out, labeled graph immediately — plain
    networkx-exported GraphML has no visual data at all, which is why labels showed
    up empty and every node landed on top of the others.
    """
    positions = _dot_layout_positions(graph)

    parts = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:y="http://www.yworks.com/xml/graphml" '
        'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
        '<key for="node" id="d0" attr.name="type" attr.type="string"/>',
        '<key for="node" id="d1" attr.name="subtype" attr.type="string"/>',
        '<key for="node" id="d2" attr.name="name" attr.type="string"/>',
        '<key for="node" id="d3" attr.name="comm_group" attr.type="string"/>',
        '<key for="node" id="d4" attr.name="size" attr.type="long"/>',
        '<key for="node" id="d5" attr.name="deps" attr.type="string"/>',
        '<key for="node" id="d_yf" yfiles.type="nodegraphics"/>',
        '<graph edgedefault="directed">',
    ]

    for n, d in graph.nodes(data=True):
        color = SUBTYPE_COLORS.get(d.get("subtype"), DEFAULT_COLOR)
        label = escape(f"{n}: {d.get('subtype', '')}\n{d.get('name', '')}")
        comm_group = ",".join(str(x) for x in d.get("comm_group", []))
        deps = ",".join(str(x) for x in d.get("deps", []))
        x, y, w, h = positions.get(str(n), (0.0, 0.0, 140.0, 40.0))

        parts.append(f'<node id="{n}">')
        parts.append(f'<data key="d0">{escape(str(d.get("type", "")))}</data>')
        parts.append(f'<data key="d1">{escape(str(d.get("subtype", "")))}</data>')
        parts.append(f'<data key="d2">{escape(str(d.get("name", "")))}</data>')
        parts.append(f'<data key="d3">{escape(comm_group)}</data>')
        parts.append(f'<data key="d4">{d.get("size", 0)}</data>')
        parts.append(f'<data key="d5">{escape(deps)}</data>')
        parts.append(
            f'<data key="d_yf"><y:ShapeNode>'
            f'<y:Geometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>'
            f'<y:Fill color="{color}"/>'
            f'<y:NodeLabel>{label}</y:NodeLabel>'
            f'<y:Shape type="rectangle"/>'
            f'</y:ShapeNode></data>'
        )
        parts.append("</node>")

    for u, v in graph.edges():
        parts.append(f'<edge source="{u}" target="{v}"/>')

    parts.append("</graph>")
    parts.append("</graphml>")

    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Saved trace DAG to {output_path}")


def write_trace_to_dot(graph: nx.DiGraph, output_path: str) -> None:
    """Renders the DAG via Graphviz's real 'dot' layout engine — proper rank-based
    crossing minimization, unlike layered_layout/multipartite_layout above. Writes a
    .dot file next to output_path, then shells out to the system 'dot' binary to
    render output_path itself (format inferred from its extension, e.g. .png/.svg/.pdf).
    Requires Graphviz's 'dot' command to be installed — not a Python package, so
    pydot/pygraphviz aren't needed.
    """
    dot_path = output_path.rsplit(".", 1)[0] + ".dot"
    with open(dot_path, "w") as f:
        f.write(_dot_source(graph))

    if shutil.which("dot") is None:
        print(
            f"Wrote {dot_path}, but the 'dot' command isn't installed — skipping render. "
            f"Install Graphviz, or render manually: dot -Tpng {dot_path} -o {output_path}"
        )
        return

    ext = output_path.rsplit(".", 1)[-1] if "." in output_path else "png"
    subprocess.run(["dot", f"-T{ext}", dot_path, "-o", output_path], check=True)
    print(f"Saved trace DAG to {output_path} (via {dot_path})")


def main():
    parser = argparse.ArgumentParser(description="Visualize a trace.json as a DAG.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace.json file.")
    parser.add_argument("--plot", type=str, default=None, help="Path to save a PNG visualization.")
    parser.add_argument("--graphml", type=str, default=None, help="Path to save a GraphML export.")
    parser.add_argument(
        "--dot", type=str, default=None,
        help="Path to save a Graphviz dot-layout render (e.g. .png/.svg/.pdf); requires the 'dot' command.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    graph = load_trace(args.trace)
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Trace is not a DAG, found cycle: {cycle}")

    if args.plot or args.show:
        draw_trace(graph, output_path=args.plot, show=args.show)
    if args.graphml:
        write_trace_to_graphml(graph, args.graphml)
    if args.dot:
        write_trace_to_dot(graph, args.dot)


if __name__ == "__main__":
    main()

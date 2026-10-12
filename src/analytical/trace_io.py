"""Load a trace.json (see trace.json) into a dependency DAG."""

from __future__ import annotations

import logging

import networkx as nx

from tracegen.compress import load_trace_json

logger = logging.getLogger("trace_io")


def load_trace(path: str) -> nx.DiGraph:
    """Returns a dag with one node per trace node (attrs preserved, including the
    literal comm_group NPU-id list), edges dep -> node. `path` may be a plain
    .json trace or a zstd-compressed .json.zst one (see compress.py) -- either
    way the data is fully decompressed in memory, never to a file on disk.
    """
    data = load_trace_json(path)

    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for node in data["nodes"]:
        for dep in node.get("deps", []):
            graph.add_edge(dep, node["id"])

    #Graph-level (not per-node) attr: the axis order this trace's comm_group
    #ids were generated with (see trace_generator.py's write_trace_to_json),
    #or None for a trace that predates this field. analytical_model.py
    #cross-checks this against the topology's own axis_order.
    graph.graph["axis_order"] = data.get("axis_order")

    return graph

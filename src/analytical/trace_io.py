"""Load a trace.json (see trace.json) into a dependency DAG."""

from __future__ import annotations

import networkx as nx

from tracegen.compress import load_trace_json


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

    return graph

"""Load a trace.json (see trace.json) into a dependency DAG."""

from __future__ import annotations

import json

import networkx as nx


def load_trace(path: str) -> nx.DiGraph:
    """Returns a dag with one node per trace node (attrs preserved, including the
    literal comm_group NPU-id list), edges dep -> node.
    """
    with open(path) as f:
        data = json.load(f)

    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for node in data["nodes"]:
        for dep in node.get("deps", []):
            graph.add_edge(dep, node["id"])

    return graph

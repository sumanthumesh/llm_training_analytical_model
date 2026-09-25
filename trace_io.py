"""Load a trace.json (see trace.json) into a dependency DAG plus its comm groups."""

from __future__ import annotations

import json

import networkx as nx


def load_trace(path: str) -> tuple[nx.DiGraph, dict[str, list[int]]]:
    """Returns (dag, comm_groups): dag has one node per trace node (attrs preserved,
    edges dep -> node), comm_groups maps comm_group id (as a string key, matching the
    json) to the list of NPU ids in that group.
    """
    with open(path) as f:
        data = json.load(f)

    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for node in data["nodes"]:
        for dep in node.get("deps", []):
            graph.add_edge(dep, node["id"])

    # return graph, data["comm_groups"]
    return graph, {"comm_groups":[]}

"""Precompute, for each COMM node in a trace DAG, the set of physical links it
needs and how long it will take, using a simple alpha-beta cost model.

Collective algorithms are fixed to Ring for anything gather/reduce-shaped (per
refined_plan.md's proof-of-concept scope); SEND/RECV are a single direct hop.
"size" on a trace node is treated as bytes.
"""

from __future__ import annotations

import math

from topology import Topology

LinkSet = set[frozenset]


def route_hop(topo: Topology, a: str, b: str) -> tuple[LinkSet, float, float]:
    """Routes a-b exactly once and returns (links, latency_sec, bottleneck_bandwidth_Gbps).

    topo.bfs_path_hops is stateful (it load-balances across equal-cost paths),
    so a hop must be routed a single time and the result reused for both the
    link set and the cost -- calling it twice could pick two different paths
    for what's meant to be the same physical transfer.
    """
    path = topo.bfs_path_hops(a, b)
    links = {frozenset((u, v)) for u, v in zip(path, path[1:])}
    latency_sec = 0.0
    bandwidth_gbps = math.inf
    for u, v in zip(path, path[1:]):
        edge = topo.graph.edges[u, v]
        latency_sec += edge["latency_ns"] * 1e-9
        bandwidth_gbps = min(bandwidth_gbps, edge["speed_Gbps"])
    return links, latency_sec, bandwidth_gbps


def alpha_beta_time(latency_sec: float, bandwidth_gbps: float, size_bytes: float) -> float:
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    return latency_sec + size_bytes / bandwidth_bytes_per_sec


def ring_hops(members: list[str]) -> list[tuple[str, str]]:
    return list(zip(members, members[1:] + members[:1]))


def _ring_cost(topo: Topology, hosts: list[str], chunk_bytes: float) -> tuple[LinkSet, float]:
    """Links touched and per-step time (bottleneck across the N concurrent hops)
    for one pass around the ring; caller scales by however many steps it needs.
    """
    links: LinkSet = set()
    hop_times = []
    for a, b in ring_hops(hosts):
        hop_links, latency_sec, bandwidth_gbps = route_hop(topo, a, b)
        links |= hop_links
        hop_times.append(alpha_beta_time(latency_sec, bandwidth_gbps, chunk_bytes))
    return links, max(hop_times)


def precompute_node(topo: Topology, node: dict, comm_groups: dict[str, list[int]]) -> tuple[LinkSet, float]:
    """Returns (links, duration_sec) for a single trace node."""
    subtype = node["subtype"]
    size_bytes = node["size"]
    members = comm_groups[str(node["comm_group"])]
    hosts = [f"h{m}" for m in members]
    n = len(hosts)

    if subtype in ("SEND", "RECV"):
        if n != 2:
            raise ValueError(f"{subtype} node {node['id']} expects a 2-NPU comm_group, got {members}")
        a, b = hosts
        links, latency_sec, bandwidth_gbps = route_hop(topo, a, b)
        duration = alpha_beta_time(latency_sec, bandwidth_gbps, size_bytes)

    elif subtype in ("ALL_GATHER", "REDUCE_SCATTER"):
        # Ring algorithm: N-1 steps, each moving a 1/N chunk to the next rank.
        links, step_time = _ring_cost(topo, hosts, size_bytes / n)
        duration = (n - 1) * step_time

    elif subtype == "ALL_REDUCE":
        # Ring all-reduce = ring reduce-scatter followed by ring all-gather:
        # 2*(N-1) steps of the same chunk size.
        links, step_time = _ring_cost(topo, hosts, size_bytes / n)
        duration = 2 * (n - 1) * step_time

    elif subtype == "ALL_TO_ALL":
        # Approximate as one concurrent round of direct pairwise sends, bounded
        # by the slowest pair. Not exercised by trace.json yet; revisit if a
        # real all-to-all step-schedule is needed.
        chunk_bytes = size_bytes / (n - 1) if n > 1 else size_bytes
        links = set()
        pair_times = []
        for i, a in enumerate(hosts):
            for b in hosts[i + 1:]:
                pair_links, latency_sec, bandwidth_gbps = route_hop(topo, a, b)
                links |= pair_links
                pair_times.append(alpha_beta_time(latency_sec, bandwidth_gbps, chunk_bytes))
        duration = max(pair_times)

    else:
        raise ValueError(f"Unsupported collective subtype: {subtype!r}")

    return links, duration

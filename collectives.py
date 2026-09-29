"""Precompute, for each node in a trace DAG, the logical hops it needs and how
long it will take, using a simple alpha-beta cost model.

Collective algorithms are fixed to Ring for anything gather/reduce-shaped (per
refined_plan.md's proof-of-concept scope); SEND/RECV/SEND_RECV are a single
direct hop; SYNC-type nodes (barriers, pre/post-layer, pre/post-pipeline) are
free. "size" on a trace node is treated as bytes.

Physical link selection is deferred to runtime (LinkManager.acquire): this
module only resolves each logical hop (e.g. a ring neighbor pair) down to its
list of equal-cost candidate physical paths via the routing table, since which
one is actually free can only be known once the simulation is running.
"""

from __future__ import annotations

from topology import PathInfo, RoutingTable

HopCandidates = list[list[PathInfo]]


def alpha_beta_time(latency_sec: float, bandwidth_gbps: float, size_bytes: float) -> float:
    bandwidth_bytes_per_sec = bandwidth_gbps * 2**30 / 8
    return latency_sec + size_bytes / bandwidth_bytes_per_sec


def ring_hops(members: list[str]) -> list[tuple[str, str]]:
    return list(zip(members, members[1:] + members[:1]))


def precompute_node(node: dict, routing_table: RoutingTable) -> tuple[HopCandidates, float]:
    """Returns (hop_candidates, duration_sec) for a single trace node.
    hop_candidates is one list of equal-cost PathInfo alternatives per logical
    hop the collective needs; every alternative for a given hop has the same
    latency/bandwidth by construction (see build_routing_table), so duration
    can be computed here without knowing which one LinkManager ends up using.
    SYNC nodes (barriers, pre/post-layer and pre/post-pipeline markers) and
    DUMMY nodes (a collective over a degree-1 parallelism dimension, e.g. a CP
    all-gather when cp=1 -- see trace_generator.py's make_comm_node) are both
    structural only -- no hops, no duration.
    """
    if node["type"] in ("SYNC", "DUMMY"):
        return [], 0.0

    subtype = node["subtype"]
    size_bytes = node["size"]
    hosts = [f"h{m}" for m in node["comm_group"]]
    n = len(hosts)

    def step_time(hop_candidates: HopCandidates, chunk_bytes: float) -> float:
        # Bottleneck across the N concurrent hops of one ring step (or, for
        # ALL_TO_ALL, one round of pairwise sends) -- any candidate in a hop's
        # list gives the same cost, so [0] is as good as any other.
        return max(alpha_beta_time(c[0].latency_sec, c[0].bandwidth_gbps, chunk_bytes) for c in hop_candidates)

    if subtype in ("SEND", "RECV", "SEND_RECV"):
        if n != 2:
            raise ValueError(f"{subtype} node {node['id']} expects a 2-NPU comm_group, got {hosts}")
        hop_candidates = [routing_table[hosts[0], hosts[1]]]
        duration = step_time(hop_candidates, size_bytes)

    elif subtype in ("ALL_GATHER", "REDUCE_SCATTER"):
        # Ring algorithm: N-1 steps, each moving a 1/N chunk to the next rank.
        hop_candidates = [routing_table[a, b] for a, b in ring_hops(hosts)]
        duration = (n - 1) * step_time(hop_candidates, size_bytes / n)

    elif subtype == "ALL_REDUCE":
        # Ring all-reduce = ring reduce-scatter followed by ring all-gather:
        # 2*(N-1) steps of the same chunk size.
        hop_candidates = [routing_table[a, b] for a, b in ring_hops(hosts)]
        duration = 2 * (n - 1) * step_time(hop_candidates, size_bytes / n)

    elif subtype == "ALL_TO_ALL":
        # Approximate as one concurrent round of direct pairwise sends, bounded
        # by the slowest pair. Not exercised by trace.json yet; revisit if a
        # real all-to-all step-schedule is needed.
        chunk_bytes = size_bytes / (n - 1) if n > 1 else size_bytes
        hop_candidates = [routing_table[a, b] for i, a in enumerate(hosts) for b in hosts[i + 1:]]
        duration = step_time(hop_candidates, chunk_bytes)

    else:
        raise ValueError(f"Unsupported collective subtype: {subtype!r}")

    return hop_candidates, duration

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

import logging

from analytical.topology import PathInfo, RoutingTable
from typing import List, Tuple

logger = logging.getLogger("collectives")

HopCandidates = list[list[PathInfo]]


def alpha_beta_time(latency_sec: float, bandwidth_gbps: float, size_bytes: float) -> float:
    bandwidth_bytes_per_sec = bandwidth_gbps * 2**30 / 8
    return latency_sec + size_bytes / bandwidth_bytes_per_sec


def ring_hops(members: list[str]) -> list[tuple[str, str]]:
    return list(zip(members, members[1:] + members[:1]))


def roofline_time(flops: float, tensor_size_bytes: float, peak_perf_tflops: float, local_mem_bw_gbps: float) -> float:
    """Roofline model: an op takes as long as whichever of the compute-bound
    or memory-bound time actually dominates -- see astra-sim's Roofline.cc /
    Workload::issue_comp, which this mirrors (their two-step
    operational_intensity -> min(bandwidth*intensity, peak_perf) -> FLOPs/perf
    collapses to this same max() closed form).
    """
    peak_perf = peak_perf_tflops * 1e12
    local_mem_bw = local_mem_bw_gbps * 1e9
    return max(flops / peak_perf, tensor_size_bytes / local_mem_bw)


def compute_time(matmuls: list[dict], peak_perf_tflops: float, local_mem_bw_gbps: float) -> float:
    """Time for one inner (parallel) list of a COMP node's comps: everything
    in it runs at once on the same NPU, so their FLOPs and bytes-moved simply
    add up before applying the roofline model once to the totals.
    """
    total_flops = sum(mm["FLOPs_numeric"] for mm in matmuls)
    total_tensor_size = sum(mm["tensor_size_numeric"] for mm in matmuls)
    return roofline_time(total_flops, total_tensor_size, peak_perf_tflops, local_mem_bw_gbps)


def precompute_node(
    node: dict,
    routing_table: RoutingTable,
    peak_perf_tflops: float,
    local_mem_bw_gbps: float,
    fixed_overhead_sec: float = 0.0,
) -> tuple[HopCandidates, float]:
    """Returns (hop_candidates, duration_sec) for a single trace node.
    hop_candidates is one list of equal-cost PathInfo alternatives per logical
    hop the collective needs; every alternative for a given hop has the same
    latency/bandwidth by construction (see build_routing_table), so duration
    can be computed here without knowing which one LinkManager ends up using.
    SYNC nodes (barriers, pre/post-layer and pre/post-pipeline markers) and
    DUMMY nodes (a collective over a degree-1 parallelism dimension, e.g. a CP
    all-gather when cp=1 -- see trace_generator.py's make_comm_node) are both
    structural only -- no hops, no duration.

    COMP nodes have no hops (pure compute, no network) -- comps is a
    List[List[Matmul]] where the inner list is a parallel group (one
    compute_time() call, roofline over the group's totals) and the outer list
    is sequential stages (their times simply sum, to honor the dependency:
    e.g. for [[X@W_1,X@W_2],[Y@W_oF]], the first two run together, but the
    third can't start until both of the first two are done).

    fixed_overhead_sec is a one-time, per-op cost (kernel launch / issue
    overhead) added once per COMM node regardless of its ring step count --
    distinct from alpha_beta_time's per-hop latency_sec, which is paid once
    per ring step. Fit against real_system_reference.csv: at the smallest
    message size, all_reduce (10 ring steps) was barely slower than
    all_gather (5 steps) -- 23.57us vs 19.01us, nowhere near 2x -- which rules
    out a per-step explanation and points to a fixed per-invocation cost
    instead. It isn't a network resource, so it doesn't touch LinkManager /
    hop_candidates -- just a flat addition to duration.
    """
    if node["type"] in ("SYNC", "DUMMY"):
        return [], 0.0

    if node["type"] == "COMP":
        duration = sum(compute_time(stage, peak_perf_tflops, local_mem_bw_gbps) for stage in node["comps"])
        return [], duration

    subtype = node["subtype"]
    size_bytes = node["size"]
    hosts = [f"h{m}" for m in node["comm_group"]]
    n = len(hosts)

    def step_time(hop_candidates: HopCandidates, chunk_bytes: float) -> float:
        # Bottleneck across the N concurrent hops of one ring step (or, for
        # ALL_TO_ALL, one round of pairwise sends) -- any candidate in a hop's
        # list gives the same cost, so [0] is as good as any other.
        return max(alpha_beta_time(c[0].latency_sec, c[0].bandwidth_gbps, chunk_bytes) for c in hop_candidates)

    def print_hop_candidates(hop_candidates: HopCandidates) -> str:
        return "\n".join(f"  {i}: {', '.join('[%s]' % str(p) for p in c)}" for i, c in enumerate(hop_candidates))

    def algorithm_hops(hops:List[Tuple[str,str]])->str:
        return ",".join(f"({a},{b})" for a,b in hops)

    if subtype in ("SEND", "RECV", "SEND_RECV"):
        if n != 2:
            raise ValueError(f"{subtype} node {node['id']} expects a 2-NPU comm_group, got {hosts}")
        logger.debug("%s, %s, %s", subtype, node["id"], node["name"])
        hop_candidates = [routing_table[hosts[0], hosts[1]]]
        duration = step_time(hop_candidates, size_bytes)
        logger.debug("Hop candidates:\n%s", print_hop_candidates(hop_candidates))

    elif subtype in ("ALL_GATHER", "REDUCE_SCATTER"):
        logger.debug("%s, %s, %s", node["id"], subtype, node["name"])
        # Ring algorithm: N-1 steps, each moving a 1/N chunk to the next rank.
        ring_paths = ring_hops(hosts)
        hop_candidates = [routing_table[a, b] for a, b in ring_paths]
        duration = (n - 1) * step_time(hop_candidates, size_bytes / n)
        logger.debug("Ring paths: %s", algorithm_hops(ring_paths))
        logger.debug("Hop candidates:\n%s", print_hop_candidates(hop_candidates))

    elif subtype == "ALL_REDUCE":
        logger.debug("%s, %s, %s", node["id"], subtype, node["name"])
        # Ring all-reduce = ring reduce-scatter followed by ring all-gather:
        # 2*(N-1) steps of the same chunk size.
        ring_paths = ring_hops(hosts)
        hop_candidates = [routing_table[a, b] for a, b in ring_paths]
        duration = 2 * (n - 1) * step_time(hop_candidates, size_bytes / n)
        logger.debug("Ring paths: %s", algorithm_hops(ring_paths))
        logger.debug("Hop candidates:\n%s", print_hop_candidates(hop_candidates))

    elif subtype == "ALL_TO_ALL":
        # Approximate as one concurrent round of direct pairwise sends, bounded
        # by the slowest pair. Not exercised by trace.json yet; revisit if a
        # real all-to-all step-schedule is needed.
        chunk_bytes = size_bytes / (n - 1) if n > 1 else size_bytes
        hop_candidates = [routing_table[a, b] for i, a in enumerate(hosts) for b in hosts[i + 1:]]
        duration = step_time(hop_candidates, chunk_bytes)

    else:
        raise ValueError(f"Unsupported collective subtype: {subtype!r}")

    return hop_candidates, fixed_overhead_sec + duration

"""Static correctness checks for a generated trace.json, without running the
full roofline simulation. The motivating check (same_rank_ordering) is what
caught the fsdp-prefetch under-serialization bug by hand: two COMP nodes that
are meant to represent the same physical rank (here, the same pipeline stage,
the only rank dimension this trace format actually replicates -- see
trace_generator.py's "one rank's worth of compute" scope) must have a path
between them in the DAG, in one direction or the other. If neither is
reachable from the other, the simulator would let them run concurrently,
which isn't physically possible on a single NPU.

COMM-node same-link correctness doesn't need an equivalent static check:
LinkManager reserves physical links at simulation runtime, so it's correct by
construction regardless of DAG wiring. COMP nodes have no analogous runtime
protection (no rank tag, no ComputeManager), so this is the one safety net
available today short of building that.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import defaultdict

import networkx as nx

from analytical.trace_io import load_trace


def infer_pipeline_stages(graph: nx.DiGraph) -> dict[int, int]:
    """Maps each COMP node id -> the pipeline stage it belongs to.

    Inferred entirely from the trace's own node names -- no external --pp/
    --layers arguments needed. Pipeline stage count comes from how many
    distinct "pipeline_stageN" markers appear (the PRE_PIPELINE/POST_PIPELINE
    sync nodes always name theirs); layer count comes from how many distinct
    "layerN" markers appear; layers_per_stage is the two divided, matching
    exactly how trace_generator.py itself assigns layers to stages.
    """
    stage_ids = set()
    layer_ids = set()
    for _, d in graph.nodes(data=True):
        name = d.get("name", "")
        m = re.search(r"pipeline_stage(\d+)", name)
        if m:
            stage_ids.add(int(m.group(1)))
        m = re.search(r"\.layer(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))

    if not stage_ids or not layer_ids:
        return {}

    num_stages = len(stage_ids)
    num_layers = len(layer_ids)
    if num_layers % num_stages != 0:
        raise ValueError(f"{num_layers} distinct layers not evenly divisible by {num_stages} distinct pipeline stages")
    layers_per_stage = num_layers // num_stages

    node_stage = {}
    for n, d in graph.nodes(data=True):
        if d.get("type") != "COMP":
            continue
        m = re.search(r"\.layer(\d+)\.", d.get("name", ""))
        if m:
            node_stage[n] = int(m.group(1)) // layers_per_stage
    return node_stage


def check_dag_valid(graph: nx.DiGraph) -> list[str]:
    if nx.is_directed_acyclic_graph(graph):
        return []
    cycle = nx.find_cycle(graph)
    return [f"trace is not a DAG, found cycle: {cycle}"]


def check_single_root(graph: nx.DiGraph) -> list[str]:
    roots = [n for n, d in graph.in_degree() if d == 0]
    if len(roots) == 1:
        return []
    names = [graph.nodes[n]["name"] for n in roots]
    return [f"expected exactly 1 root node (no deps), found {len(roots)}: {names}"]


def check_same_rank_ordering(graph: nx.DiGraph) -> list[str]:
    """SYNC/DUMMY nodes are zero-duration -- an ordering gap between them
    can't skew a completion time -- so this only checks COMP nodes, the ones
    that actually consume simulated wall-clock time.
    """
    node_stage = infer_pipeline_stages(graph)
    by_stage: dict[int, list[int]] = defaultdict(list)
    for n, stage in node_stage.items():
        by_stage[stage].append(n)

    violations = []
    for stage, nodes in sorted(by_stage.items()):
        nodes = sorted(nodes)
        # Ancestors computed once per node in this stage rather than a full
        # V x V transitive closure over the whole graph -- much cheaper once
        # a trace has many stages and thousands of nodes.
        ancestors = {n: nx.ancestors(graph, n) for n in nodes}
        for a, b in itertools.combinations(nodes, 2):
            if b not in ancestors[a] and a not in ancestors[b]:
                violations.append(
                    f"stage {stage}: '{graph.nodes[a]['name']}' (id={a}) and "
                    f"'{graph.nodes[b]['name']}' (id={b}) have no path either way -- "
                    f"the simulator would let them run concurrently despite being the same rank"
                )
    return violations


CHECKS = [
    ("single_root", check_single_root),
    ("same_rank_ordering", check_same_rank_ordering),
]


def main():
    parser = argparse.ArgumentParser(description="Static correctness checks for a trace.json.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace.json file.")
    args = parser.parse_args()

    graph = load_trace(args.trace)
    print(f"Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges from {args.trace}\n")

    total_violations = 0

    dag_violations = check_dag_valid(graph)
    print(f"[dag_valid] {'OK' if not dag_violations else f'{len(dag_violations)} issue(s)'}")
    for v in dag_violations:
        print(f"  - {v}")
    total_violations += len(dag_violations)

    if dag_violations:
        print("\nSkipping remaining checks -- they assume a valid DAG.")
    else:
        for check_name, check_fn in CHECKS:
            violations = check_fn(graph)
            print(f"[{check_name}] {'OK' if not violations else f'{len(violations)} issue(s)'}")
            for v in violations:
                print(f"  - {v}")
            total_violations += len(violations)

    print()
    if total_violations:
        print(f"FAILED: {total_violations} issue(s) found.")
        sys.exit(1)
    print("PASSED: no issues found.")


if __name__ == "__main__":
    main()

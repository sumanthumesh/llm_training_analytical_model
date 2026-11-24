"""Reports how many times each (collective_type, size, num_nodes, axis)
combination occurs in a generated trace.json, as a CSV:
collective_type,size,num_nodes,axis,number_of_times_encountered.

Counts COMM nodes only by default -- DUMMY nodes are a collective over a
degree-1 parallelism dimension (e.g. a CP all-gather when cp=1, see
trace_generator.py's make_comm_node) collapsed to zero-cost structural markers,
not real communication, so they're excluded unless --include-dummy is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
import tracegen.compress as compress

# Which parallelism axis a COMM node belongs to isn't stored as its own field
# -- it has to be inferred from the node name, going by trace_generator.py's
# own naming conventions for each collective call site (fsdp_all_gather,
# grad_dp_reduce_scatter -> DP; cp_all_gather, cp_reduce_scatter,
# grad_cp_all_reduce -> CP; tp_all_reduce -> TP; pipeline_stageN_to_M ->
# PP). Checked in this order since "fsdp" itself contains the substring "dp".
AXIS_PATTERNS = [
    ("PP", re.compile(r"pipeline_stage")),
    ("DP", re.compile(r"fsdp|_dp_")),
    ("CP", re.compile(r"cp_")),
    ("TP", re.compile(r"tp_")),
]


def classify_axis(name: str) -> str:
    for axis, pattern in AXIS_PATTERNS:
        if pattern.search(name):
            return axis
    return "UNKNOWN"


def count_collectives(trace_path: str, include_dummy: bool = False) -> Counter:
    
    trace = compress.load_trace_json(trace_path)
    
    types = {"COMM", "DUMMY"} if include_dummy else {"COMM"}
    counts: Counter = Counter()
    for node in trace["nodes"]:
        if node["type"] not in types:
            continue
        axis = classify_axis(node["name"])
        counts[node["subtype"], node["size"], len(node["comm_group"]), axis] += 1
    return counts


def write_csv(counts: Counter, output) -> None:
    writer = csv.writer(output)
    writer.writerow(["collective_type", "size", "num_nodes", "axis", "number_of_times_encountered"])
    for (collective_type, size, num_nodes, axis), count in sorted(counts.items()):
        writer.writerow([collective_type, size, num_nodes, axis, count])


def axis_transition_gaps(graph) -> dict[tuple[str, str], float]:
    """Minimum end-to-start gap between two axis-classified COMM nodes that
    land on different axes AND share at least one NPU in their comm_group,
    keyed by (prev_axis, next_axis).

    "Consecutive" is relative to one specific shared NPU's own local timeline
    of the collectives it participates in, not the trace's global timeline:
    two collectives that don't share any NPU have nothing physically to do
    with each other, so a THIRD, unrelated collective landing between them in
    global time (but sharing no NPU with either) must not block comparing
    them directly against each other -- which a simple "skip adjacent-in-
    global-time pairs with no overlap" filter would do, since it would only
    ever compare globally-adjacent pairs and never look past that unrelated
    one sitting in between. Building each NPU's own sorted sub-sequence
    avoids this: a collective neither shares nor cares about never enters
    that NPU's sequence in the first place.

    Requires `graph` to already be simulated with
    analytical_model.simulate(..., track_overlap=True), which is what
    populates every node's start_time/end_time -- this module otherwise only
    ever reads a static trace.json, which has no timing info at all.

    gap = next.start_time - prev.end_time: zero or positive is genuine idle
    time between the two; negative means they actually overlap (prev was
    still running, on different physical links, when next started).

    (prev_axis, next_axis) and (next_axis, prev_axis) are tracked separately
    -- they come from different pairs of nodes in the trace and can have
    different minimums -- see axis_pair_min_gap for a direction-agnostic
    merge of the two.

    DUMMY nodes (a degree-1 collective collapsed to a zero-cost no-op --
    see trace_generator.py's make_comm_node) never touch the network and
    have no start_time/end_time, so -- like SYNC nodes -- they're excluded;
    only real COMM nodes are considered.
    """
    events_by_npu: dict[int, list[tuple[float, float, str]]] = {}
    for _, d in graph.nodes(data=True):
        if d.get("type") != "COMM":
            continue
        event = (d["start_time"], d["end_time"], classify_axis(d["name"]))
        for npu in d["comm_group"]:
            events_by_npu.setdefault(npu, []).append(event)

    min_gap: dict[tuple[str, str], float] = {}
    for events in events_by_npu.values():
        events.sort(key=lambda e: e[0])
        for (_, prev_end, prev_axis), (next_start, _, next_axis) in zip(events, events[1:]):
            if prev_axis == next_axis:
                continue
            gap = next_start - prev_end
            key = (prev_axis, next_axis)
            if key not in min_gap or gap < min_gap[key]:
                min_gap[key] = gap
    return min_gap


def axis_pair_min_gap(graph) -> dict[frozenset[str], float]:
    """Direction-agnostic view of axis_transition_gaps: the minimum gap
    between a collective on axis A and one on axis B, in EITHER order --
    "the shortest time difference between a collective on tp and cp,
    however they're ordered" -- by taking the smaller of the two directional
    minimums for each unordered pair.
    """
    merged: dict[frozenset[str], float] = {}
    for (a, b), gap in axis_transition_gaps(graph).items():
        key = frozenset((a, b))
        if key not in merged or gap < merged[key]:
            merged[key] = gap
    return merged


def main():
    parser = argparse.ArgumentParser(description="Report collective_type,size,number_of_times_encountered for a trace.json.")
    parser.add_argument("trace", type=str, help="Path to the trace.json file.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path (default: stdout).")
    parser.add_argument("--include-dummy", action="store_true", help="Also count DUMMY nodes (degree-1 collectives collapsed to zero-cost markers).")
    args = parser.parse_args()

    counts = count_collectives(args.trace, args.include_dummy)

    if args.output:
        with open(args.output, "w", newline="") as f:
            write_csv(counts, f)
    else:
        write_csv(counts, sys.stdout)


if __name__ == "__main__":
    main()

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
    with open(trace_path) as f:
        trace = json.load(f)

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


def main():
    parser = argparse.ArgumentParser(description="Report collective_type,size,number_of_times_encountered for a trace.json.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace.json file.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: stdout).")
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

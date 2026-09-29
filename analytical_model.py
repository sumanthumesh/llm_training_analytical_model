import argparse
import json

import simpy

from collectives import HopCandidates, precompute_node
from topology import Topology, build_routing_table, draw_topology, parse_edgelist
from trace_io import load_trace


class LinkManager:
    """All-or-nothing physical link reservation, with the physical path for
    each logical hop resolved dynamically against what's actually free.

    A collective only issues once it has found, for every logical hop it
    needs, some equal-cost candidate path whose links are all free -- and then
    holds the whole resolved set for its duration. This is what lets two
    collectives run concurrently as long as their resolved links don't
    overlap, per refined_plan.md. Implemented as a simpy condition variable:
    `_changed` is replaced with a fresh event every time links are released,
    so waiters re-check from scratch rather than racing over
    individually-acquired simpy.Resources (which could deadlock).

    Links are directed (u, v) pairs, one per direction of a physical link (see
    Topology's docstring) -- so a transfer a->b and a concurrent transfer b->a
    reserve different resources and can proceed at the same time, matching
    collective implementations that drive both directions of a link at once.
    """

    def __init__(self, env: simpy.Environment, topo: Topology):
        self.env = env
        self.available = set(topo.graph.edges())
        self._changed = env.event()

    def acquire(self, hop_candidates: HopCandidates):
        """Resolves each hop to one of its candidate paths and reserves the
        union, atomically, only once every hop has found a free one.

        Walks the hops in order; for each, picks the first candidate whose
        links don't collide with a link already tentatively claimed by an
        earlier hop of this SAME attempt (self-collision within one
        collective -- two of its own hops can legitimately want the same
        physical link) and are still globally free. If any hop comes up
        empty, the whole attempt is discarded -- no partial reservation --
        and retried once links free up elsewhere.
        """
        while True:
            chosen: set[tuple[str, str]] = set()
            for candidates in hop_candidates:
                pick = None
                for path in candidates:
                    if path.links & chosen:
                        continue
                    if path.links <= self.available - chosen:
                        pick = path
                        break
                if pick is None:
                    break
                chosen |= pick.links
            else:
                self.available -= chosen
                return chosen
            yield self._changed

    def release(self, links: set[tuple[str, str]]):
        self.available |= links
        changed, self._changed = self._changed, self.env.event()
        changed.succeed()


def run_node(env, node_id, graph, link_manager: LinkManager, done_events, verbose: bool, track_overlap: bool):
    node = graph.nodes[node_id]
    deps = list(graph.predecessors(node_id))
    if deps:
        yield simpy.AllOf(env, [done_events[dep] for dep in deps])

    links = yield from link_manager.acquire(node["hop_candidates"])
    if track_overlap:
        node["start_time"] = env.now
    if verbose:
        print(f"{env.now:12.9f}  issue    {node_id:>3}  {node['subtype']:<14} {node['name']}")

    yield env.timeout(node["duration"])

    link_manager.release(links)
    if track_overlap:
        node["end_time"] = env.now
    if verbose:
        print(f"{env.now:12.9f}  complete {node_id:>3}  {node['subtype']:<14} {node['name']}")
    done_events[node_id].succeed()


def classify_overlap(graph) -> dict[str, float]:
    """Breaks the simulated timeline into exposed-compute, exposed-comm,
    overlapped (both active), and idle (neither active) time, via a sweep
    over COMP/COMM node start/end events. Requires simulate(...,
    track_overlap=True) to have populated start_time/end_time on every node
    -- SYNC/DUMMY nodes are zero-duration and excluded, they never affect
    this either way.
    """
    events = []
    for _, node in graph.nodes(data=True):
        if node["type"] not in ("COMP", "COMM"):
            continue
        events.append((node["start_time"], 1, node["type"]))
        events.append((node["end_time"], -1, node["type"]))
    if not events:
        return {"exposed_comp": 0.0, "exposed_comm": 0.0, "overlapped": 0.0, "idle": 0.0}
    events.sort(key=lambda e: (e[0], e[1]))  # ends (-1) before starts (+1) at equal timestamps

    comp_active = comm_active = 0
    last_t = events[0][0]
    exposed_comp = exposed_comm = overlapped = idle = 0.0

    for t, delta, kind in events:
        dt = t - last_t
        if comp_active and comm_active:
            overlapped += dt
        elif comp_active:
            exposed_comp += dt
        elif comm_active:
            exposed_comm += dt
        else:
            idle += dt
        if kind == "COMP":
            comp_active += delta
        else:
            comm_active += delta
        last_t = t

    return {"exposed_comp": exposed_comp, "exposed_comm": exposed_comm, "overlapped": overlapped, "idle": idle}


def write_perfetto_trace(graph, filepath: str) -> None:
    """Writes a Chrome Trace Event Format JSON, openable at ui.perfetto.dev
    or chrome://tracing. Requires simulate(..., track_overlap=True) to have
    populated start_time/end_time on every node.

    Two tracks (tid), matching classify_overlap's own categories: COMP and
    COMM. This isn't a full per-NPU Gantt chart -- COMP nodes don't carry a
    rank/NPU id the way COMM nodes' comm_group does (compute isn't replicated
    per physical rank in the trace, see trace_generator.py), so there's no
    per-rank track to assign them to yet. SYNC/DUMMY nodes are zero-duration
    and omitted -- they never contribute to "where does the time go," which
    is the point of this view.
    """
    track_tids = {"COMP": 0, "COMM": 1}
    events = [{"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "analytical_model"}}]
    for node_type, tid in track_tids.items():
        events.append({"name": "thread_name", "ph": "M", "pid": 0, "tid": tid, "args": {"name": node_type}})

    for node_id, node in graph.nodes(data=True):
        if node["type"] not in track_tids:
            continue
        events.append({
            "name": node["name"],
            "cat": node["subtype"],
            "ph": "X",
            "ts": node["start_time"] * 1e6,  # sec -> usec, Chrome trace format's unit
            "dur": (node["end_time"] - node["start_time"]) * 1e6,
            "pid": 0,
            "tid": track_tids[node["type"]],
            "args": {
                "id": node_id,
                "comm_group": node.get("comm_group", []),
                "size": node.get("size", 0),
            },
        })

    with open(filepath, "w") as f:
        json.dump({"traceEvents": events}, f)


def simulate(
    topo: Topology,
    graph,
    peak_perf_tflops: float,
    local_mem_bw_gbps: float,
    verbose: bool = True,
    track_overlap: bool = False,
) -> float:
    routing_table = build_routing_table(topo)
    for node_id, node in graph.nodes(data=True):
        node["hop_candidates"], node["duration"] = precompute_node(
            node, routing_table, peak_perf_tflops, local_mem_bw_gbps
        )

    env = simpy.Environment()
    link_manager = LinkManager(env, topo)
    done_events = {node_id: env.event() for node_id in graph.nodes}

    for node_id in graph.nodes:
        env.process(run_node(env, node_id, graph, link_manager, done_events, verbose, track_overlap))

    env.run()
    return env.now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the analytical model simulation.")
    parser.add_argument("--topology", type=str, required=True, help="Path to the topology edgelist file.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace.json file.")
    parser.add_argument("--peak-perf", type=float, required=True, help="Peak compute performance in TFLOPS, for COMP node roofline timing.")
    parser.add_argument("--local-mem-bw", type=float, required=True, help="Local memory bandwidth in GB/s, for COMP node roofline timing.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-node issue/complete logging.")
    parser.add_argument("--overlap", action="store_true", help="Track and report exposed-compute/exposed-comm/overlapped/idle time breakdown.")
    parser.add_argument("--perfetto-trace", type=str, default=None, help="Write a Perfetto/Chrome-trace-format JSON to this path (open at ui.perfetto.dev).")
    args = parser.parse_args()

    physical_topology = parse_edgelist(args.topology)
    # draw_topology(physical_topology, "topology.png")

    dag = load_trace(args.trace)
    track_overlap = args.overlap or args.perfetto_trace is not None
    finish_time = simulate(
        physical_topology, dag, args.peak_perf, args.local_mem_bw, verbose=not args.quiet, track_overlap=track_overlap
    )
    print(f"\nSimulation finished at t={finish_time:.9f}s")

    if args.overlap:
        breakdown = classify_overlap(dag)
        print(f"Exposed compute: {breakdown['exposed_comp']:.9f}s")
        print(f"Exposed comm:    {breakdown['exposed_comm']:.9f}s")
        print(f"Overlapped:      {breakdown['overlapped']:.9f}s")
        print(f"Idle:            {breakdown['idle']:.9f}s")

    if args.perfetto_trace:
        write_perfetto_trace(dag, args.perfetto_trace)
        print(f"Wrote Perfetto trace to {args.perfetto_trace}")

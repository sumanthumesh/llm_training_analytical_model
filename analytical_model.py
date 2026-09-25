import argparse

import simpy

from collectives import precompute_node
from topology import Topology, draw_topology, parse_edgelist
from trace_io import load_trace


class LinkManager:
    """All-or-nothing physical link reservation.

    A collective only issues once every physical link it needs is free, and
    then holds all of them for its whole duration -- this is what lets two
    collectives run concurrently as long as they don't share a link, per
    refined_plan.md. Implemented as a simpy condition variable: `_changed` is
    replaced with a fresh event every time links are released, so waiters
    re-check the full set of links they need rather than racing over
    individually-acquired simpy.Resources (which could deadlock).
    """

    def __init__(self, env: simpy.Environment, topo: Topology):
        self.env = env
        self.available = {frozenset(edge) for edge in topo.graph.edges()}
        self._changed = env.event()

    def acquire(self, links: set[frozenset]):
        while not links <= self.available:
            yield self._changed
        self.available -= links

    def release(self, links: set[frozenset]):
        self.available |= links
        changed, self._changed = self._changed, self.env.event()
        changed.succeed()


def run_node(env, node_id, graph, link_manager: LinkManager, done_events, verbose: bool):
    node = graph.nodes[node_id]
    deps = list(graph.predecessors(node_id))
    if deps:
        yield simpy.AllOf(env, [done_events[dep] for dep in deps])

    yield from link_manager.acquire(node["links"])
    if verbose:
        print(f"{env.now:12.9f}  issue    {node_id:>3}  {node['subtype']:<14} {node['name']}")

    yield env.timeout(node["duration"])

    link_manager.release(node["links"])
    if verbose:
        print(f"{env.now:12.9f}  complete {node_id:>3}  {node['subtype']:<14} {node['name']}")
    done_events[node_id].succeed()


def simulate(topo: Topology, graph, comm_groups, verbose: bool = True) -> float:
    for node_id, node in graph.nodes(data=True):
        node["links"], node["duration"] = precompute_node(topo, node, comm_groups)

    env = simpy.Environment()
    link_manager = LinkManager(env, topo)
    done_events = {node_id: env.event() for node_id in graph.nodes}

    for node_id in graph.nodes:
        env.process(run_node(env, node_id, graph, link_manager, done_events, verbose))

    env.run()
    return env.now


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the analytical model simulation.")
    parser.add_argument("--topology", type=str, required=True, help="Path to the topology edgelist file.")
    parser.add_argument("--trace", type=str, required=True, help="Path to the trace.json file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-node issue/complete logging.")
    args = parser.parse_args()

    physical_topology = parse_edgelist(args.topology)
    draw_topology(physical_topology, "topology.png")

    dag, comm_groups = load_trace(args.trace)
    finish_time = simulate(physical_topology, dag, comm_groups, verbose=not args.quiet)
    print(f"\nSimulation finished at t={finish_time:.9f}s")

    
"""Parse an htsim-style edgelist file into a graph and (optionally) visualize it.

Edgelist format (see astra-sim/scripts/generate_htsim_topology.py):

    # comment lines start with '#'
    Hosts <N>
    Switches <M>
    Links <L>

    Default_speed_Gbps <float>
    Default_latency_ns <float>
    Default_queue_bytes <int>
    Default_switch_latency_ns <float>

    switch s<k> switch_latency_ns <float>          # optional per-switch override
    link h<a> s<b> [speed_Gbps <float>] [latency_ns <float>]
    link s<a> s<b> [speed_Gbps <float>] [latency_ns <float>]

Hosts are named h0..h{N-1} and switches s0..s{M-1}. Both link endpoints can be
a host or a switch, so leaf-spine (s-s) links are supported as well as
host-switch links.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class Topology:
    graph: nx.Graph
    num_hosts: int
    num_switches: int
    defaults: dict = field(default_factory=dict)
    _edge_usage: dict = field(default_factory=dict, repr=False)

    @property
    def hosts(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["type"] == "host"]

    @property
    def switches(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["type"] == "switch"]

    def bfs_path_hops(self, src: str, dst: str) -> list[str]:
        """A shortest path (by hop count) between two nodes, e.g. bfs_path_hops("h0", "h5").

        Narrows ties in three stages:
        1. Hop count (nx.all_shortest_paths).
        2. Total latency -- a node with both an NVSwitch and an IB leaf has
           two hop-count-tied paths of very different cost (see
           generate_htsim_topology.py's comment on why NVSwitches are
           numbered before IB leaves), so a pure hop-count tie must not be
           treated as interchangeable.
        3. Load: among genuinely equal-cost paths (e.g. one per spine in a
           fat tree), picks whichever currently reuses the least-loaded
           edges, so concurrent flows spread across all of them instead of
           piling onto one -- routing for maximum available parallelism
           rather than replaying real ECMP hash collisions. Remaining ties
           (typically only the very first call) fall back to a hash of
           (src, dst) for a stable pick.

        This makes the method stateful: it tallies how much each edge has
        been routed over so far (across the whole topology, not scoped to a
        (src, dst) pair) and updates that tally on every call. Calling it
        again for the same pair after other routing has happened can
        therefore return a different path than an earlier call did. This
        tally is a static load-balancing heuristic over call order, not a
        model of runtime concurrency -- that's what LinkManager is for.
        """
        paths = list(nx.all_shortest_paths(self.graph, src, dst))

        if len(paths) > 1:
            def total_latency_ns(path):
                return sum(self.graph.edges[u, v]["latency_ns"] for u, v in zip(path, path[1:]))

            best_latency = min(total_latency_ns(p) for p in paths)
            paths = [p for p in paths if total_latency_ns(p) == best_latency]

        if len(paths) > 1:
            def load(path):
                usages = [self._edge_usage.get(frozenset((u, v)), 0) for u, v in zip(path, path[1:])]
                return (max(usages), sum(usages))

            best_load = min(load(p) for p in paths)
            paths = [p for p in paths if load(p) == best_load]

        if len(paths) > 1:
            digest = hashlib.sha256(f"{src}->{dst}".encode()).digest()
            path = paths[int.from_bytes(digest, "big") % len(paths)]
        else:
            path = paths[0]

        for u, v in zip(path, path[1:]):
            edge = frozenset((u, v))
            self._edge_usage[edge] = self._edge_usage.get(edge, 0) + 1
        return path

    def bfs_order(self, src: str) -> list[str]:
        """Nodes in BFS order starting from src."""
        return list(nx.bfs_tree(self.graph, src).nodes())


def _parse_kv_pairs(tokens: list[str]) -> dict:
    """Parse trailing 'key value key value ...' tokens into a dict of floats."""
    kv = {}
    for i in range(0, len(tokens) - 1, 2):
        try:
            kv[tokens[i]] = float(tokens[i + 1])
        except ValueError:
            continue
    return kv


def parse_edgelist(path: str) -> Topology:
    graph = nx.Graph()
    defaults = {
        "speed_Gbps": None,
        "latency_ns": None,
        "queue_bytes": None,
        "switch_latency_ns": None,
    }
    num_hosts = 0
    num_switches = 0

    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            keyword = tokens[0]

            if keyword == "Hosts":
                num_hosts = int(tokens[1])
                for h in range(num_hosts):
                    graph.add_node(f"h{h}", type="host")
            elif keyword == "Switches":
                num_switches = int(tokens[1])
                for s in range(num_switches):
                    graph.add_node(f"s{s}", type="switch")
            elif keyword == "Links":
                continue  # link count is implied by the 'link' lines themselves
            elif keyword == "Default_speed_Gbps":
                defaults["speed_Gbps"] = float(tokens[1])
            elif keyword == "Default_latency_ns":
                defaults["latency_ns"] = float(tokens[1])
            elif keyword == "Default_queue_bytes":
                defaults["queue_bytes"] = float(tokens[1])
            elif keyword == "Default_switch_latency_ns":
                defaults["switch_latency_ns"] = float(tokens[1])
            elif keyword == "switch":
                node = tokens[1]
                overrides = _parse_kv_pairs(tokens[2:])
                if node not in graph:
                    graph.add_node(node, type="switch")
                graph.nodes[node].update(overrides)
            elif keyword == "link":
                a, b = tokens[1], tokens[2]
                overrides = _parse_kv_pairs(tokens[3:])
                speed_gbps = overrides.get("speed_Gbps", defaults["speed_Gbps"])
                latency_ns = overrides.get("latency_ns", defaults["latency_ns"])
                for node in (a, b):
                    if node not in graph:
                        node_type = "host" if node.startswith("h") else "switch"
                        graph.add_node(node, type=node_type)
                graph.add_edge(a, b, speed_Gbps=speed_gbps, latency_ns=latency_ns)
            else:
                raise ValueError(f"Unrecognized edgelist keyword: {keyword!r} in line: {line!r}")

    return Topology(graph=graph, num_hosts=num_hosts, num_switches=num_switches, defaults=defaults)


def draw_topology(topo: Topology, output_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    graph = topo.graph
    pos = nx.spring_layout(graph, seed=0, k=1.5 / max(1, len(graph.nodes) ** 0.5))

    plt.figure(figsize=(max(6, len(topo.hosts) * 0.8), 6))
    nx.draw_networkx_nodes(graph, pos, nodelist=topo.hosts, node_color="#6fa8dc", node_shape="o", label="Hosts")
    nx.draw_networkx_nodes(graph, pos, nodelist=topo.switches, node_color="#e06666", node_shape="s", label="Switches")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, font_size=8)

    edge_labels = {
        (u, v): f"{d['speed_Gbps']:g}G/{d['latency_ns']:g}ns"
        for u, v, d in graph.edges(data=True)
        if d.get("speed_Gbps") is not None
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved topology plot to {output_path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse and visualize an htsim edgelist topology.")
    parser.add_argument("--edgelist", type=str, required=True, help="Path to the edgelist file.")
    parser.add_argument("--plot", type=str, default=None, help="Path to save a PNG visualization.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    topo = parse_edgelist(args.edgelist)
    print(f"Hosts ({topo.num_hosts}): {topo.hosts}")
    print(f"Switches ({topo.num_switches}): {topo.switches}")
    print(f"Edges ({topo.graph.number_of_edges()}):")
    for u, v, d in topo.graph.edges(data=True):
        print(f"  {u} -- {v}  {d}")

    if args.plot or args.show:
        draw_topology(topo, output_path=args.plot, show=args.show)


if __name__ == "__main__":
    main()

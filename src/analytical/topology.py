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
import logging
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger("topology")


@dataclass
class Topology:
    """graph is a DiGraph with two directed edges (a->b and b->a) per edgelist
    'link' line, each carrying its own speed_Gbps/latency_ns (identical to each
    other, since the file only specifies one value per line). This lets
    LinkManager reserve each direction independently -- some collective
    implementations drive both directions of a link concurrently, which a
    single undirected edge can't represent (it would force one direction to
    wait for the other to release).
    """

    graph: nx.DiGraph
    num_hosts: int
    num_switches: int
    defaults: dict = field(default_factory=dict)
    #Outermost-to-innermost axis nesting this topology's host numbering was
    #generated with (see coords/__init__.py and gen_dgx_topology.py /
    #gen_opt_topology.py's --axis-order), or None for an edgelist that
    #predates this field. analytical_model.py cross-checks this against the
    #trace's own axis_order and refuses to simulate a mismatched pair.
    axis_order: tuple[str, ...] | None = None

    @property
    def hosts(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["type"] == "host"]

    @property
    def switches(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["type"] == "switch"]

    def bfs_order(self, src: str) -> list[str]:
        """Nodes in BFS order starting from src."""
        return list(nx.bfs_tree(self.graph, src).nodes())


@dataclass(frozen=True)
class PathInfo:
    """One candidate physical path for a (src, dst) host pair."""

    links: frozenset[tuple[str, str]]
    latency_sec: float
    bandwidth_gbps: float

    def __str__(self) -> str:
        return ",".join(f"({l[0]},{l[1]})" for l in self.links)


def reverse_path(graph: nx.DiGraph, path: PathInfo) -> PathInfo | None:
    """Returns path's exact mirror -- same physical hops, opposite direction
    -- if every one of them has a real reverse edge in the graph, else None.

    This is the check for whether a ring collective can run a second,
    opposite-direction ring concurrently with the first: that's only possible
    if every link the (forward) path crosses is "bi", not "uni" (see
    topology.py's module docstring and gen_opt_topology.py's direction tag --
    e.g. a TP ring within one interposer is all-bi and mirrors cleanly, while
    a DP/CP/PP ring's outer-outer hop is uni and doesn't). Checked against the
    raw graph rather than routing_table[dst, src], since a uni-only ring hop
    may still have *some* route back to src -- the long way around the ring
    -- which is a real path but not this path's mirror, and using it here
    would wrongly charge the collective for a much more expensive "second
    ring" instead of correctly reporting that one doesn't exist.
    """
    reversed_links = []
    for u, v in path.links:
        if not graph.has_edge(v, u):
            return None
        reversed_links.append((v, u))
    reversed_links = frozenset(reversed_links)
    latency_sec = sum(graph.edges[v, u]["latency_ns"] for v, u in reversed_links) * 1e-9
    bandwidth_gbps = min(graph.edges[v, u]["speed_Gbps"] for v, u in reversed_links)
    return PathInfo(reversed_links, latency_sec, bandwidth_gbps)


def _path_info(graph: nx.DiGraph, path: list[str]) -> PathInfo:
    hops = list(zip(path, path[1:]))
    links = frozenset(hops)
    latency_sec = sum(graph.edges[u, v]["latency_ns"] for u, v in hops) * 1e-9
    bandwidth_gbps = min(graph.edges[u, v]["speed_Gbps"] for u, v in hops)
    return PathInfo(links, latency_sec, bandwidth_gbps)


def _tie_break(candidates: list[PathInfo]) -> list[PathInfo]:
    """Narrows to the lexicographically-best (total latency, bottleneck
    bandwidth) among candidates -- hop count isn't narrowed here since every
    candidate passed in is already guaranteed shortest-path (same hop count)
    by construction. Latency breaks ties first, bandwidth breaks remaining
    ties (e.g. between same-hop-count paths through different spine
    switches with different link speeds). Every PathInfo surviving this is
    therefore genuinely interchangeable -- same latency, same bandwidth --
    so a caller can use whichever one is actually free at runtime without it
    changing the collective's duration.
    """
    if not candidates:
        return candidates
    best_latency = min(c.latency_sec for c in candidates)
    candidates = [c for c in candidates if c.latency_sec == best_latency]
    best_bandwidth = max(c.bandwidth_gbps for c in candidates)
    candidates = [c for c in candidates if c.bandwidth_gbps == best_bandwidth]
    return candidates


def _shortest_paths_between(graph: nx.DiGraph, dist_from_src: dict[str, int], src: str, dst: str) -> list[list[str]]:
    """Every shortest src->dst path, given src's precomputed single-source
    distance map. Restricted to dst's ancestor set (nodes lying on SOME
    shortest src->dst path) rather than every node reachable from src -- the
    cost of this reconstruction therefore scales with how far dst actually
    is from src, not with the size of the whole graph.
    """
    if dst not in dist_from_src or dist_from_src[dst] == 0:
        return []
    target_dist = dist_from_src[dst]

    # Backward pass: the ancestor set.
    ancestors = {dst}
    frontier = {dst}
    for d in range(target_dist, 0, -1):
        frontier = {u for node in frontier for u in graph.predecessors(node) if dist_from_src.get(u) == d - 1}
        ancestors |= frontier

    # Forward pass: build path lists only within the ancestor set, in
    # increasing distance order -- a node's paths are its shortest-path-DAG
    # predecessors' paths, each extended by one hop.
    paths_from_src: dict[str, list[list[str]]] = {src: [[src]]}
    for node in sorted(ancestors, key=lambda n: dist_from_src[n]):
        if node == src:
            continue
        preds = [u for u in graph.predecessors(node) if dist_from_src.get(u) == dist_from_src[node] - 1 and u in ancestors]
        paths_from_src[node] = [p + [node] for u in preds for p in paths_from_src.get(u, [])]

    return paths_from_src.get(dst, [])


class LazyRoutingTable:
    """Resolves (src, dst) -> equal-cost PathInfo candidates on first query,
    memoizing the result -- unlike an eagerly-built all-pairs table, which
    precomputes and stores every one of hosts*(hosts-1) possible pairs up
    front regardless of whether a real run ever asks for them.

    It almost never does: a trace only ever queries, for a given host, the
    handful of ring-neighbor/send-recv partners its collectives actually use
    (at most one per parallelism axis -- dp/cp/pp/tp -- so at most ~4
    distinct destinations per source across an entire run), and the exact
    same small set of pairs recurs every layer and microbatch. Measured on a
    512-host ring topology: a real trace queried only 1,024 of the 261,632
    possible host pairs: eager precompute of all of them took 25.8s and
    2.4GB; resolving just the 1,024 actually needed, lazily, with no further
    sharing across different destinations from the same source (see the
    decision this was benchmarked against for the discarded alternative --
    caching each source's BFS distance map too, which traded ~40% more peak
    memory for roughly half as many BFS calls), took 0.37s and 62MB.

    Supports the same `table[src, dst]` subscript interface the old
    eagerly-built dict did, so it's a drop-in replacement everywhere a
    routing table is read from.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._cache: dict[tuple[str, str], list[PathInfo]] = {}

    def __getitem__(self, key: tuple[str, str]) -> list[PathInfo]:
        if key not in self._cache:
            src, dst = key
            dist_from_src = nx.single_source_shortest_path_length(self.graph, src)
            paths = _shortest_paths_between(self.graph, dist_from_src, src, dst)
            self._cache[key] = _tie_break([_path_info(self.graph, p) for p in paths])
        return self._cache[key]


def _parse_kv_pairs(tokens: list[str]) -> dict:
    """Parse trailing 'key value key value ...' tokens into a dict, coercing
    each value to float when it looks numeric (the common case -- speed,
    latency, axis_id, ...) and keeping it as the original string otherwise
    (e.g. gen_opt_topology.py's "direction bi"/"direction uni" tag) rather
    than silently dropping it -- a non-numeric value used to vanish from the
    result entirely with no error, which is exactly how "direction" went
    unnoticed before parse_edgelist was taught to act on it.
    """
    kv = {}
    for i in range(0, len(tokens) - 1, 2):
        value = tokens[i + 1]
        try:
            kv[tokens[i]] = float(value)
        except ValueError:
            kv[tokens[i]] = value
    return kv


def parse_edgelist(path: str) -> Topology:
    graph = nx.DiGraph()
    defaults = {
        "speed_Gbps": None,
        "latency_ns": None,
        "queue_bytes": None,
        "switch_latency_ns": None,
    }
    num_hosts = 0
    num_switches = 0
    axis_order = None

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
            elif keyword == "Axis_order":
                axis_order = tuple(tokens[1].split(","))
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
                # Defaults to "bi" for edgelists that don't specify it at all
                # (every generator before gen_opt_topology.py's direction
                # tag), preserving today's always-add-the-reverse behavior
                # for those.
                direction = overrides.get("direction", "bi")
                if direction not in ("bi", "uni"):
                    raise ValueError(f"Unrecognized link direction {direction!r} in line: {line!r}")
                # Anything beyond speed_Gbps/latency_ns/direction (e.g.
                # gen_opt_topology.py's axis_id tag) is still a real per-link
                # attribute -- pass it through to the edge instead of
                # silently dropping it.
                extra = {k: v for k, v in overrides.items() if k not in ("speed_Gbps", "latency_ns", "direction")}
                for node in (a, b):
                    if node not in graph:
                        node_type = "host" if node.startswith("h") else "switch"
                        graph.add_node(node, type=node_type)
                # Two directed edges, not one undirected edge, for a "bi"
                # link: each direction of a physical link is an
                # independently reservable resource (see Topology's
                # docstring). A "uni" link only ever gets the a->b edge --
                # the reverse direction genuinely doesn't exist (e.g.
                # gen_opt_topology.py's unidirectional ring links, see the
                # bidirectional-ring spoke-contention discussion this
                # convention exists to resolve).
                graph.add_edge(a, b, speed_Gbps=speed_gbps, latency_ns=latency_ns, direction=direction, **extra)
                if direction == "bi":
                    graph.add_edge(b, a, speed_Gbps=speed_gbps, latency_ns=latency_ns, direction=direction, **extra)
            else:
                raise ValueError(f"Unrecognized edgelist keyword: {keyword!r} in line: {line!r}")

    return Topology(graph=graph, num_hosts=num_hosts, num_switches=num_switches, defaults=defaults, axis_order=axis_order)


def draw_topology(topo: Topology, output_path: str | None = None, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    # topo.graph carries two directed edges (a->b, b->a) per physical link, with
    # identical attrs on each -- collapse to one undirected edge per link purely
    # for drawing, so the plot doesn't show every link twice.
    graph = topo.graph.to_undirected()
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

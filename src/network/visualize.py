"""Visualize an htsim-style edgelist topology (see topology.py's module
docstring for the file format), laying it out according to whichever
structural pattern it actually has, detected purely from switch connectivity
-- not from filenames or comments, so it works on any edgelist matching one
of these shapes regardless of which generator produced it:

- Two-tier (gen_dgx_topology.py's 'dgx'/'glass' types): one hub switch (the
  spine) reaches across every domain -- either directly to every host
  ('dgx') or via one link per leaf switch ('glass'). Detected as the switch
  whose degree dwarfs every other switch's. Drawn as a star: hub centered,
  leaf switches on a circle around it, each leaf's hosts fanned out just
  beyond it.

- Ring (gen_opt_topology.py): no dominating hub. Instead switches connect
  directly to a handful of other switches, forming disjoint rings -- one
  ring group per parallelism axis (PP/CP/DP), distinguished by each axis
  getting its own distinct link bandwidth. Drawn as concentric circles, one
  radius per axis, with short spokes tying together the (up to) three
  positions belonging to the same physical switch, and each switch's hosts
  fanned out beyond its outermost ring position.

- Flat: no switch-switch structure at all (e.g. a single switch) -- drawn as
  a plain star.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

from analytical.topology import Topology, parse_edgelist

_AXIS_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]


def _switch_graph(topo: Topology) -> nx.Graph:
    """Undirected simple graph over switches only (collapses the two
    directed hops of each physical link into one edge, same attrs either
    way since the edgelist only specifies one value per link line)."""
    g = nx.Graph()
    g.add_nodes_from(topo.switches)
    switches = set(topo.switches)
    for u, v, d in topo.graph.edges(data=True):
        if u in switches and v in switches:
            g.add_edge(u, v, **d)
    return g


def _hub_switch(topo: Topology, switch_graph: nx.Graph) -> str | None:
    """The spine switch of a two-tier topology, if this looks like one.
    Covers both two-tier variants, whose spines have opposite signatures:
    - 'glass': spine connects directly to (nearly) every other switch.
    - 'dgx': spine instead bypasses the leaf tier and connects directly to
      every host, so its HOST degree dwarfs every other switch's (which only
      reach their own domain's hosts), while its switch-switch degree is 0.
    """
    switches = topo.switches
    if len(switches) < 2:
        return None

    for s in switches:
        if switch_graph.degree(s) >= len(switches) - 1 and switch_graph.degree(s) > 1:
            return s

    hosts = set(topo.hosts)
    host_degree = {s: sum(1 for n in topo.graph.neighbors(s) if n in hosts) for s in switches}
    ordered = sorted(host_degree.values(), reverse=True)
    if len(ordered) >= 2 and ordered[1] > 0 and ordered[0] >= 2 * ordered[1]:
        return max(host_degree, key=host_degree.get)
    return None


def _ring_order(switch_graph: nx.Graph, nodes: set[str]) -> list[str]:
    """Cyclic visiting order for one connected ring component. The subgraph
    induced by `nodes` should be a simple cycle (or a trivial 1-2 node ring)
    by construction, so a plain walk-the-neighbors traversal recovers the
    ring order; falls back to arbitrary order if that assumption ever breaks
    (e.g. a malformed edgelist) rather than raising.
    """
    nodes = list(nodes)
    if len(nodes) <= 2:
        return nodes
    sub = switch_graph.subgraph(nodes)
    order = [nodes[0]]
    visited = {nodes[0]}
    while len(order) < len(nodes):
        current = order[-1]
        next_node = next((n for n in sub.neighbors(current) if n not in visited), None)
        if next_node is None:
            # Not a clean cycle -- bail out with whatever's left, unordered.
            order.extend(n for n in nodes if n not in visited)
            break
        order.append(next_node)
        visited.add(next_node)
    return order


_AXIS_ID_NAMES = {0: "pp", 1: "cp", 2: "dp"}


def _axis_groups(switch_graph: nx.Graph) -> list[tuple[str | None, list[list[str]]]]:
    """Groups switch-switch edges into one group per parallelism axis,
    returned as (axis_name_or_None, ring_components) pairs -- each
    ring_components entry is the list of that axis's disjoint ring
    components (each a cyclic order of switch names), e.g. a PP ring of
    size 4 shared by dp=2 x cp=3 instances yields 6 components of length 4.
    Axes are returned largest-ring-first (fewest, biggest loops first).

    Prefers gen_opt_topology.py's explicit axis_id link tag (0=pp, 1=cp,
    2=dp) when every switch-switch edge carries one, falling back to
    grouping by (rounded) bandwidth otherwise. Bandwidth alone is ambiguous
    whenever two axes share a ring bandwidth ratio -- including the default
    equal 1/3-1/3-1/3 split -- so untagged bandwidth grouping is a fallback
    for edgelists from elsewhere, not the primary signal.
    """
    edges = list(switch_graph.edges(data=True))
    tagged = bool(edges) and all("axis_id" in d for _, _, d in edges)

    by_key: dict[float, list[tuple[str, str]]] = defaultdict(list)
    names: dict[float, str | None] = {}
    for u, v, d in edges:
        if tagged:
            key = int(round(d["axis_id"]))
            names[key] = _AXIS_ID_NAMES.get(key, f"axis {key}")
        else:
            key = round(d.get("speed_Gbps", 0.0), 6)
            names[key] = None
        by_key[key].append((u, v))

    axis_groups = []
    for key, es in by_key.items():
        sub = nx.Graph()
        sub.add_edges_from(es)
        components = [_ring_order(sub, comp) for comp in nx.connected_components(sub)]
        axis_groups.append((names[key], components))

    axis_groups.sort(key=lambda item: -len(item[1][0]) if item[1] else 0)
    return axis_groups


def _fan_positions(center: tuple[float, float], anchor_angle: float, count: int, radius: float, spread: float) -> list[tuple[float, float]]:
    """`count` points on a small arc of `radius` around `center`, centered on
    `anchor_angle` and spanning `spread` radians -- used to cluster a
    switch's hosts just outside its position without overlapping neighbors.
    """
    cx, cy = center
    if count == 1:
        angles = [anchor_angle]
    else:
        angles = [anchor_angle - spread / 2 + spread * i / (count - 1) for i in range(count)]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _draw_hosts(ax, topo: Topology, anchor_pos: dict[str, tuple[float, float]], anchor_angle: dict[str, float], host_radius: float) -> dict[str, tuple[float, float]]:
    hosts_by_switch: dict[str, list[str]] = defaultdict(list)
    switches = set(topo.switches)
    for host in topo.hosts:
        switch_neighbors = [n for n in topo.graph.neighbors(host) if n in switches and n in anchor_pos]
        if not switch_neighbors:
            continue
        # A host with multiple switch neighbors (dgx's direct spine-to-host
        # bypass) anchors to whichever neighbor is its own local leaf switch
        # rather than the busy hub, i.e. the lowest-degree neighbor.
        home = min(switch_neighbors, key=lambda s: topo.graph.degree(s))
        hosts_by_switch[home].append(host)

    host_positions: dict[str, tuple[float, float]] = {}
    for switch, hosts in hosts_by_switch.items():
        positions = _fan_positions(anchor_pos[switch], anchor_angle[switch], len(hosts), host_radius, spread=0.9)
        for host, pos in zip(hosts, positions):
            host_positions[host] = pos
            ax.plot(*pos, "o", color="#6fa8dc", markersize=5, zorder=3)
            ax.plot([anchor_pos[switch][0], pos[0]], [anchor_pos[switch][1], pos[1]], color="#999999", linewidth=0.5, zorder=1)
            ax.annotate(host, pos, fontsize=5, ha="center", va="center", xytext=(0, 6), textcoords="offset points")
    return host_positions


def _draw_two_tier(ax, topo: Topology, switch_graph: nx.Graph, hub: str) -> None:
    leaves = [s for s in topo.switches if s != hub]
    ax.plot(0, 0, "s", color="#e06666", markersize=14, zorder=4)
    ax.annotate(hub, (0, 0), fontsize=7, ha="center", va="center", xytext=(0, -14), textcoords="offset points")

    leaf_radius = 3.0
    pos = {hub: (0.0, 0.0)}
    angle = {hub: 0.0}
    n = max(len(leaves), 1)
    for i, leaf in enumerate(leaves):
        theta = 2 * math.pi * i / n
        p = (leaf_radius * math.cos(theta), leaf_radius * math.sin(theta))
        pos[leaf] = p
        angle[leaf] = theta
        ax.plot(*p, "s", color="#e69138", markersize=10, zorder=4)
        ax.annotate(leaf, p, fontsize=6, ha="center", va="center", xytext=(0, -10), textcoords="offset points")
        if switch_graph.has_edge(hub, leaf):
            ax.plot([0, p[0]], [0, p[1]], color="#333333", linewidth=1.2, zorder=2)

    host_positions = _draw_hosts(ax, topo, pos, angle, host_radius=1.4)

    # dgx's bypass links (host -> hub directly, skipping the leaf switch)
    # drawn faint so the "two tiers, but hosts also reach the spine
    # directly" shape reads clearly without overwhelming the leaf fans.
    for host, hp in host_positions.items():
        if topo.graph.has_edge(host, hub):
            ax.plot([hp[0], 0], [hp[1], 0], color="#cccccc", linewidth=0.4, linestyle="--", zorder=0)
    ax.set_title(f"Two-tier topology (hub: {hub}, {len(leaves)} leaf switch(es), {len(topo.hosts)} hosts)")


def _draw_ring(ax, topo: Topology, switch_graph: nx.Graph) -> None:
    """Every switch sits at exactly one position, placed around a single
    shared circle by the axis with the biggest ring (fewest, largest
    loops) -- so that axis reads as an actual circle. The other axes'
    ring-edges are drawn as colored chords between those same positions:
    not geometrically circular themselves, but every line drawn is a real
    switch_graph edge, so the picture never implies adjacency that isn't
    there (see the two-tier star mis-render this replaced).
    """
    axis_groups = _axis_groups(switch_graph)
    num_axes = len(axis_groups)
    radius = 4.0

    # Primary axis (axis_groups[0], already sorted largest-ring-first)
    # concatenates its components' cyclic orders around the one shared
    # circle, with a gap between components so separate ring instances of
    # that axis stay visually distinct.
    _, primary = axis_groups[0]
    num_components = len(primary)
    gap = 0.15 * (2 * math.pi / max(num_components, 1))
    arc_per_component = (2 * math.pi - gap * num_components) / max(num_components, 1)
    pos: dict[str, tuple[float, float]] = {}
    angle: dict[str, float] = {}
    start = 0.0
    for ring in primary:
        L = len(ring)
        for i, switch in enumerate(ring):
            theta = start + (arc_per_component * i / L if L > 1 else arc_per_component / 2)
            pos[switch] = (radius * math.cos(theta), radius * math.sin(theta))
            angle[switch] = theta
        start += arc_per_component + gap

    for switch in topo.switches:
        if switch not in pos:
            # Shouldn't happen for a regular ring topology (every switch
            # belongs to every active axis), but keep isolated switches
            # visible rather than silently dropping them.
            pos[switch] = (0.0, 0.0)
            angle[switch] = 0.0

    for axis_idx, (name, components) in enumerate(axis_groups):
        color = _AXIS_COLORS[axis_idx % len(_AXIS_COLORS)]
        for ring in components:
            L = len(ring)
            for i in range(L):
                a, b = ring[i], ring[(i + 1) % L]
                if L > 1 and switch_graph.has_edge(a, b):
                    pa, pb = pos[a], pos[b]
                    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, linewidth=1.3, zorder=2)
        example_bw = next(iter(switch_graph.edges(components[0], data=True)))[2]["speed_Gbps"] if components and components[0] else None
        axis_label = name if name is not None else f"ring axis {axis_idx}"
        size_bw = f" (size={len(components[0])}, {example_bw:g} Gbps)" if components and example_bw is not None else ""
        ax.plot([], [], color=color, linewidth=1.3, label=f"{axis_label}{size_bw}")

    for switch, p in pos.items():
        ax.plot(*p, "s", color="#e69138", markersize=7, zorder=4)
        ax.annotate(switch, p, fontsize=5, ha="center", va="center", xytext=(0, 7), textcoords="offset points")

    _draw_hosts(ax, topo, pos, angle, host_radius=1.4)
    if num_axes:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"Ring topology ({num_axes} axis/axes, {len(topo.switches)} interposer(s), {len(topo.hosts)} hosts)")


def _draw_flat(ax, topo: Topology) -> None:
    switches = topo.switches
    pos = {s: (3.0 * math.cos(2 * math.pi * i / max(len(switches), 1)), 3.0 * math.sin(2 * math.pi * i / max(len(switches), 1))) for i, s in enumerate(switches)}
    angle = {s: 2 * math.pi * i / max(len(switches), 1) for i, s in enumerate(switches)}
    for s, p in pos.items():
        ax.plot(*p, "s", color="#e69138", markersize=10, zorder=4)
        ax.annotate(s, p, fontsize=6, ha="center", va="center", xytext=(0, -10), textcoords="offset points")
    _draw_hosts(ax, topo, pos, angle, host_radius=1.4)
    ax.set_title(f"Flat topology ({len(switches)} switch(es), {len(topo.hosts)} hosts)")


def visualize_topology(topo: Topology, output_path: str | None = None, show: bool = False) -> None:
    switch_graph = _switch_graph(topo)
    hub = _hub_switch(topo, switch_graph)

    fig, ax = plt.subplots(figsize=(9, 9))
    if hub is not None:
        _draw_two_tier(ax, topo, switch_graph, hub)
    elif switch_graph.number_of_edges() > 0:
        _draw_ring(ax, topo, switch_graph)
    else:
        _draw_flat(ax, topo)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved topology plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_edgelist_file(edgelist_path: str, output_path: str | None = None, show: bool = False) -> None:
    """Convenience wrapper for callers that only have an edgelist path on
    disk (e.g. the gen_*_topology.py scripts' --visualize option) rather
    than an already-parsed Topology."""
    visualize_topology(parse_edgelist(edgelist_path), output_path=output_path, show=show)


def main():
    parser = argparse.ArgumentParser(description="Visualize an htsim edgelist topology, pattern-aware (two-tier or ring).")
    parser.add_argument("--edgelist", type=str, required=True, help="Path to the edgelist file.")
    parser.add_argument("--plot", type=str, default=None, help="Path to save a PNG visualization.")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively.")
    args = parser.parse_args()

    topo = parse_edgelist(args.edgelist)
    visualize_topology(topo, output_path=args.plot, show=args.show)


if __name__ == "__main__":
    main()

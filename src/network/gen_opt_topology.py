"""Generate an htsim-style edgelist for a simple 2-tier topology: A NPUs per
inner domain (e.g. NVLink), B such domains connected through a single spine
switch (e.g. scale-out/IB fabric). See topology.py's module docstring for the
edgelist format this produces.

Host h<i> belongs to domain i // A -- pick A to match whatever grouping (e.g.
your trace's tensor-parallel degree) should map to a single NVLink domain.
"""

from __future__ import annotations

import argparse
import os
from typing import List

from network.visualize import visualize_edgelist_file


def generate_dgx_edgelist(
    npus_per_domain: int,
    num_domains: int,
    inner_bandwidth_gbps: float,
    inner_latency_ns: float,
    outer_bandwidth_gbps: float,
    outer_latency_ns: float,
    queue_bytes: int = 65536,
    inner_switch_latency_ns: float = 0,
    outer_switch_latency_ns: float = 0,
) -> str:
    num_hosts = npus_per_domain * num_domains
    multi_domain = num_domains > 1
    spine = f"s{num_domains}"
    num_switches = num_domains + (1 if multi_domain else 0)
    num_links = num_hosts + (num_hosts if multi_domain else 0)

    lines = [
        f"# {npus_per_domain} NPUs/domain x {num_domains} domain(s)",
        f"Hosts {num_hosts}",
        f"Switches {num_switches}",
        f"Links {num_links}",
        "",
        f"Default_speed_Gbps {inner_bandwidth_gbps}",
        f"Default_latency_ns {inner_latency_ns}",
        f"Default_queue_bytes {queue_bytes}",
        # Default_switch_latency_ns is parsed by topology.py but never applied
        # to a switch that lacks an explicit override -- it's write-only right
        # now -- so every switch below gets its own explicit 'switch' line
        # instead of relying on this default. Separately: collectives.py's
        # route_hop only sums per-link latency_ns today, so switch_latency_ns
        # doesn't affect simulated duration yet either way, until that's wired
        # into the cost model.
        f"Default_switch_latency_ns {inner_switch_latency_ns}",
        "",
    ]
    for domain in range(num_domains):
        lines.append(f"switch s{domain} switch_latency_ns {inner_switch_latency_ns}")
    if multi_domain:
        lines.append(f"switch {spine} switch_latency_ns {outer_switch_latency_ns}")
    lines.append("")

    for domain in range(num_domains):
        leaf = f"s{domain}"
        for local_idx in range(npus_per_domain):
            host = domain * npus_per_domain + local_idx
            lines.append(f"link h{host} {leaf}")
            if multi_domain:
                lines.append(
                    f"link h{host} {spine} speed_Gbps {outer_bandwidth_gbps} latency_ns {outer_latency_ns}"
                )

    return "\n".join(lines) + "\n"

def generate_opt_edgelist(
    npus_per_domain: int,
    dp: int,
    pp: int,
    cp: int,
    inner_bandwidth_gbps: float,
    inner_latency_ns: float,
    outer_bandwidth_gbps: float,
    outer_latency_ns: float,
    ring_bandwidth_ratios: List[float],
    queue_bytes: int = 65536,
    switch_latency_ns: float = 0,
) -> str:
    #Architecture description
    #There are N npus per interposer
    #Each interposer has a local OCS that connects the N npus
    #Each OCS also connects to other interposer's OCS
    #There is no spine switch that connects all the interposers together, instead they are directly connected
    #We normally have TP,PP,CP,DP parallelism axes
    #We are assuimng TP is within the interposer
    #So we will create a ring between interposers for PP,CP and DP.
    #There will be three links coming out of each interposer. One will connect to the next interposer in the PP ring, one will connect to the next interposer in the CP ring and one will connect to the next interposer in the DP ring
    #The bandwidth allocated to each of rings will be based on the ring_bandwidth_ratios parameter. The sum of the ratios should be 1.0 and correspond to outer_bandwidth_gbps. For example if the ratios are [0.5,0.3,0.2] then the PP ring will get 50% of the outer_bandwidth_gbps, CP ring will get 30% and DP ring will get 20%.
    assert len(ring_bandwidth_ratios) == 3, f"ring_bandwidth_ratios must be [pp_ratio, cp_ratio, dp_ratio], got {ring_bandwidth_ratios}"
    assert abs(sum(ring_bandwidth_ratios) - 1.0) < 1e-9, f"ring_bandwidth_ratios must sum to 1.0, got {ring_bandwidth_ratios}"
    pp_ratio, cp_ratio, dp_ratio = ring_bandwidth_ratios

    num_domains = dp * pp * cp
    num_hosts = npus_per_domain * num_domains

    def domain_coords(domain: int) -> tuple[int, int, int]:
        # Mirrors trace_generator.py's node_id_from_axes nesting (d outermost,
        # then p, then c, t innermost/within-domain): interposer `domain`'s
        # hosts are h[domain*npus_per_domain : (domain+1)*npus_per_domain],
        # same contiguous-TP-block convention as generate_dgx_edgelist.
        d = domain // (pp * cp)
        p = (domain % (pp * cp)) // cp
        c = domain % cp
        return d, p, c

    def domain_id(d: int, p: int, c: int) -> int:
        return d * pp * cp + p * cp + c

    # One ring link per interposer per non-degenerate axis -- an axis of size
    # 1 has no "next" interposer to connect to, so it's skipped entirely (its
    # slice of outer_bandwidth_gbps simply goes unused, same as a degree-1
    # collective becoming a no-op DUMMY node in trace_generator.py).
    # axis_id is a fixed code (0=pp, 1=cp, 2=dp) tagged onto each ring link
    # below, independent of ratio/bandwidth -- visualize.py uses it to tell
    # the rings apart. Grouping by speed_Gbps alone breaks whenever two axes
    # end up with equal bandwidth, which happens for any equal ring ratio
    # (e.g. the default [1/3, 1/3, 1/3], or ratios that just happen to be
    # equal like [3, 3, 3] normalized).
    ring_axes = [("pp", pp, pp_ratio, 0), ("cp", cp, cp_ratio, 1), ("dp", dp, dp_ratio, 2)]
    active_ring_axes = [(name, size, ratio, axis_id) for name, size, ratio, axis_id in ring_axes if size > 1]
    num_ring_links = num_domains * len(active_ring_axes)
    num_links = num_hosts + num_ring_links

    lines = [
        f"# {npus_per_domain} NPUs/domain x {num_domains} domain(s) (dp={dp}, pp={pp}, cp={cp}) -- optical topology, interposers ring-connected directly, no spine",
        f"Hosts {num_hosts}",
        f"Switches {num_domains}",
        f"Links {num_links}",
        "",
        f"Default_speed_Gbps {inner_bandwidth_gbps}",
        f"Default_latency_ns {inner_latency_ns}",
        f"Default_queue_bytes {queue_bytes}",
        f"Default_switch_latency_ns {switch_latency_ns}",
        "",
    ]
    for domain in range(num_domains):
        lines.append(f"switch s{domain} switch_latency_ns {switch_latency_ns}")
    lines.append("")

    # Host -> local OCS links: no override, so these inherit the defaults
    # above (inner_bandwidth_gbps/inner_latency_ns) -- same pattern as
    # generate_dgx_edgelist's host-leaf links.
    for domain in range(num_domains):
        leaf = f"s{domain}"
        for local_idx in range(npus_per_domain):
            host = domain * npus_per_domain + local_idx
            lines.append(f"link h{host} {leaf}")
    lines.append("")

    # OCS -> OCS ring links: one ring per non-degenerate PP/CP/DP axis, each
    # getting its own bandwidth slice of outer_bandwidth_gbps (per
    # ring_bandwidth_ratios) over outer_latency_ns links. Emitted once per
    # (domain, axis) as domain -> next_domain -- topology.py's edgelist parser
    # already creates both directions from a single 'link' line, so walking
    # every domain's "next" neighbor closes each ring completely without
    # needing an explicit "previous" link too.
    for domain in range(num_domains):
        d, p, c = domain_coords(domain)
        for name, size, ratio, axis_id in active_ring_axes:
            if name == "pp":
                next_domain = domain_id(d, (p + 1) % size, c)
            elif name == "cp":
                next_domain = domain_id(d, p, (c + 1) % size)
            else:  # dp
                next_domain = domain_id((d + 1) % size, p, c)
            ring_bandwidth_gbps = outer_bandwidth_gbps * ratio
            lines.append(
                f"link s{domain} s{next_domain} speed_Gbps {ring_bandwidth_gbps} latency_ns {outer_latency_ns} axis_id {axis_id}"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate an htsim edgelist for the ring-connected-OCS optical topology."
    )
    parser.add_argument("-a", "--npus-per-domain", type=int, required=True, help="NPUs per interposer (TP degree).")
    parser.add_argument("--dp", type=int, required=True, help="Data-parallel degree (number of DP-ring interposers).")
    parser.add_argument("--pp", type=int, required=True, help="Pipeline-parallel degree (number of PP-ring interposers).")
    parser.add_argument("--cp", type=int, required=True, help="Context-parallel degree (number of CP-ring interposers).")
    parser.add_argument("--inner-bandwidth-gbps", type=float, required=True, help="Bandwidth from a host to its interposer's local OCS.")
    parser.add_argument("--inner-latency-ns", type=float, required=True, help="Latency from a host to its interposer's local OCS.")
    parser.add_argument("--outer-bandwidth-gbps", type=float, required=True, help="Bandwidth budget split across the PP/CP/DP OCS-to-OCS rings, per --ring-ratios.")
    parser.add_argument("--outer-latency-ns", type=float, required=True, help="Latency of the OCS-to-OCS ring links.")
    parser.add_argument("--switch-latency-ns", type=float, default=0, help="Processing latency of each interposer's OCS (default: 0).")
    parser.add_argument("--queue-bytes", type=int, default=65536, help="Default queue size in bytes (default: 65536).")
    parser.add_argument("--ring-ratios", type=float, nargs=3, default=[1/3, 1/3, 1/3], metavar=("PP_RATIO", "CP_RATIO", "DP_RATIO"), help="Ratios of outer bandwidth to allocate to the PP, CP, DP rings. Must sum to 1.0.")
    parser.add_argument("-o", "--output", type=str, default="topology.edgelist", help="Output edgelist path.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Also render an .svg visualization next to --output (same path, .svg extension).")
    args = parser.parse_args()

    normalized_ratios = [r / sum(args.ring_ratios) for r in args.ring_ratios]

    edgelist = generate_opt_edgelist(
        args.npus_per_domain,
        args.dp,
        args.pp,
        args.cp,
        args.inner_bandwidth_gbps,
        args.inner_latency_ns,
        args.outer_bandwidth_gbps,
        args.outer_latency_ns,
        normalized_ratios,
        args.queue_bytes,
        args.switch_latency_ns,
    )
    with open(args.output, "w") as f:
        f.write(edgelist)

    num_domains = args.dp * args.pp * args.cp
    num_hosts = args.npus_per_domain * num_domains
    print(f"Wrote {args.output}: {num_hosts} hosts across {num_domains} interposer(s) (dp={args.dp}, pp={args.pp}, cp={args.cp}) of {args.npus_per_domain} NPUs each")

    if args.visualize:
        svg_path = os.path.splitext(args.output)[0] + ".svg"
        visualize_edgelist_file(args.output, output_path=svg_path)


if __name__ == "__main__":
    main()

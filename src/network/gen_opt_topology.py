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
from typing import List, Tuple, Dict

from network.visualize import visualize_edgelist_file

from itertools import product


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
    cp: int,
    pp: int,
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
    #We normally have TP,DP,CP,PP parallelism axes
    #We are assuimng TP is within the interposer
    #So we will create a ring between interposers for DP,CP and PP.
    #There will be three links coming out of each interposer. One will connect to the next interposer in the DP ring, one will connect to the next interposer in the CP ring and one will connect to the next interposer in the PP ring
    #The bandwidth allocated to each of rings will be based on the ring_bandwidth_ratios parameter. The sum of the ratios should be 1.0 and correspond to outer_bandwidth_gbps. For example if the ratios are [0.5,0.3,0.2] then the DP ring will get 50% of the outer_bandwidth_gbps, CP ring will get 30% and PP ring will get 20%.
    assert len(ring_bandwidth_ratios) == 3, f"ring_bandwidth_ratios must be [dp_ratio, cp_ratio, pp_ratio], got {ring_bandwidth_ratios}"
    assert abs(sum(ring_bandwidth_ratios) - 1.0) < 1e-9, f"ring_bandwidth_ratios must sum to 1.0, got {ring_bandwidth_ratios}"
    dp_ratio, cp_ratio, pp_ratio = ring_bandwidth_ratios

    num_domains = dp * cp * pp
    num_hosts = npus_per_domain * num_domains

    def domain_coords(domain: int) -> tuple[int, int, int]:
        # Mirrors trace_generator.py's node_id_from_axes nesting (d outermost,
        # then c, then p, t innermost/within-domain): interposer `domain`'s
        # hosts are h[domain*npus_per_domain : (domain+1)*npus_per_domain],
        # same contiguous-TP-block convention as generate_dgx_edgelist.
        d = domain // (cp * pp)
        c = (domain % (cp * pp)) // pp
        p = domain % pp
        return d, c, p

    def domain_id(d: int, c: int, p: int) -> int:
        return d * cp * pp + c * pp + p

    

    def get_all_axis_switches(axis:str)->List[List[int]]:

        axis_size = {"dp": dp, "cp": cp, "pp": pp}

        if axis_size[axis] == 1:
            print(f"Axis {axis} has size 1, skipping")
            return []

        permute_axes = set(["dp", "cp", "pp"]) - set([axis])
        print(f"Permute axes for {axis} are {permute_axes}")
        ring_axis = axis

        #Pick a unique combo of permute axes.
        #The switches in this ring will be the permutation of ring axis across all ring axis values
        rings:List[List[int]] = []
        px1,px2 = permute_axes
        for p1,p2 in product(range(axis_size[px1]), range(axis_size[px2])):
            ring = []
            for ring_idx in range(axis_size[ring_axis]):
                # print(f"Ring for {ring_axis} with {px1}={p1} and {px2}={p2} and {ring_axis}={ring_idx}")
                match ring_axis:
                    case "dp":
                        d = ring_idx
                        c = p1 if px1 == "cp" else p2
                        p = p1 if px1 == "pp" else p2
                    case "cp":
                        d = p1 if px1 == "dp" else p2
                        c = ring_idx
                        p = p1 if px1 == "pp" else p2
                    case "pp":
                        d = p1 if px1 == "dp" else p2
                        c = p1 if px1 == "cp" else p2
                        p = ring_idx
                domain_id_ = domain_id(d,c,p)
                # ring.append((d,c,p))
                ring.append(domain_id_)
            rings.append(ring)  
        return rings                  

    dp_rings = get_all_axis_switches("dp")
    cp_rings = get_all_axis_switches("cp")
    pp_rings = get_all_axis_switches("pp")
    print(f"DP rings: {dp_rings}")
    print(f"CP rings: {cp_rings}")
    print(f"PP rings: {pp_rings}")

    # Outer-switch blocks are assigned sequentially to whichever axes are
    # actually active (size > 1), in dp/cp/pp order -- NOT a fixed
    # dp=block0/cp=block1/pp=block2 split, since that breaks as soon as an
    # earlier axis is inactive while a later one isn't (e.g. dp=1, cp>1,
    # pp>1 would otherwise point cp's block at pp's switches and pp's block
    # past the end of the allocated range).
    active_axes = [name for name, size in (("pp", pp), ("cp", cp), ("dp", dp)) if size > 1]
    print(active_axes)
    axis_block_start = {name: num_domains + i * num_domains for i, name in enumerate(active_axes)}

    # Fixed by name, not by position in active_axes: visualize.py decodes
    # this back into a human label via a fixed {1:"dp",2:"cp",3:"pp"} dict,
    # which only stays correct if a given axis always gets the same id
    # regardless of which other axes happen to be active in this run (the
    # same "position-dependent assignment breaks when an axis is skipped"
    # bug as axis_block_start above, just surfacing as a wrong legend label
    # instead of a wrong switch id).
    axis_id = {"dp": 1, "cp": 2, "pp": 3}

    def outer_switch(axis: str, domain: int) -> str:
        return f"s{axis_block_start[axis] + domain}"

    #If there is a ring of size N, then there will be N links in that ring
    ring_links = sum(len(ring) for ring in dp_rings + cp_rings + pp_rings)
    #We will have one on-interposer switch per interposer and upto three outer-dim switches: one for DP,CP and PP
    num_switches = (1 + len(active_axes)) * num_domains
    #Each on-interposer switch connects to upto three outer-dim switches, one for each ring
    inner_to_outer_links = len(active_axes) * num_domains
    #Host to on-interposer (inner) links
    host_to_inner_links = num_hosts
    num_links = host_to_inner_links + inner_to_outer_links + ring_links


    lines = [
        f"# {npus_per_domain} NPUs/domain x {num_domains} domain(s) (dp={dp}, cp={cp}, pp={pp}) -- optical topology, interposers ring-connected directly, no spine",
        f"Hosts {num_hosts}",
        f"Switches {num_switches}",
        f"Links {num_links}",
        "",
        f"Default_speed_Gbps {inner_bandwidth_gbps}",
        f"Default_latency_ns {inner_latency_ns}",
        f"Default_queue_bytes {queue_bytes}",
        f"Default_switch_latency_ns {switch_latency_ns}",
        "",
    ]
    #Inner/On interposer switches
    for domain in range(num_domains):
        lines.append(f"switch s{domain} switch_latency_ns {switch_latency_ns}")
    #Outer/Ring switches
    for switch_id in range(num_domains, num_switches):
        lines.append(f"switch s{switch_id} switch_latency_ns {switch_latency_ns}")
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

    axis_ratio = {"pp": pp_ratio, "cp": cp_ratio, "dp": dp_ratio}
    axis_rings = {"pp": pp_rings, "cp": cp_rings, "dp": dp_rings}

    # Inner OCS to Outer OCS links: this is where ring_bandwidth_ratios is
    # actually enforced -- it's the local interposer's own port budget being
    # split across axes, not a property of the physical link to the next
    # interposer (that's the outer-outer link below, at full
    # outer_bandwidth_gbps). Latency is hardcoded to 0: this link doesn't
    # physically exist (the outer-dim switches are a modeling device, not a
    # real hop), so outer_latency_ns belongs entirely on the outer-outer
    # link below, which IS the real inter-interposer hop.
    for domain in range(num_domains):
        inner_switch = f"s{domain}"
        for axis in active_axes:
            lines.append(
                f"link {inner_switch} {outer_switch(axis, domain)} "
                f"speed_Gbps {outer_bandwidth_gbps * axis_ratio[axis]} latency_ns 0 axis_id 0"
            )
    lines.append("")

    # Outer OCS to Outer OCS ring links: one ring per (other two axes')
    # coordinate combination, per get_all_axis_switches -- closes from the
    # last domain in each ring back to the first. Full outer_bandwidth_gbps
    # here, not ratio-scaled: the ratio already applied once, above, to how
    # much of the inner switch's own port budget reaches this ring at all;
    # the physical link between two interposers' same-axis ports runs at
    # its own full rate once it's on the wire.
    for axis in active_axes:
        print(f"Generating links for {axis} ring(s) (outer bandwidth ratio {axis_ratio[axis]})")
        for ring in axis_rings[axis]:
            print(f"  Generating links for ring {ring}")
            for i, domain in enumerate(ring):
                print(f"    Generating link for domain {domain}")
                next_domain = ring[(i + 1) % len(ring)]
                lines.append(
                    f"link {outer_switch(axis, domain)} {outer_switch(axis, next_domain)} "
                    f"speed_Gbps {outer_bandwidth_gbps*axis_ratio[axis]} latency_ns {outer_latency_ns} axis_id {axis_id[axis]}"
                )
        lines.append("\n")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate an htsim edgelist for the ring-connected-OCS optical topology."
    )
    parser.add_argument("-a", "--npus-per-domain", type=int, required=True, help="NPUs per interposer (TP degree).")
    parser.add_argument("--dp", type=int, required=True, help="Data-parallel degree (number of DP-ring interposers).")
    parser.add_argument("--cp", type=int, required=True, help="Context-parallel degree (number of CP-ring interposers).")
    parser.add_argument("--pp", type=int, required=True, help="Pipeline-parallel degree (number of PP-ring interposers).")
    parser.add_argument("--inner-bandwidth-gbps", type=float, required=True, help="Bandwidth from a host to its interposer's local OCS.")
    parser.add_argument("--inner-latency-ns", type=float, required=True, help="Latency from a host to its interposer's local OCS.")
    parser.add_argument("--outer-bandwidth-gbps", type=float, required=True, help="Bandwidth budget split across the DP/CP/PP OCS-to-OCS rings, per --ring-ratios.")
    parser.add_argument("--outer-latency-ns", type=float, required=True, help="Latency of the OCS-to-OCS ring links.")
    parser.add_argument("--switch-latency-ns", type=float, default=0, help="Processing latency of each interposer's OCS (default: 0).")
    parser.add_argument("--queue-bytes", type=int, default=65536, help="Default queue size in bytes (default: 65536).")
    parser.add_argument("--ring-ratios", type=float, nargs=3, default=[1/3, 1/3, 1/3], metavar=("DP_RATIO", "CP_RATIO", "PP_RATIO"), help="Ratios of outer bandwidth to allocate to the DP, CP, PP rings. Must sum to 1.0.")
    parser.add_argument("-o", "--output", type=str, default="topology.edgelist", help="Output edgelist path.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Also render an .svg visualization next to --output (same path, .svg extension).")
    args = parser.parse_args()

    normalized_ratios = [r / sum(args.ring_ratios) for r in args.ring_ratios]

    edgelist = generate_opt_edgelist(
        args.npus_per_domain,
        args.dp,
        args.cp,
        args.pp,
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

    num_domains = args.dp * args.cp * args.pp
    num_hosts = args.npus_per_domain * num_domains
    print(f"Wrote {args.output}: {num_hosts} hosts across {num_domains} interposer(s) (dp={args.dp}, cp={args.cp}, pp={args.pp}) of {args.npus_per_domain} NPUs each")

    if args.visualize:
        svg_path = os.path.splitext(args.output)[0] + ".svg"
        visualize_edgelist_file(args.output, output_path=svg_path)


if __name__ == "__main__":
    main()

"""Generate an htsim-style edgelist for a simple 2-tier topology: A NPUs per
inner domain (e.g. NVLink), B such domains connected through a single spine
switch (e.g. scale-out/IB fabric). See topology.py's module docstring for the
edgelist format this produces.

Host h<i> belongs to domain i // A -- pick A to match whatever grouping (e.g.
your trace's tensor-parallel degree) should map to a single NVLink domain.

Which axes actually end up "in domain A" is controlled entirely by
--axis-order + --npus-per-domain, not a separate flag: A must match the
product of sizes of some trailing (innermost) run of axis_order -- the axes
in that run are the ones physically grouped together under one domain switch
-- see coords.inner_axes_for_domain_size. E.g. with the default axis order
dp,cp,pp,tp, A=tp alone or A=pp*tp both work (pp is adjacent to tp); A=cp*tp
does not, because pp sits between cp and tp in that order -- reorder to e.g.
dp,pp,cp,tp instead if you want cp inside the domain with tp.
"""

from __future__ import annotations

import argparse
import os

from network.visualize import visualize_edgelist_file

import coords


def generate_dgx_edgelist(
    npus_per_domain: int,
    sizes: dict,
    axis_order: tuple,
    inner_bandwidth_gbps: float,
    inner_latency_ns: float,
    outer_bandwidth_gbps: float,
    outer_latency_ns: float,
    queue_bytes: int = 65536,
    inner_switch_latency_ns: float = 0,
    outer_switch_latency_ns: float = 0,
) -> str:
    #Raises a clear error if npus_per_domain isn't achievable under
    #axis_order -- see this module's docstring and coords' own docstring for
    #why only a trailing suffix of axis_order can validly form one domain.
    inner_axes = coords.inner_axes_for_domain_size(npus_per_domain, sizes, axis_order)

    num_hosts = sizes["dp"] * sizes["cp"] * sizes["pp"] * sizes["tp"]
    num_domains = num_hosts // npus_per_domain
    multi_domain = num_domains > 1
    spine = f"s{num_domains}"
    num_switches = num_domains + (1 if multi_domain else 0)
    num_links = num_hosts + (num_hosts if multi_domain else 0)

    lines = [
        f"# {npus_per_domain} NPUs/domain x {num_domains} domain(s) -- axis_order={','.join(axis_order)}, inner_axes={','.join(inner_axes)}",
        f"Axis_order {','.join(axis_order)}",
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

    # host = domain*npus_per_domain + local_idx is only correct because
    # inner_axes was just validated to be a genuine trailing suffix of
    # axis_order: under that condition, the global rank numbering (see
    # coords/__init__.py) already groups every domain's hosts into exactly
    # this kind of contiguous block, so no per-axis coordinate loop is
    # needed here at all -- this is the same loop as before --axis-order
    # existed, just now guarded by the validation above.
    for domain in range(num_domains):
        leaf = f"s{domain}"
        for local_idx in range(npus_per_domain):
            host = domain * npus_per_domain + local_idx
            lines.append(f"link h{host} {leaf} direction bi")
            if multi_domain:
                lines.append(
                    f"link h{host} {spine} direction bi speed_Gbps {outer_bandwidth_gbps} latency_ns {outer_latency_ns}"
                )

    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2-tier (inner domain x outer domain) htsim edgelist topology."
    )
    parser.add_argument("-a", "--npus-per-domain", type=int, required=True, help="NPUs per inner domain (e.g. NVLink). Must match a trailing-suffix product of --axis-order (see module docstring).")
    parser.add_argument("--dp", type=int, required=True, help="Data-parallel degree.")
    parser.add_argument("--cp", type=int, required=True, help="Context-parallel degree.")
    parser.add_argument("--pp", type=int, required=True, help="Pipeline-parallel degree.")
    parser.add_argument("--tp", type=int, required=True, help="Tensor-parallel degree.")
    parser.add_argument(
        "--axis-order",
        type=coords.axis_order_arg,
        default=coords.DEFAULT_AXIS_ORDER,
        help="Outermost-to-innermost nesting of dp,cp,pp,tp into one global NPU rank, comma-separated "
             f"(default: {','.join(coords.DEFAULT_AXIS_ORDER)}). Must match whatever order trace_generator.py "
             "was invoked with."
    )
    parser.add_argument("--inner-bandwidth-gbps", type=float, required=True, help="Bandwidth within a domain (e.g. NVLink).")
    parser.add_argument("--inner-latency-ns", type=float, required=True, help="Latency within a domain (e.g. NVLink).")
    parser.add_argument("--outer-bandwidth-gbps", type=float, required=True, help="Bandwidth between domains (e.g. scale-out fabric).")
    parser.add_argument("--outer-latency-ns", type=float, required=True, help="Latency between domains (e.g. scale-out fabric).")
    parser.add_argument("--queue-bytes", type=int, default=65536, help="Default_queue_bytes to write (default: 65536).")
    parser.add_argument("--inner-switch-latency-ns", type=float, default=0, help="Processing latency of a leaf/domain switch (default: 0).")
    parser.add_argument("--outer-switch-latency-ns", type=float, default=0, help="Processing latency of the spine switch (default: 0).")
    parser.add_argument("-o", "--output", type=str, default="topology.edgelist", help="Output edgelist path.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Also render an .svg visualization next to --output (same path, .svg extension).")
    args = parser.parse_args()

    sizes = {"dp": args.dp, "cp": args.cp, "pp": args.pp, "tp": args.tp}

    edgelist = generate_dgx_edgelist(
            args.npus_per_domain,
            sizes,
            args.axis_order,
            args.inner_bandwidth_gbps,
            args.inner_latency_ns,
            args.outer_bandwidth_gbps,
            args.outer_latency_ns,
            args.queue_bytes,
            args.inner_switch_latency_ns,
            args.outer_switch_latency_ns,
        )

    with open(args.output, "w") as f:
        f.write(edgelist)

    num_hosts = args.dp * args.cp * args.pp * args.tp
    num_domains = num_hosts // args.npus_per_domain
    print(f"Wrote {args.output}: {num_hosts} hosts across {num_domains} domain(s) of {args.npus_per_domain} NPUs each (axis_order={','.join(args.axis_order)})")

    if args.visualize:
        svg_path = os.path.splitext(args.output)[0] + ".svg"
        visualize_edgelist_file(args.output, output_path=svg_path)


if __name__ == "__main__":
    main()

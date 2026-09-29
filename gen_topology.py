"""Generate an htsim-style edgelist for a simple 2-tier topology: A NPUs per
inner domain (e.g. NVLink), B such domains connected through a single spine
switch (e.g. scale-out/IB fabric). See topology.py's module docstring for the
edgelist format this produces.

Host h<i> belongs to domain i // A -- pick A to match whatever grouping (e.g.
your trace's tensor-parallel degree) should map to a single NVLink domain.
"""

from __future__ import annotations

import argparse


def generate_edgelist(
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2-tier (inner domain x outer domain) htsim edgelist topology."
    )
    parser.add_argument("-a", "--npus-per-domain", type=int, required=True, help="NPUs per inner domain (e.g. NVLink).")
    parser.add_argument("-b", "--num-domains", type=int, required=True, help="Number of inner domains.")
    parser.add_argument("--inner-bandwidth-gbps", type=float, required=True, help="Bandwidth within a domain (e.g. NVLink).")
    parser.add_argument("--inner-latency-ns", type=float, required=True, help="Latency within a domain (e.g. NVLink).")
    parser.add_argument("--outer-bandwidth-gbps", type=float, required=True, help="Bandwidth between domains (e.g. scale-out fabric).")
    parser.add_argument("--outer-latency-ns", type=float, required=True, help="Latency between domains (e.g. scale-out fabric).")
    parser.add_argument("--queue-bytes", type=int, default=65536, help="Default_queue_bytes to write (default: 65536).")
    parser.add_argument("--inner-switch-latency-ns", type=float, default=0, help="Processing latency of a leaf/domain switch (default: 0).")
    parser.add_argument("--outer-switch-latency-ns", type=float, default=0, help="Processing latency of the spine switch (default: 0).")
    parser.add_argument("-o", "--output", type=str, default="topology.edgelist", help="Output edgelist path.")
    args = parser.parse_args()

    edgelist = generate_edgelist(
        args.npus_per_domain,
        args.num_domains,
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

    num_hosts = args.npus_per_domain * args.num_domains
    print(f"Wrote {args.output}: {num_hosts} hosts across {args.num_domains} domain(s) of {args.npus_per_domain} NPUs each")


if __name__ == "__main__":
    main()

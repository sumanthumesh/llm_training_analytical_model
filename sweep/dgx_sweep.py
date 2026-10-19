"""Sweep (axis_order, npus_per_domain, dp, cp, pp, tp, microbatch_size,
recompute_activations, reuse_fwd_weights) configs for the DGX topology, at a
fixed (global_batch, seq_len) = (2048, 8192) -- i.e. a fixed ~16M
tokens/batch -- and report which feasible config finishes a batch fastest.

Unlike opt_sweep.py (see its own module docstring), axis_order genuinely
matters here: gen_dgx_topology.py lets the inner domain grow past a single
axis's size by absorbing whichever axis sits adjacent to tp in axis_order
(see coords.inner_axes_for_domain_size). This script restricts that to at
most two axes total -- tp alone, or tp plus exactly one other axis -- never
three or four: a domain spanning three+ parallelism axes, or one that
excludes tp entirely, isn't a physically sensible NVLink/TP-local grouping
(see the dry-run writeup this came out of).

Only the SET of axes inside the domain is a genuine modeling choice --
neither the order of the axes outside it, nor which of a 2-axis domain's two
members is innermost, changes anything. gen_dgx_topology.py builds both
tiers (the per-domain switch and the shared spine) as uniform stars, where
every member is equally connected to its switch -- so swapping two outer
axes, or swapping tp with its domain-mate, only relabels which global rank
gets which coordinate, never the topology graph (same host/switch/link
lines) or the simulated result. Confirmed both by diffing generated
edgelists for swapped orderings (identical down to the host/switch/link
lines) and directly in swept results: every permutation of the axes outside
a given domain choice, and both internal orderings of a 2-axis domain,
produced the exact same finish_time_sec. So axis_order is only varied enough
to choose which other axis (if any) joins tp in the domain -- at most 4
variants (tp alone, tp+dp, tp+cp, tp+pp), not one per permutation.

Two-phase, matching how expensive each phase is (see sweep_common.py):

1. Cheap, in-process, all candidates: compute every (dp,cp,pp,tp,mb,recompute,
   reuse) combo's peak per-rank memory footprint and keep only the ones that
   fit --mem-per-npu (axis_order-agnostic). Then attach every (axis_order,
   npus_per_domain) variant where the domain is tp alone or tp plus one
   other axis, filtered to --inner-dim-size (see
   tp_domain_axis_orders_and_sizes).

2. Expensive, one real trace_generator.py + simulate() run per surviving
   config, in parallel worker processes. Each gets its own persistent exp dir
   under --output-dir (trace + topology + a result.json), not a throwaway
   tempdir, so you can go back and inspect any specific run afterwards.
"""

from __future__ import annotations

import argparse
import os
import sys

from sweep_common import (
    memory_feasible_base_configs, generate_trace, simulate_and_report,
    flatten_config_for_result, run_sweep, write_results_csv, print_best,
)
from network.gen_dgx_topology import generate_dgx_edgelist


def tp_domain_axis_orders_and_sizes(sizes: dict, inner_dim_size: int) -> list[tuple[tuple[str, ...], int]]:
    """All (axis_order, npus_per_domain) pairs worth simulating -- one per
    distinct domain membership (tp alone, or tp plus one of dp/cp/pp), not
    one per permutation: see the module docstring for why the order of the
    axes outside the domain, and which of a 2-axis domain's two members is
    innermost, is never a genuine modeling difference here. Each entry picks
    one arbitrary representative axis_order (outer axes in a fixed order,
    tp innermost) for that domain choice.
    """
    tp_size = sizes["tp"]
    variants = []
    if tp_size <= inner_dim_size:
        variants.append((("dp", "cp", "pp", "tp"), tp_size))
    for adjacent, outer in (("dp", ("cp", "pp")), ("cp", ("dp", "pp")), ("pp", ("dp", "cp"))):
        paired_size = sizes[adjacent] * tp_size
        if paired_size <= inner_dim_size:
            variants.append((outer + (adjacent, "tp"), paired_size))
    return variants


def expand_to_topology_configs(base_configs: list[dict], inner_dim_size: int) -> list[dict]:
    """Each memory-feasible (dp,cp,pp,tp,mb,recompute,reuse) base config
    becomes one variant per (axis_order, npus_per_domain) pair from
    tp_domain_axis_orders_and_sizes -- memory footprint doesn't depend on
    either, but the simulated finish time very much does.
    """
    configs = []
    for base in base_configs:
        sizes = {"dp": base["dp"], "cp": base["cp"], "pp": base["pp"], "tp": base["tp"]}
        for axis_order, npus_per_domain in tp_domain_axis_orders_and_sizes(sizes, inner_dim_size):
            configs.append({**base, "axis_order": axis_order, "npus_per_domain": npus_per_domain})
    return configs


def run_dgx_config(config: dict, exp_dir: str, *, trace_kwargs: dict, topo_kwargs: dict) -> dict:
    trace_path = generate_trace(exp_dir, config, **trace_kwargs)

    sizes = {"dp": config["dp"], "cp": config["cp"], "pp": config["pp"], "tp": config["tp"]}
    edgelist = generate_dgx_edgelist(
        config["npus_per_domain"], sizes, config["axis_order"],
        topo_kwargs["inner_bandwidth_gbps"], topo_kwargs["inner_latency_ns"],
        topo_kwargs["outer_bandwidth_gbps"], topo_kwargs["outer_latency_ns"],
    )
    edgelist_path = os.path.join(exp_dir, "topology.edgelist")
    with open(edgelist_path, "w") as f:
        f.write(edgelist)

    return simulate_and_report(
        edgelist_path, trace_path, exp_dir,
        peak_perf_tflops=topo_kwargs["peak_perf_tflops"], local_mem_bw_gbyteps=topo_kwargs["local_mem_bw_gbyteps"],
        fixed_overhead_ns=topo_kwargs["fixed_overhead_ns"],
        global_batch=trace_kwargs["global_batch"], seq_len=trace_kwargs["seq_len"],
        extra_result_fields={**flatten_config_for_result(config), "num_npus": config["dp"] * config["cp"] * config["pp"] * config["tp"], "exp_dir": exp_dir},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-npus", type=int, required=True, help="Total NPU budget (dp*cp*pp*tp must equal this).")
    parser.add_argument("--mem-per-npu", type=float, required=True, help="Peak per-rank memory budget, in GB.")
    parser.add_argument("--inner-dim-size", type=int, default=64, help="Max achievable domain size (npus_per_domain) to consider -- e.g. an NVSwitch/NVLink domain's capacity (default: 64).")
    parser.add_argument("--min-tp", type=int, default=None, help="Minimum tp to consider (default: min(8, --inner-dim-size)). Mainly for smoke-testing with a smaller NPU budget than the default floor allows.")
    parser.add_argument("--global-batch-size", type=int, default=2048, help="Global batch size (default: 2048).")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length (default: 8192).")
    parser.add_argument("--layers", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-heads", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn-dim", type=int, default=53248)
    parser.add_argument("--vocab", type=int, default=128000)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--single-comm", action="store_true", help="Pass --single-comm to trace_generator.py (much faster/smaller traces, at the cost of modeling only one representative comm group per collective instead of all of them).")
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Bandwidth within a domain (default: NVLink 5 class).")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Bandwidth between domains, via the spine (default: IB NDR class).")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0)
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Base directory for per-config exp dirs and the results CSV. Required unless --dry-run.")
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--dry-run", action="store_true", help="Only run phase 1 (memory feasibility + axis_order/npus_per_domain expansion) and report how many configs would be simulated, then exit -- no trace/topology/simulate, no --output-dir needed.")
    args = parser.parse_args()

    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is given.")

    print("Phase 1: computing mem_footprint for all candidate configs...")
    base_configs = memory_feasible_base_configs(
        args.num_npus, args.mem_per_npu,
        args.global_batch_size, args.seq_len,
        args.layers, args.kv_heads, args.q_heads, args.head_dim, args.ffn_dim, args.vocab,
        args.bytes_per_element, max_tp=None,
        min_tp=args.min_tp if args.min_tp is not None else min(8, args.inner_dim_size),
    )
    configs = expand_to_topology_configs(base_configs, args.inner_dim_size)
    print(f"{len(base_configs)} memory-feasible (dp,cp,pp,tp,mb,recompute,reuse) configs x (axis_order, npus_per_domain) variants = {len(configs)} configs -- {'dry run, not simulating' if args.dry_run else 'running full trace+simulate for each'}.")
    if args.dry_run or not configs:
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    trace_kwargs = dict(
        global_batch=args.global_batch_size, seq_len=args.seq_len,
        num_layers=args.layers, kv_heads=args.kv_heads, q_heads=args.q_heads, head_dim=args.head_dim,
        ffn_dim=args.ffn_dim, vocab=args.vocab, bytes_per_element=args.bytes_per_element,
        single_comm=args.single_comm,
    )
    topo_kwargs = dict(
        inner_bandwidth_gbps=args.inner_bandwidth_gbps, inner_latency_ns=args.inner_latency_ns,
        outer_bandwidth_gbps=args.outer_bandwidth_gbps, outer_latency_ns=args.outer_latency_ns,
        peak_perf_tflops=args.peak_perf_tflops, local_mem_bw_gbyteps=args.local_mem_bw_gbyteps,
        fixed_overhead_ns=args.fixed_overhead_ns,
    )
    run_kwargs = dict(trace_kwargs=trace_kwargs, topo_kwargs=topo_kwargs)

    print("Phase 2: generating traces/topologies and simulating...")
    results = run_sweep(configs, args.output_dir, run_dgx_config, run_kwargs, args.max_workers)

    csv_path = os.path.join(args.output_dir, "results.csv")
    write_results_csv(results, csv_path)
    print(f"\nWrote {len(results)} results to {csv_path}")

    print_best(results, args.global_batch_size, args.seq_len, extra_fields=["npus_per_domain"])

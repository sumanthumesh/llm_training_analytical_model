"""Sweep (dp, cp, pp, tp, microbatch_size, recompute_activations,
reuse_fwd_weights) configs for the optical topology, at a fixed (global_batch,
seq_len) = (2048, 8192) -- i.e. a fixed ~16M tokens/batch -- and report which
feasible config finishes a batch fastest.

Two-phase, matching how expensive each phase is (see sweep_common.py):

1. Cheap, in-process, all candidates: compute every (dp,cp,pp,tp,mb,recompute,
   reuse) combo's peak per-rank memory footprint and keep only the ones that
   fit --mem-per-npu, then attach the one representative axis_order each
   needs (see REPRESENTATIVE_AXIS_ORDER below -- axis_order among dp/cp/pp
   doesn't affect this topology, so there's no need to sweep it here).

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
from network.gen_opt_topology import generate_opt_edgelist
from tracegen.trace_generator import ring_bandwidth_ratios_from_formulas
import coords

# generate_opt_edgelist always sizes the inner domain to exactly tp (see its
# own docstring/assertion) -- --inner-dim-size only caps how large tp is
# allowed to be during the memory-feasibility pass, it's never fed into
# topology generation as an actual domain size. So dp/cp/pp's relative order
# never changes which physical links a config ends up with here: they're
# always entirely in the outer ring, regardless of axis_order. A single
# representative order is therefore enough -- no need to multiply the sweep
# by 6 permutations that are topologically identical for this script.
#
# This stops being true once a topology generator can grow the inner domain
# past tp by absorbing an adjacent axis (dgx_sweep.py's planned
# --inner-dim-size up to 64, via coords.inner_axes_for_domain_size) -- at
# that point axis_order will matter again and this collapse needs to go.
REPRESENTATIVE_AXIS_ORDER = coords.DEFAULT_AXIS_ORDER


def expand_to_topology_configs(base_configs: list[dict]) -> list[dict]:
    """Attaches the one representative axis_order to each memory-feasible
    (dp,cp,pp,tp,mb,recompute,reuse) base config -- see
    REPRESENTATIVE_AXIS_ORDER above for why sweeping axis_order itself isn't
    needed for this topology."""
    return [{**base, "axis_order": REPRESENTATIVE_AXIS_ORDER} for base in base_configs]


def run_opt_config(config: dict, exp_dir: str, *, trace_kwargs: dict, topo_kwargs: dict) -> dict:
    trace_path = generate_trace(exp_dir, config, **trace_kwargs)

    # Per-config, not a static split -- the per-axis bandwidth each config
    # actually needs varies with (dp,cp,pp,tp,mb,recompute,reuse), so one
    # fixed ratio under- or over-provisions some rings depending on the
    # config (see ring_bandwidth_ratios_from_formulas's own docstring).
    dp_ratio, cp_ratio, pp_ratio = ring_bandwidth_ratios_from_formulas(
        num_layers=trace_kwargs["num_layers"], kv_heads=trace_kwargs["kv_heads"], q_heads=trace_kwargs["q_heads"],
        head_dim=trace_kwargs["head_dim"], ffn_dim=trace_kwargs["ffn_dim"], vocab=trace_kwargs["vocab"],
        seq_len=trace_kwargs["seq_len"], global_batch=trace_kwargs["global_batch"],
        microbatch_size=config["microbatch_size"],
        dp_degree=config["dp"], cp_degree=config["cp"], pp_degree=config["pp"], tp_degree=config["tp"],
        bytes_per_element=trace_kwargs["bytes_per_element"],
        recompute_activations=config["recompute_activations"], reuse_fwd_fsdp=config["reuse_fwd_weights"],
    )
    edgelist = generate_opt_edgelist(
        config["tp"], config["dp"], config["cp"], config["pp"], config["axis_order"],
        topo_kwargs["inner_bandwidth_gbps"], topo_kwargs["inner_latency_ns"],
        topo_kwargs["outer_bandwidth_gbps"], topo_kwargs["outer_latency_ns"],
        [dp_ratio, cp_ratio, pp_ratio],
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
    parser.add_argument("--type", type=str, choices=["opt"], default="opt", help="Topology type (only 'opt' is implemented here -- see dgx_sweep.py).")
    parser.add_argument("--mem-per-npu", type=float, required=True, help="Peak per-rank memory budget, in GB.")
    parser.add_argument("--inner-dim-size", type=int, default=8, help="Max tp (interposer size) -- default 8.")
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
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Host-to-interposer bandwidth (default: NVLink 5 class).")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Interposer-to-interposer ring bandwidth budget (default: IB NDR class).")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0)
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Base directory for per-config exp dirs and the results CSV. Required unless --dry-run.")
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--dry-run", action="store_true", help="Only run phase 1 (memory feasibility) and report how many configs would be simulated, then exit -- no trace/topology/simulate, no --output-dir needed.")
    args = parser.parse_args()

    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is given.")

    print("Phase 1: computing mem_footprint for all candidate configs...")
    base_configs = memory_feasible_base_configs(
        args.num_npus, args.mem_per_npu,
        args.global_batch_size, args.seq_len,
        args.layers, args.kv_heads, args.q_heads, args.head_dim, args.ffn_dim, args.vocab,
        args.bytes_per_element, max_tp=args.inner_dim_size,
        min_tp=args.min_tp if args.min_tp is not None else min(8, args.inner_dim_size),
    )
    configs = expand_to_topology_configs(base_configs)
    print(f"{len(base_configs)} memory-feasible (dp,cp,pp,tp,mb,recompute,reuse) configs -- {len(configs)} configs total to simulate.")
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
    results = run_sweep(configs, args.output_dir, run_opt_config, run_kwargs, args.max_workers)

    csv_path = os.path.join(args.output_dir, "results.csv")
    write_results_csv(results, csv_path)
    print(f"\nWrote {len(results)} results to {csv_path}")

    print_best(results, args.global_batch_size, args.seq_len)

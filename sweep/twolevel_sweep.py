"""Sweep (dp, cp, pp, tp, microbatch_size, recompute_activations,
reuse_fwd_weights) configs for the two-level topology
(network/gen_twolevel_topology.py), over exactly the same config list as
opt_sweep.py, and report which feasible config finishes a batch fastest.

The two-level topology is the optical topology with the static per-axis rings
replaced by one freely switched tier: each interposer (inner domain of tp
NPUs) has a single shared uplink of --outer-bandwidth-gbps to one spine
switch, and every outer-domain transfer (DP, CP or PP alike) goes
host -> leaf -> spine -> leaf -> host. There is no per-axis bandwidth split:
whichever axis is communicating gets the whole uplink. The simulator charges
no reconfiguration time for that switching, so this sweep is the
free-reconfiguration upper bound for the static-ring optical topology at the
same outer bandwidth budget -- run both sweeps with the same flags to compare.

Same config list as opt_sweep.py, by construction: this script imports
opt_sweep.py's own expand_to_topology_configs / REPRESENTATIVE_AXIS_ORDER
rather than re-deriving them, and the inner domain is always exactly tp
(npus_per_domain = tp, the same interposer-local tp that opt_sweep.py's
topology hard-codes). Reusing one representative axis_order is also valid
here: the spine is a single uniform star with one uplink per leaf, so the
order of dp/cp/pp outside the domain only relabels hosts and never changes
the topology graph (the same argument opt_sweep.py and dgx_sweep.py make for
their own topologies).

Two differences from opt_sweep.py's topology to keep in mind when comparing:
- Outer bandwidth: here --outer-bandwidth-gbps is the per-interposer uplink,
  shared by all axes and all tp NPUs of that interposer. In opt_sweep.py the
  same flag is the per-interposer budget split across the DP/CP/PP rings by
  ring_bandwidth_ratios_from_formulas. Same total, no split.
- Outer latency: an outer transfer here crosses two outer links (leaf->spine
  and spine->leaf), so --outer-latency-ns is charged twice per hop. In
  opt_sweep.py's topology it is charged once (the inter-interposer ring link;
  the inner-to-ring links have zero latency). Pass half the opt value here if
  you want identical per-hop latency. With 0 ns latency the two are identical.

Two-phase, matching how expensive each phase is (see sweep_common.py):

1. Cheap, in-process, all candidates: compute every (dp,cp,pp,tp,mb,recompute,
   reuse) combo's peak per-rank memory footprint and keep only the ones that
   fit --mem-per-npu (identical to opt_sweep.py's phase 1).

2. Expensive, one real trace_generator.py + simulate() run per surviving
   config, in parallel worker processes, each with its own persistent exp dir
   under --output-dir (trace + topology + a result.json).
"""

from __future__ import annotations

import argparse
import os
import sys

from sweep_common import (
    memory_feasible_base_configs, generate_trace, simulate_and_report,
    flatten_config_for_result, run_sweep, write_results_csv, print_best,
    add_shard_args, select_shard, results_csv_path,
)
# Imported, not copied, so the config list cannot drift from opt_sweep.py's.
from opt_sweep import expand_to_topology_configs, REPRESENTATIVE_AXIS_ORDER  # noqa: F401
from network.gen_twolevel_topology import generate_twolevel_edgelist


def run_twolevel_config(config: dict, exp_dir: str, *, trace_kwargs: dict, topo_kwargs: dict, axis_gaps: bool = False) -> dict:
    trace_path = generate_trace(exp_dir, config, **trace_kwargs)

    sizes = {"dp": config["dp"], "cp": config["cp"], "pp": config["pp"], "tp": config["tp"]}
    # The inner domain is the tp interposer, same as opt_sweep.py's topology.
    # No ring ratios: the single uplink per interposer is shared by every axis.
    edgelist = generate_twolevel_edgelist(
        config["tp"], sizes, config["axis_order"],
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
        axis_gaps=axis_gaps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-npus", type=int, required=True, help="Total NPU budget (dp*cp*pp*tp must equal this).")
    parser.add_argument("--mem-per-npu", type=float, required=True, help="Peak per-rank memory budget, in GB.")
    parser.add_argument("--inner-dim-size", type=int, default=8, help="Max tp (interposer size) -- default 8. Same meaning as opt_sweep.py's flag.")
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
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Host-to-interposer (host-to-leaf) bandwidth. Same default as opt_sweep.py.")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Per-interposer uplink to the spine, shared by all axes and all tp NPUs of that interposer (no per-axis split). Same default as opt_sweep.py, where it is the budget split across rings.")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0, help="Latency of each leaf<->spine link. An outer transfer crosses two of them, so this is charged twice per hop (opt_sweep.py's topology charges its outer latency once).")
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Base directory for per-config exp dirs and the results CSV. Required unless --dry-run.")
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() // 2))
    add_shard_args(parser)
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
    if args.num_shards > 1:
        configs = select_shard(configs, args.shard_index, args.num_shards)
        print(f"Shard {args.shard_index} of {args.num_shards} (0-based): running {len(configs)} of those configs.")
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
    results = run_sweep(configs, args.output_dir, run_twolevel_config, run_kwargs, args.max_workers)

    csv_path = results_csv_path(args.output_dir, args.shard_index, args.num_shards)
    write_results_csv(results, csv_path)
    print(f"\nWrote {len(results)} results to {csv_path}")

    print_best(results, args.global_batch_size, args.seq_len)

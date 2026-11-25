"""Run one exact (dp, cp, pp, tp, microbatch_size, recompute_activations,
reuse_fwd_weights) config end-to-end for the two-level topology -- the same
trace_generator.py + simulate() pipeline twolevel_sweep.py uses per config,
without enumerating a whole sweep around it.

Reuses twolevel_sweep.py's own run_twolevel_config (and the shared
REPRESENTATIVE_AXIS_ORDER) and sweep_common.py's run_one_config, so the exp
dir layout and naming are identical to a sweep run of this same config. See
twolevel_sweep.py's docstring for how its --outer-* flags differ from opt's
(same per-interposer budget with no split; outer latency charged twice per
hop).
"""

from __future__ import annotations

import argparse
import json

from sweep_common import run_one_config, divisibility_ok, describe_run, add_fsdp_prefetch_arg
from opt_sweep import REPRESENTATIVE_AXIS_ORDER
from twolevel_sweep import run_twolevel_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--recompute-activations", action="store_true")
    parser.add_argument("--reuse-fwd-weights", action="store_true")
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
    add_fsdp_prefetch_arg(parser)
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Host-to-leaf bandwidth. Same default as opt_sweep.py.")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Per-interposer uplink to the spine, shared by all axes and all tp NPUs of that interposer (no per-axis split).")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0, help="Latency of each leaf<->spine link; charged twice per outer hop.")
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Base directory -- the exp dir for this one config is created underneath it, same naming convention as twolevel_sweep.py's own per-config exp dirs.")
    args = parser.parse_args()

    if not divisibility_ok(
        args.global_batch_size, args.seq_len, args.dp, args.cp, args.pp, args.tp, args.microbatch_size,
        args.layers, args.kv_heads, args.ffn_dim,
    ):
        parser.error("This config fails trace_generator.py's divisibility/shape checks (see divisibility_ok in sweep_common.py).")

    config = {
        "dp": args.dp, "cp": args.cp, "pp": args.pp, "tp": args.tp,
        "microbatch_size": args.microbatch_size,
        "recompute_activations": args.recompute_activations,
        "reuse_fwd_weights": args.reuse_fwd_weights,
        "axis_order": REPRESENTATIVE_AXIS_ORDER,
    }
    trace_kwargs = dict(
        global_batch=args.global_batch_size, seq_len=args.seq_len,
        num_layers=args.layers, kv_heads=args.kv_heads, q_heads=args.q_heads, head_dim=args.head_dim,
        ffn_dim=args.ffn_dim, vocab=args.vocab, bytes_per_element=args.bytes_per_element,
        single_comm=args.single_comm, fsdp_prefetch=args.fsdp_prefetch,
    )
    topo_kwargs = dict(
        inner_bandwidth_gbps=args.inner_bandwidth_gbps, inner_latency_ns=args.inner_latency_ns,
        outer_bandwidth_gbps=args.outer_bandwidth_gbps, outer_latency_ns=args.outer_latency_ns,
        peak_perf_tflops=args.peak_perf_tflops, local_mem_bw_gbyteps=args.local_mem_bw_gbyteps,
        fixed_overhead_ns=args.fixed_overhead_ns,
    )

    print(f"Running dp={args.dp} cp={args.cp} pp={args.pp} tp={args.tp} mb={args.microbatch_size} "
          f"recompute={args.recompute_activations} reuse={args.reuse_fwd_weights}...")
    result = run_one_config(config, args.output_dir, run_twolevel_config, dict(trace_kwargs=trace_kwargs, topo_kwargs=topo_kwargs),
                            run_info=describe_run(args))

    if "error" in result:
        print(f"\nFAILED -- see {result['exp_dir']}/trace_generator.log or analytical_model.log")
        print(result["error"])
        raise SystemExit(1)

    print(f"\nfinish_time_sec={result['finish_time_sec']:.6f}  tokens_per_sec={result['tokens_per_sec']:.2f}")
    print(f"analytical_model_wall_time_sec={result['analytical_model_wall_time_sec']:.2f}")
    print(f"exp_dir={result['exp_dir']}")
    print(json.dumps(result, indent=2))

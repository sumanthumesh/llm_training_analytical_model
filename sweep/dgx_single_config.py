"""Run one exact (dp, cp, pp, tp, microbatch_size, recompute_activations,
reuse_fwd_weights, axis_order, npus_per_domain) config end-to-end for the DGX
topology -- the same trace_generator.py + simulate() pipeline dgx_sweep.py
uses per config, without enumerating a whole sweep around it.

Reuses dgx_sweep.py's own run_dgx_config and sweep_common.py's
run_one_config, so the exp dir layout and naming are identical to a sweep
run of this same config.

Unlike opt and two-level, the DGX domain can hold more than tp: pass
--npus-per-domain and an --axis-order whose innermost axes make up that
domain, e.g. --npus-per-domain 64 --axis-order dp,pp,cp,tp puts tp8 x cp8 in
one NVLink domain (see dgx_sweep.py's tp_domain_axis_orders_and_sizes).
"""

from __future__ import annotations

import argparse
import json

import coords
from sweep_common import run_one_config, divisibility_ok, describe_run, add_fsdp_prefetch_arg
from dgx_sweep import run_dgx_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dp", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--recompute-activations", action="store_true")
    parser.add_argument("--reuse-fwd-weights", action="store_true")
    parser.add_argument("--npus-per-domain", type=int, required=True, help="NVLink domain size: tp alone, or tp times the axis next to it in --axis-order.")
    parser.add_argument("--axis-order", type=coords.axis_order_arg, required=True, help="Outermost-to-innermost, comma-separated, e.g. dp,pp,cp,tp.")
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
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Bandwidth within a domain (default: NVLink 5 class).")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Bandwidth between domains, via the spine (default: IB NDR class).")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0)
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Base directory -- the exp dir for this one config is created underneath it, same naming convention as dgx_sweep.py's own per-config exp dirs.")
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
        "axis_order": args.axis_order,
        "npus_per_domain": args.npus_per_domain,
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
          f"recompute={args.recompute_activations} reuse={args.reuse_fwd_weights} "
          f"npus_per_domain={args.npus_per_domain} axis_order={','.join(args.axis_order)}...")
    result = run_one_config(config, args.output_dir, run_dgx_config, dict(trace_kwargs=trace_kwargs, topo_kwargs=topo_kwargs),
                            run_info=describe_run(args))

    if "error" in result:
        print(f"\nFAILED -- see {result['exp_dir']}/trace_generator.log or analytical_model.log")
        print(result["error"])
        raise SystemExit(1)

    print(f"\nfinish_time_sec={result['finish_time_sec']:.6f}  tokens_per_sec={result['tokens_per_sec']:.2f}")
    print(f"analytical_model_wall_time_sec={result['analytical_model_wall_time_sec']:.2f}")
    print(f"exp_dir={result['exp_dir']}")
    print(json.dumps(result, indent=2))

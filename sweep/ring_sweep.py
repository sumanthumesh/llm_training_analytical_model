"""Sweep the optical topology's static ring bandwidth split (the DP/CP/PP
ring_bandwidth_ratios of gen_opt_topology.py) for ONE fixed parallelism
config, to see how much a better split than the default can buy.

    # what would run (no output dir needed)
    python ring_sweep.py --dp 256 --cp 8 --pp 1 --tp 8 --microbatch-size 4 \\
        --recompute-activations --reuse-fwd-weights <hardware flags> --step 0.05 --dry-run

    # run it, comparing against e.g. the best DGX config's finish time
    python ring_sweep.py <same flags> -o exp_ring_sweep --reference-time-sec 1.808

The config flags are the same as opt_single_config.py's. Only the ring
ratios vary: every point on a grid over the simplex dp_ratio + cp_ratio +
pp_ratio = 1, with spacing --step, optionally narrowed per axis with
--dp-range/--cp-range/--pp-range, or an explicit list via --ratios. The
ratios ring_bandwidth_ratios_from_formulas would pick (the byte-proportional
split opt_sweep.py uses) are always added as a baseline point unless
--no-formula-baseline.

Axes of size 1 have no ring (gen_opt_topology.py only builds rings for axes
with size > 1), so any bandwidth given to them would simply be lost. Their
ratio is pinned to 0 and only the remaining axes are swept -- e.g. with pp=1
the sweep is over DP vs CP only. Every swept axis gets at least --min-ratio
(default: --step), since a 0-bandwidth ring could never finish.

Grid size: with n active axes and --min-ratio == --step, there are
C(1/step - 1, n - 1) points -- 19 for two axes at step 0.05, 171 for three
axes at step 0.05, 36 for three axes at step 0.1.

The trace does not depend on the ratios, so it is generated ONCE into
<output-dir>/trace/ and every point reuses it; each point only writes its own
topology.edgelist and simulates. The settings the trace was generated with
(config, model shape, --bytes-per-element, --fsdp-prefetch, ...) are saved
to trace/trace_args.json, and a run whose settings differ -- or a trace
copied in from before that file existed -- is refused instead of silently
reused. Each point gets its own exp dir under
--output-dir (named by its ratios) with a result.json that includes a
"run_info" block (every CLI arg, git commit), same as the other sweeps.

Sharding (--num-shards/--shard-index) splits the ratio points across
processes the same way the other sweeps do. If several shards start at once,
each generates the trace if it's missing, into a private temp dir that is
renamed into place atomically, so the first to finish wins and the rest
reuse it. To avoid generating it N times, run once with --trace-only first.
Shard results.csv files can be combined with extract_results.py over
<output-dir>/*/result.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from sweep_common import (
    generate_trace, simulate_and_report, flatten_config_for_result, divisibility_ok, describe_run,
    write_results_csv, add_shard_args, add_fsdp_prefetch_arg, select_shard, results_csv_path,
)
from opt_sweep import REPRESENTATIVE_AXIS_ORDER
from network.gen_opt_topology import generate_opt_edgelist
from tracegen.trace_generator import ring_bandwidth_ratios_from_formulas

AXES = ("dp", "cp", "pp")


def ratio_grid(active: list[str], step: float, min_ratio: float, ranges: dict) -> list[dict]:
    """Every split of the whole outer budget across the active axes on a
    grid of spacing `step`, with each active axis in [max(min_ratio, lo),
    hi] for its (lo, hi) in `ranges`. Inactive axes get 0."""
    k = round(1 / step)
    if abs(k * step - 1) > 1e-9:
        raise ValueError(f"--step must divide 1 evenly (e.g. 0.05, 0.1, 0.125), got {step}")
    points = []
    for parts in product(range(k + 1), repeat=len(active)):
        if sum(parts) != k:
            continue
        ratios = {a: 0.0 for a in AXES}
        ratios.update({a: p / k for a, p in zip(active, parts)})
        if all(max(min_ratio, ranges[a][0]) - 1e-9 <= ratios[a] <= ranges[a][1] + 1e-9 for a in active):
            points.append(ratios)
    return points


def parse_explicit_ratios(texts: list[str], active: list[str]) -> list[dict]:
    points = []
    for text in texts:
        values = [float(v) for v in text.split(",")]
        if len(values) != 3:
            raise ValueError(f"--ratios entries are DP,CP,PP (three numbers), got {text!r}")
        if abs(sum(values) - 1) > 1e-6:
            raise ValueError(f"--ratios entry {text!r} sums to {sum(values)}, not 1")
        ratios = dict(zip(AXES, values))
        for a in AXES:
            if a not in active and ratios[a] != 0:
                raise ValueError(f"--ratios entry {text!r} gives {a} bandwidth, but {a} has size 1 so it has no ring -- that bandwidth would be lost.")
            if a in active and ratios[a] <= 0:
                raise ValueError(f"--ratios entry {text!r} gives active axis {a} no bandwidth -- its ring could never finish.")
        total = sum(values)
        points.append({a: ratios[a] / total for a in AXES})
    return points


def ratio_tag(ratios: dict) -> str:
    return "_".join(f"{a}{ratios[a]:.4f}" for a in AXES)


def check_trace_args(trace_dir: str, wanted: dict) -> None:
    """Refuse to reuse a trace generated with different settings -- e.g. one
    copied or symlinked in from another sweep, or one from before
    --fsdp-prefetch existed (which used the 'layer' schedule)."""
    args_path = os.path.join(trace_dir, "trace_args.json")
    if not os.path.exists(args_path):
        raise SystemExit(
            f"{trace_dir} has no trace_args.json, so there is no record of what it was generated with (it predates "
            "this check, and so also predates --fsdp-prefetch, i.e. it uses the old 'layer' schedule). Delete or "
            "move that trace directory to have this sweep generate a fresh one.")
    with open(args_path) as f:
        recorded = json.load(f)
    diffs = {k: (recorded.get(k), wanted.get(k)) for k in set(recorded) | set(wanted) if recorded.get(k) != wanted.get(k)}
    if diffs:
        detail = ", ".join(f"{k}: trace has {old!r}, this run wants {new!r}" for k, (old, new) in sorted(diffs.items()))
        raise SystemExit(f"{trace_dir} was generated with different settings ({detail}). Use a different --output-dir, "
                         "or delete that trace directory to regenerate it.")


def ensure_trace(output_dir: str, config: dict, trace_kwargs: dict) -> str:
    """Shared trace at <output_dir>/trace/trace.json.zst, generated once.
    Built in a private temp dir and renamed into place, so concurrent shards
    never see a half-written trace; if another shard got there first, its
    copy is used and ours is discarded. The settings it was generated with
    are recorded in trace/trace_args.json and checked on every reuse."""
    trace_dir = os.path.join(output_dir, "trace")
    trace_path = os.path.join(trace_dir, "trace.json.zst")
    wanted = json.loads(json.dumps({**flatten_config_for_result(config), **trace_kwargs}))
    if os.path.exists(trace_path):
        check_trace_args(trace_dir, wanted)
        print(f"Reusing trace {trace_path}")
        return trace_path
    tmp_dir = os.path.join(output_dir, f".trace.tmp.{socket.gethostname()}.{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Generating trace (once for the whole sweep) in {tmp_dir} ...")
    generate_trace(tmp_dir, config, **trace_kwargs)
    with open(os.path.join(tmp_dir, "trace_args.json"), "w") as f:
        json.dump(wanted, f, indent=2)
    try:
        os.rename(tmp_dir, trace_dir)
    except OSError:
        if not os.path.exists(trace_path):
            raise
        print("Another process finished the trace first -- using its copy.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return trace_path


def run_one_ratio(ratios: dict, source: str, config: dict, trace_path: str, output_dir: str,
                  trace_kwargs: dict, topo_kwargs: dict, run_info: dict) -> dict:
    exp_dir = os.path.join(output_dir, ratio_tag(ratios))
    os.makedirs(exp_dir, exist_ok=True)
    fields = {
        **flatten_config_for_result(config),
        "num_npus": config["dp"] * config["cp"] * config["pp"] * config["tp"],
        "dp_ratio": ratios["dp"], "cp_ratio": ratios["cp"], "pp_ratio": ratios["pp"],
        "ratio_source": source, "exp_dir": exp_dir, "trace_path": trace_path,
    }
    try:
        # generate_opt_edgelist prints every ring it builds -- thousands of
        # lines at full scale -- so keep that out of the console.
        with open(os.path.join(exp_dir, "topology_generator.log"), "w") as log, contextlib.redirect_stdout(log):
            edgelist = generate_opt_edgelist(
                config["tp"], config["dp"], config["cp"], config["pp"], config["axis_order"],
                topo_kwargs["inner_bandwidth_gbps"], topo_kwargs["inner_latency_ns"],
                topo_kwargs["outer_bandwidth_gbps"], topo_kwargs["outer_latency_ns"],
                [ratios["dp"], ratios["cp"], ratios["pp"]],
            )
        edgelist_path = os.path.join(exp_dir, "topology.edgelist")
        with open(edgelist_path, "w") as f:
            f.write(edgelist)
        result = simulate_and_report(
            edgelist_path, trace_path, exp_dir,
            peak_perf_tflops=topo_kwargs["peak_perf_tflops"], local_mem_bw_gbyteps=topo_kwargs["local_mem_bw_gbyteps"],
            fixed_overhead_ns=topo_kwargs["fixed_overhead_ns"],
            global_batch=trace_kwargs["global_batch"], seq_len=trace_kwargs["seq_len"],
            extra_result_fields=fields,
        )
        with open(os.path.join(exp_dir, "result.json"), "w") as f:
            json.dump({**result, "run_info": run_info}, f, indent=2)
        return result
    except Exception as e:
        return {**fields, "error": str(e)}


def run_points(points: list[tuple[dict, str]], max_workers: int, **kwargs) -> list[dict]:
    """Worker pool over ratio points; same Ctrl+C handling as
    sweep_common.run_sweep (cancel queued points AND kill running workers)."""
    results = []
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(run_one_ratio, ratios, source, **kwargs) for ratios, source in points]
    try:
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            status = "ERROR: " + r["error"] if "error" in r else f"{r['finish_time_sec']:.6f}s"
            print(f"[{i}/{len(futures)}] dp={r['dp_ratio']:.4f} cp={r['cp_ratio']:.4f} pp={r['pp_ratio']:.4f} ({r['ratio_source']}) -> {status}", flush=True)
            results.append(r)
    except KeyboardInterrupt:
        print("\nInterrupted -- cancelling queued points and killing active worker processes...")
        for future in futures:
            future.cancel()
        for process in executor._processes.values():
            process.kill()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def print_summary(results: list[dict], reference_time_sec: float | None, top: int = 10) -> None:
    ok = sorted((r for r in results if "error" not in r), key=lambda r: r["finish_time_sec"])
    if not ok:
        print("\nEvery point errored -- see each exp dir's analytical_model.log.")
        return
    baseline = next((r for r in ok if r["ratio_source"] == "formula"), None)

    def line(r):
        s = (f"  dp={r['dp_ratio']:.4f} cp={r['cp_ratio']:.4f} pp={r['pp_ratio']:.4f}  "
             f"T={r['finish_time_sec']:.4f}s  exposed_comm={r['exposed_comm_sec']:.4f}s")
        if baseline is not None:
            s += f"  vs formula {100 * (r['finish_time_sec'] / baseline['finish_time_sec'] - 1):+.1f}%"
        if reference_time_sec:
            s += f"  vs reference {100 * (r['finish_time_sec'] / reference_time_sec - 1):+.1f}%"
        return s + ("  <- formula split" if r["ratio_source"] == "formula" else "")

    print(f"\nFastest {min(top, len(ok))} of {len(ok)} ratio points (this shard):")
    for r in ok[:top]:
        print(line(r))
    if baseline is not None and baseline not in ok[:top]:
        print("Formula (byte-proportional) split, for comparison:")
        print(line(baseline))
    if reference_time_sec:
        print(f"(reference: {reference_time_sec:.4f}s; negative % = faster than the reference)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # The fixed config -- same flags and defaults as opt_single_config.py.
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
    parser.add_argument("--inner-bandwidth-gbps", type=float, default=7200.0, help="Host-to-interposer bandwidth (default: NVLink 5 class).")
    parser.add_argument("--inner-latency-ns", type=float, default=1000.0)
    parser.add_argument("--outer-bandwidth-gbps", type=float, default=400.0, help="Per-interposer outer budget that the ring ratios split across the DP/CP/PP rings.")
    parser.add_argument("--outer-latency-ns", type=float, default=1000.0)
    parser.add_argument("--peak-perf-tflops", type=float, default=2250.0, help="Default: B200 BF16 dense.")
    parser.add_argument("--local-mem-bw-gbyteps", type=float, default=8000.0, help="Local memory (HBM) bandwidth in GB/s (gigaBYTES/sec, not gigabits). Default: B200 HBM3e.")
    parser.add_argument("--fixed-overhead-ns", type=float, default=0.0)
    # The ratio sweep.
    parser.add_argument("--step", type=float, default=0.05, help="Grid spacing over the ratio simplex; must divide 1 (default: 0.05).")
    parser.add_argument("--min-ratio", type=float, default=None, help="Smallest share any active ring may get (default: --step). Must be > 0.")
    for axis in AXES:
        parser.add_argument(f"--{axis}-range", type=float, nargs=2, default=[0.0, 1.0], metavar=("LO", "HI"),
                            help=f"Only grid points with {axis}_ratio in [LO, HI] (default: 0 1). Use to zoom in around a good region.")
    parser.add_argument("--ratios", type=str, nargs="+", default=None, metavar="DP,CP,PP",
                        help="Explicit ratio points instead of the grid, e.g. --ratios 0.8,0.2,0 0.75,0.25,0")
    parser.add_argument("--no-formula-baseline", action="store_true", help="Don't add the ring_bandwidth_ratios_from_formulas split as a baseline point.")
    parser.add_argument("--reference-time-sec", type=float, default=None, help="A finish time to compare against in the summary, e.g. the best DGX config's.")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Base directory: shared trace/ plus one exp dir per ratio point, and the results CSV. Required unless --dry-run.")
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() // 2), help="Parallel simulations. Each loads the full trace, so memory, not cores, is usually the limit at full scale.")
    add_shard_args(parser)
    parser.add_argument("--trace-only", action="store_true", help="Only generate the shared trace and exit (run once before launching shards).")
    parser.add_argument("--dry-run", action="store_true", help="Only list the ratio points that would run, then exit.")
    args = parser.parse_args()

    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is given.")
    if not divisibility_ok(args.global_batch_size, args.seq_len, args.dp, args.cp, args.pp, args.tp, args.microbatch_size,
                           args.layers, args.kv_heads, args.ffn_dim):
        parser.error("This config fails trace_generator.py's divisibility/shape checks (see divisibility_ok in sweep_common.py).")

    config = {
        "dp": args.dp, "cp": args.cp, "pp": args.pp, "tp": args.tp,
        "microbatch_size": args.microbatch_size,
        "recompute_activations": args.recompute_activations,
        "reuse_fwd_weights": args.reuse_fwd_weights,
        "axis_order": REPRESENTATIVE_AXIS_ORDER,
    }
    sizes = {"dp": args.dp, "cp": args.cp, "pp": args.pp}
    active = [a for a in AXES if sizes[a] > 1]
    if not active:
        parser.error("dp, cp and pp are all 1 -- there are no outer rings to split bandwidth between.")
    pinned = [a for a in AXES if a not in active]

    min_ratio = args.min_ratio if args.min_ratio is not None else args.step
    if min_ratio <= 0:
        parser.error("--min-ratio must be > 0: an active ring with no bandwidth could never finish.")
    try:
        if args.ratios:
            grid = parse_explicit_ratios(args.ratios, active)
        else:
            grid = ratio_grid(active, args.step, min_ratio, {a: getattr(args, f"{a}_range") for a in AXES})
    except ValueError as e:
        parser.error(str(e))
    points = [(r, "explicit" if args.ratios else "grid") for r in grid]

    if not args.no_formula_baseline:
        f_dp, f_cp, f_pp = ring_bandwidth_ratios_from_formulas(
            num_layers=args.layers, kv_heads=args.kv_heads, q_heads=args.q_heads, head_dim=args.head_dim,
            ffn_dim=args.ffn_dim, vocab=args.vocab, seq_len=args.seq_len, global_batch=args.global_batch_size,
            microbatch_size=args.microbatch_size, dp_degree=args.dp, cp_degree=args.cp, pp_degree=args.pp, tp_degree=args.tp,
            bytes_per_element=args.bytes_per_element,
            recompute_activations=args.recompute_activations, reuse_fwd_fsdp=args.reuse_fwd_weights,
        )
        formula = {"dp": f_dp, "cp": f_cp, "pp": f_pp}
        if not any(ratio_tag(r) == ratio_tag(formula) for r, _ in points):
            points.insert(0, (formula, "formula"))

    print(f"Config: dp={args.dp} cp={args.cp} pp={args.pp} tp={args.tp} mb={args.microbatch_size} "
          f"recompute={args.recompute_activations} reuse={args.reuse_fwd_weights}")
    print(f"Swept rings: {', '.join(active)}" + (f"; pinned to 0 (size 1, no ring): {', '.join(pinned)}" if pinned else ""))
    print(f"{len(points)} ratio points" + (" (including the formula baseline)" if not args.no_formula_baseline else ""))
    if args.num_shards > 1:
        points = select_shard(points, args.shard_index, args.num_shards)
        print(f"Shard {args.shard_index} of {args.num_shards} (0-based): running {len(points)} of them.")
    if args.dry_run:
        for r, source in points:
            print(f"  dp={r['dp']:.4f} cp={r['cp']:.4f} pp={r['pp']:.4f}  ({source})")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    trace_kwargs = dict(
        global_batch=args.global_batch_size, seq_len=args.seq_len,
        num_layers=args.layers, kv_heads=args.kv_heads, q_heads=args.q_heads, head_dim=args.head_dim,
        ffn_dim=args.ffn_dim, vocab=args.vocab, bytes_per_element=args.bytes_per_element,
        single_comm=args.single_comm, fsdp_prefetch=args.fsdp_prefetch,
    )
    trace_path = ensure_trace(args.output_dir, config, trace_kwargs)
    if args.trace_only or not points:
        sys.exit(0)

    topo_kwargs = dict(
        inner_bandwidth_gbps=args.inner_bandwidth_gbps, inner_latency_ns=args.inner_latency_ns,
        outer_bandwidth_gbps=args.outer_bandwidth_gbps, outer_latency_ns=args.outer_latency_ns,
        peak_perf_tflops=args.peak_perf_tflops, local_mem_bw_gbyteps=args.local_mem_bw_gbyteps,
        fixed_overhead_ns=args.fixed_overhead_ns,
    )
    print(f"Simulating {len(points)} ratio points, {args.max_workers} at a time...")
    results = run_points(points, args.max_workers, config=config, trace_path=trace_path, output_dir=args.output_dir,
                         trace_kwargs=trace_kwargs, topo_kwargs=topo_kwargs, run_info=describe_run(args))

    csv_path = results_csv_path(args.output_dir, args.shard_index, args.num_shards)
    write_results_csv(sorted(results, key=lambda r: r.get("finish_time_sec", float("inf"))), csv_path)
    print(f"\nWrote {len(results)} results to {csv_path}")
    print_summary(results, args.reference_time_sec)

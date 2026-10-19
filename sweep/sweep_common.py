"""Shared plumbing for opt_sweep.py and dgx_sweep.py: the parallelism-config
symbol set, the cheap mem_footprint.py-only feasibility pass, the
trace_generator.py subprocess + simulate()/classify_overlap() run, and the
parallel-worker harness + CSV/console reporting.

Topology generation itself (generate_opt_edgelist vs. generate_dgx_edgelist)
stays in each sweep script, not here -- their parameter shapes are genuinely
different (opt needs ring_bandwidth_ratios and has no npus_per_domain choice,
since tp is always the whole interposer; dgx needs an explicit
npus_per_domain and has no ring ratios). Everything else is identical.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import sympy

from tracegen import mem_footprint
from analytical.topology import parse_edgelist
from analytical.trace_io import load_trace
from analytical.analytical_model import simulate, classify_overlap, walltime_recorder

TRACE_GENERATOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "tracegen", "trace_generator.py"
)

L, H_k, H_q, D_h, D_ff, V = sympy.symbols("L H_k H_q D_h D_ff V")
dp_sym, cp_sym, pp_sym, tp_sym, B_sym, S_sym, M_sym = sympy.symbols("dp cp pp tp B S M")

POW2_RANGE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MICROBATCH_RANGE = [1, 2, 4, 8, 16, 32, 64, 128]
BYTES_TO_GB = 2**30


def divisibility_ok(global_batch: int, seq_len: int, dp: int, cp: int, pp: int, tp: int, mb: int, num_layers: int, kv_heads: int, ffn_dim: int) -> bool:
    """Mirrors trace_generator.py's check_divisibility -- filtering these out
    here avoids launching a trace_generator.py subprocess just to watch it
    fail an assert."""
    if global_batch % dp != 0:
        return False
    if (global_batch // dp) % mb != 0:
        return False
    if num_layers % pp != 0:
        return False
    if kv_heads % tp != 0:
        return False
    if ffn_dim % tp != 0:
        return False
    if seq_len % cp != 0:
        return False
    num_microbatches = global_batch // (mb * dp)
    if num_microbatches < pp:
        return False
    # More than a few pipeline flushes' worth of microbatches per step is a
    # degenerate regime for this sweep (tiny microbatches dominated by
    # per-op fixed overhead) rather than a realistic training config.
    if num_microbatches > 3 * pp:
        return False
    # Both pp and num_microbatches multiply directly into trace size (every
    # PP transition and every per-layer collective repeats once per
    # microbatch) -- without --single-comm, a large pp combined with a large
    # microbatch count has been observed to generate multi-million-node
    # traces that blow up trace_generator.py's memory well past what any
    # single worker should use (e.g. pp=128, 128 microbatches: still
    # climbing past 5.9GB after 2+ minutes, not yet finished). Capping both
    # independently keeps every surviving config's trace a sane size
    # regardless of --single-comm.
    if pp > 32:
        return False
    if num_microbatches > 64:
        return False
    return True


def memory_feasible_base_configs(
    num_npus: int,
    mem_per_npu_gb: float,
    global_batch: int,
    seq_len: int,
    num_layers: int,
    kv_heads: int,
    q_heads: int,
    head_dim: int,
    ffn_dim: int,
    vocab: int,
    bytes_per_element: int,
    max_tp: int | None = None,
    min_tp: int = 1,
) -> list[dict]:
    """All (dp, cp, pp, tp, microbatch_size, recompute_activations,
    reuse_fwd_weights) combos with dp*cp*pp*tp == num_npus, passing
    divisibility_ok, and whose mem_footprint.py-estimated peak per-rank
    memory fits mem_per_npu_gb. Topology-agnostic -- no axis_order or
    npus_per_domain here, since memory footprint doesn't depend on either
    (see mem_footprint.py's own LayerReprForFoorprint -- it only cares about
    dp/cp/pp/tp/microbatch_size/recompute/reuse). Each sweep script expands
    these into its own topology-specific variants afterwards.

    max_tp caps tp directly -- opt_sweep.py passes its --inner-dim-size here,
    since tp always equals the whole interposer there. dgx_sweep.py passes
    None: no single axis is inherently capped for dgx, since any trailing
    suffix of its --axis-order can form the inner domain; the cap there
    applies to the resulting *domain size*, enforced separately once
    axis_order is chosen (see dgx_sweep.py's inner_axes_for_domain_size use).

    min_tp floors tp instead -- both sweep scripts pass min(8, their own
    --inner-dim-size), so a config isn't considered just because dp*cp*pp*tp
    divides evenly with some tiny tp: below 8 only makes sense if the inner
    domain itself can't even fit 8 NPUs.
    """
    mem_footprint.SUBSTITUTE_VALUES.clear()
    mem_footprint.SUBSTITUTE_VALUES.update({
        L: num_layers, H_k: kv_heads, H_q: q_heads, D_h: head_dim, D_ff: ffn_dim, V: vocab,
        S_sym: seq_len, B_sym: global_batch,
    })

    configs = []
    for dp in POW2_RANGE:
        for cp in POW2_RANGE:
            for pp in POW2_RANGE:
                for tp in POW2_RANGE:
                    if dp * cp * pp * tp != num_npus:
                        continue
                    if max_tp is not None and tp > max_tp:
                        continue
                    if tp < min_tp:
                        continue
                    if pp > num_layers:
                        continue
                    mem_footprint.SUBSTITUTE_VALUES[dp_sym] = dp
                    mem_footprint.SUBSTITUTE_VALUES[cp_sym] = cp
                    mem_footprint.SUBSTITUTE_VALUES[pp_sym] = pp
                    mem_footprint.SUBSTITUTE_VALUES[tp_sym] = tp
                    for mb in MICROBATCH_RANGE:
                        if not divisibility_ok(global_batch, seq_len, dp, cp, pp, tp, mb, num_layers, kv_heads, ffn_dim):
                            continue
                        mem_footprint.SUBSTITUTE_VALUES[M_sym] = mb
                        for recompute in (True, False):
                            for reuse in (True, False):
                                layer_repr = mem_footprint.LayerReprForFoorprint(activation_recompute=recompute, reuse_fwd_weights=reuse)
                                result = layer_repr.peak_rank_memory(
                                    pp_degree=pp, dp_degree=dp, layers_per_stage=num_layers // pp, bytes_per_element=bytes_per_element,
                                )
                                if result["total_peak_bytes"] > mem_per_npu_gb * BYTES_TO_GB:
                                    continue
                                configs.append({
                                    "dp": dp, "cp": cp, "pp": pp, "tp": tp,
                                    "microbatch_size": mb,
                                    "recompute_activations": recompute,
                                    "reuse_fwd_weights": reuse,
                                    "predicted_peak_bytes": result["total_peak_bytes"],
                                })
    return configs


def exp_dir_name(config: dict) -> str:
    ao = "-".join(config["axis_order"]) if not isinstance(config["axis_order"], str) else config["axis_order"].replace(",", "-")
    npd = f"_npd{config['npus_per_domain']}" if "npus_per_domain" in config else ""
    return (
        f"dp{config['dp']}_cp{config['cp']}_pp{config['pp']}_tp{config['tp']}_ao-{ao}{npd}"
        f"_mb{config['microbatch_size']}_recompute{int(config['recompute_activations'])}_reuse{int(config['reuse_fwd_weights'])}"
    )


def generate_trace(
    exp_dir: str,
    config: dict,
    *,
    global_batch: int,
    seq_len: int,
    num_layers: int,
    kv_heads: int,
    q_heads: int,
    head_dim: int,
    ffn_dim: int,
    vocab: int,
    bytes_per_element: int,
    single_comm: bool,
) -> str:
    """Runs trace_generator.py as a subprocess for this config (kept a
    subprocess rather than an in-process call: Node._next_id/_all_nodes are
    unreset class-level state, unsafe to call repeatedly in the same worker
    process -- see opt_sweep.py's original walltime_recorder lesson, same
    class of issue). Writes trace.json.zst + trace_generator.log into
    exp_dir. Returns the trace path; raises RuntimeError (pointing at the
    log) on failure.
    """
    trace_path = os.path.join(exp_dir, "trace.json.zst")
    axis_order = config["axis_order"]
    cmd = [
        sys.executable, TRACE_GENERATOR_PATH,
        "--layers", str(num_layers),
        "--kv-heads", str(kv_heads),
        "--q-heads", str(q_heads),
        "--head-dim", str(head_dim),
        "--ffn-dim", str(ffn_dim),
        "--vocab", str(vocab),
        "--seq-len", str(seq_len),
        "--batch-size", str(global_batch),
        "--microbatch-size", str(config["microbatch_size"]),
        "--dp", str(config["dp"]), "--cp", str(config["cp"]), "--pp", str(config["pp"]), "--tp", str(config["tp"]),
        "--axis-order", ",".join(axis_order),
        "--bytes-per-element", str(bytes_per_element),
        "--output", trace_path,
    ]
    if config["recompute_activations"]:
        cmd.append("--recompute-activations")
    if config["reuse_fwd_weights"]:
        cmd.append("--reuse-fwd-fsdp")
    if single_comm:
        cmd.append("--single-comm")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(exp_dir, "trace_generator.log"), "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"trace_generator.py failed (see {exp_dir}/trace_generator.log):\n{proc.stderr[-2000:]}")
    return trace_path


def simulate_and_report(
    edgelist_path: str,
    trace_path: str,
    exp_dir: str,
    *,
    peak_perf_tflops: float,
    local_mem_bw_gbyteps: float,
    fixed_overhead_ns: float,
    global_batch: int,
    seq_len: int,
    extra_result_fields: dict,
) -> dict:
    """Loads edgelist_path + trace_path, resets walltime_recorder (module-
    level singleton inside analytical_model.py whose start() asserts a timer
    name was never used before -- fine for a single CLI run, but simulate()
    reuses those same fixed names every call, and ProcessPoolExecutor reuses
    worker processes across configs, so a second simulate() in the same
    worker hits "already recorded" without this), runs simulate() +
    classify_overlap(), and returns a result dict merging
    extra_result_fields with the measured metrics.

    simulate() runs with verbose=False/progress=True and its stdout (the
    "Parsing Edgelist"/"Preprocess nodes" progress lines, plus periodic
    "completed X/Y nodes" updates during the simpy phase) redirected to
    exp_dir/analytical_model.log -- mirroring generate_trace's
    exp_dir/trace_generator.log, since unlike that function (which shells
    out to a subprocess and can just capture its stdout), simulate() runs
    in-process here, so nothing would otherwise be captured at all.
    progress rather than verbose specifically: per-node issue/complete
    would be one line per node per event -- millions for an ungated
    config -- where progress caps it at ~200 lines regardless of trace
    size.

    analytical_model_wall_time_sec in the result is real (wall-clock) time
    spent in THIS function -- topology parse, trace load, routing,
    preprocess, the simpy run, and classify_overlap -- not to be confused
    with finish_time_sec, which is the model's own SIMULATED output (how
    long the traced training step takes on the modeled hardware).
    """
    wall_time_start = time.time()
    topo = parse_edgelist(edgelist_path)
    graph = load_trace(trace_path)

    walltime_recorder.timers.clear()

    log_path = os.path.join(exp_dir, "analytical_model.log")
    with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file):
        finish_time_sec = simulate(
            topo, graph, peak_perf_tflops, local_mem_bw_gbyteps,
            verbose=False, track_overlap=True, fixed_overhead_sec=fixed_overhead_ns * 1e-9,
            progress=True,
        )
    breakdown = classify_overlap(graph)
    analytical_model_wall_time_sec = time.time() - wall_time_start

    tokens_per_batch = global_batch * seq_len
    return {
        **extra_result_fields,
        "tokens_per_batch": tokens_per_batch,
        "finish_time_sec": finish_time_sec,
        "tokens_per_sec": tokens_per_batch / finish_time_sec,
        "exposed_comp_sec": breakdown["exposed_comp"],
        "exposed_comm_sec": breakdown["exposed_comm"],
        "overlapped_sec": breakdown["overlapped"],
        "idle_sec": breakdown["idle"],
        "analytical_model_wall_time_sec": analytical_model_wall_time_sec,
    }


def flatten_config_for_result(config: dict) -> dict:
    """axis_order as a comma string (not a tuple) for CSV/JSON -- the
    canonical post-run representation every result dict (success or error)
    uses."""
    flat = {k: v for k, v in config.items() if k != "axis_order"}
    flat["axis_order"] = ",".join(config["axis_order"])
    return flat


def run_one_config(config: dict, output_dir: str, run_fn, run_kwargs: dict) -> dict:
    """Runs one config in a worker process via run_fn(config, exp_dir,
    **run_kwargs) -> dict (each sweep script's own topology-specific
    function), capturing any failure into the result dict's "error" key
    instead of aborting the rest of the sweep.
    """
    exp_dir = os.path.join(output_dir, exp_dir_name(config))
    os.makedirs(exp_dir, exist_ok=True)
    try:
        result = run_fn(config, exp_dir, **run_kwargs)
        with open(os.path.join(exp_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        return {**flatten_config_for_result(config), "error": str(e), "exp_dir": exp_dir}


def run_sweep(configs: list[dict], output_dir: str, run_fn, run_kwargs: dict, max_workers: int) -> list[dict]:
    """Runs every config in configs across a worker pool, printing one
    progress line per completion (see run_opt_sweep.log-style output).

    All configs are submitted up front, so a plain `with ProcessPoolExecutor()
    as executor:` here would be wrong for Ctrl+C: the context manager's
    implicit shutdown(wait=True) (its default) doesn't cancel the
    not-yet-started futures still sitting in the queue -- it keeps feeding
    them to workers as each one frees up, which is exactly "new processes
    keep getting spawned after I hit Ctrl+C". Catching KeyboardInterrupt here
    instead cancels every future that hasn't started yet AND kills the
    worker processes that are already mid-config -- cancel_futures alone
    doesn't stop a worker that's already deep inside, say, a long-running
    simulate() call; only killing the OS process does. executor
    ._processes is private API, but there's no public "kill everything now"
    on ProcessPoolExecutor, and leaving a multi-hundred-GB routing table
    computation to run to completion just because Ctrl+C was pressed isn't
    an acceptable alternative.
    """
    results = []
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(run_one_config, config, output_dir, run_fn, run_kwargs) for config in configs]
    try:
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = "ERROR" if "error" in result else f"{result['finish_time_sec']:.6f}s"
            print(f"[{i}/{len(futures)}] {exp_dir_name(result)} -> {status}")
            results.append(result)
    except KeyboardInterrupt:
        print("\nInterrupted -- cancelling queued configs and killing active worker processes...")
        for future in futures:
            future.cancel()
        for process in executor._processes.values():
            process.kill()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return results


def write_results_csv(results: list[dict], path: str) -> None:
    fieldnames = []
    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_best(results: list[dict], global_batch: int, seq_len: int, extra_fields: list[str] = ()) -> None:
    succeeded = [r for r in results if "error" not in r]
    if not succeeded:
        print("\nAll configs errored -- see results.csv / each exp dir's trace_generator.log.")
        return
    best = min(succeeded, key=lambda r: r["finish_time_sec"])
    print(f"\nFastest config for {global_batch * seq_len:,} tokens/batch:")
    print(f"  dp={best['dp']} cp={best['cp']} pp={best['pp']} tp={best['tp']} axis_order={best['axis_order']}" + "".join(f" {f}={best[f]}" for f in extra_fields))
    print(f"  microbatch_size={best['microbatch_size']} recompute_activations={bool(best['recompute_activations'])} reuse_fwd_weights={bool(best['reuse_fwd_weights'])}")
    print(f"  finish_time_sec={best['finish_time_sec']:.6f}  tokens_per_sec={best['tokens_per_sec']:.2f}")
    print(f"  exp_dir={best['exp_dir']}")

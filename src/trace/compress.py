"""Compress/decompress trace JSON via the system zstd binary (no zstandard
pip dependency needed -- /usr/bin/zstd is assumed present).

trace_generator.py uses write_compressed_trace to serialize straight to
.json.zst, so the uncompressed form never touches disk at all. Readers that
need the trace's actual content (trace_io.py, used by analytical_model.py /
lint_trace.py / process_trace.py) use load_trace_json, which decompresses
entirely in memory -- it never writes a decompressed copy either. The only
way a plain .json file gets (re)created is decompress_to_file, for manual
inspection, or by running this module as a script.

zstd -3 (the default here) was measured at ~47x on a representative trace
(1.88GB -> 39.7MB in under a second, multi-threaded) -- see the project's own
notes for the comparison against gzip and zstd -19; -3 was chosen since -19's
much longer compress time (~2 min on that same file) bought comparatively
little extra ratio for this workload's generation path, where the file is
written once per run. Pass level= explicitly (e.g. for a one-off archival
compress_file call) if you want higher ratio and don't mind the wait.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Dict

DEFAULT_LEVEL = 3
COMPRESSED_SUFFIX = ".zst"


def is_compressed(path: str) -> bool:
    return path.endswith(COMPRESSED_SUFFIX)


def write_compressed_trace(json_obj: Dict[str, Any], path: str, level: int = DEFAULT_LEVEL) -> None:
    """Serializes json_obj and writes it zstd-compressed to `path` directly --
    no intermediate uncompressed file is ever created. `path` should end in
    .json.zst (not enforced, just the convention load_trace_json expects).

    json.dumps(), not json.dump(): CPython's json.dump() never uses the
    C-accelerated encoder (it calls iterencode() without _one_shot=True,
    which the C encoder requires) regardless of indent, while json.dumps()
    passes _one_shot=True by default -- measured ~5-8x faster on a
    representative multi-million-node trace payload.
    """
    payload = json.dumps(json_obj).encode()
    subprocess.run(
        ["zstd", f"-{level}", "-T0", "-q", "-f", "-o", path],
        input=payload,
        check=True,
    )
    # Compressing from stdin (no source file for zstd to copy permissions
    # from) leaves the output at a restrictive 600 -- normalize to the usual
    # umask-determined mode instead of silently handing back a file only the
    # writing user can read.
    os.chmod(path, 0o644)


def load_trace_json(path: str) -> Dict[str, Any]:
    """Loads a trace from `path`. Transparently decompresses in memory if
    `path` is zstd-compressed (.json.zst) -- the decompressed bytes are piped
    straight into json.loads and never written to disk. Plain .json still
    works unchanged.
    """
    if is_compressed(path):
        proc = subprocess.run(["zstd", "-d", "-q", "-c", path], capture_output=True, check=True)
        return json.loads(proc.stdout)
    with open(path) as f:
        return json.load(f)


def compress_file(json_path: str, zst_path: str | None = None, level: int = DEFAULT_LEVEL) -> str:
    """Compresses an existing .json file on disk to .json.zst. Used by this
    module's CLI for one-off compression of files that already exist;
    trace_generator.py itself should prefer write_compressed_trace so the
    uncompressed form is never written in the first place. Returns the
    output path.
    """
    if zst_path is None:
        zst_path = json_path + COMPRESSED_SUFFIX
    subprocess.run(["zstd", f"-{level}", "-T0", "-q", "-f", json_path, "-o", zst_path], check=True)
    return zst_path


def decompress_to_file(zst_path: str, json_path: str | None = None) -> str:
    """Materializes a human-readable .json file from a .json.zst trace, for
    manual inspection -- not part of analytical_model's own read path
    (load_trace_json decompresses in memory instead). Returns the output path.
    """
    if json_path is None:
        json_path = zst_path[: -len(COMPRESSED_SUFFIX)] if is_compressed(zst_path) else zst_path + ".json"
    subprocess.run(["zstd", "-d", "-q", "-f", zst_path, "-o", json_path], check=True)
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress/decompress trace JSON files with zstd.")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compress", help="Compress a .json trace to .json.zst")
    c.add_argument("input", help="Path to the .json trace")
    c.add_argument("-o", "--output", default=None, help="Output path (default: input + .zst)")
    c.add_argument("--level", type=int, default=DEFAULT_LEVEL, help=f"zstd compression level (default: {DEFAULT_LEVEL})")

    d = sub.add_parser("decompress", help="Decompress a .json.zst trace to .json, for viewing")
    d.add_argument("input", help="Path to the .json.zst trace")
    d.add_argument("-o", "--output", default=None, help="Output path (default: input minus .zst)")

    args = parser.parse_args()
    if args.command == "compress":
        out = compress_file(args.input, args.output, level=args.level)
    else:
        out = decompress_to_file(args.input, args.output)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

"""Collect per-config result.json files into one results CSV.

    python extract_results.py exp_new_sweep_over_opt/*/result.json
    python extract_results.py exp_new_sweep_over_dgx/*/result.json -o dgx_results.csv

Rows are sorted fastest first (by finish_time_sec).
"""
import argparse
import csv
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_files", nargs="+", help="result.json files to combine.")
    parser.add_argument("-o", "--output", default="results.csv", help="Output CSV path (default: results.csv).")
    args = parser.parse_args()

    results = []
    for path in args.result_files:
        with open(path) as f:
            results.append(json.load(f))
    results.sort(key=lambda r: r.get("finish_time_sec", float("inf")))

    fieldnames = []
    for r in results:
        for key in r:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()

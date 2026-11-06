#!/usr/bin/env python3
"""Write and summarize model fit/predict wall-clock timings."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics


FIELDS = (
    "scenario",
    "config",
    "variable",
    "model",
    "phase",
    "samples",
    "seconds",
)


def append_model_timing(
    path,
    scenario,
    config,
    variable,
    model,
    phase,
    samples,
    seconds,
):
    """Append one tab-separated timing record to a config-specific file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "scenario": scenario,
                "config": config,
                "variable": variable,
                "model": model,
                "phase": phase,
                "samples": int(samples),
                "seconds": f"{float(seconds):.9f}",
            }
        )


def summarize(results_dir, output):
    """Summarize all config-specific timing files in one results directory."""
    results_dir = Path(results_dir)
    grouped = defaultdict(list)
    files = sorted(results_dir.glob("model_timing_*.txt"))
    for path in files:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                key = (row["scenario"], row["model"], row["phase"])
                grouped[key].append(float(row["seconds"]))

    if not grouped:
        raise FileNotFoundError(
            f"No model_timing_*.txt files were found in {results_dir}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            (
                "scenario",
                "model",
                "phase",
                "count",
                "mean_seconds",
                "std_seconds",
                "min_seconds",
                "max_seconds",
            )
        )
        for key in sorted(grouped):
            values = grouped[key]
            writer.writerow(
                (
                    *key,
                    len(values),
                    f"{statistics.fmean(values):.9f}",
                    f"{statistics.pstdev(values):.9f}",
                    f"{min(values):.9f}",
                    f"{max(values):.9f}",
                )
            )
    return len(files)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize model fit/predict timing files."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = summarize(args.results_dir, args.output)
    print(f"Summarized {count} timing files in {args.output}")


if __name__ == "__main__":
    main()

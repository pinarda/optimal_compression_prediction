#!/usr/bin/env python3
"""Write final compression-label counts for every variable in a label cache."""

import argparse
import csv
import json
from pathlib import Path


PAPER_SUBMIT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PAPER_SUBMIT_DIR / "data" / "precomputed_labels"
DEFAULT_METRICS = ["dssim", "pcc"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a label-distribution table directly from precomputed label "
            "caches; model-result files are not required."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing label_manifest.json and profile directories",
    )
    parser.add_argument(
        "--profile",
        help="Profile name from label_manifest.json (selected automatically by default)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics required in the automatically selected profile",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
        help="Timestep count required in the selected cache profile (default: 2000)",
    )
    parser.add_argument(
        "--subset",
        choices=("first-1600", "model-test", "all"),
        default="first-1600",
        help=(
            "first-1600 uses timesteps 1--1600; model-test uses the final test "
            "portion after the historical 10%% train + 10%% validation split; "
            "all uses every cached timestep"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path",
    )
    return parser.parse_args()


def select_profile(manifest, requested_name, metrics, timesteps):
    profiles = manifest.get("profiles", [])
    if requested_name:
        matches = [profile for profile in profiles if profile["name"] == requested_name]
        if len(matches) != 1:
            raise ValueError(
                f"Profile {requested_name!r} was not found exactly once in the manifest"
            )
        return matches[0]

    candidates = [
        profile
        for profile in profiles
        if set(profile["spec"]["metrics"]) == set(metrics)
        and profile["spec"]["timesteps"] == timesteps
    ]
    if not candidates:
        raise ValueError(
            f"No time-{timesteps} manifest profile uses metrics {metrics}"
        )

    max_levels = max(len(profile["spec"]["comp_dirs"]) for profile in candidates)
    candidates = [
        profile
        for profile in candidates
        if len(profile["spec"]["comp_dirs"]) == max_levels
    ]
    if len(candidates) != 1:
        names = ", ".join(profile["name"] for profile in candidates)
        raise ValueError(
            f"Automatic profile selection is ambiguous ({names}); use --profile"
        )
    return candidates[0]


def select_samples(labels, spec, subset):
    timesteps = spec["timesteps"]
    collections = len(spec["sub_dirs"])
    expected = collections * timesteps
    if labels.size != expected:
        raise ValueError(f"Cache has {labels.size} labels; expected {expected}")

    by_collection = labels.reshape(collections, timesteps)
    if subset == "all":
        return by_collection.reshape(-1)
    if subset == "first-1600":
        if timesteps < 1600:
            raise ValueError(
                f"The selected profile has only {timesteps} timesteps; 1600 are required"
            )
        return by_collection[:, :1600].reshape(-1)

    train_count = int(0.10 * timesteps)
    remaining = timesteps - train_count
    validation_count = remaining // 9
    test_start = train_count + validation_count
    return by_collection[:, test_start:].reshape(-1)


def load_counts(cache_dir, profile, subset):
    import numpy as np

    spec = profile["spec"]
    levels = list(spec["comp_dirs"]) + [spec["fallback_label"]]
    rows = []
    for variable in sorted(profile["variables"], key=str.casefold):
        cache = cache_dir / profile["name"] / f"{variable}_labels.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"Missing label cache: {cache}")
        with np.load(cache, allow_pickle=False) as archive:
            labels = np.asarray(archive["final_labels"]).astype(str).reshape(-1)
        labels = select_samples(labels, spec, subset)
        unknown = sorted(set(labels) - set(levels))
        if unknown:
            raise ValueError(f"Unexpected labels in {cache}: {unknown}")
        counts = [int(np.count_nonzero(labels == level)) for level in levels]
        rows.append((variable, labels.size, counts))
    return levels, rows


def write_counts(path, levels, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variable", *levels])
        for variable, _sample_count, counts in rows:
            writer.writerow([variable, *counts])


def main():
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / "label_manifest.json"
    with manifest_path.open() as stream:
        manifest = json.load(stream)

    profile = select_profile(
        manifest,
        args.profile,
        args.metrics,
        args.timesteps,
    )
    levels, rows = load_counts(cache_dir, profile, args.subset)
    output = args.output
    if output is None:
        output = (
            PAPER_SUBMIT_DIR
            / "data"
            / f"label_distribution_{profile['name']}_{args.subset.replace('-', '_')}.csv"
        )
    output = output.resolve()

    write_counts(output, levels, rows)
    print(f"Profile: {profile['name']}")
    print(f"Variables: {len(rows)}")
    print(f"Labels per variable: {rows[0][1] if rows else 0}")
    print(f"Table: {output}")


if __name__ == "__main__":
    main()

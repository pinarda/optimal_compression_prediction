#!/usr/bin/env python3
"""Generate every unique paper labeling profile from combined metric files."""

import argparse
import json
import logging
import os
from pathlib import Path

from precomputed_labeling import (
    DEFAULT_CACHE_DIR,
    DEFAULT_METRICS_DIR,
    DEFAULT_METRICS_INFO,
    compute_and_cache_labels,
    label_cache_path,
    label_profile_name,
    label_profile_spec,
    quality_metric_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create raw classification-label caches for all unique single- and "
            "multi-variable config profiles."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing rotated_config_*.json and multi_config_*.json",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=[],
        help="Use this config instead of scanning --config-dir; may be repeated",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory containing *_quality_metrics_time2000.nc files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory in which to write precomputed label caches",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate valid existing caches",
    )
    parser.add_argument(
        "--skip-variable",
        action="append",
        default=[],
        help="Variable to omit explicitly; may be repeated",
    )
    return parser.parse_args()


def flatten_variables(items):
    variables = []
    for item in items:
        if isinstance(item, list):
            variables.extend(flatten_variables(item))
        else:
            variables.append(item)
    return variables


def config_paths(args):
    if args.config:
        return [path.resolve() for path in args.config]
    paths = list(args.config_dir.glob("rotated_config_*.json"))
    paths.extend(args.config_dir.glob("multi_config_*.json"))
    return sorted(path.resolve() for path in paths)


def collect_profiles(paths):
    profiles = {}
    for path in paths:
        with path.open() as stream:
            config = json.load(stream)
        if len(config["Times"]) != 1:
            raise ValueError(f"{path} must contain exactly one Times value")
        spec = label_profile_spec(
            int(config["Times"][0]),
            config["SubDirs"],
            config["CompDirs"],
            config["Metric"],
            DEFAULT_METRICS_INFO,
        )
        key = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        profile = profiles.setdefault(
            key,
            {"spec": spec, "variables": [], "configs": []},
        )
        profile["configs"].append(path.name)
        for variable in flatten_variables(config["VarList"]):
            if variable not in profile["variables"]:
                profile["variables"].append(variable)
    return list(profiles.values())


def write_manifest(output_dir, metrics_dir, profiles, skipped_variables):
    manifest = {
        "schema_version": 1,
        "metrics_dir": str(metrics_dir),
        "skipped_variables": skipped_variables,
        "profiles": profiles,
    }
    destination = output_dir / "label_manifest.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    paths = config_paths(args)
    if not paths:
        raise FileNotFoundError(
            f"No rotated_config_*.json or multi_config_*.json files in {args.config_dir}"
        )
    profiles = collect_profiles(paths)
    skipped_variables = list(dict.fromkeys(args.skip_variable))
    available_variables = {
        variable
        for profile in profiles
        for variable in profile["variables"]
    }
    unknown_skips = set(skipped_variables) - available_variables
    if unknown_skips:
        raise ValueError(
            f"Skipped variables are not present in the configs: {sorted(unknown_skips)}"
        )
    for profile in profiles:
        profile["skipped_variables"] = [
            variable
            for variable in profile["variables"]
            if variable in skipped_variables
        ]
        profile["variables"] = [
            variable
            for variable in profile["variables"]
            if variable not in skipped_variables
        ]
    if skipped_variables:
        logging.warning("Explicitly skipping variables: %s", skipped_variables)

    metrics_dir = args.metrics_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = sorted(
        {
            quality_metric_path(metrics_dir, variable)
            for profile in profiles
            for variable in profile["variables"]
            if not quality_metric_path(metrics_dir, variable).is_file()
        }
    )
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        remainder = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} combined quality-metric files:\n"
            f"{preview}{remainder}"
        )

    manifest_profiles = []
    total = sum(len(profile["variables"]) for profile in profiles)
    completed = 0
    for profile in profiles:
        spec = profile["spec"]
        name = label_profile_name(spec)
        logging.info(
            "Profile %s: %d variables, %d timesteps, metrics=%s, compressions=%s",
            name,
            len(profile["variables"]),
            spec["timesteps"],
            spec["metrics"],
            spec["comp_dirs"],
        )
        caches = []
        for variable in profile["variables"]:
            compute_and_cache_labels(
                variable=variable,
                timesteps=spec["timesteps"],
                sub_dirs=spec["sub_dirs"],
                comp_dirs=spec["comp_dirs"],
                metrics=spec["metrics"],
                metrics_dir=metrics_dir,
                cache_dir=output_dir,
                metrics_info=DEFAULT_METRICS_INFO,
                overwrite=args.overwrite,
            )
            completed += 1
            cache = label_cache_path(output_dir, variable, spec)
            caches.append(str(cache.relative_to(output_dir)))
            logging.info("[%d/%d] Ready: %s", completed, total, cache)
        manifest_profiles.append(
            {
                "name": name,
                "spec": spec,
                "variables": profile["variables"],
                "skipped_variables": profile["skipped_variables"],
                "configs": profile["configs"],
                "caches": caches,
            }
        )

    write_manifest(
        output_dir,
        metrics_dir,
        manifest_profiles,
        skipped_variables,
    )
    logging.info(
        "Finished %d label caches across %d unique profiles in %s",
        total,
        len(profiles),
        output_dir,
    )


if __name__ == "__main__":
    main()

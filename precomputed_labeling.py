"""Load combined quality metrics and cache the existing classification labels."""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr


CACHE_SCHEMA_VERSION = 2
FALLBACK_LABEL = "uncompressed"
DEFAULT_METRICS_INFO = {
    "dssim": {"comparison": "gt", "threshold": 0.995},
    "pcc": {"comparison": "gt", "threshold": 0.9995},
    "spre": {"comparison": "lt", "threshold": 95},
    "ks": {"comparison": "gt", "threshold": 0.05},
}
PAPER_SUBMIT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = PAPER_SUBMIT_DIR / "data" / "metrics"
DEFAULT_CACHE_DIR = PAPER_SUBMIT_DIR / "data" / "precomputed_labels"


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def label_profile_spec(
    timesteps,
    sub_dirs,
    comp_dirs,
    metrics,
    metrics_info=None,
):
    """Return the ordered inputs that uniquely define one labeling profile."""
    metrics_info = DEFAULT_METRICS_INFO if metrics_info is None else metrics_info
    if not isinstance(timesteps, int) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    if not sub_dirs:
        raise ValueError("SubDirs cannot be empty")
    if not comp_dirs:
        raise ValueError("CompDirs cannot be empty")
    if not metrics:
        raise ValueError("Metric cannot be empty")

    rules = {}
    for metric in metrics:
        if metric not in metrics_info:
            raise ValueError(f"No labeling rule is defined for metric {metric!r}")
        comparison = metrics_info[metric]["comparison"]
        if comparison not in ("gt", "lt"):
            raise ValueError(
                f"Unsupported comparison {comparison!r} for metric {metric!r}"
            )
        rules[metric] = {
            "comparison": comparison,
            "threshold": metrics_info[metric]["threshold"],
        }

    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "timesteps": timesteps,
        "sub_dirs": list(sub_dirs),
        "comp_dirs": list(comp_dirs),
        "metrics": list(metrics),
        "rules": rules,
        "fallback_label": FALLBACK_LABEL,
        "require_all_metrics_to_pass": True,
        "sample_order": "collection-major,timestep-minor",
    }


def label_profile_name(spec):
    digest = hashlib.sha256(_canonical_json(spec).encode("utf-8")).hexdigest()[:12]
    return f"time{spec['timesteps']}_{digest}"


def label_cache_path(cache_dir, variable, spec):
    return Path(cache_dir) / label_profile_name(spec) / f"{variable}_labels.npz"


def quality_metric_path(metrics_dir, variable):
    return Path(metrics_dir) / f"{variable}_quality_metrics_time2000.nc"


def source_metadata(path):
    path = Path(path)
    stat = path.stat()
    return {
        "filename": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _quality_requests(dataset, path):
    serialized = dataset.attrs.get("quality_requests")
    if serialized is None:
        raise ValueError(f"{path} has no quality_requests metadata")
    try:
        requests = json.loads(serialized) if isinstance(serialized, str) else serialized
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid quality_requests metadata in {path}") from error
    if not isinstance(requests, dict):
        raise ValueError(f"quality_requests in {path} is not a dictionary")
    return requests


def load_quality_metrics(
    path,
    variable,
    timesteps,
    sub_dirs,
    comp_dirs,
    metrics,
):
    """Load the combined metric file in the same order as the original code."""
    path = Path(path)
    expected_dims = ("collection", "compression", "timestep", "metric")
    collection_labels = [f"{subdir}_orig" for subdir in sub_dirs]

    with xr.open_dataset(path) as dataset:
        if "quality_metric_values" not in dataset:
            raise KeyError(f"quality_metric_values is missing from {path}")
        values = dataset["quality_metric_values"]
        if values.dims != expected_dims:
            raise ValueError(
                f"Unexpected quality metric dimensions in {path}: "
                f"{values.dims}; expected {expected_dims}"
            )
        stored_variable = dataset.attrs.get("variable")
        if stored_variable is not None and str(stored_variable) != variable:
            raise ValueError(
                f"{path} contains variable {stored_variable!r}, not {variable!r}"
            )

        available_collections = set(dataset["collection"].values.astype(str))
        available_compressions = set(dataset["compression"].values.astype(str))
        available_metrics = set(dataset["metric"].values.astype(str))
        missing_collections = set(collection_labels) - available_collections
        missing_compressions = set(comp_dirs) - available_compressions
        missing_metrics = set(metrics) - available_metrics
        if missing_collections:
            raise ValueError(
                f"Collections missing from {path}: {sorted(missing_collections)}"
            )
        if missing_compressions:
            raise ValueError(
                f"Compression levels missing from {path}: "
                f"{sorted(missing_compressions)}"
            )
        if missing_metrics:
            raise ValueError(f"Metrics missing from {path}: {sorted(missing_metrics)}")
        if dataset.sizes["timestep"] < timesteps:
            raise ValueError(
                f"{path} has {dataset.sizes['timestep']} timesteps; "
                f"{timesteps} are required"
            )

        expected_timestep = np.arange(1, timesteps + 1)
        actual_timestep = np.asarray(dataset["timestep"].values[:timesteps])
        if not np.array_equal(actual_timestep, expected_timestep):
            raise ValueError(
                f"The first {timesteps} timestep coordinates in {path} are not 1..{timesteps}"
            )

        requests = _quality_requests(dataset, path)
        for compression in comp_dirs:
            requested_metrics = requests.get(compression, [])
            for metric in metrics:
                if metric not in requested_metrics:
                    raise ValueError(
                        f"{path} did not compute requested pair "
                        f"({compression}, {metric}); its values may be placeholder NaNs"
                    )

        selected = values.sel(
            collection=collection_labels,
            compression=list(comp_dirs),
            metric=list(metrics),
        ).isel(timestep=slice(0, timesteps)).load()

    metrics_data = {}
    expected_length = len(collection_labels) * timesteps
    for metric in metrics:
        metrics_data[metric] = {}
        for compression in comp_dirs:
            by_collection = [
                np.asarray(
                    selected.sel(
                        collection=collection,
                        compression=compression,
                        metric=metric,
                    ).values
                ).reshape(-1)
                for collection in collection_labels
            ]
            combined = np.concatenate(by_collection)
            if combined.size != expected_length:
                raise ValueError(
                    f"Loaded {combined.size} values for {variable} {compression} "
                    f"{metric}; expected {expected_length}"
                )
            metrics_data[metric][compression] = combined
    return metrics_data


def generate_classification_labels(
    metrics_info,
    metrics_data,
    compression_level_order,
):
    """Apply the same labeling rules and ordering used by newmain*.py."""
    final_labels_dict = {}

    for metric, info in metrics_info.items():
        logging.info("Processing metric: %s", metric)
        labels = []
        compression_levels = []
        metric_values_dict = metrics_data.get(metric, {})
        if not metric_values_dict:
            logging.warning("No data for metric %r. Skipping.", metric)
            continue

        for comp_label, metric_values in metric_values_dict.items():
            if not isinstance(comp_label, str):
                comp_label = str(comp_label)
            compression_levels.append(comp_label)
            data_da = xr.DataArray(metric_values)
            if info["comparison"] == "gt":
                label_da = data_da >= info["threshold"]
            elif info["comparison"] == "lt":
                label_da = data_da < info["threshold"]
            else:
                raise ValueError(
                    f"Invalid comparison {info['comparison']!r} for metric {metric!r}"
                )
            labels.append(label_da)

        if not labels:
            final_labels_dict[metric] = None
            continue

        final_labels = xr.full_like(labels[0], "None", dtype="object")
        for comp_label, label_da in zip(compression_levels, labels):
            final_labels = xr.where(
                (final_labels == "None") & label_da,
                comp_label,
                final_labels,
            )
        final_labels_dict[metric] = final_labels

    if not final_labels_dict:
        raise ValueError("No classification labels could be generated")
    first_labels = next(
        (labels for labels in final_labels_dict.values() if labels is not None),
        None,
    )
    if first_labels is None:
        raise ValueError("No classification labels could be generated")

    combined_final_labels = xr.full_like(first_labels, "None", dtype="object")
    missing_metric_label = xr.full_like(first_labels, False, dtype=bool)
    for comp_level in compression_level_order[::-1]:
        for label_da in final_labels_dict.values():
            if label_da is None:
                continue
            combined_final_labels = xr.where(
                (combined_final_labels == "None") & (label_da == comp_level),
                comp_level,
                combined_final_labels,
            )
    for label_da in final_labels_dict.values():
        if label_da is None:
            missing_metric_label = xr.full_like(first_labels, True, dtype=bool)
        else:
            missing_metric_label = missing_metric_label | (label_da == "None")
    combined_final_labels = xr.where(
        (combined_final_labels == "None") | missing_metric_label,
        FALLBACK_LABEL,
        combined_final_labels,
    )
    return combined_final_labels, final_labels_dict


def _label_arrays(final_labels, per_metric_labels):
    final_array = np.asarray(final_labels.values).astype(str).reshape(-1)
    metric_arrays = {
        metric: np.asarray(labels.values).astype(str).reshape(-1)
        for metric, labels in per_metric_labels.items()
        if labels is not None
    }
    return final_array, metric_arrays


def write_label_cache(
    destination,
    variable,
    spec,
    final_labels,
    per_metric_labels,
    source,
):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_array, metric_arrays = _label_arrays(final_labels, per_metric_labels)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "variable": variable,
        "profile": spec,
        "source": source_metadata(source),
    }
    payload = {
        "metadata_json": np.asarray(_canonical_json(metadata)),
        "final_labels": final_array,
    }
    for metric, labels in metric_arrays.items():
        payload[f"metric__{metric}"] = labels

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_label_cache(path, variable, spec, expected_source=None):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            final_array = np.asarray(archive["final_labels"]).astype(str).reshape(-1)
            metric_arrays = {
                metric: np.asarray(archive[f"metric__{metric}"])
                .astype(str)
                .reshape(-1)
                for metric in spec["metrics"]
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid precomputed label cache {path}: {error}") from error

    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported cache schema in {path}")
    if metadata.get("variable") != variable:
        raise ValueError(f"{path} is not the label cache for {variable}")
    if metadata.get("profile") != spec:
        raise ValueError(f"Labeling profile in {path} does not match this config")
    if expected_source is not None and metadata.get("source") != expected_source:
        raise ValueError(
            f"Source metrics changed after {path} was written; rerun with --overwrite"
        )

    expected_length = len(spec["sub_dirs"]) * spec["timesteps"]
    if final_array.size != expected_length:
        raise ValueError(
            f"{path} has {final_array.size} final labels; expected {expected_length}"
        )
    allowed_final_labels = set(spec["comp_dirs"]) | {spec["fallback_label"]}
    invalid_final = set(final_array) - allowed_final_labels
    if invalid_final:
        raise ValueError(f"Unexpected final labels in {path}: {sorted(invalid_final)}")
    allowed_metric_labels = set(spec["comp_dirs"]) | {"None"}
    for metric, labels in metric_arrays.items():
        if labels.size != expected_length:
            raise ValueError(
                f"{path} has {labels.size} {metric} labels; expected {expected_length}"
            )
        invalid = set(labels) - allowed_metric_labels
        if invalid:
            raise ValueError(
                f"Unexpected {metric} labels in {path}: {sorted(invalid)}"
            )

    final_da = xr.DataArray(final_array.astype(object))
    metric_das = {
        metric: xr.DataArray(labels.astype(object))
        for metric, labels in metric_arrays.items()
    }
    return final_da, metric_das


def compute_and_cache_labels(
    variable,
    timesteps,
    sub_dirs,
    comp_dirs,
    metrics,
    metrics_dir=DEFAULT_METRICS_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    metrics_info=None,
    overwrite=False,
):
    metrics_info = DEFAULT_METRICS_INFO if metrics_info is None else metrics_info
    spec = label_profile_spec(
        timesteps,
        sub_dirs,
        comp_dirs,
        metrics,
        metrics_info,
    )
    source = quality_metric_path(metrics_dir, variable)
    destination = label_cache_path(cache_dir, variable, spec)
    if destination.exists() and not overwrite:
        return load_label_cache(
            destination,
            variable,
            spec,
            expected_source=source_metadata(source),
        )

    metrics_data = load_quality_metrics(
        source,
        variable,
        timesteps,
        sub_dirs,
        comp_dirs,
        metrics,
    )
    final_labels, per_metric_labels = generate_classification_labels(
        metrics_info,
        metrics_data,
        comp_dirs,
    )
    write_label_cache(
        destination,
        variable,
        spec,
        final_labels,
        per_metric_labels,
        source,
    )
    return load_label_cache(destination, variable, spec)


def load_precomputed_labels_for_variables(
    variables,
    timesteps,
    sub_dirs,
    comp_dirs,
    metrics,
    cache_dir=None,
    metrics_info=None,
):
    """Load pre-split labels for a model config without changing their order."""
    metrics_info = DEFAULT_METRICS_INFO if metrics_info is None else metrics_info
    if cache_dir is None:
        cache_dir = os.environ.get("PRECOMPUTED_LABEL_DIR", DEFAULT_CACHE_DIR)
    spec = label_profile_spec(
        timesteps,
        sub_dirs,
        comp_dirs,
        metrics,
        metrics_info,
    )
    final_labels = {}
    metric_labels = {}
    for variable in variables:
        path = label_cache_path(cache_dir, variable, spec)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing precomputed labels for {variable}: {path}. "
                "Run precompute_labels.pbs (or precompute_labels.py) first."
            )
        final_labels[variable], metric_labels[variable] = load_label_cache(
            path,
            variable,
            spec,
        )
    return final_labels, metric_labels

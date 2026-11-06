#!/bin/bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 RUN_DIRECTORY" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$repo_dir/configs"
cache_dir="$repo_dir/data/precomputed_labels"
run_dir="$(cd "$1" && pwd)"
results_dir="$run_dir/model_results"
single_plots="$run_dir/plots_single"
multi_plots="$run_dir/plots_multi"
table_dir="$run_dir/label_tables"

if [[ ! -d "$results_dir" ]]; then
    echo "Model-results directory does not exist: $results_dir" >&2
    exit 1
fi

mkdir -p "$single_plots" "$multi_plots" "$table_dir"

if compgen -G "$results_dir/model_timing_*.txt" >/dev/null; then
    python -u "$repo_dir/model_timing.py" \
        --results-dir "$results_dir" \
        --output "$run_dir/model_timing_summary.txt"
else
    echo "No model timing files found; timing summary skipped"
fi

python -u "$repo_dir/mainplots1.py" \
    --num_configs 47 \
    --config_prefix "$config_dir/rotated_config_" \
    --results-dir "$results_dir" \
    --output-dir "$single_plots" \
    --skip-missing

python -u "$repo_dir/mainplots1.py" \
    --num_configs 47 \
    --config_prefix "$config_dir/multi_config_" \
    --results-dir "$results_dir" \
    --output-dir "$multi_plots" \
    --feature-importance \
    --skip-missing

python -u "$repo_dir/label_distribution_table.py" \
    --cache-dir "$cache_dir" \
    --metrics dssim pcc \
    --timesteps 2000 \
    --subset first-1600 \
    --output "$table_dir/single_variable_labels_first1600.csv"

python -u "$repo_dir/label_distribution_table.py" \
    --cache-dir "$cache_dir" \
    --metrics dssim pcc \
    --timesteps 1600 \
    --subset all \
    --output "$table_dir/multi_variable_labels_1600.csv"

# Create the paper's single-variable stacked bars from each CNN/RF pair.
detail_dir="$single_plots/details"
detail_logs="$run_dir/logs/plot_details"
mkdir -p "$detail_dir" "$detail_logs"

shopt -s nullglob
for config in "$config_dir"/rotated_config_*.json; do
    if [[ "$(jq -r '.ModelType' "$config")" != cnn ]]; then
        continue
    fi
    config_name="$(basename "$config" .json)"
    if ! python -u "$repo_dir/main_plots.py" \
        --json "$config" \
        --model cnn \
        --labelsonly True \
        --results-dir "$results_dir" \
        --output-dir "$detail_dir" \
        > "$detail_logs/${config_name}.log" 2>&1
    then
        echo "Detailed plots skipped for $config_name; see $detail_logs/${config_name}.log" >&2
    fi
done

echo "Single-variable plots: $single_plots"
echo "Multi-variable plots: $multi_plots"
echo "Label tables: $table_dir"

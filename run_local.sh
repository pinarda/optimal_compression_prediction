#!/bin/bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

"$repo_dir/create_configs.sh"
"$repo_dir/create_configs_single.sh"

run_stamp="local_$(date +%Y-%m-%d_%H%M%S)"
run_dir="$repo_dir/data/runs/$run_stamp"
results_dir="$run_dir/model_results"
logs_dir="$run_dir/logs"
cache_dir="$repo_dir/data/precomputed_labels"
mkdir -p "$results_dir" "$logs_dir" "$cache_dir"

if (( $# == 0 )); then
    shopt -s nullglob
    config_files=(
        "$repo_dir"/configs/rotated_config_*.json
        "$repo_dir"/configs/multi_config_*.json
    )
    shopt -u nullglob
else
    config_files=()
    for config in "$@"; do
        if [[ "$config" != /* ]]; then
            config="$repo_dir/$config"
        fi
        if [[ ! -f "$config" ]]; then
            echo "Config does not exist: $config" >&2
            exit 1
        fi
        config_files+=("$config")
    done
fi

if (( ${#config_files[@]} == 0 )); then
    echo "No model configs were found" >&2
    exit 1
fi

python -u "$repo_dir/precompute_labels.py" \
    --config-dir "$repo_dir/configs" \
    --metrics-dir "$repo_dir/data/metrics" \
    --output-dir "$cache_dir" \
    2>&1 | tee "$logs_dir/precompute_labels.log"

for config in "${config_files[@]}"; do
    config_name="$(basename "$config" .json)"
    case "$config_name" in
        multi_config_*) main_script="$repo_dir/newmain_derecho.py" ;;
        rotated_config_*) main_script="$repo_dir/newmain.py" ;;
        *)
            echo "Unsupported config name: $config_name" >&2
            exit 1
            ;;
    esac

    echo "Running $config_name with $(basename "$main_script")"
    MODEL_RESULTS_DIR="$results_dir" \
    PRECOMPUTED_LABEL_DIR="$cache_dir" \
    MODEL_RANDOM_SEED="${MODEL_RANDOM_SEED:-2}" \
        python -u "$main_script" -c "$config" \
        2>&1 | tee "$logs_dir/$config_name.log"
done

printf '%s\n' "${config_files[@]}" > "$run_dir/modeling_configs.txt"
cat > "$run_dir/run_manifest.txt" <<EOF
mode=local
metrics=dssim,pcc
created=$run_stamp
config_count=${#config_files[@]}
precomputed_labels=$cache_dir
model_results=$results_dir
EOF
printf '%s\n' "$run_dir" > "$repo_dir/data/latest_run.txt"

"$repo_dir/generate_plots.sh" "$run_dir" \
    2>&1 | tee "$logs_dir/generate_plots.log"

echo "Local workflow finished"
echo "  Run directory: $run_dir"
echo "  Single-variable plots: $run_dir/plots_single"
echo "  Multi-variable plots: $run_dir/plots_multi"
echo "  Label tables: $run_dir/label_tables"

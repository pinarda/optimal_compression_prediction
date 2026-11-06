#!/bin/bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
template="$repo_dir/config_template.json"
output_dir="$repo_dir/configs"
overwrite=false

if [[ "${1:-}" == "--overwrite" ]]; then
    overwrite=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--overwrite]" >&2
    exit 2
fi

command -v jq >/dev/null || {
    echo "jq is required to generate configs." >&2
    exit 1
}
[[ -f "$template" ]] || {
    echo "Missing template: $template" >&2
    exit 1
}
mkdir -p "$output_dir"

# Remove configs for variables intentionally excluded from both scenarios when
# replacing a previously generated config set.
for obsolete_slot in 11 16 19 20 58 63 66 67; do
    obsolete_file="$output_dir/rotated_config_${obsolete_slot}.json"
    if [[ -e "$obsolete_file" ]]; then
        if [[ "$overwrite" != true ]]; then
            echo "Obsolete generated config exists: $obsolete_file" >&2
            echo "Re-run with --overwrite to remove excluded-variable configs." >&2
            exit 1
        fi
        rm -f "$obsolete_file"
        echo "Removed $obsolete_file"
    fi
done

var_list=(
    TREFHTMX TS LHFLX PRECSL PRECT PSL Q200 Q500 Q850 SHFLX
    T500 T850 TAUX TAUY FLNS FLNSC PRECL PRECSC QBOT T010 TMQ
    TREFHT TREFHTMN U200 U500 U850 UBOT V200 V500
    V850 VBOT WSPDSRFAV Z050 Z500 bc_a1_SRF dst_a1_SRF dst_a3_SRF
    pom_a1_SRF so4_a1_SRF so4_a2_SRF so4_a3_SRF soa_a1_SRF
    soa_a2_SRF
)

write_config() {
    local output_file=$1
    local generated_file=$2

    if [[ -e "$output_file" ]]; then
        if cmp -s "$output_file" "$generated_file"; then
            rm -f "$generated_file"
            echo "Unchanged $output_file"
            return
        fi
        if [[ "$overwrite" != true ]]; then
            rm -f "$generated_file"
            echo "Existing config differs: $output_file" >&2
            echo "Re-run with --overwrite to replace generated configs." >&2
            exit 1
        fi
    fi

    mv "$generated_file" "$output_file"
    echo "Created $output_file"
}

for model_type in cnn rf; do
    if [[ "$model_type" == cnn ]]; then
        offset=0
    else
        offset=47
    fi

    for ((i=0; i<${#var_list[@]}; i++)); do
        legacy_slot=$((i + 1))
        for omitted_slot in 11 16 19 20; do
            if ((legacy_slot >= omitted_slot)); then
                legacy_slot=$((legacy_slot + 1))
            fi
        done
        output_file="$output_dir/rotated_config_$((offset + legacy_slot)).json"
        generated_file=$(mktemp "$output_dir/.rotated_config.XXXXXX")
        var_json=$(jq -cn --arg variable "${var_list[i]}" '[[$variable]]')
        jq \
            --argjson var_list "$var_json" \
            --arg model_type "$model_type" \
            '.VarList = $var_list
             | .ModelType = $model_type
             | .Times = [2000]
             | .CompDirs = ["zfp_p_8", "zfp_p_10", "zfp_p_12", "zfp_p_14", "zfp_p_16", "zfp_p_18", "zfp_p_20", "zfp_p_22", "zfp_p_24"]
             | .Metric = ["dssim", "pcc"]' \
            "$template" > "$generated_file"
        write_config "$output_file" "$generated_file"
    done
done

echo "Generated 86 single-variable configs in $output_dir"

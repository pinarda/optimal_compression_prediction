import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import argparse
import glob
import json
import os
import re
from pathlib import Path


def paired_config_names(config_file, model_type):
    """Return the CNN and RF config names for one paired configuration."""
    config_name = Path(config_file).stem
    match = re.fullmatch(r"(.+_)(\d+)", config_name)
    if not match:
        raise ValueError(f"Cannot determine the config number from {config_file}")

    prefix, number_text = match.groups()
    number = int(number_text)
    if model_type == "cnn":
        return config_name, f"{prefix}{number + 47}"
    if model_type == "rf":
        return f"{prefix}{number - 47}", config_name
    raise ValueError(f"Unsupported ModelType {model_type!r} in {config_file}")


def load_result_array(results_dir, prefix, time, model_type, config_name,
                      variable_file, **np_load_kwargs):
    """Load the one model result matching a config and variable."""
    variable_pattern = glob.escape(str(variable_file))
    pattern = os.path.join(
        results_dir,
        f"{prefix}_*{time}{model_type}_{config_name}_{variable_pattern}.npy",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No result found for pattern {pattern}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one result for pattern {pattern}; found {len(matches)}:\n  "
            + "\n  ".join(matches)
        )
    return np.load(matches[0], **np_load_kwargs)


def process_config(config_file, results_dir, load_feature_importance=False):
    """Process a single configuration file and calculate F1 scores."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract parameters
    times = config.get('Times')                     # [60]
    time = times[0]
    var_list = config.get('VarList')                # [["TS"], ["PRECT"]]
    rf_feature_list = config.get('RFFeatureList')   # Feature names for RF
    cnn_config_name, rf_config_name = paired_config_names(
        config_file, config.get('ModelType')
    )

    # Use var_list[0] for file names directly
    variable_file = var_list[0]
    # Use the actual string for display purposes
    variable_display = var_list[0][0] if isinstance(var_list[0], list) and var_list[0] else "Unknown"

    # Load data for CNN

    test_labels_np_cnn = load_result_array(
        results_dir, "test_labels", time, "cnn", cnn_config_name, variable_file
    )
    test_predictions_cnn = load_result_array(
        results_dir, "test_predictions", time, "cnn", cnn_config_name, variable_file
    )

    # Load data for RF
    test_labels_np_rf = load_result_array(
        results_dir, "test_labels", time, "rf", rf_config_name, variable_file,
        allow_pickle=True
    )

        # np.load(f"{storageloc}/test_labels_02000rf0_{variable_file}.npy"))
    # test_predictions_rf = np.load(f"{storageloc}/test_predictions_rf_{j}{time}rf{jobid}_{variable_file}.npy")
    test_predictions_rf = load_result_array(
        results_dir, "test_predictions_rf", time, "rf", rf_config_name,
        variable_file,
        allow_pickle=True
    )

    # Compute F1 scores
    f1_weighted_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='weighted')
    f1_macro_cnn = f1_score(test_labels_np_cnn, test_predictions_cnn, average='macro')
    f1_weighted_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='weighted')
    f1_macro_rf = f1_score(test_labels_np_rf, test_predictions_rf, average='macro')

    print(f"CNN: {f1_macro_cnn} RF: {f1_macro_rf},  F1 score plot for {var_list[0]}")

    feature_importances = None
    if load_feature_importance:
        feature_importances = load_result_array(
            results_dir, "feature_importances_rf", time, "rf", rf_config_name,
            variable_file
        )

    return (variable_display, f1_weighted_cnn, f1_macro_cnn,
            f1_weighted_rf, f1_macro_rf, feature_importances, rf_feature_list)


def plot_f1_scores_alpha(var_list, cnn_scores, rf_scores, metric_name, filename):
    """Create a bar plot for F1 scores, with variables shown alphabetically."""
    import numpy as np
    import matplotlib.pyplot as plt

    # ── sort everything by variable name ──────────────────────────────
    # zip → sort by the variable string (case-insensitive) → unzip
    sorted_rows = sorted(zip(var_list, cnn_scores, rf_scores),
                         key=lambda t: t[0].lower())
    var_sorted, cnn_sorted, rf_sorted = map(list, zip(*sorted_rows))

    # ── plotting ─────────────────────────────────────────────────────
    x         = np.arange(len(var_sorted))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - bar_width/2, cnn_sorted, bar_width,
           label="CNN F1 Score", color="#e18683", edgecolor="black")
    ax.bar(x + bar_width/2, rf_sorted,  bar_width,
           label="RF F1 Score",  color="#B6D7E4", edgecolor="black")

    ax.set_xlabel("Variables")
    ax.set_ylabel(f"{metric_name} F1 Score")
    ax.set_title(f"{metric_name} F1 Score Comparison for CNN and RF Models")

    ax.set_xticks(x)
    ax.set_xticklabels(var_sorted, rotation=45, ha='right', fontsize=8)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{metric_name} F1 score plot saved to {filename}")



def plot_feature_importances(features, all_importances, variable_names, filename):
    """
    Create a grouped bar plot for feature importances across all variables.

    Parameters
    ----------
    features : list[str]
        Feature (x-axis) names (length = n_features).
    all_importances : list[list[float] or 2D array-like]
        Outer length = n_variables; each inner seq length must equal len(features).
    variable_names : list[str]
        Names of the variables corresponding to rows in all_importances.
    filename : str
        Path to save the plot (extension dictates format, e.g., .png, .pdf).
    """
    import math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if not all_importances or not features:
        print("No data for feature importance plot.")
        return

    n_features = len(features)
    n_variables = len(all_importances)

    if len(variable_names) != n_variables:
        raise ValueError(
            f"variable_names length ({len(variable_names)}) "
            f"!= all_importances length ({n_variables})"
        )

    # ensure each importance row matches feature count
    for i, imp in enumerate(all_importances):
        if len(imp) != n_features:
            raise ValueError(
                f"Row {i} ('{variable_names[i]}') len={len(imp)} "
                f"!= n_features ({n_features})"
            )

    # ------------------------------------------------------------------
    # Sort variables alphabetically (case-insensitive) BEFORE plotting
    # ------------------------------------------------------------------
    order = sorted(range(n_variables), key=lambda i: variable_names[i].lower())
    variables_plot = [variable_names[i] for i in order]
    importances_plot = [all_importances[i] for i in order]

    # ------------------------------------------------------------------
    # X positions
    # ------------------------------------------------------------------
    x = np.arange(n_features)
    bar_width = 0.8 / n_variables  # 80% of unit width shared across groups

    # ------------------------------------------------------------------
    # Figure / axes
    # widen figure slightly for many features
    fig_width = max(8, min(24, 0.35 * n_features))  # heuristic
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.rcParams.update({'font.size': 16})

    # ------------------------------------------------------------------
    # Colors (qualitative colormap; repeat if > colormap capacity)
    # ------------------------------------------------------------------
    cmap = plt.get_cmap('tab20')
    colors_plot = [cmap(i % cmap.N) for i in range(n_variables)]

    # ------------------------------------------------------------------
    # Plot grouped bars (sorted order)
    # ------------------------------------------------------------------
    bar_containers = []
    for i, (var, imp) in enumerate(zip(variables_plot, importances_plot)):
        bc = ax.bar(
            x + i * bar_width,
            imp,
            bar_width,
            color=colors_plot[i],
            label="_nolegend_"  # suppress auto legend
        )
        bar_containers.append(bc)

    # ------------------------------------------------------------------
    # Axis labels & ticks
    # ------------------------------------------------------------------
    ax.set_ylabel("Importance", fontsize=14)
    ax.set_title("Feature Importances Across Variables")

    # x ticks centered on each feature group
    ax.set_xticks(x + bar_width * (n_variables - 1) / 2)

    # smaller font; rotate for space
    ax.set_xticklabels(features, rotation=25, ha='right', fontsize=14)

    # push x-axis label away from ticks (adjust below in spacing knobs)
    # we'll set label after spacing knobs so we can change labelpad
    # (Matplotlib doesn't overwrite labelpad if unchanged, but order is clear)

    if mpl.rcParams.get("text.usetex", False):
        legend_labels = [v.replace('_', r'\_') for v in variables_plot]
    else:
        legend_labels = variables_plot

    legend_handles = [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in zip(colors_plot, legend_labels)
    ]

    # wrap legend across multiple columns (aim ≤ 4 rows)
    max_rows = 4
    ncol = int(math.ceil(len(legend_handles) / max_rows))

    # ------------------------------------------------------------------
    # Spacing knobs
    # ------------------------------------------------------------------
    XTICK_PAD   = 2     # px from axis spine to tick labels
    XLABEL_PAD  = 10    # pts from tick labels to axis label
    LEGEND_PAD  = 0.06  # fraction of axes height below axes
    # bottom margin: base for xticks + legend footprint
    BOTTOM_PAD  = 0.12 + LEGEND_PAD

    # Apply spacing tweaks
    ax.tick_params(axis='x', which='major', pad=XTICK_PAD)
    ax.set_xlabel("Features", labelpad=XLABEL_PAD, fontsize=14)

    # Figure-level legend so it spans full width
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        title="Variables",
        loc="upper center",
        bbox_to_anchor=(0.5, -LEGEND_PAD),  # move below axes
        ncol=ncol,
        fontsize=10,
        title_fontsize=12,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.4,
    )

    # Reserve space at bottom for ticks + legend
    fig.subplots_adjust(bottom=BOTTOM_PAD)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f"Feature importance comparison plot saved to {filename}")



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate F1 score and feature importance plots from multiple configuration JSON files.")
    parser.add_argument('-n', '--num_configs', type=int, default=47,
                        help="Number of configuration files to process (default: 17)")
    parser.add_argument('-p', '--config_prefix', type=str, default='rotated_config_',
                        help="Prefix for the configuration files (default: rotated_config_)")
    parser.add_argument('-s', '--suffix', type=str, default='.json',
                        help="Suffix for the configuration files (default: .json)")
    parser.add_argument('--results-dir', required=True,
                        help="Directory containing outputs from the paired CNN and RF model runs")
    parser.add_argument('--output-dir',
                        help="Directory for plots (default: RESULTS_DIR/plots)")
    parser.add_argument('--feature-importance', action='store_true',
                        help="Create the RF feature-importance plot")
    parser.add_argument('--skip-missing', action='store_true',
                        help="Skip a variable when any required CNN or RF result file is missing")

    # Parse the arguments
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(results_dir, "plots")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Store results for plotting
    var_list = []
    f1_weighted_cnn = []
    f1_macro_cnn = []
    f1_weighted_rf = []
    f1_macro_rf = []
    all_importances = []  # Collect all feature importances
    feature_names = None  # To store feature names from the first config

    # Iterate over all configuration files
    for i in range(1, args.num_configs + 1):
        config_file = f"{args.config_prefix}{i}{args.suffix}"
        if os.path.exists(config_file):
            # if i not in [23, 24, 36, 37, 40, 41, 43, 44]:
            # if i not in [1, 2, 4, 5, 12, 13, 16, 19, 20, 21, 22, 24, 26, 27, 37, 43, 44, 45, 46, 47]:
            if i not in [20, 19, 11, 16]:
                print(f"Processing {config_file}...")
                try:
                    (variable, weighted_cnn, macro_cnn,
                     weighted_rf, macro_rf, feature_importances, rf_feature_list) = process_config(
                        config_file, results_dir, args.feature_importance
                    )
                except FileNotFoundError as error:
                    if not args.skip_missing:
                        raise
                    print(f"Skipping {config_file}: {error}")
                    continue
                var_list.append(variable)
                f1_weighted_cnn.append(weighted_cnn)
                f1_macro_cnn.append(macro_cnn)
                f1_weighted_rf.append(weighted_rf)
                f1_macro_rf.append(macro_rf)

                # Save feature importances and feature names
                if feature_importances is not None:
                    all_importances.append(feature_importances)
                    if feature_names is None:  # Store feature names once
                        feature_names = rf_feature_list

        else:
            print(f"Configuration file {config_file} not found. Skipping.")

    if not var_list:
        print("No complete CNN/RF result pairs were found; no plots were created.")
        return

    # Plot Weighted F1 Scores
    plot_f1_scores_alpha(
        var_list, f1_weighted_cnn, f1_weighted_rf, "Weighted",
        os.path.join(output_dir, "f1_score_comparison_weighted.png")
    )

    # Plot Macro F1 Scores
    plot_f1_scores_alpha(
        var_list, f1_macro_cnn, f1_macro_rf, "Macro",
        os.path.join(output_dir, "f1_score_comparison_macro.png")
    )

    # Plot Feature Importances for all variables
    if args.feature_importance and all_importances and feature_names:
        plot_feature_importances(
            feature_names,
            all_importances,
            var_list,
            os.path.join(output_dir, "feature_importances_rf_comparison.png")
        )


if __name__ == "__main__":
    main()

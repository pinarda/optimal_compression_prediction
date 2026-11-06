import datetime
import glob
import json
import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import parse_command_line_arguments, read_parameters_from_json


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
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one result for pattern {pattern}; found {len(matches)}:\n  "
            + "\n  ".join(matches)
        )
    return np.load(matches[0], **np_load_kwargs)


def load_label_encoder(model, var, time, config_name, encoder_dir):
    """Load the label encoder saved by the paired model run."""
    pattern = os.path.join(
        encoder_dir,
        f"label_encoder_*{time}{model}_{config_name}_*{glob.escape(str(var))}*.pkl",
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one label encoder for pattern {pattern}; "
            f"found {len(matches)}"
        )
    with open(matches[0], "rb") as file_handle:
        return pickle.load(file_handle)


def decode(values, label_encoder, none_token="Lossless"):
    """Convert integer labels to compression-level names."""
    values = np.asarray(values).astype(int)
    if (values == -1).any():
        missing = values == -1
        encoded = values.copy()
        encoded[missing] = 0
        decoded = label_encoder.inverse_transform(encoded)
        decoded[missing] = none_token
        return decoded
    return label_encoder.inverse_transform(values)


def plot_stacked_bars(true_labels, cnn_predictions, rf_predictions,
                      metric, jobid, variable_names, output_dir):
    """Create the paper's gray, red, and blue single-variable bar chart."""
    unique_labels = sorted(np.unique(np.concatenate(
        [true_labels, cnn_predictions, rf_predictions]
    )).tolist())
    cm_cnn = confusion_matrix(true_labels, cnn_predictions, labels=unique_labels)
    cm_rf = confusion_matrix(true_labels, rf_predictions, labels=unique_labels)

    correct_cnn = np.diag(cm_cnn)
    incorrect_cnn = cm_cnn.sum(axis=0) - correct_cnn
    correct_rf = np.diag(cm_rf)
    incorrect_rf = cm_rf.sum(axis=0) - correct_rf
    true_counts = np.array([list(true_labels).count(label) for label in unique_labels])

    palette = sns.color_palette(['#FF6347', '#4682B4'])
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.2
    positions = np.arange(len(unique_labels))

    ax.bar(positions - bar_width, true_counts, bar_width,
           label='True Counts', color='gray')
    ax.bar(positions, correct_cnn, bar_width,
           label='Correct CNN', color=palette[0])
    ax.bar(positions + bar_width, correct_rf, bar_width,
           label='Correct RF', color=palette[1])
    ax.bar(positions, incorrect_cnn, bar_width, bottom=correct_cnn,
           label='Incorrect CNN', color=palette[0], alpha=0.5, hatch='//')
    ax.bar(positions + bar_width, incorrect_rf, bar_width, bottom=correct_rf,
           label='Incorrect RF', color=palette[1], alpha=0.5, hatch='//')

    maximum = np.max([
        true_counts,
        correct_cnn + incorrect_cnn,
        correct_rf + incorrect_rf,
    ])
    ax.set_ylim(0, int(maximum * 1.2))
    ax.set_xlabel('Compression Level')
    ax.set_ylabel('Count')
    ax.set_title(
        'Counts of True, Correct, and Incorrect Predictions\n'
        f'by Compression Level for CNN and RF: {", ".join(variable_names)}'
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_labels, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    variable_text = "_".join(variable_names)
    filename = os.path.join(
        output_dir,
        f"stacked_bar_{metric}_{jobid}_{date_string}_{variable_text}.png",
    )
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Stacked-bar plot saved to {filename}")


def main_plots():
    args = parse_command_line_arguments()
    parameters = read_parameters_from_json(args.json)
    vlist = parameters[1]
    times = parameters[8]
    metric = parameters[12]
    flat_vlist = [
        item
        for group in vlist
        for item in (group if isinstance(group, (list, tuple)) else [group])
    ]

    with open(args.json, 'r') as config_file:
        config_model_type = json.load(config_file).get('ModelType')
    cnn_config_name, rf_config_name = paired_config_names(
        args.json, config_model_type
    )
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(results_dir, "plots")
    )
    os.makedirs(output_dir, exist_ok=True)

    for time_value in times:
        true_encoded = load_result_array(
            results_dir, "test_labels", time_value, "cnn", cnn_config_name,
            vlist[0], allow_pickle=True,
        )
        cnn_encoded = load_result_array(
            results_dir, "test_predictions", time_value, "cnn", cnn_config_name,
            vlist[0], allow_pickle=True,
        )
        rf_encoded = load_result_array(
            results_dir, "test_predictions_rf", time_value, "rf", rf_config_name,
            vlist[0], allow_pickle=True,
        )
        label_encoder = load_label_encoder(
            "cnn", flat_vlist[0], time_value, cnn_config_name, results_dir
        )
        true_labels = decode(np.ravel(true_encoded), label_encoder)
        cnn_predictions = decode(np.ravel(cnn_encoded), label_encoder)
        rf_predictions = decode(np.ravel(rf_encoded), label_encoder)
        if not (
            len(true_labels) == len(cnn_predictions) == len(rf_predictions)
        ):
            raise ValueError(
                f"Length mismatch: true={len(true_labels)}, "
                f"cnn={len(cnn_predictions)}, rf={len(rf_predictions)}"
            )
        plot_stacked_bars(
            true_labels,
            cnn_predictions,
            rf_predictions,
            metric,
            args.jobid,
            flat_vlist,
            output_dir,
        )


if __name__ == "__main__":
    main_plots()

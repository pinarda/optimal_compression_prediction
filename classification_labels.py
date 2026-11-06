# first, grab the data from the StorageLoc location,
# saved as xarray dataarrays as train_data_dssim/test_data_dssim and so on.
# load that in for all the metrics (dssim, spre, pcc)





import numpy as np
import xarray as xr
import os
from utils import read_parameters_from_json

import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})



def extract_compression_level(filename):
    # Adjusted to extract the compression level from the new filename format
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]  # The part after 'zfp' and before the metric name
    return None

def load_and_label_data(metric_info, storageloc, ds=None):

    final_labels_dict = {}

    for metric, info in metric_info.items():
        labels = []  # To hold label DataArrays for each file
        compression_levels = []  # To hold compression levels for each file

        # ds = None
        if ds is None:
            for filename in info['filenames']:
                compression_level = extract_compression_level(filename)
                if compression_level:
                    compression_levels.append(compression_level)

                    filepath = os.path.join(storageloc, filename)
                    data = np.load(filepath)
                    data_da = xr.DataArray(data)

                    if info['comparison'] == 'gt':
                        labels.append(data_da >= info['threshold'])
                    else:  # 'lt'
                        labels.append(data_da < info['threshold'])
        else:
            for key in ds:
                compression_levels.append(key)
                data = ds[key]
                data_da = xr.DataArray(data)

                if info['comparison'] == 'gt':
                    labels.append(data_da >= info['threshold'])
                else:  # 'lt'
                    labels.append(data_da < info['threshold'])



        final_labels = xr.full_like(labels[0], "None", dtype="object")
        for compression_level, label_da in zip(compression_levels, labels):
            final_labels = xr.where((final_labels == "None") & label_da, compression_level, final_labels)

        final_labels_dict[metric] = final_labels

    return final_labels_dict




def compare_across_metrics(final_labels_dict):
    compression_levels = [
        "zfp_p_24", "zfp_p_22", "zfp_p_20", "zfp_p_18", "zfp_p_16", "zfp_p_14", "zfp_p_12", "zfp_p_10", "zfp_p_8"
    ]
    # Create a ranking dictionary for the compression levels
    level_rank = {level: rank for rank, level in enumerate(compression_levels)}

    metrics = list(final_labels_dict.keys())
    base_metric_labels = final_labels_dict[metrics[0]]

    # Initialize the final comparison array with "None" labels
    final_comparison_labels = xr.full_like(base_metric_labels, "None", dtype="object")

    # Iterate through all points and compare labels across metrics
    for metric in metrics:
        current_metric_labels = final_labels_dict[metric]
        for idx in range(current_metric_labels.size):
            current_label = current_metric_labels.values[idx]
            final_label = final_comparison_labels.values[idx]

            # If the final label is "None" and current label is not, set the current label
            if final_label == "None" and current_label != "None":
                final_comparison_labels.values[idx] = current_label
            # If both labels are not "None", choose the most conservative one
            elif final_label != "None" and current_label != "None":
                if level_rank[current_label] < level_rank[final_label]:
                    final_comparison_labels.values[idx] = current_label

    return final_comparison_labels, final_labels_dict

def setup(config_path, metrics, cdirs, storageloc):

    # split config_path to get the directory, filename, and extension
    directory, filename = os.path.split(config_path)
    filenameonly, extension = os.path.splitext(filename)


    # Metrics, their thresholds, and comparison types
    # metrics = ['dssim', 'pcc']
    thresholds = {'dssim': 0.995, 'spre': 10, 'pcc': 0, 'ks': 0.05}
    comparisons = {'dssim': 'gt', 'spre': 'lt', 'pcc': 'gt', 'ks': 'gt'}

    # Base pattern for filenames
    # deal with time, jobid
    filename_pattern = '{storageloc}{level}_{metric}_mat_alltime_10_{filenameonly}1.npy'

    # Generate metrics_info dictionary
    metrics_info = {
        metric: {
            'threshold': thresholds[metric],
            'filenames': [
                filename_pattern.format(metric=metric, level=level, filenameonly=filenameonly, storageloc=storageloc) for level in cdirs
            ],
            'comparison': comparisons[metric]
        }
        for metric in metrics
    }

    print(metrics_info)
    return metrics_info

def classify(config_path, metrics, labels=None):
    # config_path = "config.json"

    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride, m, cut_dataset, subdirs = read_parameters_from_json(
        config_path)

    # metrics = ['dssim', 'spre', 'pcc', 'ks']
    metrics_info = setup(config_path, metrics, cdirs, storageloc)

    # Load and label data for each metric
    final_labels_dict = load_and_label_data(metrics_info, storageloc, labels)

    # Compare results across all metrics
    final_comparison_labels, final_labels_dict = compare_across_metrics(final_labels_dict)

    # Save the final comparison labels
    final_labels_filepath = os.path.join(storageloc, 'final_comparison_labels.npy')
    # save the labels as a dictionary with a description of the labels, including the time steps, variable, metrics, jobid

    np.save(final_labels_filepath, final_comparison_labels.values)
    return final_comparison_labels, final_labels_dict

# Run the function to load, label, and save data
if __name__ == '__main__':
    config_path = "config_casper.json"

    save, vlist, pre, post, opath, cpath, cdirs, ldcpypath, time, storageloc, navg, stride, m, cut_dataset, subdirs = read_parameters_from_json(
        config_path)

    # metrics = ['dssim', 'spre', 'pcc', 'ks']
    # metrics = ['spre']

    metrics = ['dssim', 'pcc']
    metrics_info = setup(config_path, metrics, storageloc)

    # Load and label data for each metric
    final_labels_dict = load_and_label_data(metrics_info, storageloc)

    # Compare results across all metrics
    final_comparison_labels = compare_across_metrics(final_labels_dict)

    # Save the final comparison labels
    final_labels_filepath = os.path.join(storageloc, 'final_comparison_labels.npy')
    np.save(final_labels_filepath, final_comparison_labels.values)


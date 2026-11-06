import xarray as xr
import os
import numpy as np
import pickle
import datetime
import tensorflow as tf
from sklearn.preprocessing import quantile_transform, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn
import random
import time
from math import floor
from classification_labels import classify
# os.environ["HDF5_PLUGIN_PATH"]
#
# logging
import logging

import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})

LATS = 182
LONS = 288
WINDOWSIZE = 11

import numpy as np


def xarray_train_test_split(dataset, labels, test_size=0.5, random_state=None):
    # Determine the number of samples
    total_samples = dataset.sizes['sample']

    # Generate shuffled indices
    # if random_state is not None:
    #     np.random.seed(random_state)
    indices = np.arange(total_samples)
    # np.random.shuffle(indices)

    # Determine the size of the test set
    test_size = int(total_samples * test_size)

    # Split indices
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Select the train and test sets
    X_train = dataset.isel(sample=train_indices)
    X_test = dataset.isel(sample=test_indices)
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test


def split_data(
    dataset: xr.Dataset,
    label: np.ndarray,
    time: int,
    nvar: int,
    testset: str,
    lats: int,
    lons: int,
    cut_windows: bool = True,
    window_size: int = 11,
    encoder=None,
    storageloc=None,
    metric=None,
    modeltype=None,
    jobid=None,
    j=None,
    vars=None,
) -> tuple:
    """
    Splits the data into training, validation, and testing sets based on the `testset` variable.

    If `testset` is set to "1var", all the time steps for a single variable are used as the test set.
    For example:
    - If the dataset has shape (192, 288, 320) and `nvar` is 2, the first 160 samples are used for testing,
      and the remaining 160 samples are split between training and validation.
    - For 3 variables, two-thirds of the data are used for testing, and the remaining third for training and validation.

    Parameters:
    - dataset (xr.Dataset): The dataset containing the data samples.
    - label (np.ndarray): The labels corresponding to the data samples.
    - time (int): Not used in this function (can be removed if not needed).
    - nvar (int): Number of variables in the dataset.
    - testset (str): Strategy for splitting the data. If "1var", uses one variable for testing.
    - lats (int): Number of latitude points.
    - lons (int): Number of longitude points.
    - cut_windows (bool): Whether to cut the data into smaller windows.
    - window_size (int): Size of the window if `cut_windows` is True.
    - encoder: Optional encoder for labels.
    - storageloc: Storage location (not used here).
    - metric: Metric used (not used here).
    - modeltype: Type of model (not used here).
    - jobid: Job ID (not used here).
    - j: Index (not used here).

    Returns:
    - X_train: Training data samples.
    - y_train: Training labels.
    - X_val: Validation data samples.
    - y_val: Validation labels.
    - X_test: Testing data samples.
    - y_test: Testing labels.
    """

    # Total number of samples
    if modeltype == "cnn":
        total_samples = dataset.dims['sample']
    else:
        total_samples = len(dataset)

    if testset == '1var':

        random.seed(1)
        # Calculate the number of samples per variable
        samples_per_var = total_samples // nvar

        # Number of samples to use for testing (all samples from one variable)

        logging.info(f"Total samples: {total_samples}")
        # first, find out how big the first variable group is (length of vars[0])
        # then, use that to determine the number of samples to use for testing
        test_samples = len(vars[0]) * samples_per_var
        # Indices for the test set (first variable)
        test_indices = np.arange(0, test_samples)
        # indices for the validation set (second variable)
        # val_indices = np.arange(test_samples, 2*test_samples)

        logging.info(f"Test indices: {test_indices}")
        # Indices for training and validation sets (remaining variables)
        train_val_indices = np.arange(test_samples, total_samples)

        logging.info(f"Train/val indices: {train_val_indices}")

        # let's printn out all the labels (without any newlines)
        np.set_printoptions(threshold=np.inf)

        logging.info(f"Labels: {label}")

        # set the print options back to default
        np.set_printoptions(threshold=1000)


        # Extract test data and labels
        if modeltype == "cnn":
            X_test_val = dataset.isel(sample=test_indices)
        else:
            X_test_val = dataset[test_indices]
        y_test_val = label[test_indices]


        # Extract training and validation data and labels
        if modeltype == "cnn":
            X_train = dataset.isel(sample=train_val_indices)
        else:
            X_train = dataset[train_val_indices]
        y_train = label[train_val_indices]


        # Further split training and validation sets (e.g., 80% training, 20% validation)
        if modeltype == "cnn":
            X_test, X_val, y_test, y_val = xarray_train_test_split(
                X_test_val, y_test_val, test_size=0.5, random_state=None
            )
        else:
            X_test, X_val, y_test, y_val = train_test_split(
                X_test_val, y_test_val, test_size=0.5, random_state=None
            )

    else:

        random.seed(1)
        if modeltype == "cnn":
            # Default random split if testset is not '1var'
            X_train_val, X_test, y_train_val, y_test = xarray_train_test_split(
                dataset, label, test_size=0.2, random_state=42
            )
            # Further split training and validation sets
            X_train, X_val, y_train, y_val = xarray_train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42
            )
        else:
            # Default random split if testset is not '1var'
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                dataset, label, test_size=0.2, random_state=42
            )
            # Further split training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42
            )
        # This results in 60% train, 20% val, 20% test

    # Optionally cut the data into smaller windows
    if cut_windows:
        def cut_into_windows(data):
            # Assuming data has dimensions (sample, lat, lon, variable)
            data_windows = []
            for i in range(data.dims['sample']):
                sample = data.isel(sample=i)
                # Convert to numpy array
                sample_array = sample['combined'].values  # Shape: (lat, lon, variable)
                # Pad the sample if necessary
                pad_size = window_size // 2
                sample_padded = np.pad(
                    sample_array,
                    pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                    mode='reflect'
                )
                # Extract windows
                lat_indices = range(pad_size, pad_size + lats)
                lon_indices = range(pad_size, pad_size + lons)
                for lat in lat_indices:
                    for lon in lon_indices:
                        window = sample_padded[
                            lat - pad_size: lat + pad_size + 1,
                            lon - pad_size: lon + pad_size + 1,
                            :
                        ]
                        data_windows.append(window)
            return np.array(data_windows)

        # Cut data into windows
        X_train = cut_into_windows(X_train)
        X_val = cut_into_windows(X_val)
        X_test = cut_into_windows(X_test)

    else:
        pass


    return X_train, y_train, X_val, y_val, X_test, y_test

def split_data_old(dataset: xr.Dataset, label: np.ndarray, time: int, nvar: int, testset: str, lats:int, lons:int, cut_windows:bool = True, window_size: int = 11, encoder=None, storageloc=None, metric=None, modeltype=None, jobid=None, j=None, vars=None) -> tuple:
    """
    Splits the dataset into training, validation, and testing sets based on the specified testset parameter.

    Parameters:
     dataset (xr.Dataset): The input xarray dataset.
     dssim (np.ndarray): The SSIM matrix corresponding to the dataset.
     time (int): The number of time slices to consider in the dataset.
     nvar (int): The number of variables to consider in the dataset.
     testset (str): The method of splitting the data ("random", "oneout", "10pct", "1var", "60pct", "60_25_wholeslice").
     comp (int): The component index to use for extracting labels from the SSIM matrix.

    Returns:
     tuple: A tuple containing the training, validation, and testing data and labels.
    """
    # Calculate the number of windows based on the dataset's dimensions
    # num_windows = lats * lons # 182 * 288
    num_windows = 1

    total_data_points = num_windows * time * nvar
    # Calculate the indices corresponding to 60% and 75% of the data
    index_60pct = int(total_data_points * 0.6)
    index_75pct = int(total_data_points * 0.75)
    index_10pct = int(total_data_points * 0.1)
    index_50pct = int(total_data_points * 0.5)

    # use 90% of the data for training, 9% for validation, and 1% for testing

    random.seed(1)
    if testset == "random":

        train_data, test_data, train_labels, test_labels = train_test_split(dataset[0:(num_windows * time * nvar)],
                                                                            label, test_size=0.5)
        val_data, test_data, val_labels, test_labels = train_test_split(test_data, test_labels, test_size=0.5)
    elif testset == "oneout":
        # train_data = dataset[0:(50596*time*nvar-1)]
        # train_labels = label[0:(50596*time*nvar-1)]
        val_data = dataset[(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        val_labels = label[(num_windows * time * nvar - 2):(num_windows * time * nvar - 1)]
        # test_data = dataset[(50596*time*nvar-1):(50596*time*nvar)]
        # test_labels = label[(50596*time*nvar-1):(50596*time*nvar)]

        # Currently, this will always make the first time slice of the data the test set for consistency
        test_data = dataset[0:num_windows]
        test_labels = label[0:num_windows]
        train_data = dataset[num_windows:(num_windows * time * nvar)]
        train_labels = label[num_windows:(num_windows * time * nvar)]
    elif testset == "10pct":
        # use the first 10% of the data for testing
        test_data = dataset[0:int(num_windows * time * nvar * 0.1)]
        test_labels = label[0:int(num_windows * time * nvar * 0.1)]
        # Note: this randomizes the training and validation data, an alternative would be to use the last 10% of the data for validation
        train_data, val_data, train_labels, val_labels = train_test_split(
            dataset[int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)],
            label[int(num_windows * time * nvar * 0.1):(num_windows * time * nvar)], test_size=0.1)

    elif testset == "1var":
        # leave out a single variable for testing, and use the rest for training
        # test_data = dataset[0:(num_windows * time)]
        stop = num_windows * time
        if modeltype == "cnn":
            test_data = dataset.isel(sample=slice(0, stop))
        else:
            test_data = dataset[0:stop]

        if label is not None:
            test_labels = label[0:(num_windows * time)]

        # Use all remaining samples for training; leave validation empty.
        if label is not None:
            # -------------------------------------------------------------------
            # 1.  Choose the rows (samples) you want
            # -------------------------------------------------------------------
            start = num_windows * time
            stop = num_windows * time * nvar  # ← sanity-check that this is correct!
            y_full = label[start:stop]

            # -------------------------------------------------------------------
            # 2.  Build the feature array X_full
            # -------------------------------------------------------------------
            if modeltype == "cnn":
                # keep the 2-D spatial structure for the CNN
                # result shape: (N_samples, lat, lon)
                subset = dataset.isel(sample=slice(start, stop))
                X_full = subset['combined'] \
                    .transpose('sample', 'lat', 'lon') \
                    .values  # -> ndarray (N, H, W)
            else:
                X_full = dataset[start:stop]

            # -------------------------------------------------------------------
            # 3.  Keep every non-test sample in the training set
            # -------------------------------------------------------------------
            train_data, train_labels = X_full, y_full
            val_data, val_labels = X_full[:0], y_full[:0]



        else:
            train_data, val_data = train_test_split(dataset[(num_windows * time):(num_windows * time * nvar)],
                                                                              test_size=0.1)

    elif testset == "60pct":
        # Use the first 60% of the data for training
        train_data = dataset[0:index_60pct]
        train_labels = label[0:index_60pct]

        # Use the last 25% of the data for testing and validation
        last_25pct_data = dataset[index_75pct:]
        last_25pct_labels = label[index_75pct:]

        # Randomly split the last 25% into 10% for testing and 15% for validation
        test_data, val_data, test_labels, val_labels = train_test_split(
            last_25pct_data, last_25pct_labels, test_size=0.4)  # 10% of 25% is 40% of the last 25%

    elif testset == "60_25_wholeslice":
        # Calculate the number of time slices in the last 25% of the data (rounding down)
        num_time_slices_last_25pct = (total_data_points - index_75pct) // num_windows

        # Calculate the number of time slices for test and validation (rounding down)
        num_time_slices_val = num_time_slices_last_25pct * 40 // 100.0
        num_time_slices_test = num_time_slices_last_25pct - num_time_slices_val

        # Calculate the number of windows for test and validation
        num_windows_test = num_time_slices_test * num_windows
        num_windows_val = num_time_slices_val * num_windows

        # Calculate the start index for the remaining time slice (if any)
        remaining_start_index = index_75pct + num_windows_test + num_windows_val

        # Use the first 60% of the data for training
        train_data = dataset[0:index_60pct]
        train_labels = label[0:index_60pct]

        # Use the calculated number of windows for test and validation
        test_data = dataset[index_75pct:int(index_75pct + num_windows_test)]
        test_labels = label[index_75pct:int(index_75pct + num_windows_test)]
        val_data = dataset[int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]
        val_labels = label[int(index_75pct + num_windows_test):int(index_75pct + num_windows_test + num_windows_val)]

        # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
        if remaining_start_index < total_data_points:
            remaining_windows = total_data_points - remaining_start_index
            split_index = remaining_windows * 40 // 100
            test_data = np.concatenate(
                (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
            test_labels = np.concatenate(
                (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
            val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
            val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))


    elif testset == "10_90_wholeslice":

        random.seed(1)
        if modeltype == 'rf':
            logging.info(f"getting 10% train, 90% test")
            # Calculate the number of time slices in the last 90% of the data (rounding down)
            num_time_slices_last_90pct = (total_data_points - index_10pct) // num_windows

            # Calculate the number of time slices for test and validation (rounding down)


            num_time_slices_val = num_time_slices_last_90pct // 9
            num_time_slices_test = num_time_slices_last_90pct - num_time_slices_val
            num_time_slices_train = time*nvar - num_time_slices_last_90pct

            # Calculate the number of windows for test and validation
            num_windows_test = num_time_slices_test * num_windows
            num_windows_val = num_time_slices_val * num_windows

            # Calculate the start index for the remaining time slice (if any)
            remaining_start_index = index_10pct + num_windows_test + num_windows_val

            # Use the first 10% of the data for training
            train_data = dataset[0:index_10pct]
            if label is not None:
                train_labels = label[0:index_10pct]

            # Use the calculated number of windows for test and validation
            test_data = dataset[index_10pct:int(index_10pct + num_windows_test)]
            if label is not None:
                test_labels = label[index_10pct:int(index_10pct + num_windows_test)]
            val_data = dataset[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]
            if label is not None:
                val_labels = label[int(index_10pct + num_windows_test):int(index_10pct + num_windows_test + num_windows_val)]

            # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
            if remaining_start_index < total_data_points:
                remaining_windows = total_data_points - remaining_start_index
                split_index = remaining_windows * 90 // 100
                test_data = np.concatenate(
                    (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
                if label is not None:
                    test_labels = np.concatenate(
                        (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
                val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
                if label is not None:
                    val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))

            if not cut_windows:
                test_data = dataset[0:int(num_windows_test/num_windows)]
                if label is not None:
                    test_labels = label[0:int(num_windows_test/num_windows)]
                val_data = dataset[int(num_windows_test/num_windows):(int(num_windows_test/num_windows)+int(num_windows_val/num_windows))]
                if label is not None:
                    val_labels = label[int(num_windows_test/num_windows):(int(num_windows_test/num_windows)+int(num_windows_val/num_windows))]
                train_data = dataset[(int(num_windows_test/num_windows)+int(num_windows_val/num_windows)):(int(num_windows_test/num_windows)+int(num_windows_val/num_windows)+int(index_10pct/num_windows))]
                if label is not None:
                    train_labels = label[(int(num_windows_test/num_windows)+int(num_windows_val/num_windows)):(int(num_windows_test/num_windows)+int(num_windows_val/num_windows)+int(index_10pct/num_windows))]

            len_train = len(train_labels)
            len_test = len(test_labels)
            logging.info(f"length of train and test data: {len_train} train, {len_test} test")
        else:
            # ------------------------------------------------------------------
            # 10 % / 90 % split  (CNN branch)
            # ------------------------------------------------------------------
            logging.info("getting 10% train, 90% test, CNN")
            dsdims = dataset.dims
            print(f"{dsdims}")

            ntime = dataset.dims["sample"]  # total number of windows
            index_10pct = int(0.10 * ntime)  # first 10 % for training

            # --- last 90 %: split 1 : 9  →  val : test ------------------------
            remaining = ntime - index_10pct  # 90 %
            n_val = remaining // 9  # 10 % of the 90 %
            n_test = remaining - n_val  # 90 % of the 90 %

            # ------------------------------------------------------------------
            # slices (xarray-friendly)
            # ------------------------------------------------------------------
            train_data = dataset.isel(sample=slice(0, index_10pct))
            val_data = dataset.isel(sample=slice(index_10pct,
                                                  index_10pct + n_val))
            test_data = dataset.isel(sample=slice(index_10pct + n_val, None))

            if label is not None:
                train_labels = label[0:index_10pct]
                val_labels = label[index_10pct:index_10pct + n_val]
                test_labels = label[index_10pct + n_val:]

            # ------------------------------------------------------------------
            # optional: collapse windows → treat each **time slice** as one sample
            # ------------------------------------------------------------------
            if not cut_windows:
                # number of *time slices* (not windows)
                ntime_slices = ntime // num_windows

                # convert window indices → slice indices
                idx10 = index_10pct // num_windows
                n_val_slices = n_val // num_windows
                n_test_slices = n_test // num_windows

                test_data = dataset.isel(sample=slice(0, n_test_slices))
                val_data = dataset.isel(sample=slice(n_test_slices,
                                                      n_test_slices + n_val_slices))
                train_data = dataset.isel(sample=slice(n_test_slices + n_val_slices,
                                                        n_test_slices + n_val_slices + idx10))

                if label is not None:
                    test_labels = label[0:n_test_slices]
                    val_labels = label[n_test_slices:n_test_slices + n_val_slices]
                    train_labels = label[n_test_slices + n_val_slices:
                                         n_test_slices + n_val_slices + idx10]

            # ------------------------------------------------------------------
            # log sizes
            # ------------------------------------------------------------------
            len_train = len(train_labels) if label is not None else train_data.dims["time"]
            len_test = len(test_labels) if label is not None else test_data.dims["time"]
            logging.info(f"length of train and test data: {len_train} train, {len_test} test")

    elif testset == "50_50_wholeslice":

        # Calculate the number of time slices in the last 90% of the data (rounding down)
        num_time_slices_last_50pct = (total_data_points - index_50pct) // num_windows

        # Calculate the number of time slices for test and validation (rounding down)


        num_time_slices_val = num_time_slices_last_50pct // 2
        num_time_slices_test = num_time_slices_last_50pct - num_time_slices_val
        num_time_slices_train = time*nvar - num_time_slices_last_50pct

        # Calculate the number of windows for test and validation
        num_windows_test = num_time_slices_test * num_windows
        num_windows_val = num_time_slices_val * num_windows

        # Calculate the start index for the remaining time slice (if any)
        remaining_start_index = index_50pct + num_windows_test + num_windows_val
        if cut_windows:
            # Use the first 10% of the data for training
            train_data = dataset[0:index_50pct]
            if label is not None:
                train_labels = label[0:index_50pct]

            # Use the calculated number of windows for test and validation
            test_data = dataset[index_50pct:int(index_50pct + num_windows_test)]
            if label is not None:
                test_labels = label[index_50pct:int(index_50pct + num_windows_test)]
            val_data = dataset[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]
            if label is not None:
                val_labels = label[int(index_50pct + num_windows_test):int(index_50pct + num_windows_test + num_windows_val)]

            # If there is a remaining time slice, split it between test and validation to preserve the 60-40 split
            if remaining_start_index < total_data_points:
                remaining_windows = total_data_points - remaining_start_index
                split_index = remaining_windows * 50 // 100
                test_data = np.concatenate(
                    (test_data, dataset[int(remaining_start_index):int(remaining_start_index + split_index)]))
                if label is not None:
                    test_labels = np.concatenate(
                        (test_labels, label[int(remaining_start_index):int(remaining_start_index + split_index)]))
                val_data = np.concatenate((val_data, dataset[int(remaining_start_index + split_index):]))
                if label is not None:
                    val_labels = np.concatenate((val_labels, label[int(remaining_start_index + split_index):]))
        elif not cut_windows:
            train_data = dataset[0:int(index_50pct/num_windows)]
            if label is not None:
                train_labels = label[0:int(index_50pct/num_windows)]
                # use the encoder to transform the labels back to the original labels
                original_labels = encoder.inverse_transform(train_labels)
                # compute the number of labels in each class
                unique, counts = np.unique(original_labels, return_counts=True)
                # save the number of labels in each class as a text file
                with open(f"{storageloc}train_labels_{metric}_{j}{time}{modeltype}_{jobid}.txt", "w") as f:
                    for i in range(len(unique)):
                        f.write(f"{unique[i]}: {counts[i]}\n")
            val_data = dataset[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            if label is not None:
                val_labels = label[int(index_50pct/num_windows):(int(index_50pct/num_windows)+int(num_windows_val/num_windows))]
            test_data = dataset[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]
            if label is not None:
                test_labels = label[(int(index_50pct/num_windows)+int(num_windows_val/num_windows)):(int(index_50pct/num_windows)+int(num_windows_val/num_windows)+int(num_windows_test/num_windows))]

    if label is not None:
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    else:
        return train_data, val_data, test_data



def convert_np_to_xr(np_arrays, titles=None):
    das = []
    set_values = [f'set{i+1}' for i in range(len(np_arrays))]
    if titles:
        set_values = titles
    # stick the dssims and predictions into an xarray DataArray
    da = xr.DataArray(np_arrays)
    lat_values = np.linspace(-90, 90, np_arrays.shape[1])
    lon_values = np.linspace(0, 360, np_arrays.shape[2])
    len_values = np.arange(np_arrays.shape[0])
    # Rotate the longitude values by 180 degrees
    lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
    da.coords['latitude'] = (('dim_1'), lat_values)
    da.coords['longitude'] = (('dim_2'), lon_values)
    # da = da.expand_dims({'length': len_values})
    # Convert the DataArray to a Dataset
    da = da.rename({'dim_0': 'time', 'dim_1': 'latitude', 'dim_2': 'longitude'})
    # Assuming dssims_da is your DataArray and you've renamed the dimensions to 'latitude' and 'longitude'
    da['latitude'].attrs['units'] = 'degrees_north'
    da['longitude'].attrs['units'] = 'degrees_east'
    da.attrs['units'] = ''
    # ds = xr.concat(das, dim=xr.DataArray(set_values, dims='collection'))
    ldcpy_da = da.to_dataset(name='array')
    ldcpy_da.attrs['data_type'] = 'cam-fv'
    ldcpy_da.attrs["dtype"] = np.float64
    return ldcpy_da

def get_data_labels(dataset: xr.Dataset, labels: np.ndarray, time, varname, nvar, storageloc,
                                   testset="random", j=None, plotdir=None, window_size=WINDOWSIZE, only_data=False, modeltype="cnn", feature=None, featurelist=None, transform="quantile", jobid=0, cut_windows=True, metric=["dssim"], vars=None) -> float:
    """
    Train a CNN for DSSIM regression and return the average error.

    Parameters:
        dataset (xr.Dataset): The input dataset.
        dssim (np.ndarray): The DSSIM values.
        time (int): The time parameter.
        varname (str): The variable name.
        nvar (int): The number of variables.
        storageloc (str): The storage location.
        testset (str, optional): The test set type. Defaults to "random".
        j (int, optional): An optional parameter. Defaults to None.
        plotdir (str, optional): The plot directory. Defaults to None.

    Returns:
        float: The average error.
    """
    # model_path = f"model_{j}{comp}.h5"
    # model_path = ""
    average_error_path = os.path.join(storageloc, "average_error.txt")

    if not only_data:

        label_encoder = LabelEncoder()
        integer_encoded_labels = label_encoder.fit_transform(labels)

    # let's go backwards to the original labels
    scores = []
    av_preds = []
    av_dssims = []


    #### SECTIONS TO REVIEW ^^^^^^^^ ####

    # conver the data to a numpy array

    if not only_data:
        newlabels = np.array(integer_encoded_labels)

        random.seed(1)
        train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data_old(dataset, newlabels,
                                                                                            time, nvar, testset,
                                                                                                LATS, LONS, cut_windows,
                                                                                                encoder=label_encoder,
                                                                                                storageloc=storageloc,
                                                                                                metric=metric,
                                                                                                modeltype=modeltype,
                                                                                                jobid=jobid, j=j, vars=vars)
    else:

        random.seed(1)
        train_data, val_data, test_data = split_data_old(dataset, None, time, nvar, testset, LATS, LONS, cut_windows, encoder=None, storageloc=storageloc, metric=metric, modeltype=modeltype, jobid=jobid, j=j, vars=vars)



    if modeltype == "cnn":
        train_data_np = (
            train_data['combined'].transpose('sample', 'lat', 'lon').values
            if isinstance(train_data, xr.Dataset) else train_data
        )
        val_data_np = (
            val_data['combined'].transpose('sample', 'lat', 'lon').values
            if isinstance(val_data, xr.Dataset) else val_data
        )
        test_data_np = (
            test_data['combined'].transpose('sample', 'lat', 'lon').values
            if isinstance(test_data, xr.Dataset) else test_data
        )
    else:
        train_data_np = train_data
        val_data_np = val_data
        test_data_np = test_data

    # Adjust the labels if necessary (ensure they are integers starting from 0)
    train_labels_np = train_labels.astype(int)
    val_labels_np = val_labels.astype(int)
    test_labels_np = test_labels.astype(int)

    # save the data and labels as numpy arrays
    np.save(f"{storageloc}/train_data_{j}{time}{modeltype}{jobid}.npy", train_data_np)
    np.save(f"{storageloc}/val_data_{j}{time}{modeltype}{jobid}.npy", val_data_np)
    np.save(f"{storageloc}/test_data_{j}{time}{modeltype}{jobid}.npy", test_data_np)
    np.save(f"{storageloc}/train_labels_{j}{time}{modeltype}{jobid}.npy", train_labels_np)
    np.save(f"{storageloc}/val_labels_{j}{time}{modeltype}{jobid}.npy", val_labels_np)
    np.save(f"{storageloc}/test_labels_{j}{time}{modeltype}{jobid}_{vars[0]}.npy", test_labels_np)
    # save the encoding also
    with open(f"{storageloc}/label_encoder_{j}{time}{modeltype}{jobid}_{vars[0]}.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    return

def train_cnn(
        train_data_np,
        train_labels_np,
        val_data_np,
        val_labels_np,
        test_data_np,
        test_labels_np,
        modeltype,
        transform,
        featurelist=None,
        return_fit_seconds=False,
):
    """
    Train a model for the given data and labels.

    Parameters:
        train_data_np (np.ndarray): The training data.
        train_labels_np (np.ndarray): The training labels.
        val_data_np (np.ndarray): The validation data.
        val_labels_np (np.ndarray): The validation labels.
        test_data_np (np.ndarray): The testing data.
        test_labels_np (np.ndarray): The testing labels.
        modeltype (str): The model type ("cnn", "rf", "mlp").
        transform (str): The transformation type (e.g., "quantile").
        featurelist (list of str, optional): List of features to compute using ldcpy.
        return_fit_seconds (bool): Return the model.fit wall time with the model.

    Returns:
        Trained model, optionally paired with model.fit elapsed seconds.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Apply transformation if specified
    if transform == "quantile":
        # Flatten the data for quantile_transform
        n_samples_train = train_data_np.shape[0]
        n_samples_val = val_data_np.shape[0]
        n_samples_test = test_data_np.shape[0]

        train_data_reshaped = train_data_np.reshape(n_samples_train, -1)
        val_data_reshaped = val_data_np.reshape(n_samples_val, -1)
        test_data_reshaped = test_data_np.reshape(n_samples_test, -1)

        # Apply the quantile transform
        train_data_transformed = quantile_transform(
            train_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )
        val_data_transformed = quantile_transform(
            val_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )
        test_data_transformed = quantile_transform(
            test_data_reshaped,
            output_distribution="uniform",
            copy=True,
            n_quantiles=1000,
        )

        # Reshape back to original shape
        train_data_np = train_data_transformed.reshape(train_data_np.shape)
        val_data_np = val_data_transformed.reshape(val_data_np.shape)
        test_data_np = test_data_transformed.reshape(test_data_np.shape)

    # Compute features using ldcpy if featurelist is provided

    # Now, depending on modeltype, build and train the model
    # Define filter sizes, dropout rate, and number of convolutional layers as needed
    filter1 = 32
    filter2 = 64
    dropout = 0.5
    conv_layers = 2
    # Seed Python, NumPy, and TensorFlow and require deterministic GPU ops.
    model_random_seed = int(os.environ.get("MODEL_RANDOM_SEED", "2"))
    tf.keras.utils.set_random_seed(model_random_seed)
    tf.config.experimental.enable_op_determinism()

    all_labels = np.concatenate([train_labels_np, test_labels_np])
    num_classes = len(np.unique(all_labels))


    if modeltype == "cnn":
        # Update input_shape
        input_shape = train_data_np.shape[1:]  # Now input_shape is (192, 288, 1)
        i = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(filter1, kernel_size=(3, 3), activation="relu")(i)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(3, 3), activation="relu")(x)
        if conv_layers >= 3:
            x = tf.keras.layers.Conv2D(filter1, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        if conv_layers == 4:
            x = tf.keras.layers.Conv2D(filter2, kernel_size=(2, 2), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        # softmax for multi-class classification, (number of classes will be the number of unique training validation labels)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=i, outputs=outputs)
        # Compile the model for classification
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )

        model.summary()

        # Train the model
        num_epochs = 5  # Adjust as needed
        batch_size = 32  # Adjust as needed


        random.seed(model_random_seed)

        fit_started = time.perf_counter()
        model.fit(
            train_data_np,
            train_labels_np,
            validation_data=(val_data_np, val_labels_np) if val_data_np.shape[0] > 0 else None,
            epochs=num_epochs,
            batch_size=batch_size,
        )
        fit_seconds = time.perf_counter() - fit_started



        # Evaluate the model on test data
        test_loss, test_accuracy = model.evaluate(test_data_np, test_labels_np)
        print(f"Test Accuracy: {test_accuracy}")

        if return_fit_seconds:
            return model, fit_seconds
        return model

def evaluate_model(model, test_data, test_labels, return_predict_seconds=False):
    """
    Evaluate the trained model on test data.

    Parameters:
        model: The trained model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The true labels for the test data.
        return_predict_seconds (bool): Include model.predict wall time in the return.

    Returns:
        accuracy (float): The accuracy of the model on the test data.
        predictions (np.ndarray): The predicted class for each test sample.
    """
    from sklearn.metrics import accuracy_score

    # Get predicted probabilities
    predict_started = time.perf_counter()
    predictions_prob = model.predict(test_data)
    predict_seconds = time.perf_counter() - predict_started

    # Convert probabilities to class labels
    predictions = np.argmax(predictions_prob, axis=1)

    # Compute accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    result = (accuracy, predictions)
    if return_predict_seconds:
        return (*result, predict_seconds)
    return result

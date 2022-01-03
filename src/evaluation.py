import numpy as np
import pandas as pd


def corr_score(target, prediction, weights):
    """
    Calculates the weighted correlation score as defined in competition
    """
    w = np.ravel(weights)
    a = np.ravel(target)
    b = np.ravel(prediction)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return corr


def purged_walked_forward_cv(data: pd.DataFrame,
                             train_size_days: int,
                             purge_window_days: int,
                             test_size_days: int,
                             start_date: str = "2018-01-01 00:00:000",
                             dadjust: int = 1440):
    """
    Generate indices for purged walk forward cross validation.
    Returns array of tuples (train_indices, test_indices, fold_id)
    """
    # Create series of all timestamps
    timestamp_series = (pd.Series([pd.Timestamp(start_date)])
                        .append(data['timestamp'], ignore_index=True))
    all_timestamps = np.sort(timestamp_series.unique())

    # Adjust windows
    train_size = train_size_days * dadjust
    purge_size = purge_window_days * dadjust
    test_size = test_size_days * dadjust
    total_size = train_size + purge_size + test_size

    # Resulting splits
    splits = []

    # Loop over data
    train_start = 0
    fold = 1
    done = False

    while not done:

        # Check if enough remaining, otherwise reduce test_size and finish
        if train_start + total_size > len(all_timestamps):
            done = True

            train_end = train_start + train_size
            test_start = train_end + purge_size

            train_ts = all_timestamps[train_start:train_end]
            test_ts = all_timestamps[test_start:]

        train_end = train_start + train_size
        test_start = train_end + purge_size
        test_end = test_start + test_size

        train_ts = all_timestamps[train_start:train_end]
        test_ts = all_timestamps[test_start:test_end]

        # Iterate train_start
        train_start = test_end
        fold += 1

        splits.append((train_ts, test_ts, fold))

    return splits

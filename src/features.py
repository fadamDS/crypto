import pandas as pd
import numpy as np


def ohlcv_features(asset):

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    features['direct_return'] = (asset.Close - asset.Open) / asset.Open
    features['log_return'] = np.log(asset.Close / asset.Open)
    features['high_low_ratio'] = asset.High / asset.Low

    return features


def relative_features(asset, feature_cols, period):

    log_change_colnames = ['log_change_' + feature
                           + '_' + str(period) + 'min' for feature in feature_cols]
    rel_change_colnames = ['rel_change_' + feature
                           + '_' + str(period) + 'min' for feature in feature_cols]

    log_changes = np.log(asset[feature_cols] / asset[feature_cols].shift(
        period)).rename(columns=dict(zip(feature_cols, log_change_colnames)))
    rel_changes = (asset[feature_cols].diff(periods=period, axis=0) / asset[feature_cols].shift(
        period)).rename(columns=dict(zip(feature_cols, rel_change_colnames)))

    log_changes['timestamp'] = asset.timestamp
    log_changes['Asset_ID'] = asset.Asset_ID
    rel_changes['timestamp'] = asset.timestamp
    rel_changes['Asset_ID'] = asset.Asset_ID

    return log_changes, rel_changes

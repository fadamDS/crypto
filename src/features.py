import pandas as pd
import numpy as np


def create_ohlcv_features(asset):

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    features['direct_return'] = (asset.Close - asset.Open) / asset.Open
    features['log_return'] = np.log(asset.Close / asset.Open)
    features['high_low_ratio'] = asset.High / asset.Low

    return features


def create_relative_features(asset, feature_cols, period):
    assert(period > 0)

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


def create_lagged_features(asset, feature_cols, period):
    assert(period > 0)
    lagged_col_names = [
        'lag_' + str(period) + '_min_' + feature for feature in feature_cols]

    lagged_features = asset[feature_cols].shift(period)
    lagged_features = lagged_features.rename(
        columns=dict(zip(feature_cols, lagged_col_names)))

    lagged_features['timestamp'] = asset.timestamp
    lagged_features['Asset_ID'] = asset.Asset_ID

    return lagged_features


def engineer_features(asset):

    # Ensure sorting
    asset = asset.sort_values('timestamp')
    assert(
        all((asset.timestamp.diff() > pd.to_timedelta('00:00:00')).values[1:]))

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    ohlcv_features = create_ohlcv_features(asset)

    features = features.merge(ohlcv_features,
                              on=['timestamp', 'Asset_ID'],
                              how='left')

    relative_cols = ['Count', 'Open', 'High', 'Low', 'Close',
                     'Volume', 'VWAP', 'Target']
    for period in [1, 10, 30, 60]:

        log_features, rel_features = create_relative_features(asset,
                                                              feature_cols=relative_cols,
                                                              period=period)

        features = features.merge(log_features,
                                  on=['timestamp', 'Asset_ID'],
                                  how='left')

        features = features.merge(rel_features,
                                  on=['timestamp', 'Asset_ID'],
                                  how='left')
    # lagged features
    lagged_cols = ["direct_return", "direct_return", "high_low_ratio",
                   'log_change_Count_1min', 'log_change_Open_1min',
                   'log_change_High_1min', 'log_change_Low_1min',
                   'log_change_Close_1min',
                   'log_change_Volume_1min', 'log_change_VWAP_1min',
                   'log_change_Target_1min']

    for period in [1, 2, 3, 4, 5]:
        lagged_features = create_lagged_features(features,
                                                 feature_cols=lagged_cols,
                                                 period=period)

        features = features.merge(lagged_features,
                                  on=['timestamp', 'Asset_ID'],
                                  how='left')
    return features, features.columns[2:]

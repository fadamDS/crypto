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

    log_changes = np.log(asset[feature_cols] / asset[feature_cols].shift(
        period)).rename(columns=dict(zip(feature_cols, log_change_colnames)))

    log_changes['timestamp'] = asset.timestamp
    log_changes['Asset_ID'] = asset.Asset_ID

    return log_changes


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


def create_rolling_features(asset,
                            func,
                            feature_cols,
                            period,
                            quantile=None):

    assert(func in ['mean', 'median', 'std', 'quantile'])
    assert(period > 0)

    feature_colnames = ['rolling_' + func + '_' + feature
                        + '_' + str(period) + 'min' for feature in feature_cols]
    if func == 'mean':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).mean()
    elif func == 'median':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).median()
    elif func == 'std':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).std()
    elif func == 'quantile':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).median()
        feature_colnames = [str(quantile) + '_'
                            + col for col in feature_colnames]

    features = features.rename(columns=dict(
        zip(feature_cols, feature_colnames)))

    features['timestamp'] = asset.timestamp
    features['Asset_ID'] = asset.Asset_ID

    return features


def engineer_all_features(asset):

    # Ensure sorting
    #asset = asset.sort_values('timestamp')
    #assert(
    #    all((asset.timestamp.diff() > pd.to_timedelta('00:00:00')).values[1:]))

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    ohlcv_features = create_ohlcv_features(asset)

    features = features.merge(ohlcv_features,
                              on=['timestamp', 'Asset_ID'],
                              how='left')

    return features, features.columns[2:]

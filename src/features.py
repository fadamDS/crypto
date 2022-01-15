import pandas as pd
import numpy as np


def create_ohlcv_features(asset):

    features = asset[['timestamp', 'Asset_ID', 'Count',
                      'Open', 'High', 'Low', 'Close',
                      'Volume', 'VWAP']].copy()
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


def engineer_all_features(asset,
                          relative_cols,
                          relative_periods,
                          lagged_cols,
                          lagged_periods):

    # Ensure sorting
    asset = asset.sort_values('timestamp')

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    ohlcv_features = create_ohlcv_features(asset)
    for col in ohlcv_features.columns:
        features[col] = ohlcv_features[col]

    for period in relative_periods:

        log_features = create_relative_features(
            asset, relative_cols, period=period)
        for col in log_features.columns:
            features[col] = log_features[col]

    for period in lagged_periods:
        lagged_features = create_lagged_features(features,
                                                 feature_cols=lagged_cols,
                                                 period=period)
        for col in lagged_features.columns:
            features[col] = lagged_features[col]

    return features


def fast_ohlcv_features(df, out_features):

    out_features[0] = df['Count']
    out_features[1] = df['Open']
    out_features[2] = df['High']
    out_features[3] = df['Low']
    out_features[4] = df['Close']
    out_features[5] = df['Volume']
    out_features[6] = df['VWAP']
    out_features[7] = (df.Close - df.Open) / df.Open
    out_features[8] = (np.log(df.Close / df.Open))
    out_features[9] = (df.High / df.Low)

    return out_features


# Relative change features
def fast_relative_features(feature_array, columns, period, out_features):

    for j in columns:

        value = np.log(out_features[j]
                       / feature_array[-period, j])
        i = np.min(np.where(np.isnan(out_features)))
        out_features[i] = value

    return out_features


def fast_lagged_features(feature_array, columns, period, out_features):

    for j in columns:
        value = feature_array[-period, j]
        i = np.min(np.where(np.isnan(out_features)))
        out_features[i] = value

    return out_features


def fast_engineer_all_features(asset,
                               asset_features,
                               feature_names,
                               relative_cols,
                               relative_periods,
                               lagged_cols,
                               lagged_periods):

    # Initialize new feature array
    current_features = np.repeat(np.nan,
                                 repeats=asset_features.shape[1])

    current_features = fast_ohlcv_features(asset, current_features)

    # Relative change features
    for period in relative_periods:
        current_features = fast_relative_features(asset_features,
                                                  columns=np.where(
                                                      np.isin(feature_names,
                                                              relative_cols))[0],
                                                  period=period,
                                                  out_features=current_features)

    # Lagged features
    for period in lagged_periods:
        current_features = fast_lagged_features(asset_features,
                                                np.where(np.isin(feature_names,
                                                                 lagged_cols))[
                                                         0],
                                                period=period,
                                                out_features=current_features)

    current_features = np.array(current_features).reshape(1, -1)

    return current_features

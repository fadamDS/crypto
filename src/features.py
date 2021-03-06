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

    feature_colnames = ['rolling$' + func + '$' + feature
                        + '$' + str(period) + '$min' for feature in feature_cols]
    if func == 'mean':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).mean()
    elif func == 'median':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).median()
    elif func == 'std':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).std(ddof=1)
    elif func == 'quantile':
        features = asset[feature_cols].rolling(
            period, min_periods=period, axis=0).quantile(quantile)
        feature_colnames = ['rolling$' + func + ';' + str(quantile) + '$' + feature
                            + '$' + str(period) + '$min' for feature in feature_cols]

    features = features.rename(columns=dict(
        zip(feature_cols, feature_colnames)))

    features['timestamp'] = asset.timestamp
    features['Asset_ID'] = asset.Asset_ID

    return features


def engineer_all_features(asset,
                          relative_cols,
                          relative_periods,
                          lagged_cols,
                          lagged_periods,
                          rolling_cols,
                          rolling_periods
                          ):

    # Ensure sorting
    asset = asset.sort_values('timestamp')

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    ohlcv_features = create_ohlcv_features(asset)
    for col in ohlcv_features.columns:
        features[col] = ohlcv_features[col]

    # relative
    for period in relative_periods:

        log_features = create_relative_features(
            asset, relative_cols, period=period)
        for col in log_features.columns:
            features[col] = log_features[col]

    # Lagged
    for period in lagged_periods:
        lagged_features = create_lagged_features(features,
                                                 feature_cols=lagged_cols,
                                                 period=period)
        for col in lagged_features.columns:
            features[col] = lagged_features[col]

    # Rolling
    for period in rolling_periods:
        for fun in ['mean', 'median', 'std']:
            rolling_features = create_rolling_features(features,
                                                       fun,
                                                       rolling_cols,
                                                       period,
                                                       quantile=None)

            for col in rolling_features.columns:
                features[col] = rolling_features[col]

        for q in [0.1, 0.25, 0.75, 0.9]:
            rolling_features = create_rolling_features(features,
                                                       'quantile',
                                                       rolling_cols,
                                                       period,
                                                       quantile=q)

            for col in rolling_features.columns:
                features[col] = rolling_features[col]

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


def fast_rolling_feature(feature_array, feature, feature_names):

    # Get function parameters
    func, variable, period, q = params_from_roll_feature_name(feature)

    # Out location in array
    i = np.where(feature_names == feature)[0][0]

    # Location of the variable needed for calculation
    j = np.where(feature_names == variable)[0][0]

    if func == 'mean':
        value = np.mean(feature_array[-period:, j])

    elif func == 'median':
        value = np.median(feature_array[-period:, j])

    elif func == 'std':
        value = np.std(feature_array[-period:, j], ddof=1)

    elif func == 'quantile':
        value = np.quantile(feature_array[-period:, j], q)

    # Now it goes to the last one which is still nan
    feature_array[-1, i] = value

    return feature_array


def params_from_roll_feature_name(feature_name):

    splitted = feature_name.split('$')
    func = splitted[1]
    period = int(splitted[-2])
    variable = splitted[2]

    if func[:5] == 'quant':
        q = float(func.split(';')[-1])
        func = func.split(';')[0]
    else:
        q = None

    return func, variable, period, q


def fast_engineer_all_features(asset,
                               asset_features,
                               feature_names,
                               relative_cols,
                               relative_periods,
                               lagged_cols,
                               lagged_periods,
                               rolling_features):

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

    # Already set up the new feature array to calculate the rolling features
    asset_features = np.append(asset_features[1:, :],
                               current_features.reshape(1, -1), axis=0)

    # Create rolling features
    for feature in rolling_features:
        asset_features = fast_rolling_feature(asset_features,
                                              feature,
                                              feature_names)

    return asset_features

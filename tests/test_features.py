import pandas as pd
import numpy as np
from src.utils import load_gresearch_raw
from src.features import (create_ohlcv_features,
                          create_relative_features,
                          create_lagged_features,
                          create_rolling_features,
                          engineer_all_features)


# Constants
col_order = ['Target', 'Weight', 'timestamp',
             'Asset_ID', 'Count', 'Open', 'High',
             'Low', 'Close', 'Volume', 'VWAP']

data_path = "data/gresearch/"
data_path_raw = data_path + 'raw/'

train, asset_info = load_gresearch_raw(data_path_raw)


def test_ohlcv_features():

    asset = train[train.Asset_ID == 1].sort_values('timestamp')

    features = create_ohlcv_features(asset)

    assert(features.direct_return.values[0] == (
        (asset.Close - asset.Open) / asset.Open).values[0])

    assert(features.log_return.values[0] == (
        np.log(asset.Close / asset.Open)).values[0])

    assert(features.high_low_ratio.values[0] == (
        asset.High / asset.Low).values[0])


def test_relative_feature():

    asset = train[train.Asset_ID == 1].sort_values('timestamp')

    feature_cols = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']

    for period in [1, 10, 30]:

        log_changes = create_relative_features(asset,
                                               feature_cols,
                                               period)

        # Check log changes
        test_case_2 = np.log(
                          asset[feature_cols].iloc[period]
                          / asset[feature_cols].iloc[0])
        same = log_changes.iloc[period, :-2].values == test_case_2.values
        assert(all(same))


def test_lagged_features():

    asset = train[train.Asset_ID == 1].sort_values('timestamp')

    feature_cols = ['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']

    for period in [1, 10, 30]:

        features = create_lagged_features(asset, feature_cols, period)
        same = (features.iloc[period, :
                              - 2].values == asset[feature_cols].iloc[0].values)
        assert(all(same))


def test_rolling_features():

    asset = pd.DataFrame({'value': np.arange(1, 101, 1),
                          'timestamp': np.arange(1, 101, 1),
                          'Asset_ID': np.repeat(999, 100)
                          })

    for period in [5, 10, 30]:
        roll1 = create_rolling_features(asset, 'mean', ['value'], period)
        name1 = f'rolling_mean_value_{period}min'
        assert(roll1[name1].iloc[period-1]
               == asset.value.iloc[:period].mean())

        #roll3 = create_rolling_features(asset, 'median', ['value'], period)
        #roll1 = create_rolling_features(asset, 'mean', ['value'], period)


def test_engineer_all_features():

    asset = train[train.Asset_ID == 1]

    features, _ = engineer_all_features(asset)

    # Same length
    assert(features.shape[0] == asset.shape[0])

    # Same timestamps
    assert(all(asset.timestamp.values == features.timestamp.values))

    # No duplicate features
    assert(features.columns.duplicated().sum() == 0)
    assert(asset.columns.duplicated().sum() == 0)

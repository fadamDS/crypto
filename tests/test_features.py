import pandas as pd
import numpy as np
from src.features import ohlcv_features, relative_features


# Constants
col_order = ['Target', 'Weight', 'timestamp', 'Asset_ID', 'Count', 'Open', 'High',
             'Low', 'Close', 'Volume', 'VWAP']

data_path = "data/gresearch/"
train = pd.read_csv(data_path + 'raw/train.csv')
train['timestamp'] = pd.to_datetime(train.timestamp, unit='s')
asset_info = pd.read_csv(data_path + 'raw/asset_details.csv')
assets = list(asset_info.Asset_ID)
train = train.merge(asset_info[['Asset_ID', 'Weight']],
                    on='Asset_ID', how='left')[col_order]


def test_ohlcv_features():

    asset = train[train.Asset_ID == 1].sort_values('timestamp')

    features = ohlcv_features(asset)

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

        log_changes, rel_changes = relative_features(asset,
                                                     feature_cols,
                                                     period)

        # Check relative changes
        test_case_1 = ((asset[feature_cols].iloc[period]
                        - asset[feature_cols].iloc[0]) / asset[feature_cols].iloc[0])
        assert(all(rel_changes.iloc[period, :-2].values == test_case_1.values))

        # Check log changes
        test_case_2 = np.log(
            asset[feature_cols].iloc[period] / asset[feature_cols].iloc[0])
        assert(all(log_changes.iloc[period,
                                    :-2].values == test_case_2.values))

import pandas as pd
import numpy as np


def ohlcv_features(asset):

    features = pd.DataFrame({'timestamp': asset.timestamp,
                             'Asset_ID': asset.Asset_ID})

    features['direct_return'] = (asset.Close - asset.Open) / asset.Open
    features['log_return'] = np.log(asset.Close / asset.Open)
    features['high_low_ratio'] = asset.High / asset.Low

    return features

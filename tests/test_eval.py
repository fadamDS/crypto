import pandas as pd
import numpy as np
from src.evaluation import corr_score


# Constants
data_path = "data/gresearch/"
train = pd.read_csv(data_path + 'raw/train.csv')
train['timestamp'] = pd.to_datetime(train.timestamp, unit='s')
asset_info = pd.read_csv(data_path + 'raw/asset_details.csv')


def test_corr_score(train,asset_info):

    # Public LB min date
    min_date = pd.to_datetime('2021-06-13 00:00:00')

    test_df = train[train.timestamp >= min_date].copy()

    # Naive predictions
    test_df['naive_prediction'] = np.log(test_df.Close / test_df.Open)
    test_df = (test_df.merge(asset_info[['Asset_ID', 'Weight']],
                             on='Asset_ID', how='left')
               .dropna(subset=['Target', 'naive_prediction']))

    score = corr_score(test_df.Target,
                       test_df.naive_prediction,
                       test_df.Weight)

    assert(score == -0.008372884483520249)

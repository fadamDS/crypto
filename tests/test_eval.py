import pandas as pd
import numpy as np
from src.evaluation import corr_score
from src.utils import recreate_gresearch_target


# Constants
data_path = "data/gresearch/"
train = pd.read_csv(data_path + 'raw/train.csv')
train['timestamp'] = pd.to_datetime(train.timestamp, unit='s')
asset_info = pd.read_csv(data_path + 'raw/asset_details.csv')
train = train.merge(asset_info[['Asset_ID', 'Weight']],
                    on='Asset_ID', how='left')


def test_corr_score():

    # Public LB min date
    min_date = pd.to_datetime('2021-06-13 00:00:00')

    test_df = train[train.timestamp >= min_date].copy()

    # Naive predictions
    test_df['naive_prediction'] = np.log(test_df.Close / test_df.Open)
    test_df = (test_df.dropna(subset=['Target', 'naive_prediction']))

    score = corr_score(test_df.Target,
                       test_df.naive_prediction,
                       test_df.Weight)

    assert(score == -0.008372884483520249)


def test_recreate_gresearch_target():

    df = recreate_gresearch_target(data=train, asset_info=asset_info,
                                   price_column='Close')

    # NaN's match between Target and recon_Target?
    assert(all(df.Target.isna() == df.recon_Target.isna()))

    # None of the targets dropped
    assert(len(df.Target) == len(df.Target))

    abserror = abs(df['Target'] - df['recon_Target']).dropna()

    # Erros sufficiently small
    np.all(abserror < 1e-13)

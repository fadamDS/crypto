import pandas as pd
import numpy as np
from src.evaluation import corr_score, purged_walked_forward_cv, score_model
from src.models import BaseCryptoLearner
from src.utils import recreate_gresearch_target, load_fold, load_gresearch_raw


# Constants
col_order = ['Target', 'Weight', 'timestamp', 'Asset_ID', 'Count',
             'Open', 'High',
             'Low', 'Close', 'Volume', 'VWAP']

data_path = "data/gresearch/"
data_path_raw = data_path + 'raw/'

train, asset_info = load_gresearch_raw(data_path_raw)


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


def test_purged_walke_forward_cv():
    splits = purged_walked_forward_cv(data=train,
                                      train_size_days=90,
                                      purge_window_days=14,
                                      test_size_days=30,
                                      start_date="2018-01-01 00:00:000",
                                      dadjust=1440)
    # Should contain something
    assert(len(splits) > 0)

    # Last equal taking into account that the last array is "incomplete"
    assert(splits[-1][0][-1] == train.timestamp.max())

    train_idx = []
    test_idx = []
    for i in range(len(splits)-1):
        train_idx.append(splits[i][0])
        test_idx.append(splits[i][1])

    # No Overlap
    assert(len(set(np.concatenate(train_idx))
               & set(np.concatenate(test_idx))) == 0)


def test_score_model():

    train_df = load_fold(data_path + 'processed/fold_1/train/')
    test_df = load_fold(data_path + 'processed/fold_1/test/')

    base_model = BaseCryptoLearner(assets=test_df.Asset_ID.unique())
    base_model.train(train_df)

    score = score_model(test_df, base_model)

    assert(score == -0.0031253889651972947)

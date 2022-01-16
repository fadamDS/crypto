import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.features import engineer_all_features
from src.evaluation import purged_walked_forward_cv
from src.utils import load_gresearch_raw
from src.settings import (relative_cols, relative_periods,
                          lagged_cols, lagged_periods,
                          rolling_cols, rolling_periods,
                          max_lookback_minutes)


def main(head_path='../data/gresearch/',
         log=True):

    raw_data_dir = head_path + 'raw/'

    if log:
        print('Loading data')

    data, asset_info = load_gresearch_raw(raw_data_dir)

    # Get splits
    print('Getting Splits')
    splits = purged_walked_forward_cv(data,
                                      train_size_days=90,
                                      purge_window_days=7,
                                      test_size_days=40,
                                      start_date="2018-01-01 00:00:000",
                                      dadjust=1440)

    # Save as info
    with open(head_path + 'info/splits.pkl', 'wb') as f:
        pickle.dump(splits, f)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot and save as info
    for split in splits:

        train_ts = split[0]
        test_ts = split[1]
        fold = split[2]

        ax.plot(train_ts, np.repeat(fold, len(train_ts)),
                color='blue', linewidth=10.0)
        ax.plot(test_ts, np.repeat(fold, len(test_ts)),
                color='red', linewidth=10.0)
    ax.set_yticks(np.arange(1, 11, 1))
    ax.set_ylabel('Fold Number')
    plt.tight_layout()
    fig.savefig(head_path + 'info/folds.png')

    asset_ids = asset_info.Asset_ID.values

    processed_path = head_path + 'processed/'

    for asset_id in asset_ids:

        asset_name = (asset_info[asset_info.Asset_ID == asset_id]
                      .Asset_Name.iloc[0].replace(' ', '_'))

        asset_full = data[data.Asset_ID == asset_id]
        print(f'Working on {asset_name}')

        # Save splits
        for split in splits:

            train_ts = split[0]
            test_ts = split[1]
            fold = split[2]

            start_time = train_ts[0] - pd.to_timedelta(max_lookback_minutes*60,
                                                       unit='s')
            end_time = test_ts[-1]

            print(f'Features for split {fold} for {asset_name}')
            asset_current_split = asset_full[(asset_full.timestamp >= start_time)
                                             & (asset_full.timestamp <= end_time)]

            # Make whole time series and forward fill if missing
            full_ts = pd.date_range(start=start_time, end=end_time,
                                    freq='min')

            full_ts = pd.DataFrame({'timestamp': full_ts})
            asset_current_split = full_ts.merge(asset_current_split,
                                                on=['timestamp'],
                                                how='left')

            if asset_current_split.isna().any().any():
                print(
                    f'Forward filling {asset_current_split.Close.isna().sum()} rows')

            asset_current_split = asset_current_split.fillna(method='ffill',
                                                             axis=0)

            features = engineer_all_features(asset_current_split,
                                             relative_cols,
                                             relative_periods,
                                             lagged_cols,
                                             lagged_periods,
                                             rolling_cols,
                                             rolling_periods)

            asset_current_split = asset_current_split[['timestamp', 'Asset_ID']].merge(
                features, on=['timestamp', 'Asset_ID'])

            train_df = asset_current_split[asset_current_split.timestamp.isin(
                train_ts)]
            test_df = asset_current_split[asset_current_split.timestamp.isin(
                test_ts)]

            # Directory
            fold_path = f'{processed_path}/fold_{fold}'
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
                os.makedirs(fold_path + '/train')
                os.makedirs(fold_path + '/test')

            # Save Full Data
            print(f'Saving {fold} split for {asset_name}')
            train_df.to_pickle(f'{fold_path}/train/{asset_name}.pkl')
            test_df.to_pickle(f'{fold_path}/test/{asset_name}.pkl')

    pd.DataFrame({'colname': train_df.columns}).to_csv(head_path
                                                       + 'info/colnames.csv')


if __name__ == '__main__':
    main(head_path='data/gresearch/')

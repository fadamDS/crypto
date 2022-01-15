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
                          rolling_cols, rolling_periods)


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
        print(f'Writing {asset_name}')

        asset = data[data.Asset_ID == asset_id]

        features = engineer_all_features(asset,
                                         relative_cols,
                                         relative_periods,
                                         lagged_cols,
                                         lagged_periods,
                                         rolling_cols,
                                         rolling_periods)

        asset_full = asset.merge(features,
                                 on=['timestamp', 'Asset_ID'])

        # To Pickle
        asset_full.to_pickle(f'{processed_path}/full/{asset_name}.pkl')

        # Save splits
        print(f'Writing splits for {asset_name}')
        for split in splits:
            train_ts = split[0]
            test_ts = split[1]
            fold = split[2]

            train_df = asset_full[asset_full.timestamp.isin(train_ts)]
            test_df = asset_full[asset_full.timestamp.isin(test_ts)]

            # Directory
            fold_path = f'{processed_path}/fold_{fold}'
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
                os.makedirs(fold_path + '/train')
                os.makedirs(fold_path + '/test')

            # Save Full Data
            train_df.to_pickle(f'{fold_path}/train/{asset_name}.pkl')
            test_df.to_pickle(f'{fold_path}/test/{asset_name}.pkl')

    pd.DataFrame({'colname': train_df.columns}).to_csv(head_path
                                                       + 'info/colnames.csv')


if __name__ == '__main__':
    main(head_path='data/gresearch/')

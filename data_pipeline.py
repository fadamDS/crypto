import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from src.features import engineer_all_features
from src.evaluation import purged_walked_forward_cv


def main(head_path='../data/gresearch/',
         log=True):

    raw_data_dir = head_path + 'raw/'

    col_order = ['Target', 'Weight',
                 'timestamp', 'Asset_ID',
                 'Count', 'Open', 'High',
                 'Low', 'Close', 'Volume', 'VWAP']

    if log:
        print('Loading data')

    data = pd.read_csv(raw_data_dir + 'train.csv')

    asset_info = pd.read_csv(
        raw_data_dir + 'asset_details.csv').sort_values('Asset_ID')

    data['timestamp'] = pd.to_datetime(data.timestamp, unit='s')

    data = data.merge(asset_info[['Asset_ID', 'Weight']],
                      on='Asset_ID',
                      how='left')[col_order]

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
    fig.savefig('../data/gresearch/' + 'info/folds.png')

    asset_ids = asset_info.Asset_ID.values

    processed_path = head_path + 'processed/'

    for asset_id in asset_ids:

        asset_name = (asset_info[asset_info.Asset_ID == asset_id]
                      .Asset_Name.iloc[0].replace(' ', '_'))
        print(f'Writing {asset_name}')

        asset = data[data.Asset_ID == asset_id]

        features, _ = engineer_all_features(asset)

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


if __name__ == '__main__':
    main(head_path='data/gresearch/')

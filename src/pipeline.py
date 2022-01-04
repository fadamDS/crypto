import pandas as pd
from src.features import engineer_all_features


def create_training_data(raw_data_dir,
                         output_dir=None,
                         log=True):

    col_order = ['Target', 'Weight',
                 'timestamp', 'Asset_ID',
                 'Count', 'Open', 'High',
                 'Low', 'Close', 'Volume', 'VWAP']

    data = pd.read_csv(raw_data_dir + 'train.csv')
    asset_info = pd.read_csv(
        raw_data_dir + 'asset_details.csv').sort_values('Asset_ID')

    data['timestamp'] = pd.to_datetime(data.timestamp, unit='s')

    data = data.merge(asset_info[['Asset_ID', 'Weight']],
                      on='Asset_ID',
                      how='left')[col_order]

    asset_ids = asset_info.Asset_ID.values

    asset_names = asset_info.Asset_Name.values

    training_data = []

    for i, asset_id in enumerate(asset_ids):
        if log:
            print(asset_names[i])

        asset = data[data.Asset_ID == asset_id]

        features, feature_names = engineer_all_features(asset)

        asset_full = asset.merge(features, on=['timestamp', 'Asset_ID'])
        training_data.append(asset_full)

    training_data = pd.concat(training_data)

    if output_dir is None:
        return training_data

    else:
        training_data.to_csv(output_dir + 'train_process.csv')

import pandas as pd
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

    asset_ids = asset_info.Asset_ID.values

    for asset_id in asset_ids:

        asset_name = (asset_info[asset_info.Asset_ID == asset_id]
                      .Asset_Name.iloc[0].replace(' ', '_'))
        print(f'Writing {asset_name}')

        asset = data[data.Asset_ID == asset_id]

        features, _ = engineer_all_features(asset)

        training_data = asset.merge(features, on=['timestamp', 'Asset_ID'])

        # To Pickle
        training_data.to_pickle(f'{head_path}/processed/{asset_name}.pkl')


if __name__ == '__main__':
    main(head_path='data/gresearch/')

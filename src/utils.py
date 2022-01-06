# Data Utils
import pandas as pd
import numpy as np
import os

klines_colnames = ['openTime', 'open', 'high',
                   'low', 'close', 'volume',
                   'closeTime', 'quoteAssetVolume',
                   'numberTrades', 'takerBuyBaseAssetVolume',
                   'takerBuyQuoteAssetVolume', 'Ignore']

assets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def load_klines(path, colnames):

    data = pd.read_csv(path, names=colnames)

    data['openTime'] = pd.to_datetime(data.openTime, unit='ms')
    data['closeTime'] = pd.to_datetime(data.closeTime, unit='ms')

    return data


def recreate_gresearch_target(data: pd.DataFrame, asset_info: pd.DataFrame,
                              price_column: str):

    timestamp_series = (pd.Series([pd.Timestamp("2018-01-01 00:00")])
                        .append(data['timestamp'], ignore_index=True))

    all_timestamps = np.sort(timestamp_series.unique())

    targets = pd.DataFrame(index=all_timestamps)

    # Calculate future returns
    for asset_id in asset_info.Asset_ID.values:
        asset = data[data.Asset_ID == asset_id].set_index('timestamp')
        price = pd.Series(index=all_timestamps, data=asset['Close'])
        targets[asset_id] = (
            (price.shift(periods=-16) / price.shift(periods=-1)) - 1)

    targets['market_future_return'] = np.average(
        targets.fillna(0), axis=1, weights=asset_info.Weight)
    market = targets.market_future_return

    # Numerator of the beta calculation
    num = targets.multiply(market.values, axis=0).rolling(3750).mean().values

    # Denominator of the beta calculation
    denom = market.multiply(market.values, axis=0).rolling(3750).mean().values

    # Calculate Beta
    beta = np.nan_to_num(num.T / denom, nan=0., posinf=0., neginf=0.)

    # Reconstructed target
    recon_targets = targets - (beta * market.values).T

    recon_targets = pd.melt(recon_targets.reset_index(), id_vars='index')

    recon_targets = recon_targets.rename(
        columns={'index': 'timestamp',
                 'variable': 'Asset_ID',
                 'value': 'recon_Target'})
    data = data.merge(recon_targets, on=['Asset_ID', 'timestamp'])

    return data


def load_fold(data_dir):

    pickles = os.listdir(data_dir)
    data = []
    for pkl in pickles:
        if pkl[-3:] == 'pkl':
            data.append(pd.read_pickle(data_dir + pkl))

    data = pd.concat(data)

    return data

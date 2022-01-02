# Data Utils
import pandas as pd

klines_colnames = ['openTime', 'open', 'high',
                   'low', 'close', 'volume',
                   'closeTime', 'quoteAssetVolume',
                   'numberTrades', 'takerBuyBaseAssetVolume',
                   'takerBuyQuoteAssetVolume', 'Ignore']


def load_klines(path, colnames):

    data = pd.read_csv(path, names=colnames)

    data['openTime'] = pd.to_datetime(data.openTime, unit='ms')
    data['closeTime'] = pd.to_datetime(data.closeTime, unit='ms')

    return data

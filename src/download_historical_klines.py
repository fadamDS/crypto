
# Download historical clines data
import requests
from zipfile import ZipFile
from io import BytesIO


def download_montly_clines(base_url, year, month, freq, instrument, save_dir):

    url = (f'{base_url}/{instrument}/'
           f'{freq}/{instrument}-{freq}-{year}-{month}.zip')

    r = requests.get(url)
    if r.status_code == 200:
        print(url)
        zipfile = ZipFile(BytesIO(r.content))
        zipfile.extractall(path=save_dir)
    else:
        print(f'{url} not found - response {r.status_code}')


if __name__ == '__main__':

    save_dir = '/Users/adam/Documents/Code/crypto/data/binance'

    base_url = 'https://data.binance.vision/data/spot/monthly/klines'

    years = [2017, 2018, 2019, 2020, 2021]
    months = ['01',  '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']

    freq = '5m'

    for year in years:
        for month in months:
            download_montly_clines(base_url,
                                   year,
                                   month,
                                   freq,
                                   instrument='BTCUSDT',
                                   save_dir=save_dir)

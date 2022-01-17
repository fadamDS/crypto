import numpy as np

assets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

weights = {2: 2.3978952727983707,
           0: 4.30406509320417,
           1: 6.779921907472252,
           5: 1.3862943611198906,
           7: 2.079441541679836,
           6: 5.8944028342648505,
           9: 2.3978952727983707,
           11: 1.6094379124341005,
           13: 1.791759469228055,
           12: 2.079441541679836,
           3: 4.406719247264253,
           8: 1.0986122886681098,
           10: 1.0986122886681098,
           4: 3.555348061489413}

names = {2: 'Bitcoin_Cash',
         0: 'Binance_Coin',
         1: 'Bitcoin',
         5: 'EOS.IO',
         7: 'Ethereum_Classic',
         6: 'Ethereum',
         9: 'Litecoin',
         11: 'Monero',
         13: 'TRON',
         12: 'Stellar',
         3: 'Cardano',
         8: 'IOTA',
         10: 'Maker',
         4: 'Dogecoin'}

non_train_cols = ['Target', 'Weight',
                  'timestamp', 'Asset_ID', 'forward_filled']

target = 'Target'

relative_cols = ['Count', 'Open',
                 'High', 'Low', 'Close',
                 'Volume', 'VWAP']

relative_periods = [1, 5, 15, 30, 60, 120, 180, 360]

lagged_cols = ['direct_return', 'log_return', 'high_low_ratio',
               'log_change_Count_1min', 'log_change_Open_1min',
               'log_change_High_1min', 'log_change_Low_1min',
               'log_change_Close_1min',
               'log_change_Volume_1min', 'log_change_VWAP_1min',

               'log_change_Count_5min', 'log_change_Open_5min',
               'log_change_High_5min', 'log_change_Low_5min',
               'log_change_Close_5min',
               'log_change_Volume_5min', 'log_change_VWAP_5min',

               'log_change_Count_60min', 'log_change_Open_60min',
               'log_change_High_60min', 'log_change_Low_60min',
               'log_change_Close_60min',
               'log_change_Volume_60min', 'log_change_VWAP_60min',

               'log_change_Count_180min', 'log_change_Open_180min',
               'log_change_High_180min', 'log_change_Low_180min',
               'log_change_Close_180min',
               'log_change_Volume_180min', 'log_change_VWAP_180min']

lagged_periods = [1, 5, 60, 90]

rolling_cols = ['log_return',
                'log_change_Count_1min', 'log_change_Open_1min',
                'log_change_High_1min', 'log_change_Low_1min',
                'log_change_Close_1min',
                'log_change_Volume_1min', 'log_change_VWAP_1min']

rolling_periods = [15, 30, 90, 180, 360]

max_lookback_minutes = np.max(
                  np.concatenate((lagged_periods,
                                  relative_periods,
                                  rolling_periods))
                  )

dart_base_params = {'objective': 'mae',
                    'boosting': 'dart',
                    'num_iterations': 200,
                    'learning_rate': 0.1,
                    'num_leaves': 20,
                    'tree_learner': 'feature',
                    'num_threads': 2,
                    'max_depth': 40,
                    'min_data_in_leaf': 40,
                    'feature_fraction': 0.8,
                    'lambda_l1': 0.01,
                    'lambda_l2': 0.01,
                    'drop_rate': 0.15,
                    'skip_drop': 0.5}

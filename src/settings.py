import numpy as np


non_train_cols = ['Target', 'Weight', 'timestamp',
                  'Asset_ID', 'Count', 'Open',
                  'High', 'Low', 'Close',
                  'Volume', 'VWAP']

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

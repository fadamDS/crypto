import numpy as np

relative_cols = ['Count', 'Open',
                 'High', 'Low', 'Close',
                 'Volume', 'VWAP']

relative_periods = [1, 60]

lagged_cols = ['direct_return', 'log_return', 'high_low_ratio',
               'log_change_Count_1min', 'log_change_Open_1min',
               'log_change_High_1min', 'log_change_Low_1min',
               'log_change_Close_1min',
               'log_change_Volume_1min', 'log_change_VWAP_1min']

lagged_periods = [1, 2, 3, 4, 5]

non_train_cols = ['Target', 'Weight', 'timestamp',
                  'Asset_ID', 'Count', 'Open',
                  'High', 'Low', 'Close',
                  'Volume', 'VWAP']

rolling_cols = ['log_return']

rolling_periods = [15, 30, 90, 180]

max_lookback_minutes = np.max(
                              np.concatenate((lagged_periods,
                                              relative_periods,
                                              rolling_periods))
                              )

import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from scipy import stats
from src.settings import assets, dart_base_params, weights, names
from src.evaluation import corr_score


class CryptoDART():

    def __init__(self,
                 assets: list = assets,
                 weights: dict = weights,
                 names: dict = names,
                 params: dict = dart_base_params,
                 ):
        self.assets = assets
        self.weights = weights
        self.names = names
        self.params = params
        self.models = {}

    def make_data(self,
                  features: list,
                  target: str,
                  train: pd.DataFrame,
                  test: pd.DataFrame = None):
        """
        Training and testing lgb data for each coin
        """
        self.features = features
        self.target = target
        self.data = {}

        for asset_id in self.assets:
            print(f'Making data for {asset_id}')

            asset_train = train[train.Asset_ID == asset_id]
            asset_lgb_train = lgb.Dataset(data=asset_train[features],
                                          label=asset_train[target],
                                          feature_name=features,)
            self.data[asset_id] = {'lgb_train': asset_lgb_train}

            if test is not None:
                self.test_available = True
                asset_test = test[test.Asset_ID == asset_id]

                asset_lgb_test = lgb.Dataset(data=asset_test[features],
                                             label=asset_test[target])

                self.data[asset_id]['lgb_test'] = asset_lgb_test

        print('Data creation done')
        print('------------\n')

    def train(self):
        """
        Train models for each coin
        """

        for asset_id in self.assets:

            print(f'Training {asset_id}')

            lgb_train = self.data[asset_id]['lgb_train']

            dart = lgb.train(params=self.params,
                             train_set=lgb_train)

            self.models[asset_id] = dart
            print('------------\n')

        print(f'Training for {asset_id} done')
        print('------------\n')

    def run_full_test(self):

        if self.test_available:
            print('Running test')
            results_list = []
            for asset_id in self.assets:
                print(asset_id)
                lgb_test = self.data[asset_id]['lgb_test']
                dart = self.models[asset_id]
                predictions = dart.predict(lgb_test.data)
                target = lgb_test.label['Target'].values

                result = pd.DataFrame(
                    {'prediction': predictions, 'target': target})
                result['Asset_ID'] = asset_id
                result['Weight'] = self.weights[asset_id]
                results_list.append(result)

            self.results_df = pd.concat(results_list)

            self.test_score = corr_score(self.results_df.target,
                                         self.results_df.prediction,
                                         self.results_df.Weight)
        else:
            print('No test data available')

    def save_models(self,
                    save_dir: str,
                    base_name: str):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for asset_id in self.assets:

            dart = self.models[asset_id]
            name = self.names[asset_id]

            save_path = save_dir + base_name + name + '.txt'
            dart.save_model(save_path)

    def save_test_results(self, save_path, remove_ffil=False):

        pearsons = []
        perason_p_vals = []
        mae = []
        names = []
        for asset_id in self.assets:

            result_data = self.results_df[self.results_df.Asset_ID == asset_id]

            # Remove forward fill to avoid overestimation
            if remove_ffil:
                result_data = result_data[result_data.forward_filled == False]

            pearson_res = stats.pearsonr(
                result_data.target, result_data.prediction)
            pearsons.append(pearson_res[0])
            perason_p_vals.append(pearson_res[1])
            mae.append(
                np.mean(np.abs(result_data.prediction - result_data.target)))
            names.append(self.names[asset_id])

        pd.DataFrame({'full_score': self.test_score,
                      'asset_id': self.assets,
                      'asset_name': names,
                      'pearson': pearsons,
                      'pearson_pval': perason_p_vals,
                      'mae': mae}).to_csv(save_path)

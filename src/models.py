import numpy as np
import pandas as pd


class BaseCryptoLearner():

    def __init__(self, assets=None):
        self.initialized = True
        self.learned_mean = {}
        self.assets = assets

    def train(self, data):
        # Just take the target median over the sample for each coin
        for asset in self.assets:
            mean = np.mean(data[data.Asset_ID == asset].Target)
            if np.isnan(mean):
                mean = 0
            self.learned_mean[asset] = mean

    def predict(self, X):
        # Should always be with (AssetID & Timestamp)
        # Predict the learned mean
        self.predictions = []

        for asset in self.assets:

            pred = pd.DataFrame({'Asset_ID': asset,
                                 'timestamp': X[X.Asset_ID == asset].timestamp})

            pred['prediction'] = np.repeat(
                self.learned_mean[asset], repeats=pred.shape[0])
            self.predictions.append(pred)

        self.predictions_df = pd.concat(self.predictions)

        return self.predictions_df

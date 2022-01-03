import numpy as np
import pandas as pd


class BaseCryptoLearner():

    def __init__(self, assets=None):
        self.initialized = True
        self.learned_mean = {}
        self.assets = assets

    def train(self, data):
        # Just take the target distribution median over the sample
        self.learned_mean = np.mean(data.Target)

    def predict(self, X):  # Should always be with (AssetID & Timestamp)
        # Predict the learned mean
        pred = pd.DataFrame({'Asset_ID': X.Asset_ID,
                             'timestamp': X.timestamp})
        pred['prediction'] = np.repeat(self.learned_mean, repeats=X.shape[0])

        return pred

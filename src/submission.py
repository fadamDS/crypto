import numpy as np
from src.features import fast_engineer_all_features


def make_prediction(assets,
                    features,
                    feature_names,
                    relative_cols,
                    relative_periods,
                    lagged_cols,
                    lagged_periods,
                    model,
                    test_df,
                    prediction_df):

    new_features = []

    for asset_id in assets:

        # Check if asset in test_df, if not forward fill
        if asset_id not in test_df.Asset_ID.values:
            new_features.append(features[asset_id][-1].reshape(1, -1))
            continue

        asset = test_df[test_df.Asset_ID == asset_id].iloc[0]

        row_id = asset['row_id']

        asset_features = features[asset_id]

        current_features = fast_engineer_all_features(asset,
                                                      asset_features,
                                                      feature_names,
                                                      relative_cols,
                                                      relative_periods,
                                                      lagged_cols,
                                                      lagged_periods)

        pred = model.predict(np.array(current_features))[0]
        prediction_df.loc[prediction_df.row_id == row_id, 'Target'] = pred

        # Append data
        new_features.append(current_features)

    new_features = np.array(new_features)

    # Append to old array and shift one forward
    features = np.append(features, new_features, axis=1)
    features = features[:, 1:, :]

    return features, prediction_df

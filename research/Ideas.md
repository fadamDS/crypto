# Crypto Market Prediction G-Research

# Overview
- Forecast Residual log-returns for the asset over a 15 minute horizon.
- Residual: Market component removed
- Minute by minute data is used

# Target Definition
- Target at time of time series alpha t:
- Predict the log return in the next 15 minutes

# Evaluation
- Calculate average weighted market return
- Sum of weighted individual returns divided by sum of weights
- Weights are provided in asset information
- Calculate beta of an asset as rolling average correlation between individual return and market
- target is the individual return minus the beta times the market return
- Finally: Score is weighted Pearson correlation coefficient between target and actual


## To Do's

- Set up that resulting models can be applied to submission setting
  - Fitting 
  - Updating of time series params
  - LightGBM (no hyperparameter / other fancy stuff)

- Pipeline
  - Try modelling on one split.
- Features -> Make features on whole dataset (e.g lagged, EWMA)
  - Vola
  - Rolling averages etc.
- Model
  - Single model vs. model for each coin
  - boosting
  - Simple model, e.g. ARMA
  - Combine models, learn jointly
- Still overfitting

- Adjust the missing time deltas

## Modelling
- Ensembles of simple models
- XGB, LGBM, etc..
- Throw many simple models at it, join them in a smart way potentially
- Some smart feature engineering!
  - Lagged features
  - Market features
  - Modelled Vola

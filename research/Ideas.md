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

# Score Tracker
- 0.026 btc fold 9

## To Do's
- Features
    - Check which features to add / remove based on importance

- Submission:
  - Check whether something can be done to avoid empty warm up period

- Supplemental train updated & how to incorporate?

- Model
  - Top features?
  - Correlation over time?
  - Hyperparameters, especially avoid overfitting
  - Deep Learning / MLP

### Backlog
- Features
  - Time features
  - Transforms (e.g. fft)

- Model
  - Combine models, learn jointly
  - Single model vs. model for each coin
  - Deep Learning models
  - Simple model, e.g. ARMA

# Stock Price Prediction - Nifty 50 Forecasting

## Project Overview

This project implements and compares multiple machine learning and deep learning models for predicting next-day closing prices of Nifty 50 stock index. The study evaluates model performance across four distinct market regimes to test robustness and generalization capability.

## Dataset

- Source: Nifty 50 historical data (2000-2025)
- Features: Date, Open, High, Low, Close
- Total records: 6315 daily observations

## Feature Engineering

The following technical indicators were created from raw price data:

- Returns (daily percentage change)
- Simple Moving Averages (SMA-10, SMA-50)
- Exponential Moving Average (EMA-10)
- Volatility (10-day rolling standard deviation of returns)
- Relative Strength Index (RSI - 14 day)
- Bollinger Bands (Upper, Middle, Lower - 20 day)

All features use shift(1) to prevent data leakage, ensuring only past information is used for prediction.

## Models Implemented

### Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression

### Tree-Based Models
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gradient Boosting Regressor

### Deep Learning Models
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

### Time Series Models
- ARIMA (5,1,0)

### Ensemble Methods
- Stacking Regressor (Linear + Ridge + RandomForest)

## Train-Test Splits

Four temporal splits created to test robustness across different market conditions:

| Split | Training Period | Testing Period | Market Regime |
|-------|-----------------|----------------|---------------|
| 1 | 2000-2015 | 2016-2018 | Pre-COVID Stable |
| 2 | 2000-2018 | 2019-2020 | COVID Era |
| 3 | 2000-2020 | 2021-2022 | Post-COVID Volatility |
| 4 | 2000-2022 | 2023-2024 | Recent Recovery |

## Evaluation Metrics

- MAE (Mean Absolute Error) - Prediction accuracy in rupees
- RMSE (Root Mean Square Error) - Penalizes large errors
- MAPE (Mean Absolute Percentage Error) - Relative error percentage
- R2 Score - Variance explained by model
- Directional Accuracy - Percentage of correctly predicted price movement direction

## Results Summary

### Top Performing Models (Average across all splits)

| Model | R2 Score | MAE | Directional Accuracy |
|-------|----------|-----|----------------------|
| LightGBM | 0.9878 | 104.21 | 53.80% |
| XGBoost | 0.9878 | 104.03 | 53.93% |
| RandomForest | 0.9865 | 109.58 | 53.51% |
| Stacking | 0.9774 | 149.62 | 52.01% |
| Linear | 0.9772 | 149.27 | 52.10% |

### Robustness Analysis (R2 Standard Deviation)

| Model | Std Dev | Rank |
|-------|---------|------|
| LightGBM | 0.0092 | Most Robust |
| XGBoost | 0.0093 | Most Robust |
| RandomForest | 0.0107 | Robust |
| Stacking | 0.0164 | Robust |
| Linear | 0.0168 | Robust |

### Models Not Recommended

- ARIMA: R2 = -1.93, Directional Accuracy = 4.56% (worse than random)
- LSTM: Poor performance on volatile periods
- GRU: R2 drops to -0.18 on Split 3

## Key Findings

1. Tree-based models (LightGBM, XGBoost) outperform linear and deep learning models for this task
2. Feature shifting (using shift(1)) is critical to prevent data leakage
3. Deep learning models show instability during volatile market periods
4. ARIMA is unsuitable for stock price prediction with directional accuracy near 5%
5. All models achieve directional accuracy only slightly above 50%, indicating the stochastic nature of daily price movements

## Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow statsmodels matplotlib seaborn tqdm

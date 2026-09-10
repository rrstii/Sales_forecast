# Sales Forecasting with Lag Features

A time-series forecasting project that predicts daily store sales using lagged sales features and linear regression.

## Overview

This project demonstrates a simple, reproducible forecasting workflow:

- Generate a synthetic daily sales series with trend, weekly seasonality, and noise.
- Create lag features from previous observations.
- Add day-of-week and 7-day rolling-mean features.
- Preserve temporal order when splitting training and test data.
- Compare the regression model against a naive previous-day baseline.
- Evaluate performance with MAE and RMSE.
- Visualize actual sales, model predictions, and the baseline.

> **Note:** The current dataset is synthetic and is intended for demonstration. The data-generation function can be replaced with a CSV containing `date` and `sales` columns for real-world use.

## Model

The forecasting features include:

- `lag_1` through `lag_7`
- `day_of_week`
- `rolling_mean_7`

A `LinearRegression` model from scikit-learn is trained using earlier observations and evaluated on the final 30 days.

## Evaluation

In the example run, the model achieved approximately:

| Metric | Result |
|---|---:|
| Model MAE | ~7 |
| Naive baseline MAE | ~13 |
| Improvement vs. baseline | ~48% |

Results can vary if the data-generation process or random seed is changed.

## Installation

```bash
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

```bash
python forecast.py
```

The script prints the evaluation metrics and saves the forecast comparison as `forecast_result.png`.

## Project Structure

```text
.
├── forecast.py
├── requirements.txt
└── README.md
```

## Limitations & Next Steps

This is intentionally a lightweight forecasting example. A production-oriented version could add:

- real historical sales data
- stronger time-series models
- time-series cross-validation
- holiday and promotional features
- prediction intervals
- automated experiment tracking

## Tech Stack

Python · NumPy · Pandas · Matplotlib · scikit-learn

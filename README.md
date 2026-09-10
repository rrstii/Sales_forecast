# Sales Forecasting with Lag Features

A practical time-series forecasting project that predicts daily store sales from historical sales patterns using lag features, rolling statistics, and linear regression.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://github.com/rrstii/Sales_forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/rrstii/Sales_forecast/actions/workflows/ci.yml)

## Project Overview

The project demonstrates a compact end-to-end forecasting workflow:

1. Generate a synthetic daily sales series with trend, weekly seasonality, and noise.
2. Build lagged sales features (`lag_1` ... `lag_7`).
3. Add calendar and rolling-statistics features.
4. Preserve chronological order when creating the train/test split.
5. Train a `LinearRegression` model.
6. Compare the model against a naive previous-day baseline.
7. Evaluate and visualize the forecasts.

> **Data note:** The current experiment uses synthetic data for reproducibility. The forecasting functions also accept a DataFrame containing `date` and `sales` columns.

## Features

The model uses:

- **Lag features:** previous 1–7 days of sales
- **Calendar feature:** day of week
- **Rolling feature:** 7-day mean of prior sales

The final 30 observations are reserved for evaluation, while earlier observations are used for training.

## Results

With the default seed and settings, the included experiment reports:

| Metric | Value |
| --- | ---: |
| Model MAE | 6.93 |
| Naive baseline MAE | 13.23 |
| Model RMSE | 9.15 |
| Improvement vs. baseline | 47.66% |

These metrics describe the synthetic demonstration dataset and should not be interpreted as production forecasting performance.

## Visualization

The script saves a comparison of actual sales, model predictions, and the naive baseline to:

`forecast_result.png`

## Installation

```bash
git clone https://github.com/rrstii/Sales_forecast.git
cd Sales_forecast
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Usage

```bash
python forecast.py
```

Run the test suite with:

```bash
python -m pytest -q
```

## Project Structure

```text
Sales_forecast/
├── forecast.py
├── tests/
│   └── test_forecast.py
├── forecast_result.png
├── requirements.txt
├── requirements-dev.txt
├── .github/workflows/ci.yml
├── .gitignore
└── README.md
```

## Limitations and Next Steps

This repository is intentionally a lightweight forecasting example rather than a production forecasting system. Natural next steps include:

- Replace synthetic data with a real sales dataset.
- Add holiday, promotion, and external-demand features.
- Compare against stronger forecasting models.
- Use rolling time-series cross-validation.
- Add prediction intervals and uncertainty estimates.
- Track experiments and model versions systematically.

## Tech Stack

**Python · NumPy · Pandas · Matplotlib · scikit-learn · Time Series Forecasting**

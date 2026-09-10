"""Daily sales forecasting with lag-based linear regression.

The project uses reproducible synthetic data by default so the example can run
without an external dataset. A DataFrame containing ``date`` and ``sales`` can
also be passed directly to the forecasting functions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_SEED = 7
DEFAULT_DAYS = 400
DEFAULT_LAGS = 7
DEFAULT_TEST_SIZE = 30
OUTPUT_PATH = Path("forecast_result.png")


def make_fake_data(n_days: int = DEFAULT_DAYS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate reproducible synthetic daily sales data."""
    if n_days <= 0:
        raise ValueError("n_days must be greater than 0")

    rng = np.random.RandomState(seed)
    dates = pd.date_range(start="2024-06-01", periods=n_days, freq="D")
    trend = np.linspace(100, 180, n_days)
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = rng.normal(0, 8, n_days)
    sales = np.round(np.clip(trend + weekly + noise, 20, None))

    return pd.DataFrame({"date": dates, "sales": sales})


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate the columns and basic types required by the pipeline."""
    required_columns = {"date", "sales"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input data must not be empty")

    if not pd.api.types.is_numeric_dtype(df["sales"]):
        raise ValueError("'sales' must be numeric")


def add_features(df: pd.DataFrame, n_lags: int = DEFAULT_LAGS) -> pd.DataFrame:
    """Create lag, calendar, and rolling-mean features without leakage."""
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")

    result = df.copy()
    validate_input_data(result)
    result["date"] = pd.to_datetime(result["date"])

    for lag in range(1, n_lags + 1):
        result[f"lag_{lag}"] = result["sales"].shift(lag)

    result["day_of_week"] = result["date"].dt.dayofweek
    result["rolling_mean_7"] = result["sales"].shift(1).rolling(7).mean()

    return result.dropna().reset_index(drop=True)


def train_test_split_time(
    df: pd.DataFrame, test_size: int = DEFAULT_TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series chronologically, keeping future data out of training."""
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and len(df) - 1")

    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def fit_and_evaluate(
    df: pd.DataFrame, test_size: int = DEFAULT_TEST_SIZE
) -> Tuple[LinearRegression, Dict[str, float], pd.DataFrame]:
    """Train the regression model and return metrics plus test predictions."""
    feature_cols: List[str] = [
        column for column in df.columns if column.startswith("lag_")
    ] + ["day_of_week", "rolling_mean_7"]

    train, test = train_test_split_time(df, test_size=test_size)
    model = LinearRegression()
    model.fit(train[feature_cols], train["sales"])

    predictions = model.predict(test[feature_cols])
    naive_baseline = test["lag_1"].to_numpy()

    mae_model = mean_absolute_error(test["sales"], predictions)
    mae_naive = mean_absolute_error(test["sales"], naive_baseline)
    rmse_model = mean_squared_error(test["sales"], predictions) ** 0.5
    improvement = (mae_naive - mae_model) / mae_naive * 100

    results = test[["date", "sales", "lag_1"]].copy()
    results["prediction"] = predictions
    results = results.rename(columns={"lag_1": "naive_baseline"})

    metrics = {
        "mae_model": float(mae_model),
        "mae_naive": float(mae_naive),
        "rmse_model": float(rmse_model),
        "improvement_percent": float(improvement),
    }
    return model, metrics, results


def plot_forecast(results: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save a comparison plot for actual, model, and naive forecasts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(results["date"], results["sales"], label="Actual sales", marker="o", markersize=3)
    plt.plot(results["date"], results["prediction"], label="Model forecast", marker="o", markersize=3)
    plt.plot(
        results["date"],
        results["naive_baseline"],
        label="Naive baseline",
        linestyle="--",
        alpha=0.6,
    )
    plt.legend()
    plt.title("Daily Sales Forecast — Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Run the complete forecasting example."""
    raw = make_fake_data()
    featured = add_features(raw)
    _, metrics, results = fit_and_evaluate(featured)

    print(f"Model MAE:          {metrics['mae_model']:.2f}")
    print(f"Naive baseline MAE: {metrics['mae_naive']:.2f}")
    print(f"Model RMSE:         {metrics['rmse_model']:.2f}")
    print(f"Improvement:        {metrics['improvement_percent']:.1f}%")

    plot_forecast(results)
    print(f"\nChart saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import pandas as pd
import pytest

from forecast import add_features, fit_and_evaluate, make_fake_data, train_test_split_time


def test_make_fake_data_is_reproducible():
    first = make_fake_data(20, seed=11)
    second = make_fake_data(20, seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_add_features_creates_expected_rows_and_columns():
    data = make_fake_data(20)
    result = add_features(data, n_lags=7)

    assert len(result) == 13
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns
    assert "rolling_mean_7" in result.columns
    assert result.isna().sum().sum() == 0


def test_time_split_preserves_chronological_order():
    data = add_features(make_fake_data(50))
    train, test = train_test_split_time(data, test_size=10)

    assert len(test) == 10
    assert train["date"].max() < test["date"].min()


def test_invalid_test_size_raises():
    data = add_features(make_fake_data(20))
    with pytest.raises(ValueError):
        train_test_split_time(data, test_size=len(data))


def test_fit_and_evaluate_returns_finite_metrics():
    data = add_features(make_fake_data())
    _, metrics, predictions = fit_and_evaluate(data)

    assert set(metrics) == {
        "mae_model",
        "mae_naive",
        "rmse_model",
        "improvement_percent",
    }
    assert all(pd.notna(value) for value in metrics.values())
    assert len(predictions) == 30

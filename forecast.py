"""
پیش‌بینی فروش روزانه یک فروشگاه با استفاده از رگرسیون روی ویژگی‌های تاخیری (lag features)
داده واقعی نداشتم برای تست، پس یه دیتاست مصنوعی با روند + نوسان هفتگی + نویز ساختم.
اگه بخوای رو داده واقعی اجرا کنی، کافیه فایل csv با ستون‌های date و sales بدی به جای تابع make_fake_data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(7)


def make_fake_data(n_days=400):
    dates = pd.date_range(start="2024-06-01", periods=n_days, freq="D")

    trend = np.linspace(100, 180, n_days)          # فروش کم‌کم داره زیاد میشه
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # آخر هفته‌ها معمولا پرفروش‌تره
    noise = np.random.normal(0, 8, n_days)

    sales = trend + weekly + noise
    sales = np.round(np.clip(sales, 20, None))       # فروش منفی که معنی نداره

    return pd.DataFrame({"date": dates, "sales": sales})


def add_features(df, n_lags=7):
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split_time(df, test_size=30):
    # چون سری زمانیه، نمیشه رندوم split کرد - باید ترتیب زمانی حفظ بشه
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


def main():
    raw = make_fake_data()
    df = add_features(raw)

    feature_cols = [c for c in df.columns if c.startswith("lag_")] + ["day_of_week", "rolling_mean_7"]
    train, test = train_test_split_time(df, test_size=30)

    X_train, y_train = train[feature_cols], train["sales"]
    X_test, y_test = test[feature_cols], test["sales"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # baseline ساده برای مقایسه: فرض کن فردا هم مثل امروزه
    naive_baseline = test["lag_1"].values

    mae_model = mean_absolute_error(y_test, preds)
    mae_naive = mean_absolute_error(y_test, naive_baseline)
    rmse_model = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE مدل رگرسیون:  {mae_model:.2f}")
    print(f"MAE baseline ساده: {mae_naive:.2f}")
    print(f"RMSE مدل:          {rmse_model:.2f}")
    improvement = (mae_naive - mae_model) / mae_naive * 100
    print(f"مدل نسبت به baseline حدود {improvement:.1f}% بهتره")

    # رسم نمودار مقایسه
    plt.figure(figsize=(10, 5))
    plt.plot(test["date"], y_test.values, label="فروش واقعی", marker="o", markersize=3)
    plt.plot(test["date"], preds, label="پیش‌بینی مدل", marker="o", markersize=3)
    plt.plot(test["date"], naive_baseline, label="baseline ساده", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("پیش‌بینی فروش روزانه - ۳۰ روز آخر")
    plt.xlabel("تاریخ")
    plt.ylabel("فروش")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("forecast_result.png", dpi=150)
    print("\nنمودار در forecast_result.png ذخیره شد.")


if __name__ == "__main__":
    main()

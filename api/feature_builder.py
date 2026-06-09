import numpy as np
import pandas as pd


DATA_PATH = "data/raw/train.csv"


# Load once when API starts
sales_df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)


def build_features(
    store: int,
    item: int,
    forecast_date: str
):

    forecast_date = pd.to_datetime(
        forecast_date
    )

    # =====================================
    # Historical data for store-item
    # =====================================

    history = sales_df[
        (sales_df["store"] == store)
        &
        (sales_df["item"] == item)
    ].copy()

    if len(history) < 90:
        raise ValueError(
            f"Not enough history for Store={store}, Item={item}"
        )

    history = history.sort_values(
        "date"
    )

    sales = history["sales"]

    # =====================================
    # Date Features
    # =====================================

    year = forecast_date.year
    month = forecast_date.month
    day = forecast_date.day

    weekday = forecast_date.weekday()

    weekofyear = int(
        forecast_date.isocalendar().week
    )

    is_weekend = int(
        weekday in [5, 6]
    )

    # =====================================
    # Cyclical Features
    # =====================================

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    weekday_sin = np.sin(
        2 * np.pi * weekday / 7
    )

    weekday_cos = np.cos(
        2 * np.pi * weekday / 7
    )

    # =====================================
    # Lag Features
    # =====================================

    sales_lag_1 = sales.iloc[-1]
    sales_lag_7 = sales.iloc[-7]
    sales_lag_14 = sales.iloc[-14]
    sales_lag_30 = sales.iloc[-30]
    sales_lag_60 = sales.iloc[-60]
    sales_lag_90 = sales.iloc[-90]

    # =====================================
    # Rolling Means
    # =====================================

    rolling_mean_7 = (
        sales.iloc[-7:]
        .mean()
    )

    rolling_mean_30 = (
        sales.iloc[-30:]
        .mean()
    )

    # =====================================
    # Rolling Std
    # =====================================

    rolling_std_7 = (
        sales.iloc[-7:]
        .std()
    )

    rolling_std_30 = (
        sales.iloc[-30:]
        .std()
    )

    # =====================================
    # EMA Features
    # =====================================

    ema_7 = (
        sales
        .ewm(
            span=7,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    ema_30 = (
        sales
        .ewm(
            span=30,
            adjust=False
        )
        .mean()
        .iloc[-1]
    )

    # =====================================
    # Expanding Mean
    # =====================================

    expanding_mean = (
        sales.mean()
    )

    # =====================================
    # Build Feature Row
    # =====================================

    features = pd.DataFrame([
        {
            "store": store,
            "item": item,

            "year": year,
            "month": month,
            "day": day,

            "weekday": weekday,
            "weekofyear": weekofyear,
            "is_weekend": is_weekend,

            "month_sin": month_sin,
            "month_cos": month_cos,

            "weekday_sin": weekday_sin,
            "weekday_cos": weekday_cos,

            "sales_lag_1": sales_lag_1,
            "sales_lag_7": sales_lag_7,
            "sales_lag_14": sales_lag_14,
            "sales_lag_30": sales_lag_30,
            "sales_lag_60": sales_lag_60,
            "sales_lag_90": sales_lag_90,

            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_30": rolling_mean_30,

            "rolling_std_7": rolling_std_7,
            "rolling_std_30": rolling_std_30,

            "ema_7": ema_7,
            "ema_30": ema_30,

            "expanding_mean": expanding_mean
        }
    ])

    return features
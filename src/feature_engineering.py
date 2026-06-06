import numpy as np


def create_date_features(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["weekday"] = df["date"].dt.weekday
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (
        df["weekday"].isin([5, 6])
    ).astype(int)

    return df


def create_cyclical_features(df):

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["weekday_sin"] = np.sin(
        2 * np.pi * df["weekday"] / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df["weekday"] / 7
    )

    return df


def create_lag_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    for lag in [1, 7, 14, 30]:

        df[f"sales_lag_{lag}"] = (
            df.groupby(["store", "item"])["sales"]
            .shift(lag)
        )

    return df


def create_rolling_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    grouped = df.groupby(
        ["store", "item"]
    )["sales"]

    df["rolling_mean_7"] = grouped.transform(
        lambda x:
        x.shift(1)
         .rolling(7)
         .mean()
    )

    df["rolling_mean_30"] = grouped.transform(
        lambda x:
        x.shift(1)
         .rolling(30)
         .mean()
    )

    df["rolling_std_7"] = grouped.transform(
        lambda x:
        x.shift(1)
         .rolling(7)
         .std()
    )

    df["rolling_std_30"] = grouped.transform(
        lambda x:
        x.shift(1)
         .rolling(30)
         .std()
    )

    return df


def create_ema_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    grouped = df.groupby(
        ["store", "item"]
    )["sales"]

    df["ema_7"] = grouped.transform(
        lambda x:
        x.shift(1)
         .ewm(span=7, adjust=False)
         .mean()
    )

    df["ema_30"] = grouped.transform(
        lambda x:
        x.shift(1)
         .ewm(span=30, adjust=False)
         .mean()
    )

    return df
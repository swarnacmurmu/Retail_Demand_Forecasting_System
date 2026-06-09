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

    group = df.groupby(
        ["store", "item"]
    )["sales"]

    df["sales_lag_1"] = group.shift(1)

    df["sales_lag_7"] = group.shift(7)

    df["sales_lag_14"] = group.shift(14)

    df["sales_lag_30"] = group.shift(30)

    # NEW FEATURES

    df["sales_lag_60"] = group.shift(60)

    df["sales_lag_90"] = group.shift(90)

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


def create_expanding_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    grouped = df.groupby(
        ["store", "item"]
    )["sales"]

    df["expanding_mean"] = grouped.transform(
        lambda x:
        x.shift(1)
         .expanding()
         .mean()
    )

    return df
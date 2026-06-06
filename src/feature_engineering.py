def create_date_features(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["weekday"] = df["date"].dt.weekday

    df["quarter"] = df["date"].dt.quarter

    df["weekofyear"] = df["date"].dt.isocalendar().week

    df["is_weekend"] = (
        df["weekday"].isin([5, 6])
    ).astype(int)

    return df


def create_lag_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    for lag in [1, 7, 14, 30]:

        df[f"sales_lag_{lag}"] = (
            df.groupby(["store", "item"])
            ["sales"]
            .shift(lag)
        )

    return df

def create_rolling_features(df):

    df = df.sort_values(
        ["store", "item", "date"]
    )

    df["rolling_mean_7"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(window=7)
             .mean()
        )
    )

    df["rolling_mean_30"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(window=30)
             .mean()
        )
    )

    return df
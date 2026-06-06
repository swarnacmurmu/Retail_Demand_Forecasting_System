from data_preprocessing import load_data
from feature_engineering import (
    create_date_features,
    create_lag_features,
    create_rolling_features
)

df = load_data("data/raw/train.csv")

df = create_date_features(df)

df = create_lag_features(df)

df = create_rolling_features(df)

sample = df[
    (df["store"] == 1)
    & (df["item"] == 1)
]

print(
    sample[
        [
            "date",
            "sales",
            "sales_lag_1",
            "sales_lag_7",
            "rolling_mean_7",
            "rolling_mean_30"
        ]
    ].head(40)
)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
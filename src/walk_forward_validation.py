import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from data_preprocessing import load_data

from feature_engineering import (
    create_date_features,
    create_cyclical_features,
    create_lag_features,
    create_rolling_features,
    create_ema_features
)

print("Loading data...")

df = load_data("data/raw/train.csv")

print("Creating features...")

df = create_date_features(df)
df = create_cyclical_features(df)
df = create_lag_features(df)
df = create_rolling_features(df)
df = create_ema_features(df)

df = df.dropna()

print("Final Shape:", df.shape)

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

folds = [
    ("2015-01-01", "2016-01-01"),
    ("2016-01-01", "2017-01-01"),
    ("2017-01-01", "2018-01-01")
]

mae_scores = []

for fold_num, (valid_start, valid_end) in enumerate(folds, start=1):

    print("\n" + "=" * 50)
    print(f"FOLD {fold_num}")
    print("=" * 50)

    train_df = df[df["date"] < valid_start]

    valid_df = df[
        (df["date"] >= valid_start)
        & (df["date"] < valid_end)
    ]

    print("Train Shape:", train_df.shape)
    print("Valid Shape:", valid_df.shape)

    X_train = train_df[features]
    X_valid = valid_df[features]

    y_train = np.log1p(train_df["sales"])
    y_valid = valid_df["sales"]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    preds_log = model.predict(X_valid)

    preds = np.expm1(preds_log)

    mae = mean_absolute_error(
        y_valid,
        preds
    )

    mae_scores.append(mae)

    print(f"Fold {fold_num} MAE: {mae:.4f}")

print("\n")
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

print("MAE Scores:")

for i, score in enumerate(mae_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(
    f"\nAverage MAE: {np.mean(mae_scores):.4f}"
)

print(
    f"Std Dev: {np.std(mae_scores):.4f}"
)
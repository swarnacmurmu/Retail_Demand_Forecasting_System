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

# =====================================
# Train on 2013-2016
# Test on 2017
# =====================================

train_df = df[df["date"] < "2017-01-01"]

test_df = df[df["date"] >= "2017-01-01"]

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

X_train = train_df[features]
X_test = test_df[features]

y_train = np.log1p(train_df["sales"])
y_test = test_df["sales"]

print("Training model...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    loss_function="RMSE",
    verbose=False,
    random_seed=42
)

model.fit(X_train, y_train)

pred_log = model.predict(X_test)

pred = np.expm1(pred_log)

mae = mean_absolute_error(y_test, pred)

print(f"\nMAE: {mae:.4f}")

# =====================================
# Build Analysis DataFrame
# =====================================

results = test_df[
    ["date", "store", "item", "sales"]
].copy()

results["prediction"] = pred

results["absolute_error"] = (
    results["sales"] - results["prediction"]
).abs()

print("\nSaving predictions...")

results.to_csv(
    "predictions_analysis.csv",
    index=False
)

print("Saved: predictions_analysis.csv")

print("\n")
print("=" * 60)
print("TOP 20 WORST PREDICTIONS")
print("=" * 60)

worst = results.sort_values(
    "absolute_error",
    ascending=False
)

print(
    worst[
        [
            "date",
            "store",
            "item",
            "sales",
            "prediction",
            "absolute_error"
        ]
    ].head(20)
)




print("\n")
print("=" * 60)
print("MAE BY STORE")
print("=" * 60)

store_mae = (
    results.groupby("store")
    .apply(
        lambda x:
        mean_absolute_error(
            x["sales"],
            x["prediction"]
        )
    )
    .reset_index(name="mae")
)

print(
    store_mae.sort_values(
        by="mae",
        ascending=False
    )
)



print("\n")
print("=" * 60)
print("TOP 20 HARDEST ITEMS")
print("=" * 60)

item_mae = (
    results.groupby("item")
    .apply(
        lambda x:
        mean_absolute_error(
            x["sales"],
            x["prediction"]
        )
    )
    .reset_index(name="mae")
)

print(
    item_mae.sort_values(
        by="mae",
        ascending=False
    ).head(20)
)
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

# =====================================
# LOAD DATA
# =====================================

print("Loading data...")

df = load_data("data/raw/train.csv")

# =====================================
# FEATURE ENGINEERING
# =====================================

print("Creating features...")

df = create_date_features(df)

df = create_cyclical_features(df)

df = create_lag_features(df)

df = create_rolling_features(df)

df = create_ema_features(df)

df = df.dropna()

print("Final Shape:", df.shape)

# =====================================
# FEATURES
# =====================================

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

# =====================================
# WALK FORWARD SPLITS
# =====================================

folds = [
    (
        "2013-01-31",
        "2014-12-31",
        "2015-01-01",
        "2015-12-31"
    ),
    (
        "2013-01-31",
        "2015-12-31",
        "2016-01-01",
        "2016-12-31"
    ),
    (
        "2013-01-31",
        "2016-12-31",
        "2017-01-01",
        "2017-12-31"
    )
]

mae_scores = []

# =====================================
# WALK FORWARD VALIDATION
# =====================================

for fold_num, (
    train_start,
    train_end,
    valid_start,
    valid_end
) in enumerate(folds, start=1):

    print("\n" + "=" * 50)
    print(f"FOLD {fold_num}")
    print("=" * 50)

    train_df = df[
        (df["date"] >= train_start)
        & (df["date"] <= train_end)
    ]

    valid_df = df[
        (df["date"] >= valid_start)
        & (df["date"] <= valid_end)
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

    model.fit(
        X_train,
        y_train
    )

    pred_log = model.predict(X_valid)

    predictions = np.expm1(pred_log)

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    mae_scores.append(mae)

    print(
        f"Fold {fold_num} MAE: {mae:.4f}"
    )

# =====================================
# FINAL RESULTS
# =====================================

print("\n")
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

for idx, score in enumerate(
    mae_scores,
    start=1
):
    print(
        f"Fold {idx}: {score:.4f}"
    )

avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

print()
print(
    f"Average MAE: {avg_mae:.4f}"
)

print(
    f"Std Dev: {std_mae:.4f}"
)

# =====================================
# INTERPRETATION
# =====================================

print("\n")
print("=" * 50)
print("INTERPRETATION")
print("=" * 50)

if std_mae < 0.30:
    print(
        "Model is stable across folds."
    )
else:
    print(
        "Model performance varies noticeably across time."
    )

print(
    f"Trusted Walk-Forward MAE: {avg_mae:.4f}"
)
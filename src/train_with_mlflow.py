import os
import time
import numpy as np
import pandas as pd

import mlflow
import mlflow.catboost

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
# CONFIG
# =====================================

EXPERIMENT_NAME = "Retail_Demand_Forecasting"

WALK_FORWARD_MAE = 5.9568

MODEL_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 10,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False
}

# =====================================
# CREATE EXPERIMENT
# =====================================

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# =====================================
# LOAD DATA
# =====================================

print("Loading data...")

df = load_data(
    "data/raw/train.csv"
)

print(
    f"Original Shape: {df.shape}"
)

# =====================================
# FEATURE ENGINEERING
# =====================================

print("Creating date features...")
df = create_date_features(df)

print("Creating cyclical features...")
df = create_cyclical_features(df)

print("Creating lag features...")
df = create_lag_features(df)

print("Creating rolling features...")
df = create_rolling_features(df)

print("Creating EMA features...")
df = create_ema_features(df)

print("Dropping NaNs...")
df = df.dropna()

print(
    f"Final Shape: {df.shape}"
)

# =====================================
# TIME-BASED SPLIT
# =====================================

train_df = df[
    df["date"] < "2017-01-01"
]

valid_df = df[
    df["date"] >= "2017-01-01"
]

print(
    f"\nTrain Shape: {train_df.shape}"
)

print(
    f"Validation Shape: {valid_df.shape}"
)

# =====================================
# FEATURE LIST
# =====================================

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

print(
    f"\nNumber of Features: {len(features)}"
)

# =====================================
# TRAIN DATA
# =====================================

X_train = train_df[features]
X_valid = valid_df[features]

y_train = np.log1p(
    train_df["sales"]
)

y_valid = valid_df["sales"]

# =====================================
# START RUN
# =====================================

with mlflow.start_run():

    print("\nTraining CatBoost...")

    start_time = time.time()

    model = CatBoostRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.time() - start_time
    )

    # =====================================
    # PREDICTIONS
    # =====================================

    pred_log = model.predict(
        X_valid
    )

    predictions = np.expm1(
        pred_log
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    # =====================================
    # LOG PARAMETERS
    # =====================================

    mlflow.log_params(
        MODEL_PARAMS
    )

    # =====================================
    # LOG METRICS
    # =====================================

    mlflow.log_metric(
        "validation_mae",
        mae
    )

    mlflow.log_metric(
        "walk_forward_avg_mae",
        WALK_FORWARD_MAE
    )

    mlflow.log_metric(
        "training_time_seconds",
        training_time
    )

    mlflow.log_metric(
        "num_features",
        len(features)
    )

    mlflow.log_metric(
        "train_rows",
        len(train_df)
    )

    mlflow.log_metric(
        "validation_rows",
        len(valid_df)
    )

    # =====================================
    # SAVE LOCAL MODEL
    # =====================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save_model(
        "models/catboost_demand_forecaster.cbm"
    )

    # =====================================
    # LOG MODEL TO MLFLOW
    # =====================================

    mlflow.catboost.log_model(
        cb_model=model,
        name="catboost_model"
    )

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance":
            model.get_feature_importance()
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    mlflow.log_artifact(
        "feature_importance.csv"
    )

    # =====================================
    # RESULTS
    # =====================================

    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    print(
        f"Validation MAE: {mae:.4f}"
    )

    print(
        f"Walk Forward MAE: {WALK_FORWARD_MAE:.4f}"
    )

    print(
        f"Training Time: "
        f"{training_time:.2f} sec"
    )

    print("\nTop 10 Features")

    print(
        importance_df.head(10)
    )

print("\n" + "=" * 40)
print("MLFLOW RUN COMPLETED")
print("=" * 40)
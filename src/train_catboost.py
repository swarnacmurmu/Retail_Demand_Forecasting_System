import numpy as np
import pandas as pd
import time

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

print("Original Shape:", df.shape)

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

# =====================================
# DROP NANs
# =====================================

print("Dropping NaNs...")

df = df.dropna()

print("Shape After Dropping NaNs:", df.shape)

# =====================================
# TIME-BASED SPLIT
# =====================================

train_df = df[df["date"] < "2017-01-01"]

valid_df = df[df["date"] >= "2017-01-01"]

print("\nTrain Shape:", train_df.shape)
print("Validation Shape:", valid_df.shape)

# =====================================
# FEATURES
# =====================================

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

print("\nNumber of Features:", len(features))

X_train = train_df[features]
X_valid = valid_df[features]

# =====================================
# LOG TARGET
# =====================================

y_train = np.log1p(train_df["sales"])
y_valid = valid_df["sales"]

# =====================================
# TRAIN MODEL
# =====================================

print("\nTraining CatBoost...")

start_time = time.time()

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    eval_set=(
        X_valid,
        np.log1p(valid_df["sales"])
    ),
    use_best_model=True
)

training_time = time.time() - start_time

# =====================================
# PREDICTIONS
# =====================================

log_preds = model.predict(X_valid)

preds = np.expm1(log_preds)

# =====================================
# EVALUATION
# =====================================

mae = mean_absolute_error(
    y_valid,
    preds
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"MAE: {mae:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

# =====================================
# FEATURE IMPORTANCE
# =====================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.get_feature_importance()
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Features")

print(
    importance_df.head(15)
)

# =====================================
# SAMPLE PREDICTIONS
# =====================================

print("\nSample Predictions")

for i in range(10):

    print(
        f"Actual: {y_valid.iloc[i]:.0f} | "
        f"Predicted: {preds[i]:.2f}"
    )

# =====================================
# SAVE MODEL
# =====================================

model.save_model(
    "models/catboost_demand_forecaster_v3.cbm"
)

print("\nModel Saved Successfully!")
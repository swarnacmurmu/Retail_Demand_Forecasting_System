# src/train.py

from data_preprocessing import load_data
from feature_engineering import (
    create_date_features,
    create_lag_features,
    create_rolling_features
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================
# Load Data
# =========================

print("Loading data...")

df = load_data("data/raw/train.csv")

print("Original Shape:", df.shape)


# =========================
# Feature Engineering
# =========================

print("Creating date features...")

df = create_date_features(df)

print("Creating lag features...")

df = create_lag_features(df)

print("Creating rolling features...")

df = create_rolling_features(df)


# =========================
# Remove NaNs
# =========================

print("Dropping NaN rows...")

df = df.dropna()

print("Shape After Dropping NaNs:", df.shape)


# =========================
# Time-Based Split
# =========================

train_df = df[df["date"] < "2017-01-01"]

valid_df = df[df["date"] >= "2017-01-01"]

print("\nTrain Shape:", train_df.shape)
print("Validation Shape:", valid_df.shape)


# =========================
# Features and Target
# =========================

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

print("\nFeatures Used:")
print(features)

X_train = train_df[features]
y_train = train_df["sales"]

X_valid = valid_df[features]
y_valid = valid_df["sales"]


# =========================
# Train Model
# =========================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================
# Predictions
# =========================

print("Generating Predictions...")

preds = model.predict(X_valid)


# =========================
# Evaluation
# =========================

mae = mean_absolute_error(
    y_valid,
    preds
)

print("\n=========================")
print("MODEL EVALUATION")
print("=========================")

print(f"MAE: {mae:.4f}")


# =========================
# Sample Predictions
# =========================

print("\nSample Predictions:")

for i in range(10):
    print(
        f"Actual: {y_valid.iloc[i]:.0f} | "
        f"Predicted: {preds[i]:.2f}"
    )
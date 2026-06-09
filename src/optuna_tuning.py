import numpy as np
import optuna

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

# =====================================
# TRAIN / VALID SPLIT
# =====================================

train_df = df[df["date"] < "2017-01-01"]
valid_df = df[df["date"] >= "2017-01-01"]

features = [
    col
    for col in df.columns
    if col not in ["date", "sales"]
]

X_train = train_df[features]
X_valid = valid_df[features]

y_train = np.log1p(train_df["sales"])
y_valid = valid_df["sales"]

print("Train Shape:", train_df.shape)
print("Valid Shape:", valid_df.shape)


# =====================================
# OPTUNA OBJECTIVE
# =====================================

def objective(trial):

    params = {
        "iterations": trial.suggest_int(
            "iterations",
            500,
            1500
        ),
        "depth": trial.suggest_int(
            "depth",
            4,
            10
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1
        ),
        "l2_leaf_reg": trial.suggest_int(
            "l2_leaf_reg",
            1,
            20
        ),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": False,
        "random_seed": 42
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train,
        y_train
    )

    pred_log = model.predict(
        X_valid
    )

    pred = np.expm1(pred_log)

    mae = mean_absolute_error(
        y_valid,
        pred
    )

    print(
        f"Trial {trial.number} | "
        f"MAE = {mae:.4f}"
    )

    return mae


# =====================================
# CREATE STUDY
# =====================================

print("\nStarting Optuna Search...\n")

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=10
)

# =====================================
# RESULTS
# =====================================

print("\n")
print("=" * 50)
print("BEST RESULT")
print("=" * 50)

print(
    f"Best MAE: "
    f"{study.best_value:.4f}"
)

print("\nBest Parameters:")

for key, value in study.best_params.items():
    print(
        f"{key}: {value}"
    )
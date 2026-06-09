import numpy as np

from catboost import CatBoostRegressor

from api.feature_builder import build_features


MODEL_PATH = "models/final_catboost_model.cbm"

model = CatBoostRegressor()
model.load_model(MODEL_PATH)


def predict_sales(
    store: int,
    item: int,
    forecast_date: str
):

    features = build_features(
        store=store,
        item=item,
        forecast_date=forecast_date
    )

    pred_log = model.predict(features)

    pred = np.expm1(pred_log)

    return float(pred[0])
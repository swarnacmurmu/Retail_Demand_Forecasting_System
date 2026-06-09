from fastapi import FastAPI

from api.schemas import ForecastRequest
from api.predictor import predict_sales

app = FastAPI(
    title="Retail Demand Forecasting API"
)


@app.get("/")
def home():
    return {
        "message": "Retail Demand Forecasting API Running"
    }


@app.post("/predict")
def predict(request: ForecastRequest):

    prediction = predict_sales(
        store=request.store,
        item=request.item,
        forecast_date=request.forecast_date
    )

    return {
        "predicted_sales": round(prediction, 2)
    }
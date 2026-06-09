from pydantic import BaseModel



class ForecastRequest(BaseModel):
    store: int
    item: int
    forecast_date: str
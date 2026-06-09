from pydantic import BaseModel


class ForecastRequest(BaseModel):

    store: int
    item: int

    year: int
    month: int
    day: int

    weekday: int
    weekofyear: int
    is_weekend: int

    month_sin: float
    month_cos: float

    weekday_sin: float
    weekday_cos: float

    sales_lag_1: float
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_30: float
    sales_lag_60: float
    sales_lag_90: float

    rolling_mean_7: float
    rolling_mean_30: float

    rolling_std_7: float
    rolling_std_30: float

    ema_7: float
    ema_30: float

    expanding_mean: float
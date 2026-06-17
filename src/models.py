import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def seasonal_naive_forecast(train_series, horizon=16, period=7):
    """Predict each future day as the value from [period] days earlier in the last observed cycle."""

    last_cycle = train_series.iloc[-period:].values
    reps = int(np.ceil(horizon / period))

    return np.tile(last_cycle, reps)[:horizon]


def ets_forecast(train_series, horizon=16, seasonal_periods=7):
    """Holt-Winters additive trend + weekly seasonality."""

    s = train_series.astype(float)

    if len(s) < 2 * seasonal_periods or (s <= 0).all():
        return seasonal_naive_forecast(s, horizon, seasonal_periods)
    
    model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated").fit()

    return np.clip(model.forecast(horizon).values, 0, None)

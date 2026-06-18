import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def seasonal_naive_forecast(train_series, horizon=16, period=7):
    """Predict each future day as the value from [period] days earlier in the last observed cycle.
    Expects a continuous daily series (no time gaps)."""

    last_cycle = train_series.iloc[-period:].values
    reps = int(np.ceil(horizon / period))

    return np.tile(last_cycle, reps)[:horizon]


def ets_forecast(train_series, horizon=16, seasonal_periods=7, return_method=False):
    """Holt-Winters additive trend + weekly seasonality.
    Expects a continuous daily series (no time gaps)."""

    s = train_series.astype(float)

    if len(s) < 2 * seasonal_periods or (s <= 0).all():
        preds = seasonal_naive_forecast(s, horizon, seasonal_periods)
        return (preds, "naive fallback") if return_method else preds
    
    model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated").fit()

    preds = np.clip(model.forecast(horizon).values, 0, None)
    return (preds, "ets") if return_method else preds

import numpy as np
import pandas as pd
import lightgbm as lgb

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


LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "seed": 1221,
    "deterministic": True,
    "verbose": -1
}

def train_lgbm(X_train, y_train, X_valid=None, y_valid=None, params=None, categorical=None, num_boost_round=2000):
    """Train a single global LGBM regressor on a log1p sales target.
    Expects y to be already log-transformed. Returns the fitted booster. If a validation set is provided, it uses early stopping to choose the tree count.
    """
    params = {**LGBM_PARAMS, **(params or {})}

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical or "auto")
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = [train_set]

    if X_valid is not None and y_valid is not None:
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, categorical_feature=categorical or "auto")
        valid_sets.append(valid_set)
        callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks
    )

    return model


import numpy as np
import pandas as pd

def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error: Penalizes proportional error and under-forecasting. Defined on non-negative values."""

    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error: Total absolute error as a % of total volume, and robust to values near zero."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true).sum()

    return np.abs(y_true - y_pred).sum() / denominator if denominator else np.nan

def score_table(preds, model_name):
    overall = pd.DataFrame([{
        "model": model_name,
        "RMSLE": rmsle(preds.y_true, preds.y_pred),
        "WAPE": wape(preds.y_true, preds.y_pred)
    }])

    by_family = (preds.groupby("family").apply(lambda g: pd.Series({
        "RMSLE": rmsle(g.y_true, g.y_pred),
        "WAPE": wape(g.y_true, g.y_pred)
    })).reset_index().assign(model=model_name))

    return overall, by_family


import numpy as np
import pandas as pd

from src.rolling_origin_cv import rolling_origin_cutoffs
from src.features import (add_lags, add_rolling, add_calendar, build_holiday_lookup, add_holidays, add_promo, add_oil)
from src.models import train_lgbm

FEATURE_COLS = None
CATEGORICAL = ["family", "city", "state", "type", "cluster"]


def build_features(df, stores, oil, national, local):
    """Assemble dataframe with all features.
    Note that stores must be merged before holidays since local/regional holidays rely on the city/state information from stores."""

    df = df.merge(stores, on="store_nbr", how="left")
    df = add_oil(df, oil)
    df = add_calendar(df)
    df = add_holidays(df, national, local)
    df = add_promo(df)
    df = add_lags(df)
    df = add_rolling(df)

    return df


def evaluate_lgbm(df, stores, oil, holidays, horizon=16, n_folds=4, feature_cols=None, params=None):
    """Run global LightGBM thorugh the same rolling-origin folds as the baselines.
    Returns predictions in long format to be fed into score_table() function."""

    cutoffs = rolling_origin_cutoffs(df["date"], horizon, n_folds)
    national, local = build_holiday_lookup(holidays)
    feats = build_features(df.copy(), stores, oil, national, local)

    for c in CATEGORICAL:
        if c in feats.columns:
            feats[c] = feats[c].astype("category")

    drop = {"id", "date", "sales", "store_nbr"}

    if feature_cols is None:
        feature_cols = [c for c in feats.columns if c not in drop]

    all_preds = []

    for fold, cutoff in enumerate(cutoffs):
        train_mask = (feats["date"] <= cutoff)
        test_mask = ((feats["date"] > cutoff) & (feats["date"] <= cutoff + pd.Timedelta(days=horizon)))

        # Validation is the last 'horizon' days of train for early stopping
        valid_start = cutoff - pd.Timedelta(days=horizon)
        fit_mask = train_mask & (feats["date"] <= valid_start)
        valid_mask = train_mask & (feats["date"] > valid_start)

        X_fit = feats.loc[fit_mask, feature_cols]
        X_val = feats.loc[valid_mask, feature_cols]
        X_test = feats.loc[test_mask, feature_cols]

        # log1p target with negatives clipped
        y_fit = np.log1p(feats.loc[fit_mask, "sales"].clip(lower=0))
        y_val = np.log1p(feats.loc[valid_mask, "sales"].clip(lower=0))

        model = train_lgbm(X_fit, y_fit, X_val, y_val, params=params, categorical=CATEGORICAL)

        preds_log = model.predict(X_test, num_iteration=model.best_iteration)
        preds = np.clip(np.expm1(preds_log), 0, None)

        test_rows = feats.loc[test_mask, ["store_nbr", "family", "date", "sales"]].copy()
        test_rows = test_rows.rename(columns={"sales": "y_true"})
        test_rows["y_pred"] = preds
        test_rows["fold"] = fold
        all_preds.append(test_rows)

    return pd.concat(all_preds, ignore_index=True)
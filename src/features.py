import pandas as pd
import numpy as np

def add_lags(df, lags=(12,21,28)):
    """Adds lags from a given set of lag lengths."""
    df = df.sort_values(["store_nbr", "family", "date"])

    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby(["store_nbr", "family"], observed=True)["sales"].shift(lag)
    
    return df


def add_rolling(df, windows=(7,14,30), horizon=16):
    """Calculates the rolling mean and standard deviation in a given window."""
    df = df.sort_values(["store_nbr", "family", "date"])
    g = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    base = g.shift(horizon)
    grouped_base = base.groupby([df["store_nbr"], df["family"]], observed=True)

    for w in windows:
        df[f"roll_mean_{w}"] = base.rolling(w).mean().reset_index(level=[0,1], drop=True)
        df[f"roll_std_{w}"] = base.rolling(w).std().reset_index(level=[0,1], drop=True)

    return df


def add_calendar(df):
    """Calendar features"""
    d = df["date"].dt
    df["dow"] = d.dayofweek
    df["month"] = d.month
    df["day"] = d.day
    df["year"] = d.year
    df["payday"] = ((d.day == 15) | (d.is_month_end)).astype("int8")

    return df


def build_holiday_lookup(holidays):
    """Resolve whether it was a real day off and return a national set and a local/regional set for holidays."""
    h = holidays.copy()
    real = ((h["type"] == "Holiday") & (~h["transferred"])) | (h["type"].isin(["Transfer", "Bridge", "Additional"]))

    h = h[real]
    national = set(h.loc[h["locale"] == "National", "date"])
    local = set(zip(h.loc[h["locale"] != "National", "date"],
                    h.loc[h["locale"] != "National", "locale_name"]))
    
    return national, local


def add_holidays(df, national, local):
    """Vectorized holiday flags."""
    is_natl = df["date"].isin(national)

    city_match = pd.Series(list(zip(df["date"], df["city"])), index=df.index).isin(local)
    state_match = pd.Series(list(zip(df["date"], df["state"])), index=df.index).isin(local)

    df["is_holiday"] = (is_natl | city_match | state_match).astype("int8")

    return df


def add_promo(df):
    """Adds column for items with date-specific promotions."""
    df["onpromotion"] = df["onpromotion"].fillna(0)

    return df


def add_oil(df, oil):
    o = oil.sort_values("date").copy()

    o["dcoilwtico"] = o["dcoilwtico"].ffill()
    o["oil_roll_14"] = o["dcoilwtico"].rolling(14, min_periods=1).mean()

    return df.merge(o, on="date", how="left")
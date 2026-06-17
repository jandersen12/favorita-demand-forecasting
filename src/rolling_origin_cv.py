import pandas as pd

def rolling_origin_cutoffs(dates, horizon=16, n_folds=4, step=16):
    """Return cutoff dates for rolling-origin cross-validation."""

    last = pd.Timestamp(dates.max())
    cutoffs = []

    for k in range(n_folds):
        test_end = last - pd.Timedelta(days= step * k)
        cutoff = test_end - pd.Timedelta(days=horizon)
        cutoffs.append(cutoff)
    
    return sorted(cutoffs)

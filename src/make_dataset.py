"""Convert raw Kaggle CSVs to parquet. Reads from data/raw, writes to data/processed"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def load_and_save():
    # Main sales table - parse dates, downcast categoricals to save memory

    train = pd.read_csv(
        RAW / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "family": "category", "onpromotion": "int32"},
    )
    train["sales"]= train["sales"].astype("float32")

    stores = pd.read_csv(RAW / "stores.csv",
                         dtype={"store_nbr": "int16", "city": "category", "state": "category",
                                "type": "category", "cluster": "int8"})
    
    oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(RAW / "transactions.csv", parse_dates=["date"],
                               dtype={"store_nbr": "int16", "transactions": "int32"})
    
    for name, df in {
        "train": train, 
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }.items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name}: {df.shape[0]:,} rows -> {name}.parquet")

if __name__ == "__main__":
    load_and_save()
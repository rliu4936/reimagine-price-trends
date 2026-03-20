from __future__ import print_function, division
import os.path as op, os, time
import numpy as np, pandas as pd
from DataHandler import dgp_config as dcf


def get_format_raw_us_stock_data() -> pd.DataFrame:
    """
    Load the feather cache if it exists; otherwise read the raw CRSP‑like file,
    clean it, derive features, cache to feather, and return a MultiIndex frame.
    """
    processed_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")

    if op.exists(processed_path):                       # fast‑path
        print(f"Loading processed data from {processed_path}")
        t0 = time.time()
        df = pd.read_feather(processed_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finished in {(time.time() - t0) / 60:.2f} min")
        return df.copy()

    # ─── slow‑path: read raw ──────────────────────────────────────────
    raw_path = op.join(dcf.RAW_DATA_DIR, "us_920101-200731.csv")
    print(f"Reading raw data from {raw_path}")
    t0 = time.time()
    df = pd.read_csv(
        raw_path,
        parse_dates=["date"],
        dtype={
            "PERMNO": str, "BIDLO": np.float64, "ASKHI": np.float64,
            "PRC": np.float64, "VOL": np.float64, "SHROUT": np.float64,
            "OPENPRC": np.float64, "RET": object, "EXCHCD": np.float64,
        },
        compression="gzip", header=0,
    )
    print(f"Finished reading in {(time.time() - t0):.2f} s")


    # … << original rename / cleaning steps >> …
    df = df.rename(
        columns={
            "date": "Date", "PERMNO": "StockID",
            "BIDLO": "Low", "ASKHI": "High",
            "PRC": "Close", "VOL": "Vol", "SHROUT": "Shares",
            "OPENPRC": "Open", "RET": "Ret",
        }
    )
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"] = pd.to_numeric(df["Ret"], errors="coerce")
    # sentinel replacements, abs‑casting, indexing …
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)
    
    df.reset_index().to_feather(processed_path)
    return df.copy()


def get_format_raw_us_index_data():
    pass
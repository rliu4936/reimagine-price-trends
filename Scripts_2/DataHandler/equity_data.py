from __future__ import print_function, division

import os
import os.path as op
import time
import numpy as np
import pandas as pd

from DataHandler import dgp_config as dcf
from DataHandler.cn_data_adapter import (
    get_format_raw_cn_stock_data, 
    get_format_raw_cn_index_data
)
from DataHandler.us_data_adapter import (
    get_format_raw_us_stock_data, 
    get_format_raw_us_index_data
)


def get_index_ret(freq: str = "week", country: str = "CN") -> pd.DataFrame:
    assert freq in ("week", "month", "quarter")
    country = country.upper()
    path = os.path.join(dcf.CACHE_DIR, f"{country}_index_ret_{freq}.csv")

    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])

    # Load raw data
    if country == "CN":
        df = get_format_raw_cn_index_data()
    elif country == "US":
        df = get_format_raw_us_index_data()
    else:
        raise ValueError(f"Unsupported country: {country}")

    df["log_ret"] = np.log1p(df["ret"])
    df["cum_log_ret"] = df["log_ret"].cumsum()

    for f in ("week", "month", "quarter"):
        period_ends = get_period_end_dates(f, country=country)
        period_df = df[df["date"].isin(period_ends)].copy()

        period_df[f"{f}_ret"] = (
            np.exp(period_df["cum_log_ret"] - period_df["cum_log_ret"].shift(1)) - 1
        )

        out_df = period_df[["date", f"{f}_ret"]].copy()
        out_df.columns = ["date", "Ret"]
        out_df.reset_index(drop=True, inplace=True)
        out_df["Ret"] = out_df["Ret"].shift(-2)

        out_path = os.path.join(dcf.CACHE_DIR, f"{country}_index_ret_{f}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"💾 Saved: {out_path} — shape={out_df.shape}")

    return pd.read_csv(path, parse_dates=["date"])



def get_stock_ret(country: str = "CN") -> pd.DataFrame:

    # Return cleaned + derived-feature stock returns for given country
    country = country.upper()
    path = op.join(dcf.PROCESSED_DATA_DIR, f"{country}_stock_ret.feather")

    if op.exists(path):
        print(f"📂 Loaded: {path}")
        df = pd.read_feather(path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
    else:
        if country == "CN":
            df = get_format_raw_cn_stock_data()
        elif country == "US":
            df = get_format_raw_us_stock_data()
        else:
            raise ValueError(f"Unsupported country: {country}")

        df = _add_derived_features(df)
        df.reset_index().to_feather(path)
        print(f"💾 Saved: {path}")

    return df

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["MarketCap"] = np.abs(df["Close"] * df["Shares"])
    df["log_ret"] = np.log1p(df["Ret"])
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)

    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )

    # period‑end returns
    for freq in ("week", "month", "quarter"):
        period_ends = get_period_end_dates(freq)
        freq_df = df.loc[df.index.get_level_values("Date").isin(period_ends)].copy()
        freq_df["freq_ret"] = freq_df.groupby(
            "StockID", group_keys=False
        )["cum_log_ret"].apply(lambda x: np.exp(x.shift(-1) - x) - 1)
        df[f"Ret_{freq}"] = freq_df["freq_ret"]

    # rolling‑window horizon returns
    for d in (5, 20, 60, 65, 180, 250, 260):
        df[f"Ret_{d}d"] = df.groupby(
            "StockID", group_keys=False
        )["cum_log_ret"].apply(lambda x: np.exp(x.shift(-d) - x) - 1)
    return df




def get_period_end_dates(freq: str, country: str = "CN") -> pd.DatetimeIndex:
    # Return period-end dates for given freq and country, compute all 3 if missing
    assert freq in ("week", "month", "quarter")
    country = country.upper()

    def cache_path(f): return op.join(dcf.CACHE_DIR, f"{country}_period_end_dates_{f}.csv")
    path = cache_path(freq)

    if op.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])["Date"]

    if country == "CN":
        df = get_format_raw_cn_stock_data()
    elif country == "US":
        df = get_format_raw_us_stock_data()

    df = df.reset_index() if "Date" not in df.columns else df
    df["Date"] = pd.to_datetime(df["Date"])

    for f, code in {"week": "W", "month": "M", "quarter": "Q"}.items():
        ends = df.groupby(df["Date"].dt.to_period(code))["Date"].max().sort_values().unique()
        pd.DataFrame({"Date": ends}).to_csv(cache_path(f), index=False)
        print(f"💾 Saved: {cache_path(f)}")

    return pd.read_csv(path, parse_dates=["Date"])["Date"]



def get_stock_ret_freqs(freq: str, country: str = "CN") -> pd.DataFrame:
    # Compute and return stock returns for given freq, caching all 3 if needed
    assert freq in ("week", "month", "quarter")
    country = country.upper()

    def cache_path(f): return op.join(dcf.CACHE_DIR, f"{country}_stock_ret_{f}.pq")
    path = cache_path(freq)
    if op.exists(path):
        print(f"📂 Loaded from cache: {path}")
        return pd.read_parquet(path)

    df = get_stock_ret(country = country)
    if "log_ret" not in df.columns:
        df["log_ret"] = np.log1p(df["Ret"])
    if "cum_log_ret" not in df.columns:
        df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)

    for d in (5, 20, 60):
        col = f"Ret_{d}d"
        df[col] = df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-d) - x) - 1
        )

    for start, end in ((6, 20), (6, 60)):
        col = f"Ret_{start}-{end}d"
        df[col] = df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-start) - x.shift(-end)) - 1
        )

    for f in ("week", "month", "quarter"):
        ends = get_period_end_dates(f, country)
        freq_col = f"{f}_ret"
        next_col = f"next_{f}_ret"
        next_col_delay = f"{next_col}_0delay"

        period_df = df.loc[df.index.get_level_values("Date").isin(ends)].copy()
        period_df[freq_col] = period_df.groupby(
            "StockID", group_keys=False
        )["cum_log_ret"].apply(lambda x: np.exp(x.shift(-1) - x) - 1)
        df.loc[period_df.index, freq_col] = period_df[freq_col]

        period_df[next_col] = period_df.groupby("StockID")[freq_col].shift(-1)
        df.loc[period_df.index, next_col] = period_df[next_col]
        df[next_col_delay] = df[next_col]

        base_cols = [
            "MarketCap", "log_ret",
            "Ret_5d", "Ret_20d", "Ret_60d", "Ret_6-20d", "Ret_6-60d",
            freq_col, next_col, next_col_delay,
        ]
        mask = df[freq_col].notna()
        out_df = (
            df.loc[mask, base_cols]
              .reset_index()
              .sort_values(["StockID", "Date"])
              .copy()
        )

        out_df.to_parquet(cache_path(f), index=False)
        print(f"💾 Saved: {cache_path(f)} — shape={out_df.shape}")

    return pd.read_parquet(path)


def main():
    pass

if __name__ == "__main__":
    main()
    
from __future__ import print_function, division

import os
import os.path as op
import time
import numpy as np
import pandas as pd

from DataHandler import dgp_config as dcf



def get_format_raw_cn_stock_data() -> pd.DataFrame:
    raw_path = op.join(dcf.RAW_DATA_DIR, "CN_raw_stock_1993_2025.csv")
    print(f"📄 Reading: {raw_path}")
    t0 = time.time()
    df = pd.read_csv(
        raw_path,
        parse_dates=["Trddt"],
        dtype={
            "Stkcd": str, "Opnprc": np.float64, "Hiprc": np.float64,
            "Loprc": np.float64, "Clsprc": np.float64,
            "Dnshrtrd": np.float64, "Dnvaltrd": np.float64,
            "Dretwd": object, "Dsmvosd": np.float64,
            "Markettype": np.float64, "Adjprcwd": np.float64,
        },
        header=0,
    )
    print(f"✅ Done in {(time.time() - t0):.2f}s")

    df = df.rename(columns={
        "Trddt": "Date", "Stkcd": "StockID", "Loprc": "Low", "Hiprc": "High",
        "Clsprc": "Close", "Opnprc": "Open", "Dnshrtrd": "Vol", "Dretwd": "Ret"
    })

    cols = ["Date", "StockID", "Low", "High", "Close", "Open", "Vol", "Ret", "Dsmvosd", "Markettype", "Adjprcwd"]
    df = df[cols]

    df["Date"] = pd.to_datetime(df["Date"])
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"] = pd.to_numeric(df["Ret"], errors="coerce")
    df["EXCHCD"] = pd.to_numeric(df["Markettype"], errors="coerce")
    df.drop(columns="Markettype", inplace=True)

    if "Adjprcwd" in df.columns:
        ratio = df["Adjprcwd"] / df["Close"]
        df["Close"] = df["Adjprcwd"]
        df["Open"] *= ratio
        df["High"] *= ratio
        df["Low"] *= ratio
    df.drop(columns="Adjprcwd", inplace=True, errors="ignore")

    if "Dsmvosd" in df.columns:
        df["Shares"] = df["Dsmvosd"] / df["Close"] / ratio if "ratio" in locals() else df["Dsmvosd"] / df["Close"]
    else:
        df["Shares"] = 0.0
    df.drop(columns="Dsmvosd", inplace=True, errors="ignore")

    df.replace({
        "Close": {0: np.nan}, "Open": {0: np.nan},
        "High": {0: np.nan}, "Low": {0: np.nan},
        "Vol": {0: np.nan, -99: np.nan},
    }, inplace=True)

    df.dropna(subset=["Ret"], inplace=True)
    df[["Close", "Open", "High", "Low", "Vol", "Shares"]] = df[["Close", "Open", "High", "Low", "Vol", "Shares"]].abs()
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    return df.copy()



def get_format_raw_cn_index_data():
    file_path = os.path.join(dcf.RAW_DATA_DIR, "CN_raw_index_2010_2025.csv")

    df = (
        pd.read_csv(file_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    df["ret"] = df["Ret"].str.rstrip("%").astype(float) / 100
    return df
import os
import os.path as op
from datetime import datetime, timedelta
import time

import akshare as ak
import efinance as ef
import pandas as pd
from tqdm import tqdm  # added tqdm

from DataHandler import dgp_config as dcf  # must define STOCKS_SAVEPATH

def download_cn_stocks(save_date: str | None = None,
                       lookback_days: int = 150,
                       min_rows: int = 60,
                       max_tickers: int | None = None,
                       verbose: bool = False) -> str:
    if save_date is None:
        save_date = datetime.today().strftime("%Y%m%d")

    beg_date = datetime.today() - timedelta(days=lookback_days)
    beg_str = beg_date.strftime("%Y%m%d")

    base_dir = op.join(dcf.WORK_DIR, "current", save_date)
    save_dir = op.join(base_dir, "price_data")
    os.makedirs(save_dir, exist_ok=True)

    ticker_csv_path = op.join(base_dir, "ticker_list.csv")

    # ----- Load or generate ticker list -----
    if op.exists(ticker_csv_path):
        if verbose:
            print(f"📄 Using cached ticker list from {ticker_csv_path}")
        tickers_df = pd.read_csv(ticker_csv_path, dtype={"code": str})
    else:
        try:
            tickers_df_all = ak.stock_info_a_code_name()
            print(f"📊 全部A股数量: {len(tickers_df_all)}")

            st_df = ak.stock_zh_a_st_em()
            print(f"🚫 ST列表数量: {len(st_df)}")

            st_codes = set(st_df['代码'].astype(str))
            tickers_df = tickers_df_all[~tickers_df_all['code'].astype(str).isin(st_codes)]

            tickers_df.to_csv(ticker_csv_path, index=False)
            if verbose:
                print(f"✅ 保留正常股票: {len(tickers_df)}，已保存到 {ticker_csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch or filter ticker list: {e}")

    # ----- Download each ticker -----
    tickers = tickers_df["code"].astype(str).tolist()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    for i in tqdm(range(len(tickers)), desc="📥 Downloading stocks"):
        code = tickers[i]
        out_path = op.join(save_dir, f"{code}.csv")
        if op.exists(out_path):
            if verbose:
                tqdm.write(f"[{i+1}/{len(tickers)}] Skip {code}: already exists")
            continue

        try:
            df = ef.stock.get_quote_history(stock_codes=code, beg=beg_str)
            if df is None or df.empty:
                if verbose:
                    tqdm.write(f"[{i+1}/{len(tickers)}] Skip {code}: empty")
                continue

            needed_cols = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
            if not set(needed_cols).issubset(df.columns):
                if verbose:
                    tqdm.write(f"[{i+1}/{len(tickers)}] Skip {code}: missing columns")
                continue

            df = df[needed_cols].copy()
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            if len(df) < min_rows:
                if verbose:
                    tqdm.write(f"[{i+1}/{len(tickers)}] Skip {code}: only {len(df)} rows (<{min_rows})")
                continue

            df["ma20"] = df["Close"].rolling(window=20, min_periods=20).mean()
            df.dropna(subset=["ma20"], inplace=True)

            if len(df) < 20:
                if verbose:
                    tqdm.write(f"[{i+1}/{len(tickers)}] Skip {code}: <20 rows after ma20")
                continue

            df.to_csv(out_path)
            if verbose:
                tqdm.write(f"[{i+1}/{len(tickers)}] Saved {code} ({len(df)} rows)")

        except Exception as e:
            tqdm.write(f"[{i+1}/{len(tickers)}] ❌ Error {code}: {e}")
            input("Press Enter to retry this ticker...")

    if verbose:
        print(f"📁 Done. Files saved in: {save_dir}")
    return save_dir


if __name__ == "__main__":
    download_cn_stocks(verbose=True)

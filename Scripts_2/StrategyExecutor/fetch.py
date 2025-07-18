import akshare as ak
import efinance as ef
import random
import pandas as pd
import mplfinance as mpf
import os
from datetime import datetime, timedelta

# Setup output folder
os.makedirs("stock_charts", exist_ok=True)

# Get today's and 2-month-ago date
end_date = datetime.today()
beg_date = end_date - timedelta(days=60)  # ~2 months
end_str = end_date.strftime('%Y%m%d')
beg_str = beg_date.strftime('%Y%m%d')

# Get all A-share tickers
stock_info = ak.stock_info_a_code_name()
tickers = stock_info['code'].tolist()
random.shuffle(tickers)

valid_count = 0
i = 0

while valid_count < 50 and i < len(tickers):
    code = tickers[i]
    i += 1

    try:
        df = ef.stock.get_quote_history(
            stock_codes=code,
            beg=beg_str,
            end=end_str,
            klt=101,  # Daily
            fqt=1     # Front-adjusted
        )

        if df is None or df.empty or len(df) < 10:
            continue

        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        save_path = f"stock_charts/{code}.png"
        mpf.plot(df, type='candle', volume=True, style='charles', title=code, savefig=save_path)

        valid_count += 1
        print(f"[{valid_count}/50] Saved candlestick chart for {code}")

    except Exception as e:
        print(f"Error processing {code}: {e}")

if valid_count < 50:
    print(f"Only {valid_count} valid stocks found.")
else:
    print("✅ Done: 50 candlestick charts saved.")

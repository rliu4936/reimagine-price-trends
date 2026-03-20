import os
import os.path as op
from datetime import datetime, timedelta
import time
import pandas as pd

from DataHandler import dgp_config as dcf  # must define STOCKS_SAVEPATH

def get_fundamental_weights(output_path) -> pd.DataFrame:
    # Load the Excel file
    save_date = datetime.today().strftime("%Y%m%d")
    pi_dir = op.join(dcf.WORK_DIR, "current", save_date, '沪深A股非ST(PI).xlsx')
    df = pd.read_excel(pi_dir)
    
    # Remove the last 3 characters from the stock codes
    df['证券代码'] = df['证券代码'].str.slice(0, -3)
    
    # Rename columns
    df.rename(columns={
        '证券代码': 'StockID',
        'PI7\n[交易日期]最新\n[复权方式]不复权\n': 'pi',
        '市盈率PE(TTM,扣除非经常性损益)\n[交易日期]最新\n': 'pe',
        '资产负债率\n[报告期]去年年报\n[报告类型]合并报表\n[单位]%\n': 'debt-to-asset',
        '市净率PB(扣除商誉)\n[交易日期]最新\n':'pb'
    }, inplace=True)

    df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
    df['pb'] = pd.to_numeric(df['pb'], errors='coerce')
    df['debt-to-asset'] = pd.to_numeric(df['debt-to-asset'], errors='coerce')
    
    # Filter out rows with negative PE or debt-to-asset ratio > 80
    df = df[
        (df['pe'] > 0) &
        (df['debt-to-asset'] <= 80) &
        (df['pb'] > 0)
    ]

    
    # Add a new column for fund rank based on sorting "pi" from low to high
    df['fund_rank'] = df['pi'].rank(method='min')
    df = df[['StockID', 'fund_rank', 'pi', 'pe', 'pb', 'debt-to-asset']]
    df.dropna()
    df.to_csv(output_path)
    return df


# Example usage:

from pathlib import Path
import pandas as pd
import numpy as np


# add current portfolio
# sell first and buy second

def export_algo_orders(
    account_name: str ,
    total_amount: float ,
    begin_time: str ,
    end_time: str ,
    algo_type: str = "TWAP",
    input_csv: str | Path = "current_deciles.csv",
    output_csv: str | Path = "算法指令.csv",
    top: int = 100
):
    """Split total_amount over top decile; round shares to 100s; write algo CSV."""
    df = pd.read_csv(input_csv)
    top = df.head(top)
    if top.empty:
        raise ValueError("No rows in the top decile")

    per_stock = total_amount / len(top)
    shares = (np.round(per_stock / top["Close"] / 100) * 100).astype(int)

    params = f"beginTime={begin_time};endTime={end_time};limitPrice=0;participateRate=0;tradingStyle=1"

    out = pd.DataFrame({
        0: "S0",
        1: account_name,
        2: top["StockID"].astype(str),
        3: shares,
        4: 0,              # side
        5: algo_type,
        6: params,
    })

    out.to_csv(output_csv, index=False, header=False)
    print(f"Wrote {len(out)} lines to {output_csv}")
    return out


if __name__ == "__main__":
    export_algo_orders()

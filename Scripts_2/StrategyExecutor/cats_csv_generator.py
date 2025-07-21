from pathlib import Path
import pandas as pd
import numpy as np


def export_algo_orders(
    input_csv: str | Path = "current_deciles.csv",
    output_csv: str | Path = "算法指令.csv",
    tag: str = "technical",
    side: int = 0,
    amount: int = 10000,
    algo_type: str = "TWAP",
    params: str = "beginTime=145700;endTime=145800;limitPrice=0;participateRate=0;tradingStyle=1",
):
    df = pd.read_csv(input_csv)
    top_dec = df["decile"].max()
    top = df[df["decile"] == top_dec].sort_values("up_prob", ascending=False)
    out = pd.DataFrame({
        0: "S0",
        1: tag,
        2: top["StockID"].astype(str),
        3: np.round(amount / top["Close"] / 100) * 100,
        4: side,
        5: algo_type,
        6: params,
    })
    out.to_csv(output_csv, index=False, header=False)
    print(f"Wrote {len(out)} lines to {output_csv}")

if __name__ == "__main__":
    export_algo_orders()

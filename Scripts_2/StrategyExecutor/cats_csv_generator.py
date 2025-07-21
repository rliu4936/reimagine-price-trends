from pathlib import Path
import pandas as pd

def export_algo_orders(
    input_csv: str | Path = "current_deciles.csv",
    output_csv: str | Path = "算法指令3.csv",
    tag: str = "technical",
    side: int = 0,
    qty_flag: int = 1000,
    algo_type: str = "TWAP",
    params: str = "beginTime=93000;endTime=150000;limitPrice=0;participateRate=0;tradingStyle=1",
):
    df = pd.read_csv(input_csv)
    top_dec = df["decile"].max()
    top = df[df["decile"] == top_dec].sort_values("up_prob", ascending=False)
    out = pd.DataFrame({
        0: "S0",
        1: tag,
        2: top["StockID"].astype(str),
        3: qty_flag,
        4: side,
        5: algo_type,
        6: params,
    })
    out.to_csv(output_csv, index=False, header=False)
    print(f"Wrote {len(out)} lines to {output_csv}")

if __name__ == "__main__":
    export_algo_orders()

# import os
import os.path as op
import pdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from DataHandler import equity_data as eqd


class PortfolioManager(object):
    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        portfolio_dir: str,
        start_year=1800,
        end_year=2200,
        country="USA",
        delay_list=None,
        load_signal=True,
        custom_ret=None,
        transaction_cost=False,
    ):
        assert freq in ["week", "month", "quarter"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        self.no_delay_ret_name = f"next_{freq}_ret"
        self.custom_ret = custom_ret
        self.delay_list = [0] if delay_list is None else delay_list
        self.transaction_cost = transaction_cost
        if load_signal:
            assert "up_prob" in signal_df.columns
            print(f"🧮 Debug: signal_df has {len( signal_df)} rows in load_signal")
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)
            print(f"🧮 Debug: self.signal_df has {len( self.signal_df)} rows in load_signal")

    


    

    def generate_portfolio(self, cut=10, delay=0):
        assert delay in self.delay_list
        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating {pf_name}")
            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type, cut=cut, delay=delay
            )
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            portfolio_ret.to_csv(op.join(data_dir, f"pf_data_{pf_name}.csv"))

            summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            print(f"Summary saved to {smry_path}")
            summary_df.to_csv(smry_path)
            with open(os.path.join(self.portfolio_dir, f"{pf_name}.txt"), "w+") as file:
                summary_df = summary_df.astype(float).round(2)
                file.write(ut.to_latex_w_turnover(summary_df, cut=cut))

def save_decile_weights(signal_df, weight_type, cut, output_dir):
    """
    Step 1: Split signal_df into deciles based on up_prob and save the portfolio weights.
    This does not require any knowledge of future returns, so it can be run in a live setting.
    """
    assert weight_type in ["ew", "vw"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Starting save_decile_weights: {len(signal_df)} rows, weight_type={weight_type}, deciles={cut}")
    dates = np.sort(signal_df.index.unique())
    print(f"📅 Found {len(dates)} unique dates to process")

    decile_weight_dict = {}

    for i, date in enumerate(dates):
        print(f"🔄 Processing date {date} ({i+1}/{len(dates)})")
        rebalance_df = signal_df.loc[date].copy()
        low = np.percentile(rebalance_df.up_prob, 10)
        high = np.percentile(rebalance_df.up_prob, 90)
        if low == high:
            print(f"⚠️ Skipping date {date} due to no variation in up_prob")
            continue

        date_weights = {}
        for j in range(cut):
            rebalance_up_prob = rebalance_df.up_prob
            low = np.percentile(rebalance_up_prob, j * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (j + 1) * 100.0 / cut)
            if j == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = rebalance_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                if stock_num == 0:
                    print(f"⚠️ Empty decile {j} on date {date}")
                    continue
                _decile_df["weight"] = (
                    pd.Series(np.ones(stock_num), dtype=float, index=_decile_df.index)
                    / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=float) / np.sum(value)

            date_weights[j] = _decile_df["weight"]
            print(f"✅ Decile {j} on {date}: {len(_decile_df)} stocks")

        decile_weight_dict[date] = date_weights

    weight_file_path = os.path.join(output_dir, f"decile_weights_{weight_type}.pkl")
    with open(weight_file_path, "wb") as f:
        import pickle
        pickle.dump(decile_weight_dict, f)

    print(f"💾 Saved decile weights to {weight_file_path}")
    return weight_file_path

def save_current_decile_weights(
    signal_df: pd.DataFrame,
    cut: int = 10,
    output_path: str | None = None,
) -> pd.DataFrame:

    if "up_prob" not in signal_df.columns:
        raise KeyError("signal_df must contain column 'up_prob'")

    n = len(signal_df)
    if n == 0:
        raise ValueError("signal_df is empty")

    df = signal_df.copy()

    # rank up_prob ascending → lowest gets rank 1
    ranks = df["up_prob"].rank(method="first", ascending=True)
    df["decile"] = ((ranks - 1) / n * cut).astype(int)

    # final ordering: highest up_prob at the top
    df = df.sort_values("up_prob", ascending=False).reset_index(drop=True)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Saved decile assignments to {output_path}")

    return df


def main():
    pass


if __name__ == "__main__":
    main()
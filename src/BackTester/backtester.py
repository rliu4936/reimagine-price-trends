
# import os
import os.path as op
import pdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from DataHandler import equity_data as eqd
from PortfolioManager.portfolio import PortfolioManager


class BackTester(object):
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

    def backtest_decile_weights(weight_file_path, signal_df, ret_name, transaction_cost, output_dir):
        """
        Step 2: Backtest using pre-saved decile weights and compute portfolio returns and turnover.
        """
        import pickle
        with open(weight_file_path, "rb") as f:
            decile_weight_dict = pickle.load(f)

        dates = sorted(decile_weight_dict.keys())
        cut = len(decile_weight_dict[dates[0]])
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None

        for i, date in enumerate(dates):
            decile_weights = decile_weight_dict[date]
            rebalance_df = signal_df.loc[date].copy()

            for j in range(cut):
                decile_df = rebalance_df.loc[decile_weights[j].index].copy()
                decile_df["weight"] = decile_weights[j]
                decile_df["inv_ret"] = decile_df["weight"] * decile_df[ret_name]

                if transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )

                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            # Turnover
            sell_decile = rebalance_df.loc[decile_weight_dict[date][0].index].copy()
            sell_decile["weight"] = decile_weight_dict[date][0]
            sell_decile["inv_ret"] = sell_decile["weight"] * sell_decile[ret_name]
            sell_decile[["weight", "inv_ret"]] *= -1

            buy_decile = rebalance_df.loc[decile_weight_dict[date][cut - 1].index].copy()
            buy_decile["weight"] = decile_weight_dict[date][cut - 1]
            buy_decile["inv_ret"] = buy_decile["weight"] * buy_decile[ret_name]

            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(index=np.unique(list(to_df.index) + list(prev_to_df.index)))
                tto_df["cur_weight"] = to_df["weight"]
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[
                    ["weight", ret_name, "inv_ret"]
                ]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (
                        tto_df["cur_weight"]
                        - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom
                    ).abs()
                )
                turnover[i - 1] = turnover[i - 1] * 0.5
            prev_to_df = to_df

        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]

        return portfolio_ret, np.mean(turnover)


    def __add_period_ret_to_us_res_df_w_delays(self, signal_df):
        print("🔍 [DEBUG] Entered __add_period_ret_to_us_res_df_w_delays")

        # Load period returns
        period_ret = eqd.get_stock_ret_freqs(self.freq, country=self.country)
        print(f"📂 Loaded from cache: {period_ret.shape[0]} rows")

        # Determine required columns
        columns = ["MarketCap"] + [
            f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list
        ]
        if self.custom_ret is not None:
            columns += [self.custom_ret]

        # Show structure of signal_df
        print("🔢 [DEBUG] signal_df before merge:")
        print(f"       Index: {signal_df.index.names}")
        print(f"       Columns: {list(signal_df.columns)}")
        print("       Head:\n", signal_df.head())

        # Show structure of period_ret
        print("📦 [DEBUG] period_ret:")
        print(f"       Index: {period_ret.index.names if period_ret.index.names[0] is not None else '[Not Set]'}")
        print(f"       Columns: {list(period_ret.columns)}")
        print("       Head:\n", period_ret.head())

        # Ensure period_ret has same MultiIndex as signal_df
        period_ret = period_ret.set_index(["Date", "StockID"])
        print("🔧 [DEBUG] Set period_ret index to ['Date', 'StockID']")
        print("       period_ret index:", period_ret.index.names)

        # Merge on index
        signal_df = signal_df.copy()
        signal_df = signal_df.merge(
            period_ret[columns],
            left_index=True,
            right_index=True,
            how="left",
        )
        print(f"✅ [DEBUG] Merged signal_df has {len(signal_df)} rows")

        # Drop rows with any NaNs
        signal_df.dropna(inplace=True)
        print(f"🧹 [DEBUG] Dropped rows with NaNs: now {len(signal_df)} rows remain")

        # Store no-delay return
        signal_df[self.no_delay_ret_name] = signal_df[f"next_{self.freq}_ret_0delay"]

        # Show delay-specific debug info
        for dl in self.delay_list:
            dl_ret_name = f"next_{self.freq}_ret_{dl}delay"
            print(
                f"📊 [DEBUG] {len(signal_df)} samples, delay={dl} | "
                f"NaNs={np.sum(signal_df[dl_ret_name].isna())}, "
                f"Zeros={np.sum(signal_df[dl_ret_name] == 0.)}"
            )

        return signal_df.copy()



    def get_up_prob_with_period_ret(self, signal_df):
        print(f"\n🔍 [DEBUG] Entered get_up_prob_with_period_ret")
        print(f"📅 [DEBUG] Filtering signal_df by year from {self.start_year} to {self.end_year}")

        before_filter = len(signal_df)
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]
        after_filter = len(signal_df)
        print(f"📉 [DEBUG] Filtered by year: {before_filter} → {after_filter} rows")

        signal_df = self.__add_period_ret_to_us_res_df_w_delays(signal_df)

        if self.country not in ["future", "new_future"]:
            before_mktcap = len(signal_df)
            signal_df.MarketCap = signal_df.MarketCap.abs()
            signal_df = signal_df[~signal_df.MarketCap.isnull()].copy()
            after_mktcap = len(signal_df)
            print(f"💰 [DEBUG] Filtered MarketCap NaNs: {before_mktcap} → {after_mktcap} rows")

        print(f"✅ [DEBUG] Finished get_up_prob_with_period_ret with {len(signal_df)} rows\n")
        return signal_df

    @staticmethod
    def _ret_to_cum_log_ret(rets):
        log_rets = np.log(rets.astype(float) + 1)
        return log_rets.cumsum()

    @staticmethod
    def _ret_to_cum_ret(rets):
        """
        Convert a series of period returns to cumulative returns (not in log space).
        rets: pandas Series or numpy array of returns (e.g., 0.05 for 5%)
        Returns: cumulative return series
        """
        cum_rets = (1 + rets.astype(float)).cumprod()
        return cum_rets 



    def make_portfolio_plot(
        self, portfolio_ret, cut, weight_type, save_path, plot_title, log=False
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        print(f"\n[DEBUG] Starting make_portfolio_plot")
        print(f"[DEBUG] cut = {cut}, weight_type = {weight_type}")
        print(f"[DEBUG] portfolio_ret shape = {portfolio_ret.shape}")

        # Choose return type
        ret_name = "ewretx" if weight_type == "ew" else "vwretx"
        print(f"[DEBUG] Using return type column: {ret_name}")

        # Copy and rename columns
        df = portfolio_ret.copy()
        expected_columns = ["Low(L)"] + [f"{i}" for i in range(2, cut)] + ["High(H)", "H-L"]
        df.columns = expected_columns
        print(f"[DEBUG] Renamed df columns: {df.columns.tolist()}")

        # # Add SPY benchmark returns
        # spy = eqd.get_spy_freq_rets(self.freq)
        # df["SPY"] = spy[ret_name]
        # print(f"[DEBUG] Added SPY column. df shape: {df.shape}")
        
        # Add SPY benchmark returns
        spy = eqd.get_benchmark_returns(self.freq, "CN")
        df["Index"] = spy
        print(f"[DEBUG] Added SPY column. df shape: {df.shape}")

        # Drop NA values
        # df.dropna(inplace=True)
        # print(f"[DEBUG] After dropna, df shape: {df.shape}")

        # Convert returns to cumulative log returns
        log_ret_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            if log:
                log_ret_df[col] = self._ret_to_cum_log_ret(df[col])
            else:
                log_ret_df[col] = self._ret_to_cum_ret(df[col])
        print(f"[DEBUG] Cumulative log return columns: {log_ret_df.columns.tolist()}")

        # Plot all lines
        plt.figure(figsize=(20, 10))
        for col in log_ret_df.columns:
          lw=1
          if col == "High(H)" or col == "Index":
            lw = 4
          plt.plot(log_ret_df.index, log_ret_df[col], label=col, lw=lw)
        print(f"[DEBUG] Finished plotting all lines.")

        
        
        # Finalize plot
        plt.ylim(0.6, 2)
        plt.title(plot_title)
        plt.legend(loc="best", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[DEBUG] Plot saved to {save_path}\n")

    def portfolio_res_summary(self, portfolio_ret, turnover, cut=10):
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))
        period = 52 if self.freq == "week" else 12 if self.freq == "month" else 4
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 / 4
            if self.freq == "week"
            else 1
            if self.freq == "month"
            else 3
            if self.freq == "quarter"
            else 12
        )
        print(summary_df)
        return summary_df



    def get_portfolio_name(self, weight_type, delay, cut):
        # assert weight_type.lower() in ["ew", "vw"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_surfix = "" if self.custom_ret is None else self.custom_ret
        tc_surfix = "" if not self.transaction_cost else "_w_transaction_cost"
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_surfix}{custom_ret_surfix}{tc_surfix}"
        return pf_name

    def load_portfolio_ret(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}_100.csv"),
                index_col=0,
            )
        df.index = pd.to_datetime(df.index)
        return df

    def load_portfolio_summary(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(op.join(self.portfolio_dir, f"{pf_name}.csv"), index_col=0)
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, f"{pf_name}_100.csv"), index_col=0
            )
        return df





def main():
    pass


if __name__ == "__main__":
    main()

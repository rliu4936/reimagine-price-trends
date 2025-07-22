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
            print("✅ DF LENGTH 86", len(signal_df))
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)
            print("✅ DF LENGTH 86", len(self.signal_df))

    def __add_period_ret_to_us_res_df_w_delays(self, signal_df):
        # ---------- prep ----------
        period_ret = eqd.get_stock_ret_freqs(self.freq, country=self.country)

        # Only the return columns you really need to merge in
        ret_cols = [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        if self.custom_ret is not None:
            ret_cols.append(self.custom_ret)

        # Work on copies
        signal_df = signal_df.copy()
        signal_df = signal_df.reset_index()  # Date, StockID -> columns
        signal_df["StockID"] = signal_df["StockID"].astype(str).str.zfill(6)
        period_ret["StockID"] = period_ret["StockID"].astype(str).str.zfill(6)

        # ---------- merge ----------
        merged = pd.merge(
            signal_df,
            period_ret[["Date", "StockID", "MarketCap"] + ret_cols],
            on=["Date", "StockID"],
            how="left",
            suffixes=("", "_pr")   # keep signal_df's names untouched
        )

        # If MarketCap missing in signal_df, fill from period_ret
        if "MarketCap" not in merged.columns:
            merged["MarketCap"] = merged["MarketCap_pr"]
        else:
            merged["MarketCap"] = merged["MarketCap"].fillna(merged["MarketCap_pr"])
        if "MarketCap_pr" in merged.columns:
            merged.drop(columns="MarketCap_pr", inplace=True)

        # ---------- derive no-delay ret ----------
        merged[self.no_delay_ret_name] = merged[f"next_{self.freq}_ret_0delay"]

        # ---------- debug before drop ----------
        print("📉 NaNs before dropna (only ret cols):")
        print(merged[ret_cols].isna().sum())

        # Drop only rows missing the needed return columns (NOT everything)
        merged.dropna(subset=ret_cols, inplace=True)
        print("✅ DF LENGTH after dropna:", len(merged))

        # Optional diagnostics for each delay
        for dl in self.delay_list:
            col = f"next_{self.freq}_ret_{dl}delay"
            print(f"{dl} delay → NaNs: {merged[col].isna().sum()}, Zeros: {(merged[col]==0).sum()}")

        # Restore MultiIndex if you need it
        merged.set_index(["Date", "StockID"], inplace=True)

        return merged


    def get_up_prob_with_period_ret(self, signal_df):
        print("✅ DF LENGTH 63", len(signal_df))
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]
        print("✅ DF LENGTH 69", len(signal_df))
        signal_df = self.__add_period_ret_to_us_res_df_w_delays(signal_df)
        print("✅ DF LENGTH 71", len(signal_df))
        if self.country not in ["future", "new_future"]:
            signal_df.MarketCap = signal_df.MarketCap.abs()
            signal_df = signal_df[~signal_df.MarketCap.isnull()].copy()
        return signal_df

    def calculate_portfolio_rets(self, weight_type, cut=10, delay=0):
        assert weight_type in ["ew", "vw"]
        assert delay in self.delay_list
        if self.custom_ret:
            print(f"Calculating portfolio using {self.custom_ret}")
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name
                if delay == 0
                else f"next_{self.freq}_ret_{delay}delay"
            )
            
        df = self.signal_df.copy()
        print("✅ DF LENGTH 86", len(df))
        def __get_decile_df_with_inv_ret(reb_df, decile_idx):
            rebalance_up_prob = reb_df.up_prob
            low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = reb_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                _decile_df["weight"] = (
                    pd.Series(
                        np.ones(stock_num), dtype=float, index=_decile_df.index #
                    )
                    / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=float) / np.sum(value) #
            _decile_df["inv_ret"] = _decile_df["weight"] * _decile_df[ret_name]
            return _decile_df
        
        print("✅ DF LENGTH 110", len(df))
        dates = np.sort(np.unique(df.index.get_level_values("Date")))

        print(f"Calculating portfolio from {dates[0]}, {dates[1]} to {dates[-1]}")
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None
        prob_ret_corr = []
        prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []

        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()
            rank_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="spearman"
            )
            pearson_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="pearson"
            )
            prob_ret_corr.append(rank_corr)
            prob_ret_pearson_corr.append(pearson_corr)

            low = np.percentile(rebalance_df.up_prob, 10)
            high = np.percentile(rebalance_df.up_prob, 90)
            if low == high:
                print(f"Skipping {date}")
                continue
            for j in range(cut):
                decile_df = __get_decile_df_with_inv_ret(rebalance_df, j)
                if self.transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = __get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = __get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            buy_sell_decile = pd.concat([sell_decile, buy_decile])
            prob_inv_ret_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
            )
            prob_inv_ret_pearson_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            )

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (
                -1
            )
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(
                    index=np.unique(list(to_df.index) + list(prev_to_df.index))
                )
                try:
                    tto_df["cur_weight"] = to_df["weight"]
                except ValueError:
                    pdb.set_trace()
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
        print(
            f"Spearman Corr between Prob and Stock Return is {np.nanmean(prob_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Stock Return is {np.nanmean(prob_ret_pearson_corr):.4f}"
        )
        print(
            f"Spearman Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_pearson_corr):.4f}"
        )

        plot_title = f"{self.freq.capitalize()} Portfolio ({weight_type.upper()} Weighting)"
        save_path = op.join(self.portfolio_dir, f"portfolio_plot_{weight_type}_{cut}.png")
        self.make_portfolio_plot(portfolio_ret=portfolio_ret, cut=cut, weight_type=weight_type, plot_title=plot_title, save_path=save_path)

        return portfolio_ret, np.mean(turnover)

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
        self, portfolio_ret, cut, weight_type, save_path, plot_title
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
        spy = eqd.get_index_ret(self.freq, "CN")
                # Set 'date' as index and select the 'Ret' column
        spy_series = spy.set_index("date")["Ret"]

        # Align the index type to match df (if needed)
        spy_series.index = pd.to_datetime(spy_series.index)
        df.index = pd.to_datetime(df.index)

        # Assign to df (only if indices match)
        df["Index"] = spy_series
        print(f"[DEBUG] Added SPY column. df shape: {df.shape}")

        # Drop NA values
        # df.dropna(inplace=True)
        # print(f"[DEBUG] After dropna, df shape: {df.shape}")

        # Convert returns to cumulative log returns
        log_ret_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            # log_ret_df[col] = self._ret_to_cum_log_ret(df[col])
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

    def generate_portfolio(self, cut=5, delay=0):
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
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}_{cut}.csv")
            print(f"Summary saved to {smry_path}")
            summary_df.to_csv(smry_path)
            with open(os.path.join(self.portfolio_dir, f"{pf_name}_{cut}.txt"), "w+") as file:
                summary_df = summary_df.astype(float).round(2)
                file.write(ut.to_latex_w_turnover(summary_df, cut=cut))

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


def _latest_close_map(price_data_dir: str) -> dict[str, float]:
    """Read every CSV in price_data_dir and return {StockID: latest Close}"""
    close_map = {}
    for csv in os.listdir(price_data_dir):
        if csv.endswith(".csv"):
            stock = csv[:-4]                       # remove ".csv"
            df = pd.read_csv(os.path.join(price_data_dir, csv))
            if len(df):
                close_map[stock] = float(df.iloc[-1]["Close"])
    return close_map


def save_current_decile_weights(
    signal_df: pd.DataFrame,
    cut: int = 10,
    top: int | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Assign deciles by 'up_prob', ensure 'Close', optional top-N, save CSV."""

    # need up_prob
    if "up_prob" not in signal_df.columns:
        raise KeyError("signal_df must contain 'up_prob'")

    df = signal_df.copy()  # copy

    # ensure Close
    if "Close" not in df.columns:
        if output_path is None:
            raise KeyError("'Close' missing and no output_path to infer data dir")
        date_dir = os.path.dirname(output_path)              # date dir
        price_data_dir = os.path.join(date_dir, "price_data")  # price dir
        if not os.path.isdir(price_data_dir):
            raise FileNotFoundError(f"price_data not found: {price_data_dir}")
        close_map = _latest_close_map(price_data_dir)        # load map
        df["Close"] = df["StockID"].map(close_map)           # fill Close
        if df["Close"].isna().any():
            missing = df[df["Close"].isna()]["StockID"].tolist()
            raise ValueError(f"Close price not found for: {missing}")

    # assign deciles
    n = len(df)
    if n == 0:
        raise ValueError("signal_df is empty")
    ranks = df["up_prob"].rank(method="first", ascending=True)
    df["decile"] = ((ranks - 1) / n * cut).astype(int)

    df = df.sort_values("up_prob", ascending=False).reset_index(drop=True)  # sort by prob

    # top‑N slice
    if top is not None:
        if top <= 0:
            raise ValueError("top must be > 0")
        df = df.iloc[:top].copy()

    # save
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✅ Saved decile assignments to {output_path}")

    return df


def main():
    pass


if __name__ == "__main__":
    main()
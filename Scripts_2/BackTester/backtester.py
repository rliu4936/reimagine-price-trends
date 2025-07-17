
# import os
import os.path as op
import pdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from DataHandler import equity_data as eqd
from Scripts_2.PortfolioManager.portfolio import PortfolioManager

# TODO
# add inversion based metric

# ==============================================================
#  BACKTESTER – return maths, plotting, I/O, summaries
# ==============================================================
class Backtester:
    """Wrap a PortfolioManager and perform full back‑tests."""

    # ---------------------- utility helpers ----------------------
    @staticmethod
    def _ret_to_cum_log_ret(rets: pd.Series) -> pd.Series:
        return np.log1p(rets.astype(float)).cumsum()

    @staticmethod
    def _ret_to_cum_ret(rets: pd.Series) -> pd.Series:
        return (1 + rets.astype(float)).cumprod() - 1

    # ---------------------- life‑cycle ---------------------------
    def __init__(self, manager: PortfolioManager, portfolio_dir: str):
        self.mgr = manager
        self.portfolio_dir = portfolio_dir
        os.makedirs(op.join(portfolio_dir, "pf_data"), exist_ok=True)

    # ------------------------------------------------------------------
    # Core calculation (calls PortfolioManager.build_decile)
    # ------------------------------------------------------------------
    def _calculate_portfolio_returns(
        self,
        weight_type: str,
        *,
        cut: int = 10,
        delay: int = 0,
    ) -> tuple[pd.DataFrame, float]:
        """Return (portfolio_ret_df, turnover_float)."""
        assert weight_type in {"ew", "vw"}
        assert delay in self.mgr.delay_list

        # pick the return column
        ret_name = (
            self.mgr.custom_ret
            if self.mgr.custom_ret
            else (
                self.mgr.no_delay_ret_name
                if delay == 0
                else f"next_{self.mgr.freq}_ret_{delay}delay"
            )
        )

        df = self.mgr.signal_df.copy()
        dates = np.sort(np.unique(df.index.get_level_values("Date")))

        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        turnover = np.zeros(len(dates) - 1)
        prev_to_df = None

        # --------------------------------------------------------------
        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()

            # skip degenerate distribution
            if np.percentile(rebalance_df.up_prob, 10) == np.percentile(
                rebalance_df.up_prob, 90
            ):
                continue

            for j in range(cut):
                decile_df = self.mgr.build_decile(
                    rebalance_df,
                    j,
                    weight_type=weight_type,
                    cut=cut,
                    ret_name=ret_name,
                )

                # optional transaction cost handling
                if self.mgr.transaction_cost:
                    factor = decile_df["transaction_fee"] * 2
                    if j == cut - 1:
                        decile_df["inv_ret"] -= decile_df["weight"] * factor
                    elif j == 0:
                        decile_df["inv_ret"] += decile_df["weight"] * factor

                portfolio_ret.loc[date, j] = decile_df["inv_ret"].sum()

            # turnover calc – use top & bottom deciles
            sell_decile = self.mgr.build_decile(
                rebalance_df,
                0,
                weight_type=weight_type,
                cut=cut,
                ret_name=ret_name,
            )
            buy_decile = self.mgr.build_decile(
                rebalance_df,
                cut - 1,
                weight_type=weight_type,
                cut=cut,
                ret_name=ret_name,
            )
            sell_decile[["weight", "inv_ret"]] *= -1
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                idx = np.unique(list(to_df.index) + list(prev_to_df.index))
                tdf = pd.DataFrame(index=idx)
                tdf["cur_w"] = to_df["weight"].reindex(idx).fillna(0)
                tdf["prev_w"] = prev_to_df["weight"].reindex(idx).fillna(0)
                tdf["ret"] = prev_to_df[ret_name].reindex(idx).fillna(0)
                denom = 1 + prev_to_df["inv_ret"].sum()
                turnover[i - 1] = (
                    (tdf["cur_w"] - tdf["prev_w"] * (1 + tdf["ret"]) / denom)
                    .abs()
                    .sum()
                    * 0.5
                )
            prev_to_df = to_df

        portfolio_ret.fillna(0, inplace=True)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        return portfolio_ret, turnover.mean()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        cut: int = 10,
        delay: int = 0,
        weight_types: tuple[str, ...] = ("ew", "vw"),
    ) -> None:
        for wt in weight_types:
            pf_name = self._portfolio


def main():
    pass


if __name__ == "__main__":
    main()

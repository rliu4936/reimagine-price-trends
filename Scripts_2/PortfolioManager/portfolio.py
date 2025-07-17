# import os
import os.path as op
import pdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from DataHandler import equity_data as eqd


# ==============================================================
#  PORTFOLIO MANAGER – data wrangling **and weight assignment**
# ==============================================================
class PortfolioManager:
    """Prepare signals **and** apply portfolio weights for a given cut."""

    # ---------------------- life‑cycle ---------------------------
    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        country: str = "USA",
        start_year: int = 1800,
        end_year: int = 2200,
        delay_list: list[int] | None = None,
        custom_ret: str | None = None,
        transaction_cost: bool = False,
        load_signal: bool = True,
    ) -> None:
        assert freq in {"week", "month", "quarter"}, "freq must be week, month or quarter"
        self.freq = freq
        self.country = country
        self.start_year = start_year
        self.end_year = end_year
        self.delay_list = [0] if delay_list is None else delay_list
        self.custom_ret = custom_ret
        self.transaction_cost = transaction_cost
        self.no_delay_ret_name = f"next_{freq}_ret"

        # prepare signal_df (attach returns, filter years, etc.)
        self.signal_df = (
            self._prepare_signal(signal_df) if load_signal else signal_df.copy()
        )

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _prepare_signal(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """Filter year range & attach forward returns / market‑cap info."""
        mask_year = signal_df.index.get_level_values("Date").year.isin(
            range(self.start_year, self.end_year + 1)
        )
        signal_df = signal_df[mask_year]

        # attach per‑period returns for each requested delay
        signal_df = self._add_period_ret(signal_df)

        # drop rows with missing / negative market cap (for cash equity)
        if self.country not in {"future", "new_future"} and "MarketCap" in signal_df.columns:
            signal_df["MarketCap"] = signal_df.MarketCap.abs()
            signal_df = signal_df[~signal_df.MarketCap.isna()].copy()
        return signal_df

    def _add_period_ret(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        period_ret = eqd.get_period_ret(self.freq, country=self.country)
        base_cols = [f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list]
        columns = ["MarketCap"] + base_cols
        if self.custom_ret is not None:
            columns.append(self.custom_ret)

        signal_df = signal_df.copy()
        signal_df[columns] = period_ret[columns]
        signal_df[self.no_delay_ret_name] = signal_df[f"next_{self.freq}_ret_0delay"]
        signal_df.dropna(inplace=True)
        return signal_df

    # ------------------------------------------------------------------
    # Weighting / decile helper
    # ------------------------------------------------------------------
    def build_decile(
        self,
        rebalance_df: pd.DataFrame,
        decile_idx: int,
        *,
        weight_type: str,
        cut: int,
        ret_name: str,
    ) -> pd.DataFrame:
        """Return a DataFrame for one decile with weights & inv_ret calculated."""
        assert weight_type in {"ew", "vw"}
        up_prob = rebalance_df.up_prob
        low = np.percentile(up_prob, decile_idx * 100.0 / cut)
        high = np.percentile(up_prob, (decile_idx + 1) * 100.0 / cut)
        filt = (
            (up_prob >= low) & (up_prob <= high)
            if decile_idx == 0
            else (up_prob > low) & (up_prob <= high)
        )
        ddf = rebalance_df[filt].copy()

        # assign weights
        if weight_type == "ew":
            ddf["weight"] = 1.0 / len(ddf)
        else:
            ddf["weight"] = ddf.MarketCap / ddf.MarketCap.sum()

        # investment return contribution
        ddf["inv_ret"] = ddf["weight"] * ddf[ret_name]
        return ddf

def main():
    pass


if __name__ == "__main__":
    main()

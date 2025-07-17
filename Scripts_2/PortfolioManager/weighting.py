import os
import os.path as op
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from DataHandler import equity_data as eqd

# ------------------------------------------------------------------
#  WEIGHT‑SCHEME INTERFACE & IMPLEMENTATIONS
# ------------------------------------------------------------------
class WeightScheme(ABC):
    """Return a *pd.Series* of portfolio weights indexed the same as *df*."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError


class EqualWeight(WeightScheme):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n, index=df.index, dtype=float)


class ValueWeight(WeightScheme):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        if "MarketCap" not in df.columns:
            raise KeyError("MarketCap column required for value weighting")
        total = df.MarketCap.sum()
        if total == 0:
            return pd.Series(dtype=float)
        return df.MarketCap.astype(float) / total


# --------‑‑‑ Skeletons (to be implemented later) -------------------
class MarkowitzPastWeight(WeightScheme):
    """Weights from historical mean/variance (classic Markowitz)."""

    def __call__(self, df: pd.DataFrame) -> pd.Series:  # noqa: D401
        # TODO: implement Markowitz weights based on past returns
        raise NotImplementedError("Markowitz‑Past weighting not implemented yet.")

class MarkowitzPredWeight(WeightScheme):
    """Markowitz, but using *predicted* μ and Σ instead of historical ones."""

    def __call__(self, df: pd.DataFrame) -> pd.Series:  # noqa: D401
        # TODO: implement Markowitz using predicted returns / covariances
        raise NotImplementedError("Markowitz‑Pred weighting not implemented yet.")
    
class MLWeight(WeightScheme):
    """Let an external ML model map features → weights."""

    def __call__(self, df: pd.DataFrame) -> pd.Series:  # noqa: D401
        # TODO: plug‑in machine‑learning optimiser (e.g. reinforcement learning)
        raise NotImplementedError("ML weighting not implemented yet.")



# Central registry for easy look‑up
WEIGHT_SCHEMES: dict[str, WeightScheme] = {
    "ew": EqualWeight(),
    "vw": ValueWeight(),
    "markowitz_past": MarkowitzPastWeight(),
    "ml": MLWeight(),
    "markowitz_pred": MarkowitzPredWeight(),
}
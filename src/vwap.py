"""VWAP computation and VWAP-trend signal.

The Zarattini/Aziz (2023) paper uses an intraday session-anchored VWAP reset
at the 9:30 ET open: go long when price > VWAP, short when price < VWAP.
With daily bars we cannot reset each session, so we approximate by a rolling
window VWAP computed from typical price (H+L+C)/3 and volume. This is the
same idea as "Anchored VWAP over the last N days" used widely by practitioners.

Signal definition:
    +1 when Close > VWAP  (trend-up, demand > supply)
    -1 when Close < VWAP  (trend-down, supply > demand)
     0 if either is NaN

Optional 'min_dist_bps' requires price to be sufficiently far from VWAP
to filter out chop — left as 0 in the baseline to match the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling volume-weighted average price over `window` sessions."""
    tp = typical_price(df)
    vol = df["Volume"].astype(float)
    pv = (tp * vol).rolling(window, min_periods=max(5, window // 4)).sum()
    v = vol.rolling(window, min_periods=max(5, window // 4)).sum()
    return pv / v


def vwap_signal(df: pd.DataFrame,
                window: int = 20,
                min_dist_bps: float = 0.0) -> pd.Series:
    """Return {-1,0,+1} VWAP-trend signal."""
    vwap = rolling_vwap(df, window)
    close = df["Close"]
    dist = (close - vwap) / vwap
    sig = pd.Series(0, index=df.index, dtype=float)
    if min_dist_bps > 0:
        thr = min_dist_bps / 10_000.0
        sig[dist > thr] = 1.0
        sig[dist < -thr] = -1.0
    else:
        sig[close > vwap] = 1.0
        sig[close < vwap] = -1.0
    sig[vwap.isna()] = 0.0
    return sig


def vwap_trend_signal(df: pd.DataFrame,
                      fast: int = 10,
                      slow: int = 50) -> pd.Series:
    """Two-window VWAP crossover variant: trade only when both point the same way."""
    vwap_f = rolling_vwap(df, fast)
    vwap_s = rolling_vwap(df, slow)
    close = df["Close"]
    sig = pd.Series(0.0, index=df.index)
    sig[(close > vwap_f) & (vwap_f > vwap_s)] = 1.0
    sig[(close < vwap_f) & (vwap_f < vwap_s)] = -1.0
    sig[vwap_f.isna() | vwap_s.isna()] = 0.0
    return sig

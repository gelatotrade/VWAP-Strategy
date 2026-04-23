"""VWAP computation, VWAP-bands, and signal generators.

Signal families implemented
---------------------------
1. `vwap_momentum_signal` — Zarattini/Aziz (2023) style trend rule:
       +1 when Close > VWAP, -1 when Close < VWAP.
   Best in trending, low-vol regimes ("price walks along VWAP").

2. `vwap_mean_reversion_signal` — fade extreme deviations from VWAP using
   standard-deviation bands (analog of Bollinger Bands with VWAP as mean).
   Entry when |z| > k_entry, exit when |z| < k_exit.
   Best in range-bound / choppy regimes where price oscillates around VWAP.

3. `vwap_trend_signal` — fast vs slow VWAP crossover (two windows agree).

VWAP is computed as rolling volume-weighted typical price over N sessions;
this is the daily-bar approximation of the paper's intraday session-anchored
VWAP (see README for the tradeoff).
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


def rolling_vwap_std(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-weighted std-deviation of typical price around the rolling VWAP.

    Used to build VWAP ± k*σ bands (GoCharting / TrendSpider convention).
    """
    tp = typical_price(df)
    vol = df["Volume"].astype(float)
    vwap = rolling_vwap(df, window)
    dev2 = (tp - vwap) ** 2
    pv2 = (dev2 * vol).rolling(window,
                               min_periods=max(5, window // 4)).sum()
    v = vol.rolling(window, min_periods=max(5, window // 4)).sum()
    return np.sqrt(pv2 / v)


# ---------------------------------------------------------------------------
# Signal 1: momentum (a.k.a. VWAP trend, Zarattini/Aziz)
# ---------------------------------------------------------------------------
def vwap_momentum_signal(df: pd.DataFrame,
                         window: int = 20,
                         min_dist_bps: float = 0.0) -> pd.Series:
    """+1 above VWAP, -1 below VWAP. Optional buffer to filter chop."""
    vwap = rolling_vwap(df, window)
    close = df["Close"]
    dist = (close - vwap) / vwap
    sig = pd.Series(0.0, index=df.index)
    if min_dist_bps > 0:
        thr = min_dist_bps / 10_000.0
        sig[dist > thr] = 1.0
        sig[dist < -thr] = -1.0
    else:
        sig[close > vwap] = 1.0
        sig[close < vwap] = -1.0
    sig[vwap.isna()] = 0.0
    return sig


# alias for backwards compatibility
vwap_signal = vwap_momentum_signal


# ---------------------------------------------------------------------------
# Signal 2: mean reversion with VWAP ± k*σ bands
# ---------------------------------------------------------------------------
def vwap_mean_reversion_signal(df: pd.DataFrame,
                               window: int = 20,
                               k_entry: float = 2.0,
                               k_exit: float = 0.5) -> pd.Series:
    """Fade price deviations beyond ±k_entry standard deviations of VWAP.

    Enters a counter-trend position when z = (close - VWAP) / σ crosses the
    entry band, stays in it until z pulls back inside ±k_exit. This models
    the common "VWAP reversion with stdev bands" practitioner rule.

    * Long  when  z < -k_entry   (oversold below VWAP)
    * Short when  z >  k_entry   (overbought above VWAP)
    * Flat otherwise, with hysteresis so the position is held through a
      small mean-reversion move rather than flipping each bar.

    k_entry, k_exit are in units of standard deviations.
    """
    vwap = rolling_vwap(df, window)
    sd = rolling_vwap_std(df, window)
    close = df["Close"]
    z = (close - vwap) / sd.replace(0, np.nan)

    sig = np.zeros(len(df))
    pos = 0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            pos = 0
        elif pos == 0:
            if zi < -k_entry:
                pos = 1
            elif zi > k_entry:
                pos = -1
        elif pos == 1 and zi >= -k_exit:
            pos = 0
        elif pos == -1 and zi <= k_exit:
            pos = 0
        sig[i] = pos
    return pd.Series(sig, index=df.index)


# ---------------------------------------------------------------------------
# Signal 3: fast/slow VWAP crossover (optional)
# ---------------------------------------------------------------------------
def vwap_trend_signal(df: pd.DataFrame,
                      fast: int = 10,
                      slow: int = 50) -> pd.Series:
    """Trade only when price and fast VWAP agree on the trend vs slow VWAP."""
    vwap_f = rolling_vwap(df, fast)
    vwap_s = rolling_vwap(df, slow)
    close = df["Close"]
    sig = pd.Series(0.0, index=df.index)
    sig[(close > vwap_f) & (vwap_f > vwap_s)] = 1.0
    sig[(close < vwap_f) & (vwap_f < vwap_s)] = -1.0
    sig[vwap_f.isna() | vwap_s.isna()] = 0.0
    return sig

"""Lightweight daily backtester + performance metrics.

A position series in {-1, 0, +1} is applied to next-day returns with a
per-turnover commission. Returns equity curve and summary stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series          # cumulative equity (1.0 = start)
    returns: pd.Series         # daily strategy returns (net of costs)
    position: pd.Series        # traded position series (shifted)
    stats: dict                # summary metrics


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def _sharpe(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / sd)


def _cagr(equity: pd.Series, periods: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    years = len(equity) / periods
    return float(equity.iloc[-1] ** (1 / years) - 1.0)


def run_backtest(prices: pd.DataFrame,
                 signal: pd.Series,
                 commission_bps: float = 1.0,
                 slippage_bps: float = 0.5,
                 allow_short: bool = True) -> BacktestResult:
    """Apply a daily signal in {-1, 0, +1} to next-day close-to-close returns.

    Args:
        prices: OHLCV dataframe with at least 'Close'.
        signal: desired position at the end of each day (same index as prices).
        commission_bps: per-turnover cost in bps (e.g. 1 bp = 0.0001 of notional).
        slippage_bps: extra bps on every position change.
        allow_short: if False, clip signal to non-negative.
    """
    close = prices["Close"].astype(float)
    sig = signal.reindex(close.index).fillna(0.0).clip(-1, 1)
    if not allow_short:
        sig = sig.clip(lower=0.0)

    # shift by 1 day — decision at t becomes position for t+1
    pos = sig.shift(1).fillna(0.0)
    daily_ret = close.pct_change().fillna(0.0)

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (commission_bps + slippage_bps) / 10_000.0
    strat_ret = pos * daily_ret - cost

    # If a day's strategy return is <= -100 %, the account is ruined;
    # floor at -99.5 % so cumprod stays strictly positive (otherwise
    # negative equity values compound to absurd drawdowns).
    strat_ret = strat_ret.clip(lower=-0.995)
    equity = (1 + strat_ret).cumprod()
    stats = {
        "CAGR":        _cagr(equity),
        "Sharpe":      _sharpe(strat_ret),
        "MaxDD":       _max_drawdown(equity),
        "TotalReturn": float(equity.iloc[-1] - 1.0),
        "Volatility":  float(strat_ret.std() * np.sqrt(252)),
        "HitRate":     float((strat_ret > 0).mean()),
        "AvgExposure": float(pos.abs().mean()),
        "Turnover":    float(turnover.sum()),
        "N":           int(len(strat_ret)),
    }
    return BacktestResult(equity=equity, returns=strat_ret, position=pos, stats=stats)


def buy_and_hold(prices: pd.DataFrame,
                 commission_bps: float = 1.0) -> BacktestResult:
    sig = pd.Series(1.0, index=prices.index)
    return run_backtest(prices, sig, commission_bps=commission_bps, slippage_bps=0.0)

"""Crypto data loader — pulls real daily price + volume from coinmetrics/data.

Coinmetrics publishes a CSV per asset with either
    * `PriceUSD`            — direct USD close (BTC, ETH, XRP)
    * or only `CapMrktEstUSD` + recent `ReferenceRateUSD` (SOL, HYPE).

For the latter case we recover an implied circulating supply from the
overlap of `CapMrktEstUSD` and `ReferenceRateUSD` on recent days and
reconstruct a full historical price series as
    Close_t  =  CapMrktEstUSD_t / supply_median
This is accurate enough for strategy backtesting — supply changes over
time are small relative to daily price moves.

Because coinmetrics gives only Close + Volume, we build a daily OHLCV
frame by setting Open[t] = Close[t-1] (crypto trades 24/7, so "gap" is
the midnight UTC step) and synthesising a small High/Low range from
rolling log-return volatility. VWAP on daily bars reduces to a
volume-weighted moving average of the close price, which is the standard
crypto-VWAP convention anyway.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CRYPTO_TICKERS = ("BTC", "ETH", "XRP", "SOL", "HYPE")

COINMETRICS_BASE = "https://raw.githubusercontent.com/coinmetrics/data/master/csv"


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _approx_ohlc(close: pd.Series, range_vol_lookback: int = 20) -> pd.DataFrame:
    """Build approximate Open/High/Low from a Close-only series.

    Open[t]  = Close[t-1]
    High[t]  = max(Open, Close) * (1 + k*σ)
    Low[t]   = min(Open, Close) * (1 - k*σ)
    where σ is rolling-20d log-return stdev and k=0.5 (roughly the
    expected max-min vs close-close range for daily bars).
    """
    log_ret = np.log(close / close.shift(1))
    sigma = (log_ret.rolling(range_vol_lookback).std()
             .bfill().fillna(0.02))
    open_ = close.shift(1).fillna(close.iloc[0])
    hi = np.maximum(open_, close) * (1 + 0.5 * sigma)
    lo = np.minimum(open_, close) * (1 - 0.5 * sigma)
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo,
                         "Close": close})


def _coinmetrics_close(raw_df: pd.DataFrame) -> pd.Series:
    """Extract a Close price series from a coinmetrics dataframe.

    Falls back to an implied-supply reconstruction if no direct USD
    price column has enough history. Reconstructed series are clipped
    to reasonable daily-return bounds to filter supply-change artifacts.
    """
    for c in ("PriceUSD", "ReferenceRateUSD"):
        if c in raw_df.columns and raw_df[c].notna().sum() > 200:
            return raw_df[c].astype(float).dropna()

    # reconstruct from CapMrktEstUSD + implied supply
    if {"CapMrktEstUSD", "ReferenceRateUSD"}.issubset(raw_df.columns):
        ref = raw_df["ReferenceRateUSD"]
        cap = raw_df["CapMrktEstUSD"]
        mask = ref.notna() & cap.notna()
        if mask.sum() > 0:
            supply = (cap[mask] / ref[mask]).median()
            close = (cap / supply).astype(float).dropna()
            # Clip extreme daily returns that are almost certainly the
            # result of supply-inflation inside CapMrktEstUSD rather than
            # a real 500%+ one-day price move.
            log_ret = np.log(close / close.shift(1))
            bad = log_ret.abs() > 0.7            # > ~100% one-day
            if bad.any():
                log_ret = log_ret.clip(-0.7, 0.7)
                close = float(close.iloc[0]) * np.exp(log_ret.cumsum().fillna(0))
            return close
    raise RuntimeError("no usable close/price column in coinmetrics row")


def load_crypto(ticker: str,
                start: Optional[str] = None,
                end: Optional[str] = None,
                refresh: bool = False) -> pd.DataFrame:
    """Load daily OHLCV for a crypto ticker.

    Returns columns Open/High/Low/Close/Volume/Adj Close.
    """
    cache = DATA_DIR / f"{ticker}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        url = f"{COINMETRICS_BASE}/{ticker.lower()}.csv"
        raw = _http_get(url)
        raw_df = pd.read_csv(io.StringIO(raw))
        raw_df["time"] = pd.to_datetime(raw_df["time"])
        raw_df = raw_df.set_index("time").sort_index()

        close = _coinmetrics_close(raw_df)

        vol_col = None
        for c in ("volume_reported_spot_usd_1d", "VolReportedSpotUSD1d"):
            if c in raw_df.columns and raw_df[c].notna().sum() > 50:
                vol_col = c
                break
        if vol_col is None:
            # last-resort: market-cap-proxy volume
            raise RuntimeError(f"no Volume column in coinmetrics {ticker}")
        volume = (raw_df[vol_col].astype(float)
                  .reindex(close.index).ffill().fillna(0))

        ohlc = _approx_ohlc(close)
        df = ohlc.assign(Volume=volume)
        df["Adj Close"] = df["Close"]
        df = df.dropna()
        df.to_csv(cache)

    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df.dropna()


def load_all_crypto(start: Optional[str] = None,
                    end: Optional[str] = None,
                    refresh: bool = False,
                    tickers: tuple[str, ...] = CRYPTO_TICKERS
                    ) -> dict[str, pd.DataFrame]:
    return {t: load_crypto(t, start, end, refresh) for t in tickers}


if __name__ == "__main__":
    for t in CRYPTO_TICKERS:
        d = load_crypto(t, refresh=True)
        print(f"{t:<5}: {len(d):5d} rows  "
              f"{d.index.min().date()} -> {d.index.max().date()}  "
              f"last_close=${d['Close'].iloc[-1]:,.2f}  "
              f"avg_vol=${d['Volume'].mean():,.0f}")


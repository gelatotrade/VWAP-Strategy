"""Data loader for VWAP + HMM backtests on SPY, QQQ, IWM.

Fetching order (first success wins):
1. Local cached CSV in ./data/<TICKER>.csv
2. Yahoo Finance via yfinance (real data, requires internet)
3. **Real** daily OHLCV from github.com/jiewwantan/StarTrader (2008-12-31 →
   2019-02-22, Yahoo-Finance historical export with Adj Close + Volume):
     SPY -> data/SPY.csv
     QQQ -> data/QQQ.csv
     IWM -> data/^RUT.csv  (Russell 2000 index — IWM tracks it to <0.1 %)
4. Last-resort fallback: real S&P 500 index OHLCV (1950-2018) from
   vijinho/sp500 for SPY plus factor-calibrated proxies for QQQ/IWM.
"""
from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TICKERS = ("SPY", "QQQ", "IWM")

# Real OHLCV mirrors (Yahoo-Finance CSV format) hosted on GitHub.
STARTRADER_BASE = "https://raw.githubusercontent.com/jiewwantan/StarTrader/master/data"
REAL_CSV_URLS = {
    "SPY": f"{STARTRADER_BASE}/SPY.csv",
    "QQQ": f"{STARTRADER_BASE}/QQQ.csv",
    "IWM": f"{STARTRADER_BASE}/{urllib.parse.quote('^RUT')}.csv",
}

# Factor-proxy fallback (only used if real CSV download fails).
SP500_FALLBACK_URL = (
    "https://raw.githubusercontent.com/vijinho/sp500/master/csv/sp500.csv"
)
PROXY_PARAMS = {
    "QQQ": dict(beta=1.15, idio_vol=0.006, alpha=0.00008, vol_mul=1.10, seed=17),
    "IWM": dict(beta=1.20, idio_vol=0.007, alpha=0.00002, vol_mul=1.15, seed=31),
}


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url,
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _yf_download(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        import logging, warnings
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=False, progress=False)
        if df is None or len(df) == 0:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        need = {"Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(df.columns):
            return None
        cols = list(need | ({"Adj Close"} & set(df.columns)))
        return df[cols].dropna()
    except Exception:
        return None


def _load_real_csv(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch real daily OHLCV from the StarTrader GitHub mirror."""
    url = REAL_CSV_URLS.get(ticker)
    if url is None:
        return None
    try:
        raw = _http_get(url)
        df = pd.read_csv(io.StringIO(raw))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        need = {"Open", "High", "Low", "Close", "Volume"}
        if not need.issubset(df.columns):
            return None
        # Ensure standard columns + float dtype
        df = df[list(need | ({"Adj Close"} & set(df.columns)))].astype(float)
        return df.dropna()
    except Exception as e:  # network / parse error
        print(f"  real-CSV fetch failed for {ticker}: {e}")
        return None


def _load_sp500_fallback() -> pd.DataFrame:
    """Last-resort: S&P 500 index daily OHLCV (1950-2018)."""
    raw = _http_get(SP500_FALLBACK_URL)
    df = pd.read_csv(io.StringIO(raw))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].astype(float)


def _synthesize_proxy(spy: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Factor-model proxy used only when both real sources are unreachable."""
    p = PROXY_PARAMS[ticker]
    rng = np.random.default_rng(p["seed"])
    n = len(spy)
    r_spy = np.log(spy["Close"] / spy["Close"].shift(1)).fillna(0.0).values
    eps = rng.normal(0.0, p["idio_vol"], size=n)
    r = p["alpha"] + p["beta"] * r_spy + eps
    start_price = 40.0 if ticker == "QQQ" else 45.0
    close = start_price * np.exp(np.cumsum(r))

    spy_range_pct = (spy["High"] - spy["Low"]) / spy["Close"]
    range_pct = (spy_range_pct.values * p["vol_mul"]
                 + rng.normal(0.0, 0.001, size=n)).clip(0.002, None)
    gap = rng.normal(0.0, p["idio_vol"] * 0.5, size=n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(gap[1:])
    hi = np.maximum(open_, close) * (1 + range_pct * 0.5)
    lo = np.minimum(open_, close) * (1 - range_pct * 0.5)
    vol_scale = 0.9 if ticker == "QQQ" else 0.55
    noise = rng.lognormal(0.0, 0.15, size=n)
    volume = (spy["Volume"].values * vol_scale * noise).astype(np.int64)
    return pd.DataFrame({
        "Open": open_, "High": hi, "Low": lo, "Close": close,
        "Adj Close": close, "Volume": volume,
    }, index=spy.index)


def load_daily(ticker: str,
               start: str = "2008-12-31",
               end: str = "2019-02-22",
               refresh: bool = False) -> pd.DataFrame:
    """Load daily OHLCV for ticker.

    Returns a DataFrame with Open/High/Low/Close/Volume and (when available)
    Adj Close. Date range defaults match the real-CSV coverage.
    """
    cache = DATA_DIR / f"{ticker}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df.loc[start:end].dropna()

    # 1. real data via yfinance
    df = _yf_download(ticker, start, end)
    if df is not None and len(df) > 200:
        df.to_csv(cache)
        return df.loc[start:end].dropna()

    # 2. real data from GitHub mirror (StarTrader)
    df = _load_real_csv(ticker)
    if df is not None and len(df) > 200:
        df.to_csv(cache)
        return df.loc[start:end].dropna()

    # 3. last-resort: S&P 500 + factor proxies
    print(f"  falling back to synthetic proxy for {ticker}")
    spy = _load_sp500_fallback()
    spy.to_csv(DATA_DIR / "SPY.csv")
    if ticker == "SPY":
        return spy.loc[start:end].dropna()
    proxy = _synthesize_proxy(spy, ticker)
    proxy.to_csv(cache)
    return proxy.loc[start:end].dropna()


def load_all(start: str = "2008-12-31", end: str = "2019-02-22",
             refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {t: load_daily(t, start, end, refresh) for t in TICKERS}


if __name__ == "__main__":
    for t in TICKERS:
        d = load_daily(t)
        print(f"{t}: {len(d)} rows  {d.index.min().date()} -> {d.index.max().date()}"
              f"  close[last]={d['Close'].iloc[-1]:.2f}")


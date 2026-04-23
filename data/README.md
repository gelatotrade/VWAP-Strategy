# data/

This directory holds cached CSVs populated at runtime by `src/data.py`.
The CSVs themselves are gitignored so the repo stays small.

Load order per ticker:
1. `data/<TICKER>.csv` (local cache from a previous run)
2. Yahoo Finance via `yfinance`
3. Fallback: real S&P 500 index OHLCV from
   [github.com/vijinho/sp500](https://github.com/vijinho/sp500) for SPY,
   plus factor-calibrated proxies for QQQ and IWM.

Run `python src/run.py` to (re)populate.

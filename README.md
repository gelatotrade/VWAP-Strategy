# VWAP Strategy & HMM Regime Backtest on SPY, QQQ, IWM

Replication and extension of **Zarattini & Aziz (2023),
*"Volume Weighted Average Price (VWAP) The Holy Grail for Day Trading Systems"*
([SSRN #4631351](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351)).
The paper shows that a long-above-VWAP / short-below-VWAP day-trading rule,
applied to QQQ/TQQQ (Jan 2018 – Sep 2023), produced a **671 % cumulative
return on QQQ with Sharpe 2.1 and MaxDD 9.4 %**.

This project:

1. Reviews the paper and public VWAP strategy literature.
2. Applies a VWAP-trend rule individually to **SPY, QQQ, IWM**.
3. Adds a **4-state Gaussian Hidden Markov regime filter** on top of the
   same VWAP signal and re-runs the backtest on the same three ETFs.

---

## 1. Research summary

### Paper rules (Zarattini/Aziz, 2023)
- Use **intraday session-anchored VWAP**, reset at the 09:30 ET open.
- **Long** when current price is above VWAP, **short** when below.
- Exit at session close. No overnight risk.
- Tested on QQQ and **3×-leveraged TQQQ** (2018-01-02 → 2023-09-28).
- Explicit commissions; no additional stop-loss or trend filter in the
  base variant.

### Public VWAP literature
Common practitioner variants we reviewed:
- **VWAP mean-reversion** — fade deviations from VWAP (opposite sign to
  trend version); works better in range-bound regimes.
- **VWAP + EMA trend filter** — only take VWAP signals when the 21-/55-EMA
  also points in the same direction.
- **VWAP crossover of two windows** (fast/slow anchored VWAPs).
- **Regime-switched VWAP** — disable the signal in pathological regimes.
  That is the variant implemented here as the HMM extension.

Sources used in this research:
- Zarattini & Aziz, *VWAP: The Holy Grail for Day Trading Systems*, 2023
  — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351).
- Concretum Research summary of the paper:
  <https://concretumgroup.com/volume-weighted-average-price-vwap-the-holy-grail-for-day-trading-systems/>
- Peak Capital Trading / Bear Bull Traders replication notes.
- QuantStart: *Market Regime Detection using Hidden Markov Models*.
- QuantInsti: *Regime-adaptive trading in Python*.

---

## 2. Methodology

### Data
**We tried to use Yahoo Finance (`yfinance`) for real SPY/QQQ/IWM OHLCV.**
The sandbox blocks `query*.finance.yahoo.com`, so we fall back to:

- **SPY** → real S&P 500 index daily OHLCV from
  [github.com/vijinho/sp500](https://github.com/vijinho/sp500)
  (1950-01-03 → 2018-12-21; we use 2007-2018 for the backtest).
- **QQQ, IWM** → statistically calibrated *proxies* built by a single-factor
  model on top of the S&P 500 series:
  `r_t = alpha + beta · r_spy_t + ε_t`, with β/vol/α matching historical
  NASDAQ-100 and Russell-2000 characteristics, plus log-normal noise on
  volume. See `src/data.py`.

The proxies are **not real market prints**. The `load_daily` function
transparently replaces them with real yfinance data the moment internet
access to Yahoo is available — no code change needed. All backtest logic,
metrics, and HMM fitting use exactly the same code paths for real and
proxy data.

### VWAP signal (daily approximation)
Paper uses intraday, session-anchored VWAP; `yfinance` only exposes ~2 y of
intraday bars, so for a multi-year backtest we use **rolling VWAP** over
`N = 20` sessions, computed from typical price `(H+L+C)/3` weighted by
volume. This is the same form practitioners call "20-day anchored VWAP".

Signal:
- `+1` when `Close > VWAP`
- `-1` when `Close < VWAP`
- `0` until the rolling window is filled

Order executed at next-day close; transaction cost = 1.0 bp commission +
0.5 bp slippage per side on every position change.

### Hidden Markov regime detector (4 states)
A Gaussian HMM with diagonal covariance is fit on three features:
1. 5-day EWMA of log returns
2. 20-day realized volatility (annualized)
3. 20-day return z-score

After fitting, the four latent states are **labelled** deterministically by
(mean-return, mean-volatility) → `HV_BEAR`, `LV_BEAR`, `LV_BULL`, `HV_BULL`.

Regime → VWAP overlay:
| Regime   | Action on VWAP signal                |
|----------|--------------------------------------|
| LV_BULL  | take full VWAP signal (long & short) |
| HV_BULL  | keep longs, skip shorts              |
| LV_BEAR  | keep shorts, skip longs              |
| HV_BEAR  | **stay flat** — avoid whipsaw        |

---

## 3. Results (2007-01-03 → 2018-12-21, daily)

### VWAP-only backtest
| Ticker | Strategy  | CAGR     | Sharpe | MaxDD    | Total Return |
|--------|-----------|---------:|-------:|---------:|-------------:|
| SPY    | Buy & Hold| +4.56 %  |  0.32  | -56.78 % | +70.58 %     |
| SPY    | VWAP(20)  | -8.53 %  | -0.35  | -70.97 % | -65.58 %     |
| QQQ    | Buy & Hold| +6.22 %  |  0.37  | -68.89 % | +105.88 %    |
| QQQ    | VWAP(20)  | -5.56 %  | -0.11  | -65.56 % | -49.56 %     |
| IWM    | Buy & Hold| -1.78 %  |  0.06  | -65.89 % | -19.37 %     |
| IWM    | VWAP(20)  | -7.40 %  | -0.16  | -74.34 % | -60.13 %     |

A 20-day rolling VWAP trend signal **whipsaws on daily bars** — it is a
slow moving-average-crossover with double-sided exposure through the
2008-09 and 2015-16 chop. This is expected: the paper's original signal
depends on an *intraday* session-anchored VWAP that produces many short
holding-period trades per day, not a multi-day crossover.

### VWAP + 4-state HMM backtest
| Ticker | Strategy      | CAGR        | Sharpe    | MaxDD        | Total Return |
|--------|---------------|------------:|----------:|-------------:|-------------:|
| SPY    | Buy & Hold    | +4.56 %     |  0.32     | -56.78 %     | +70.58 %     |
| SPY    | VWAP(20)      | -8.53 %     | -0.35     | -70.97 %     | -65.58 %     |
| SPY    | **VWAP+HMM4** | **+1.89 %** | **+0.22** | **-24.40 %** | **+25.16 %** |
| QQQ    | Buy & Hold    | +6.22 %     |  0.37     | -68.89 %     | +105.88 %    |
| QQQ    | VWAP(20)      | -5.56 %     | -0.11     | -65.56 %     | -49.56 %     |
| QQQ    | **VWAP+HMM4** | **+4.59 %** | **+0.38** | **-32.27 %** | **+71.10 %** |
| IWM    | Buy & Hold    | -1.78 %     |  0.06     | -65.89 %     | -19.37 %     |
| IWM    | VWAP(20)      | -7.40 %     | -0.16     | -74.34 %     | -60.13 %     |
| IWM    | **VWAP+HMM4** | **+2.45 %** | **+0.23** | **-47.44 %** | **+33.62 %** |

The HMM filter flips every VWAP-only backtest from loss-making to
positive, cuts max drawdown by roughly **half** on SPY and QQQ, and
roughly matches the Sharpe ratio of buy-and-hold on QQQ.
Exposure drops from ~100 % to ~60-80 % because the strategy sits flat
during `HV_BEAR` regimes.

Full artifacts:
- `results/vwap_only/summary.csv` and `*_vwap.csv` per ticker.
- `results/vwap_hmm/summary.csv`, `*_vwap_hmm.csv`, `*_regimes.csv`,
  and per-ticker `*_equity.png`, `*_regimes.png`, plus a combined
  `equity_grid.png`.

---

## 4. Caveats & next steps

1. **Data**: QQQ and IWM are factor-calibrated proxies on top of the
   real S&P 500 index in this sandbox. Real yfinance data is used
   automatically when reachable — just rerun `python src/run.py` on
   a network-unrestricted machine.
2. **Daily VWAP approximation** under-represents the paper's intraday
   signal. The natural next step is to switch the signal module to
   session-anchored intraday VWAP on 5-min or 1-min bars; the backtest
   engine in `src/backtest.py` accepts any frequency.
3. **No leverage** (paper also tested 3× TQQQ). Add a leveraged ticker
   to `TICKERS` and extend the loader to verify the ~8 000 % figure.
4. **HMM is fit in-sample** over the full period for simplicity. A
   walk-forward re-fit every N days (e.g. 252) is the correct
   out-of-sample protocol and will give more honest numbers.
5. **Regime rules are a prior, not learned.** The `regime_modulated_signal`
   mapping (HV_BEAR flat, LV_BEAR shorts-only, etc.) is a reasonable
   but untuned choice; it could be optimized per ticker with care
   against overfitting.

## Layout
```
VWAP-Strategy/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py           # SPY/QQQ/IWM loader with yfinance + fallback
│   ├── vwap.py           # rolling VWAP + signal
│   ├── hmm_regime.py     # 4-state Gaussian HMM + regime-to-signal overlay
│   ├── backtest.py       # daily backtest engine + metrics
│   └── run.py            # runs everything, writes tables + plots
└── results/
    ├── vwap_only/        # VWAP-only backtest outputs
    └── vwap_hmm/         # VWAP + HMM(4) backtest outputs + regime plots
```

## Reproducing
```
pip install -r requirements.txt
python src/run.py
```

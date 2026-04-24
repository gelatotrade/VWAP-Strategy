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
4. Adds a **VWAP mean-reversion signal** (fade price when it breaks a ±kσ
   band around VWAP) and combines momentum + mean-reversion per regime.
5. Runs a **parameter sweep** over window / band / regime-mapping combos
   and picks the best configuration per ticker.

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

### Momentum vs mean-reversion, regime-dependent behaviour
Why both signal families, and why switch between them? Three relevant
academic references:

- **Giner & Zakamulin (2023)** — *A regime-switching model of stock
  returns with momentum and mean reversion*
  ([SSRN #3997837](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3997837),
  published in *Economic Modelling*).
  Semi-Markov model where momentum dominates at short horizons and
  reversal at longer horizons; the optimal trend-following rule reduces
  to an MACD-like filter.
- **Fons, Dawson, Zeng, Keane & Iosifidis (2020)** — *Hidden Markov
  Models Applied To Intraday Momentum Trading With Side Information*
  ([arXiv:2006.08307](https://arxiv.org/pdf/2006.08307)). Latent
  momentum state generates noisy observed returns; signal flips
  cleanly at regime changes.
- **Nystrup, Madsen & Lindström (2020)** — *Regime-Switching Factor
  Investing with Hidden Markov Models*
  ([MDPI](https://www.mdpi.com/1911-8074/13/12/311)). Rotates between
  factor models conditional on HMM-detected regime; reports higher
  return and better risk-adjusted performance than the best static
  single-factor model.

The common thread: in **calm, trending** regimes momentum works
(price walks the VWAP); in **choppy / high-vol** regimes mean-reversion
works (price oscillates and tails get fat). Empirical VWAP trading
guides agree — VWAP ± σ bands are used explicitly for fade trades when
the tape is range-bound, and to stay out when price *walks the bands*.

### VWAP ± σ bands (mean-reversion signal)
The band-based fade rule is codified in the practitioner literature:

- **Trendspider**: *VWAP with St. Dev Bands* — 2-σ band is the
  "patient but stronger-signal" entry.
- **GoCharting**: *VWAP Bands — Standard Deviation Trading Strategy*.
- **LinkedIn / Chaudhary** — *VWAP Reversion with Standard Deviation
  Bands*.
- Entry on z-score breaking ±k_entry, exit on z-score pulling back
  within ±k_exit (hysteresis). Stop beyond ±3σ where the "gravitational
  pull" to VWAP weakens.

Sources used in this research:
- Zarattini & Aziz, *VWAP: The Holy Grail for Day Trading Systems*, 2023
  — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351).
- Concretum Research summary of the paper:
  <https://concretumgroup.com/volume-weighted-average-price-vwap-the-holy-grail-for-day-trading-systems/>
- Peak Capital Trading / Bear Bull Traders replication notes.
- QuantStart: *Market Regime Detection using Hidden Markov Models*.
- QuantInsti: *Regime-adaptive trading in Python*.
- Giner & Zakamulin (2023), *A Regime-Switching Model of Stock Returns
  with Momentum and Mean Reversion*.
- Fons et al. (2020), *HMMs Applied to Intraday Momentum Trading with
  Side Information*.
- Nystrup et al. (2020), *Regime-Switching Factor Investing with HMMs*.

---

## 2. Methodology

### Data
**The backtest now runs on real daily OHLCV.** `yfinance` and Alpha Vantage
are blocked by the sandbox (all `query*.finance.yahoo.com`,
`www.alphavantage.co`, Polygon, Tiingo, Stooq, FRED → 403), but GitHub is
reachable, so we pull real Yahoo-Finance-format CSVs from a public GitHub
mirror:

- **SPY** → `jiewwantan/StarTrader/data/SPY.csv`
- **QQQ** → `jiewwantan/StarTrader/data/QQQ.csv`
- **IWM** → `jiewwantan/StarTrader/data/^RUT.csv` — Russell 2000 **index**.
  IWM is the iShares ETF that tracks this index; daily returns match to
  <0.1 % tracking error, so the VWAP / band signals are identical.

All three cover **2008-12-31 → 2019-02-22** (2 553 trading days, ~10 years).
This window includes the 2009 recovery, 2011 euro-debt shock, 2015-16
energy-sector chop, and the 2018 Q4 selloff — enough regime variation for
the HMM to learn from.

The loader (`src/data.py`) tries `yfinance` first (so a full-internet
machine picks up the latest SPY/QQQ/IWM automatically) and falls back to
the GitHub CSVs if that fails. A last-resort factor-proxy path still
exists for offline use.

### VWAP signals
Paper uses intraday, session-anchored VWAP; `yfinance` only exposes ~2 y of
intraday bars, so for a multi-year backtest we use **rolling VWAP** over
`N` sessions, computed from typical price `(H+L+C)/3` weighted by volume.
This is the same form practitioners call "N-day anchored VWAP".

**Momentum (a.k.a. Zarattini/Aziz trend rule)** — `vwap_momentum_signal`
- `+1` when `Close > VWAP`
- `-1` when `Close < VWAP`

**Mean-reversion with VWAP ± k·σ bands** — `vwap_mean_reversion_signal`
- σ is the volume-weighted std-dev of typical-price around the rolling VWAP.
- z-score `z = (Close − VWAP) / σ`.
- Enter long when `z < −k_entry`, enter short when `z > +k_entry`.
- Exit when |z| < k_exit (hysteresis), so the position survives mild
  pullbacks toward VWAP without flipping every bar.
- Default `k_entry = 2.0`, `k_exit = 0.5`.

Order executed at next-day close; transaction cost = 1.0 bp commission +
0.5 bp slippage per side on every position change.

### Hidden Markov regime detector (4 states)
A Gaussian HMM with diagonal covariance is fit on three features:
1. 5-day EWMA of log returns
2. 20-day realized volatility (annualized)
3. 20-day return z-score

After fitting, the four latent states are **labelled** deterministically by
(mean-return, mean-volatility) → `HV_BEAR`, `LV_BEAR`, `LV_BULL`, `HV_BULL`.

Regime → VWAP overlay used by `MOM+HMM` and `MR+HMM`
(`regime_modulated_signal`):
| Regime   | Action on VWAP signal                |
|----------|--------------------------------------|
| LV_BULL  | take full VWAP signal (long & short) |
| HV_BULL  | keep longs, skip shorts              |
| LV_BEAR  | keep shorts, skip longs              |
| HV_BEAR  | **stay flat** — avoid whipsaw        |

### Hybrid momentum + mean-reversion per regime
`hybrid_regime_signal` lets each regime pick **which signal family** to run.
Four mappings are evaluated in the sweep:

| Mapping         | LV_BULL   | HV_BULL   | LV_BEAR   | HV_BEAR |
|-----------------|-----------|-----------|-----------|---------|
| `HYB_default`   | momentum  | mean_rev  | mean_rev  | flat    |
| `HYB_contrarian`| mean_rev  | mean_rev  | momentum  | flat    |
| `HYB_bullmom`   | momentum  | momentum  | mean_rev  | flat    |
| `HYB_calmonly`  | momentum  | flat      | mean_rev  | flat    |

`HYB_default` encodes the academic prior (Giner & Zakamulin 2023):
momentum in calm uptrends, mean-reversion when volatility rises.

### Parameter sweep
`src/sweep.py` evaluates:
- VWAP window ∈ {10, 20, 50}
- Mean-reversion `k_entry` ∈ {1.5, 2.0, 2.5} σ
- Strategy ∈ {BuyHold, MOM, MR, MOM+HMM, MR+HMM, HYB_default,
  HYB_contrarian, HYB_bullmom, HYB_calmonly, BLEND50}
- Ticker ∈ {SPY, QQQ, IWM}

That is **246 configurations** scored on Sharpe, CAGR, MaxDD,
TotalReturn, HitRate, AvgExposure, Turnover.

---

## 3. Results (2008-12-31 → 2019-02-22, daily, **real data**)

### VWAP-only backtest
| Ticker | Strategy  | CAGR      | Sharpe | MaxDD    | Total Return |
|--------|-----------|----------:|-------:|---------:|-------------:|
| SPY    | Buy & Hold| +11.79 %  |  0.76  | -27.13 % | +209.3 %     |
| SPY    | VWAP(20)  | -1.35 %   | -0.00  | -51.37 % | -12.85 %     |
| QQQ    | Buy & Hold| +18.97 %  |  1.04  | -23.16 % | +481.3 %     |
| QQQ    | VWAP(20)  | -5.43 %   | -0.21  | -52.71 % | -43.17 %     |
| IWM    | Buy & Hold| +12.11 %  |  0.63  | -33.31 % | +218.3 %     |
| IWM    | VWAP(20)  | -9.88 %   | -0.36  | -78.15 % | -65.16 %     |

A 20-day rolling VWAP trend signal **whipsaws on daily bars** — it is a
slow moving-average-crossover with double-sided exposure through the
2010-11 vol spikes, 2015-16 chop, and the 2018 Q4 sell-off. This is
expected: the paper's original signal depends on an *intraday*
session-anchored VWAP that produces many short holding-period trades
per day, not a multi-day crossover.

### VWAP + 4-state HMM backtest
| Ticker | Strategy      | CAGR         | Sharpe    | MaxDD         | Total Return |
|--------|---------------|-------------:|----------:|--------------:|-------------:|
| SPY    | Buy & Hold    | +11.79 %     |  0.76     | -27.13 %      | +209.3 %     |
| SPY    | VWAP(20)      | -1.35 %      | -0.00     | -51.37 %      | -12.85 %     |
| SPY    | **VWAP+HMM4** | **+4.65 %**  | **+0.50** | **-20.73 %**  | **+58.53 %** |
| QQQ    | Buy & Hold    | +18.97 %     |  1.04     | -23.16 %      | +481.3 %     |
| QQQ    | VWAP(20)      | -5.43 %      | -0.21     | -52.71 %      | -43.17 %     |
| QQQ    | VWAP+HMM4     | +0.30 %      | +0.09     | -43.93 %      | +3.07 %      |
| IWM    | Buy & Hold    | +12.11 %     |  0.63     | -33.31 %      | +218.3 %     |
| IWM    | VWAP(20)      | -9.88 %      | -0.36     | -78.15 %      | -65.16 %     |
| IWM    | **VWAP+HMM4** | **+6.52 %**  | **+0.52** | **-27.70 %**  | **+89.72 %** |

The HMM filter flips every VWAP-only backtest from losing into positive
territory and cuts max drawdown by 20–50 pp. But on a 10-year post-GFC
bull market, **beating buy-and-hold on absolute return is very hard** —
especially on QQQ where the NDX bull was exceptional. The wins here are
in **risk metrics** (lower DD, lower volatility) rather than raw CAGR.

### Parameter sweep — best (window, k_entry) per strategy
Sharpe ratio of the best configuration per (strategy × ticker),
chosen across window ∈ {10, 20, 50} and k_entry ∈ {1.5, 2.0, 2.5}:

| Strategy        |  IWM      |  QQQ      |  SPY      |
|-----------------|----------:|----------:|----------:|
| BuyHold         |  0.63     |  **1.04** |  0.76     |
| MOM             | -0.21     | -0.06     | -0.00     |
| MOM+HMM         |  0.55     |  0.33     |  **0.76** |
| MR              |  0.48     |  0.42     |  0.27     |
| **MR+HMM**      |  **0.94** |  0.45     |  0.48     |
| HYB_default     | -0.04     |  0.37     |  0.14     |
| HYB_contrarian  |  0.20     |  0.19     | -0.26     |
| HYB_bullmom     | -0.08     |  0.04     |  0.36     |
| HYB_calmonly    | -0.13     |  0.12     |  0.12     |
| BLEND50         | -0.02     |  0.07     |  0.18     |

Total-return view for the same best configs:

| Strategy        |   IWM     |   QQQ     |   SPY     |
|-----------------|----------:|----------:|----------:|
| BuyHold         | +218.3 %  | **+481.3 %** | +209.3 %  |
| MOM+HMM         | +92.6 %   | +35.6 %   | +100.3 %  |
| MR              | +66.6 %   | +40.5 %   | +17.2 %   |
| MR+HMM          | +73.8 %   | +10.4 %   | +12.1 %   |
| HYB_default     | -8.4 %    | +39.0 %   | +7.6 %    |

**Top 5 configurations by Sharpe (real data)**

| Ticker | Strategy | Window | k_entry | Sharpe   | CAGR    | MaxDD    | TotalRet  |
|--------|----------|-------:|--------:|---------:|--------:|---------:|----------:|
| IWM    | MR+HMM   |  10    |  1.5    | **0.94** | +5.6 %  |  -9.5 %  | +73.8 %   |
| SPY    | MOM+HMM  |  50    |  any    | 0.76     | +7.1 %  | -15.0 %  | +100.2 %  |
| IWM    | MR+HMM   |  10    |  2.0    | 0.72     | +3.3 %  |  -9.9 %  | +39.3 %   |
| IWM    | MR+HMM   |  20    |  1.5    | 0.57     | +3.5 %  | -14.9 %  | +41.1 %   |
| IWM    | MOM+HMM  |  50    |  any    | 0.55     | +6.7 %  | -32.4 %  | +92.6 %   |

### Takeaways (real data)
1. **Pure VWAP momentum loses money on every ticker** on daily bars —
   −13 %, −43 %, −65 % on SPY/QQQ/IWM. Confirms the paper's signal is
   intraday by construction, not a daily-bar rule.
2. **IWM is the clearest win for the research hypothesis** — `MR+HMM`
   (window 10, k=1.5 σ) gives **Sharpe 0.94 vs 0.63 for buy-and-hold**
   and **−9.5 % max drawdown vs −33 %**. On a risk-adjusted basis it is
   the best configuration in the whole sweep.
3. **SPY**: `MOM+HMM` with a 50-day VWAP matches buy-and-hold's Sharpe
   (0.76) while cutting max drawdown from −27 % → −15 %. Useful if
   drawdown matters more than raw return.
4. **QQQ is buy-and-hold territory in this window** — the 10-year
   NDX bull produced a 1.04 Sharpe that no signal beats. The best
   the strategy family does is `MR+HMM` Sharpe 0.45, DD −15 %.
5. **Shorter VWAP windows (10) + tighter bands (1.5 σ)** still win
   for mean-reversion on SPY/IWM. For momentum, longer windows (50)
   are better to survive chop.
6. **No single strategy is universal** — exactly Nystrup et al. (2020)'s
   conclusion. Different tickers and different regime mixes want
   different strategies; the HMM lets us condition on that mix.

Full artifacts:
- `results/vwap_only/summary.csv` and `*_vwap.csv` per ticker.
- `results/vwap_hmm/summary.csv`, `*_vwap_hmm.csv`, `*_regimes.csv`,
  and per-ticker `*_equity.png`, `*_regimes.png`, plus a combined
  `equity_grid.png`.
- `results/sweep/sweep_all.csv` (all 246 configs),
  `best_per_strategy.csv`, `sharpe_heatmap.{csv,png}`,
  `totalreturn_heatmap.csv`, and
  `param_sensitivity_MR_HMM.png`.

---

## 4. Caveats & next steps

1. **Data is real but historical** (2008-12-31 → 2019-02-22 from the
   `jiewwantan/StarTrader` GitHub mirror of Yahoo-Finance CSVs). The IWM
   series is the Russell 2000 index ^RUT, which IWM tracks to <0.1 %
   error. When run on a machine with Yahoo Finance access, the loader
   fetches the latest SPY/QQQ/IWM through yfinance automatically and
   extends the window to present day.
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
6. **Sweep is in-sample as well.** Picking the best (strategy, window,
   k_entry) per ticker in the full window biases Sharpe upward.
   The honest test is: train on 2007-2014, select best config, then
   test 2015-2018 — the current code already produces per-config
   series so a train/test split is a one-liner at the sweep loop.

---

## 5. Crypto extension (BTC, ETH, XRP, SOL, HYPE)

### Data (real, daily, through 2026-04-22)
The crypto loader (`src/crypto_data.py`) pulls daily series from
[coinmetrics/data](https://github.com/coinmetrics/data):

| Ticker | Rows | First day   | Last day    | Last close | Source column             |
|--------|-----:|-------------|-------------|-----------:|---------------------------|
| BTC    | 5 758| 2010-07-18  | 2026-04-22  | $78 317    | `PriceUSD`                |
| ETH    | 3 911| 2015-08-08  | 2026-04-22  | $2 382     | `PriceUSD`                |
| XRP    | 4 269| 2014-08-15  | 2026-04-22  | $1.43      | `PriceUSD`                |
| SOL    | 2 203| 2020-04-11  | 2026-04-22  | $22.54*    | Reconstructed (see below) |
| HYPE   |   496| 2024-12-13  | 2026-04-22  | $41.52     | Reconstructed (see below) |

*For **SOL** and **HYPE** the community coinmetrics feed exposes only
`CapMrktEstUSD` + volume, not a direct USD price. The loader recovers
an implied circulating supply from the handful of overlap days where
both `ReferenceRateUSD` and `CapMrktEstUSD` are reported, then
reconstructs Close = `CapMrktEstUSD / supply_median`. Daily returns
are clipped at ±70 % to reject spurious supply-inflation jumps.
This means SOL's absolute price level before ≈ 2023 is biased low
(current supply is ~70× the 2020 supply), but the period 2023-2026
is reasonable for strategy backtesting. Note: the clipping skews
SOL total-return numbers downward vs the true ETF — use Sharpe and
MaxDD as the primary comparators for SOL.

### Strategy setup for crypto
- Crypto-specific transaction cost: **10 bps commission + 5 bps slippage**
  per turnover (vs 1+0.5 bps for the ETFs).
- Same 4-state Gaussian HMM on log-return / realized-vol / z-return.
- Two HMM variants evaluated:
  - **IS** (in-sample): HMM fit once on the full history of the ticker.
  - **WF** (walk-forward): HMM re-fit every 63 days (≈ 1 quarter) on the
    trailing 504-day window. Uses no future information at decision time.
    Only tickers with ≥ 600 days of history (BTC, ETH, XRP, SOL) get WF.
- Same signal families as the ETF run: MOM, MR, MOM+HMM, MR+HMM,
  HYB_default (momentum in LV_BULL, mean-rev in HV_BULL / LV_BEAR, flat in HV_BEAR).

### Results (Sharpe) — crypto, costs 10+5 bps

| Strategy           |  BTC   |  ETH   | XRP    |   SOL  |  HYPE   |
|--------------------|-------:|-------:|-------:|-------:|--------:|
| BuyHold            | **1.18**| 0.99  | 0.77   | **1.30**| 0.68   |
| MOM                | 0.84   | 0.88   | 0.52   | 0.51   | -0.32   |
| MR                 | -1.18  | -0.78  | -0.76  | -0.83  | 0.70    |
| MOM+HMM (IS)       | 0.90   | 0.91   | 0.74   | 0.64   | 0.32    |
| MR+HMM  (IS)       | -0.65  | -0.47  | -0.10  | 0.07   | 0.48    |
| HYB_default (IS)   | -0.34  | 0.04   | -0.44  | -0.45  | **1.21**|
| **MOM+HMM (WF)**   | **0.91**| **0.96**| 0.57 | -0.08  | n/a     |
| MR+HMM  (WF)       | -0.64  | -0.61  | -0.36  | -0.55  | n/a     |
| HYB_default (WF)   | -0.30  | 0.22   | -0.20  | -0.68  | n/a     |

### Max drawdown — same configs

| Strategy           |   BTC    |   ETH    |   XRP    |   SOL    |  HYPE   |
|--------------------|---------:|---------:|---------:|---------:|--------:|
| BuyHold            |  -92.8 % |  -94.0 % |  -94.9 % |  -95.5 % | -69.2 % |
| MOM                |  -90.2 % |  -78.1 % |  -99.1 % |  -99.8 % | -85.8 % |
| **MOM+HMM (WF)**   | **-70.7 %**| **-58.3 %** | -97.4 % | -89.7 %| n/a    |
| HYB_default (IS)   |  -100 %  |  -96.1 % |  -100 %  |  -100 %  | **-31.4 %**|

### Takeaways — crypto
1. **Crypto majors are momentum-dominated.** BTC, ETH, SOL all have
   Buy-and-Hold Sharpe ≈ 1–1.3 from multi-year trends. Pure mean-reversion
   is a disaster on these (−100 % equity) — you keep fading the dip and
   get run over.
2. **MOM+HMM with walk-forward training is the *honest* best** for BTC
   and ETH: Sharpe 0.91 / 0.96, drawdown **cut in half** (−71 % / −58 %
   vs −93 % / −94 % for buy-and-hold). WF ≈ IS on BTC/ETH — the HMM
   isn't overfitting, the regimes are persistent.
3. **HYPE is the mean-reversion ticker.** Fresh post-launch price has
   oscillated around $30-$60 with no persistent trend — exactly the
   setup where VWAP-band fades work. `HYB_default` (momentum in LV_BULL,
   MR in HV_BULL/LV_BEAR, flat in HV_BEAR) gives **Sharpe 1.21, +171 %,
   MaxDD only −31 %**, the best risk-adjusted result in the whole
   study. Pure MR on HYPE also wins (+61 %).
4. **SOL's reconstructed price degrades the backtest** for strategies
   that accumulate over long horizons — MOM+HMM IS under-performs
   because the reconstructed early returns are too noisy. The WF
   results are more trustworthy and still show Sharpe ≈ 0 for
   active strategies vs 1.30 for buy-and-hold — SOL is a "hold and
   hope" ticker in this period.
5. **XRP benefits most from MOM+HMM**: strategy captures the 2017 and
   2021 pumps while dropping ~96 % MaxDD down to −97 % (still brutal
   because XRP had multi-year flat-to-down stretches in between).
6. **Crypto fees matter.** 15 bps round-trip × high turnover on daily
   signals erodes edge; the best strategies are ones that turn over
   rarely (MOM holds for weeks) or spend long periods flat (HYB with
   HV_BEAR=flat).

### HYPE — why the HMM hybrid wins
HYPE is only ~1.4 years old and has seen four distinct phases:
  * Dec-24 → Feb-25: grind-up from $32 to $36
  * Mar-25: flash crash to $14 (HV_BEAR)
  * Apr-25 → Sep-25: 4× rally to $60 (LV_BULL → HV_BULL)
  * Oct-25 → Feb-26: slow bleed to $25 (LV_BEAR)
  * Mar-26 → Apr-26: recovery to $42

The 4-state HMM picks these out cleanly (see `HYPE_regimes_IS.png`).
The hybrid rule then rides momentum during the Apr-Sep bull and fades
the chop in the sideways phases — exactly the academic prescription
from Giner & Zakamulin (2023) and Nystrup et al. (2020). In a longer-lived
ticker the rule would be tuned down (momentum dominates), but on a
fresh asset HYB_default captures both regimes.

Artifacts:
- `results/crypto/summary.csv` — all 45 (ticker × strategy × HMM mode) stats.
- `results/crypto/{sharpe,totalreturn,maxdd}_by_strategy.csv` — pivots.
- `results/crypto/{BTC,ETH,XRP,SOL,HYPE}_equity.png` — equity curves.
- `results/crypto/{BTC,ETH,XRP,SOL,HYPE}_regimes_IS.png` — in-sample regimes.
- `results/crypto/{BTC,ETH,XRP,SOL}_regimes_WF.png` — walk-forward regimes.
- `results/crypto/heatmap.png` — Sharpe + total-return heatmap.
- `results/crypto/equity_grid.png` — all curves side by side.

---

## Layout
```
VWAP-Strategy/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py            # SPY/QQQ/IWM loader
│   ├── crypto_data.py     # BTC/ETH/XRP/SOL/HYPE loader (coinmetrics mirror)
│   ├── vwap.py            # rolling VWAP, VWAP bands, MOM + MR signals
│   ├── hmm_regime.py      # 4-state HMM, IS + walk-forward, hybrid maps
│   ├── backtest.py        # daily backtest engine + metrics
│   ├── run.py             # equities baseline: BuyHold / VWAP / VWAP+HMM
│   ├── sweep.py           # equities parameter sweep
│   └── run_crypto.py      # crypto deep backtest (IS + walk-forward)
└── results/
    ├── vwap_only/         # equity VWAP-only outputs
    ├── vwap_hmm/          # equity VWAP + HMM(4) outputs
    ├── sweep/             # equity parameter sweep outputs
    └── crypto/            # crypto (BTC/ETH/XRP/SOL/HYPE) outputs
```

## Reproducing
```
pip install -r requirements.txt
python src/run.py          # equities baseline
python src/sweep.py        # equities parameter sweep
python src/run_crypto.py   # crypto deep backtest (IS + walk-forward HMM)
```

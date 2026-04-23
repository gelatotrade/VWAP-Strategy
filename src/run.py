"""Run full VWAP vs VWAP+HMM backtest on SPY, QQQ, IWM and write results."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import TICKERS, load_all
from vwap import vwap_signal, rolling_vwap
from hmm_regime import fit_regime, regime_modulated_signal, REGIME_LABELS
from backtest import run_backtest, buy_and_hold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_VWAP = ROOT / "results" / "vwap_only"
OUT_HMM = ROOT / "results" / "vwap_hmm"
OUT_VWAP.mkdir(parents=True, exist_ok=True)
OUT_HMM.mkdir(parents=True, exist_ok=True)

# backtest parameters
START = "2007-01-01"
END = "2018-12-31"           # fallback data ends 2018; yfinance will extend further
VWAP_WINDOW = 20
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 0.5
REGIME_COLORS = {"HV_BEAR": "#8b0000", "LV_BEAR": "#ffa07a",
                 "LV_BULL": "#9acd32", "HV_BULL": "#228b22"}


def _fmt_stats(s: dict) -> str:
    return (f"CAGR={s['CAGR']*100:6.2f}%  "
            f"Sharpe={s['Sharpe']:5.2f}  "
            f"MaxDD={s['MaxDD']*100:6.2f}%  "
            f"Ret={s['TotalReturn']*100:7.2f}%  "
            f"HitRate={s['HitRate']*100:5.2f}%  "
            f"AvgExp={s['AvgExposure']:.2f}")


def _plot_equity(curves: dict[str, pd.Series], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, eq in curves.items():
        ax.plot(eq.index, eq.values, label=label, linewidth=1.3)
    ax.set_title(title)
    ax.set_ylabel("Equity (start = 1.0)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_regime(prices: pd.DataFrame, regimes: pd.Series, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    close = prices["Close"].reindex(regimes.index)
    ax.plot(close.index, close.values, color="black", linewidth=0.8)
    ymin, ymax = close.min(), close.max()
    for label in REGIME_LABELS:
        mask = (regimes == label).values
        if not mask.any():
            continue
        ax.fill_between(regimes.index, ymin, ymax, where=mask,
                        color=REGIME_COLORS[label], alpha=0.25, label=label)
    ax.set_ylabel("Close")
    ax.set_title(out_path.stem.replace("_", " "))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    data = load_all(START, END)
    print(f"Loaded: " + ", ".join(f"{t}={len(d)}" for t, d in data.items()))

    rows_vwap: list[dict] = []
    rows_hmm: list[dict] = []
    equity_curves_by_ticker: dict[str, dict[str, pd.Series]] = {}

    for ticker, df in data.items():
        print(f"\n=== {ticker} ===")

        # ------- baseline buy&hold -------
        bh = buy_and_hold(df, commission_bps=COMMISSION_BPS)

        # ------- VWAP-only -------
        sig = vwap_signal(df, window=VWAP_WINDOW)
        r_vwap = run_backtest(df, sig, commission_bps=COMMISSION_BPS,
                              slippage_bps=SLIPPAGE_BPS)
        print(f"  Buy&Hold  : {_fmt_stats(bh.stats)}")
        print(f"  VWAP (N={VWAP_WINDOW}): {_fmt_stats(r_vwap.stats)}")

        rows_vwap.append(dict(ticker=ticker, strategy="BuyHold", **bh.stats))
        rows_vwap.append(dict(ticker=ticker, strategy=f"VWAP_{VWAP_WINDOW}d",
                              **r_vwap.stats))

        # save per-ticker signal + equity csv
        pd.DataFrame({
            "Close": df["Close"],
            "VWAP": rolling_vwap(df, VWAP_WINDOW),
            "signal": sig,
            "equity": r_vwap.equity,
        }).to_csv(OUT_VWAP / f"{ticker}_vwap.csv")

        # ------- HMM + VWAP -------
        regime = fit_regime(df, n_states=4, seed=42)
        sig_hmm = regime_modulated_signal(sig, regime.states)
        r_hmm = run_backtest(df, sig_hmm, commission_bps=COMMISSION_BPS,
                             slippage_bps=SLIPPAGE_BPS)
        print(f"  VWAP+HMM  : {_fmt_stats(r_hmm.stats)}")

        rows_hmm.append(dict(ticker=ticker, strategy="BuyHold", **bh.stats))
        rows_hmm.append(dict(ticker=ticker, strategy=f"VWAP_{VWAP_WINDOW}d",
                             **r_vwap.stats))
        rows_hmm.append(dict(ticker=ticker, strategy="VWAP+HMM4",
                             **r_hmm.stats))

        # regime breakdown
        rcount = regime.states.value_counts().reindex(REGIME_LABELS).fillna(0).astype(int)
        print(f"  Regime days: " +
              ", ".join(f"{k}={v}" for k, v in rcount.items()))

        regime.states.to_csv(OUT_HMM / f"{ticker}_regimes.csv")
        regime.probs.to_csv(OUT_HMM / f"{ticker}_regime_probs.csv")
        pd.DataFrame({
            "Close": df["Close"].reindex(sig_hmm.index),
            "VWAP": rolling_vwap(df, VWAP_WINDOW).reindex(sig_hmm.index),
            "regime": regime.states,
            "vwap_signal": sig.reindex(sig_hmm.index),
            "hmm_signal": sig_hmm,
            "equity": r_hmm.equity,
        }).to_csv(OUT_HMM / f"{ticker}_vwap_hmm.csv")

        # plots
        equity_curves = {
            "Buy & Hold": bh.equity,
            f"VWAP ({VWAP_WINDOW}d)": r_vwap.equity,
            "VWAP + HMM(4)": r_hmm.equity,
        }
        equity_curves_by_ticker[ticker] = equity_curves
        _plot_equity(equity_curves,
                     f"{ticker} — VWAP strategy variants",
                     OUT_HMM / f"{ticker}_equity.png")
        _plot_regime(df, regime.states, OUT_HMM / f"{ticker}_regimes.png")

    # ------- write summary tables -------
    vwap_df = pd.DataFrame(rows_vwap).drop_duplicates(subset=["ticker", "strategy"])
    hmm_df = pd.DataFrame(rows_hmm).drop_duplicates(subset=["ticker", "strategy"])
    for df_, path in [(vwap_df, OUT_VWAP / "summary.csv"),
                      (hmm_df, OUT_HMM / "summary.csv")]:
        df_.to_csv(path, index=False)

    print("\n================ VWAP-only summary =================")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(vwap_df.to_string(index=False))

    print("\n================ VWAP+HMM(4) summary ================")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(hmm_df.to_string(index=False))

    # combined cross-ticker equity plot (normalized)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, (t, curves) in zip(axes, equity_curves_by_ticker.items()):
        for label, eq in curves.items():
            ax.plot(eq.index, eq.values, label=label, linewidth=1.2)
        ax.set_title(t)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Equity (log scale): Buy&Hold vs VWAP vs VWAP+HMM")
    fig.tight_layout()
    fig.savefig(OUT_HMM / "equity_grid.png", dpi=120)
    plt.close(fig)
    print(f"\nWrote results under {ROOT/'results'}")


if __name__ == "__main__":
    main()

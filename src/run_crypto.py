"""Deep VWAP + HMM backtest on BTC, ETH, XRP, SOL, HYPE.

Differences vs the equity runner:
  * Crypto-specific costs: 10 bps commission + 5 bps slippage per turn.
  * Both in-sample HMM (fit once on full history) and walk-forward HMM
    (retrain every quarter on the trailing 2-year window) are evaluated.
  * All 7 strategy variants from the sweep are scored per ticker:
      BuyHold, MOM, MR, MOM+HMM, MR+HMM, HYB_default, HYB_calmonly.
  * Outputs equity curves, regime shading, and per-ticker summary CSVs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_data import CRYPTO_TICKERS, load_all_crypto
from vwap import vwap_momentum_signal, vwap_mean_reversion_signal, rolling_vwap
from hmm_regime import (fit_regime, walk_forward_regimes,
                        regime_modulated_signal, hybrid_regime_signal,
                        DEFAULT_HYBRID_MAP, REGIME_LABELS)
from backtest import run_backtest, buy_and_hold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "crypto"
OUT.mkdir(parents=True, exist_ok=True)

# Crypto-specific costs (10 bps taker fee + 5 bps slippage on both sides).
COMM_BPS = 10.0
SLIP_BPS = 5.0
VWAP_WINDOW = 20
MR_K_ENTRY = 1.5
MR_K_EXIT = 0.5

REGIME_COLORS = {"HV_BEAR": "#8b0000", "LV_BEAR": "#ffa07a",
                 "LV_BULL": "#9acd32", "HV_BULL": "#228b22"}


def _bt(df, sig):
    return run_backtest(df, sig, commission_bps=COMM_BPS, slippage_bps=SLIP_BPS)


def _fmt(s: dict) -> str:
    return (f"CAGR={s['CAGR']*100:7.2f}%  "
            f"Sharpe={s['Sharpe']:5.2f}  "
            f"MaxDD={s['MaxDD']*100:7.2f}%  "
            f"Ret={s['TotalReturn']*100:8.1f}%  "
            f"AvgExp={s['AvgExposure']:.2f}")


def _plot_regime(prices: pd.DataFrame, regimes: pd.Series, ticker: str,
                 path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    close = prices["Close"].reindex(regimes.index).dropna()
    ax.plot(close.index, close.values, color="black", linewidth=0.8)
    ax.set_yscale("log")
    ymin, ymax = close.min(), close.max()
    for label in REGIME_LABELS:
        mask = (regimes == label)
        if not mask.any():
            continue
        mask = mask.reindex(close.index).fillna(False)
        ax.fill_between(close.index, ymin, ymax, where=mask.values,
                        color=REGIME_COLORS[label], alpha=0.25, label=label)
    ax.set_title(title)
    ax.set_ylabel("Close (USD, log scale)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_equity(curves: dict[str, pd.Series], title: str, path: Path):
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
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_ticker(ticker: str, df: pd.DataFrame) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    curves: dict[str, pd.Series] = {}

    bh = buy_and_hold(df, commission_bps=COMM_BPS)
    rows.append({"ticker": ticker, "strategy": "BuyHold",
                 "hmm_mode": "-", **bh.stats})
    curves["Buy & Hold"] = bh.equity

    sig_mom = vwap_momentum_signal(df, window=VWAP_WINDOW)
    sig_mr = vwap_mean_reversion_signal(df, window=VWAP_WINDOW,
                                        k_entry=MR_K_ENTRY, k_exit=MR_K_EXIT)

    r_mom = _bt(df, sig_mom)
    r_mr = _bt(df, sig_mr)
    rows.append({"ticker": ticker, "strategy": "MOM",
                 "hmm_mode": "-", **r_mom.stats})
    rows.append({"ticker": ticker, "strategy": "MR",
                 "hmm_mode": "-", **r_mr.stats})
    curves["MOM (VWAP trend)"] = r_mom.equity
    curves["MR (VWAP ±σ fade)"] = r_mr.equity

    # ----- in-sample HMM -----
    reg_is = fit_regime(df, n_states=4, seed=42)
    for mode_name, regimes in [("IS", reg_is.states)]:
        mom_h = regime_modulated_signal(sig_mom, regimes)
        mr_h = regime_modulated_signal(sig_mr, regimes)
        hyb = hybrid_regime_signal(sig_mom, sig_mr, regimes,
                                   DEFAULT_HYBRID_MAP)
        rows += [
            {"ticker": ticker, "strategy": f"MOM+HMM",
             "hmm_mode": mode_name, **_bt(df, mom_h).stats},
            {"ticker": ticker, "strategy": f"MR+HMM",
             "hmm_mode": mode_name, **_bt(df, mr_h).stats},
            {"ticker": ticker, "strategy": f"HYB_default",
             "hmm_mode": mode_name, **_bt(df, hyb).stats},
        ]
        curves["MR+HMM (IS)"] = _bt(df, mr_h).equity
        curves["HYB_default (IS)"] = _bt(df, hyb).equity

    # ----- walk-forward HMM (only if enough history) -----
    if len(df) > 600:
        reg_wf = walk_forward_regimes(df, train_days=504, step_days=63,
                                      n_states=4, seed=42)
        mom_wf = regime_modulated_signal(sig_mom, reg_wf.states)
        mr_wf = regime_modulated_signal(sig_mr, reg_wf.states)
        hyb_wf = hybrid_regime_signal(sig_mom, sig_mr, reg_wf.states,
                                      DEFAULT_HYBRID_MAP)
        rows += [
            {"ticker": ticker, "strategy": "MOM+HMM",
             "hmm_mode": "WF", **_bt(df, mom_wf).stats},
            {"ticker": ticker, "strategy": "MR+HMM",
             "hmm_mode": "WF", **_bt(df, mr_wf).stats},
            {"ticker": ticker, "strategy": "HYB_default",
             "hmm_mode": "WF", **_bt(df, hyb_wf).stats},
        ]
        curves["MR+HMM (walk-fwd)"] = _bt(df, mr_wf).equity
        curves["HYB_default (walk-fwd)"] = _bt(df, hyb_wf).equity

        # Regime plot — use walk-forward regimes (out of sample)
        _plot_regime(df, reg_wf.states, ticker,
                     OUT / f"{ticker}_regimes_WF.png",
                     f"{ticker} — walk-forward HMM regimes")
    # Always plot in-sample regime for visual comparison
    _plot_regime(df, reg_is.states, ticker,
                 OUT / f"{ticker}_regimes_IS.png",
                 f"{ticker} — in-sample HMM regimes")

    _plot_equity(curves, f"{ticker} — strategy equity curves",
                 OUT / f"{ticker}_equity.png")

    # save trace CSV
    trace = pd.DataFrame({"Close": df["Close"],
                          "VWAP": rolling_vwap(df, VWAP_WINDOW),
                          "signal_MOM": sig_mom,
                          "signal_MR": sig_mr,
                          "regime_IS": reg_is.states})
    if len(df) > 600:
        trace["regime_WF"] = reg_wf.states
    trace.to_csv(OUT / f"{ticker}_trace.csv")

    return rows, curves


def main() -> None:
    data = load_all_crypto()
    print("Loaded:")
    for t, d in data.items():
        print(f"  {t:<5}: {len(d):5d} rows  "
              f"{d.index.min().date()} -> {d.index.max().date()}  "
              f"last_close=${d['Close'].iloc[-1]:,.2f}")
    print()

    all_rows: list[dict] = []
    all_curves: dict[str, dict[str, pd.Series]] = {}

    for ticker, df in data.items():
        print(f"=== {ticker} ===")
        rows, curves = run_ticker(ticker, df)
        all_rows.extend(rows)
        all_curves[ticker] = curves
        # terse per-strategy summary
        for r in rows:
            label = f"{r['strategy']:<12}({r['hmm_mode']}):"
            print(f"  {label:<20} {_fmt(r)}")
        print()

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    # Compact pivot: Sharpe by (strategy, hmm_mode) rows and ticker cols
    summary["strat_full"] = summary["strategy"] + "(" + summary["hmm_mode"] + ")"
    sharpe_pv = summary.pivot_table(index="strat_full", columns="ticker",
                                    values="Sharpe", aggfunc="first")
    ret_pv = summary.pivot_table(index="strat_full", columns="ticker",
                                 values="TotalReturn", aggfunc="first")
    dd_pv = summary.pivot_table(index="strat_full", columns="ticker",
                                values="MaxDD", aggfunc="first")
    order = ["BuyHold(-)", "MOM(-)", "MR(-)",
             "MOM+HMM(IS)", "MR+HMM(IS)", "HYB_default(IS)",
             "MOM+HMM(WF)", "MR+HMM(WF)", "HYB_default(WF)"]
    sharpe_pv = sharpe_pv.reindex([r for r in order if r in sharpe_pv.index])
    ret_pv = ret_pv.reindex([r for r in order if r in ret_pv.index])
    dd_pv = dd_pv.reindex([r for r in order if r in dd_pv.index])
    sharpe_pv.to_csv(OUT / "sharpe_by_strategy.csv")
    ret_pv.to_csv(OUT / "totalreturn_by_strategy.csv")
    dd_pv.to_csv(OUT / "maxdd_by_strategy.csv")

    print("=============== Sharpe (strategy × ticker) ===============")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(sharpe_pv.to_string())
    print("\n=============== Total Return (strategy × ticker) ========")
    with pd.option_context("display.float_format", lambda x: f"{x:.2%}"):
        print(ret_pv.to_string())
    print("\n=============== Max Drawdown (strategy × ticker) ========")
    with pd.option_context("display.float_format", lambda x: f"{x:.2%}"):
        print(dd_pv.to_string())

    # heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (df_, title, cmap, vmin, vmax) in zip(axes, [
            (sharpe_pv, "Sharpe", "RdYlGn", -1, 1.5),
            (ret_pv,    "Total Return", "RdYlGn", -1, 5.0)]):
        im = ax.imshow(df_.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(df_.columns)), df_.columns)
        ax.set_yticks(range(len(df_.index)), df_.index, fontsize=8)
        for i in range(df_.shape[0]):
            for j in range(df_.shape[1]):
                v = df_.values[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.2f}" if title == "Sharpe" else f"{v*100:.0f}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Crypto VWAP/HMM backtest "
                 f"(costs {COMM_BPS:.0f}+{SLIP_BPS:.0f} bps)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.png", dpi=120)
    plt.close(fig)

    # cross-ticker equity grid
    n = len(all_curves)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    for ax, (t, curves) in zip(axes, all_curves.items()):
        for label, eq in curves.items():
            ax.plot(eq.index, eq.values, label=label, linewidth=1.0)
        ax.set_title(t)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Equity (log scale): Buy&Hold vs VWAP variants")
    fig.tight_layout()
    fig.savefig(OUT / "equity_grid.png", dpi=120)
    plt.close(fig)
    print(f"\nArtifacts: {OUT}")


if __name__ == "__main__":
    main()

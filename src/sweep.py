"""Parameter sweep over VWAP strategies.

Sweeps:
  * VWAP window              [10, 20, 50]
  * Mean-reversion k_entry   [1.5, 2.0, 2.5]
  * Strategy family          MOM | MR | MOM+HMM | MR+HMM | HYBRID_default
                             | HYBRID_contrarian | BLEND50
  * Tickers                  SPY, QQQ, IWM

Writes results/sweep/summary.csv plus a heatmap of Sharpe by
(strategy, ticker) for the best window/k per family, and a ranking
of the top-N configurations.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import TICKERS, load_all
from vwap import (vwap_momentum_signal, vwap_mean_reversion_signal,
                  rolling_vwap)
from hmm_regime import (fit_regime, regime_modulated_signal,
                        hybrid_regime_signal, DEFAULT_HYBRID_MAP)
from backtest import run_backtest, buy_and_hold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sweep"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "2007-01-01", "2018-12-31"
COMM, SLIP = 1.0, 0.5

WINDOWS = [10, 20, 50]
K_ENTRIES = [1.5, 2.0, 2.5]
K_EXIT = 0.5

# alternative hybrid mappings to try
HYBRID_MAPS = {
    "HYB_default":    DEFAULT_HYBRID_MAP,
    "HYB_contrarian": {"LV_BULL": "mean_rev", "HV_BULL": "mean_rev",
                       "LV_BEAR": "momentum", "HV_BEAR": "flat"},
    "HYB_bullmom":    {"LV_BULL": "momentum", "HV_BULL": "momentum",
                       "LV_BEAR": "mean_rev", "HV_BEAR": "flat"},
    "HYB_calmonly":   {"LV_BULL": "momentum", "HV_BULL": "flat",
                       "LV_BEAR": "mean_rev", "HV_BEAR": "flat"},
}


def _bt(df, sig):
    return run_backtest(df, sig, commission_bps=COMM, slippage_bps=SLIP)


def run_sweep() -> pd.DataFrame:
    data = load_all(START, END)
    rows: list[dict] = []

    for ticker, df in data.items():
        bh = buy_and_hold(df, commission_bps=COMM)
        rows.append(dict(ticker=ticker, strategy="BuyHold",
                         window=None, k_entry=None, **bh.stats))

        # precompute HMM once per ticker
        regime = fit_regime(df, n_states=4, seed=42)

        for w, k in itertools.product(WINDOWS, K_ENTRIES):
            mom = vwap_momentum_signal(df, window=w)
            mr = vwap_mean_reversion_signal(df, window=w,
                                            k_entry=k, k_exit=K_EXIT)
            # ---- base pure strategies ----
            rows.append(dict(ticker=ticker, strategy="MOM",
                             window=w, k_entry=k, **_bt(df, mom).stats))
            rows.append(dict(ticker=ticker, strategy="MR",
                             window=w, k_entry=k, **_bt(df, mr).stats))
            # ---- HMM-modulated one-sided ----
            mom_hmm = regime_modulated_signal(mom, regime.states)
            mr_hmm = regime_modulated_signal(mr, regime.states)
            rows.append(dict(ticker=ticker, strategy="MOM+HMM",
                             window=w, k_entry=k, **_bt(df, mom_hmm).stats))
            rows.append(dict(ticker=ticker, strategy="MR+HMM",
                             window=w, k_entry=k, **_bt(df, mr_hmm).stats))
            # ---- hybrid regime-switched combos ----
            for name, mapping in HYBRID_MAPS.items():
                hyb = hybrid_regime_signal(mom, mr, regime.states, mapping)
                rows.append(dict(ticker=ticker, strategy=name,
                                 window=w, k_entry=k, **_bt(df, hyb).stats))
            # ---- simple 50/50 blend of momentum + mean-reversion ----
            blend = 0.5 * (mom + mr)
            rows.append(dict(ticker=ticker, strategy="BLEND50",
                             window=w, k_entry=k, **_bt(df, blend).stats))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sweep_all.csv", index=False)
    return out


def summarize(df: pd.DataFrame) -> None:
    # best-per-strategy (highest Sharpe across windows/k) per ticker
    best = (
        df[df["strategy"] != "BuyHold"]
        .sort_values("Sharpe", ascending=False)
        .groupby(["ticker", "strategy"], as_index=False)
        .first()
    )
    bh = df[df["strategy"] == "BuyHold"]
    combined = pd.concat([bh, best], ignore_index=True)
    combined = combined.sort_values(["ticker", "Sharpe"], ascending=[True, False])
    combined.to_csv(OUT / "best_per_strategy.csv", index=False)

    # pivot of Sharpe
    pv = combined.pivot_table(index="strategy", columns="ticker",
                              values="Sharpe", aggfunc="first")
    # fix row order
    order = ["BuyHold", "MOM", "MOM+HMM", "MR", "MR+HMM",
             "HYB_default", "HYB_contrarian", "HYB_bullmom",
             "HYB_calmonly", "BLEND50"]
    pv = pv.reindex([r for r in order if r in pv.index])
    pv.to_csv(OUT / "sharpe_heatmap.csv")

    pv_ret = combined.pivot_table(index="strategy", columns="ticker",
                                  values="TotalReturn", aggfunc="first")
    pv_ret = pv_ret.reindex([r for r in order if r in pv_ret.index])
    pv_ret.to_csv(OUT / "totalreturn_heatmap.csv")

    print("\n================ Best-per-strategy summary (Sharpe) ================")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(pv.to_string())

    print("\n================ Best-per-strategy summary (Total Return) ==========")
    with pd.option_context("display.float_format", lambda x: f"{x:.2%}"):
        print(pv_ret.to_string())

    print("\n================ Top 10 configurations by Sharpe ===================")
    top = (df[df["strategy"] != "BuyHold"]
           .sort_values("Sharpe", ascending=False)
           .head(15)[["ticker", "strategy", "window", "k_entry",
                      "Sharpe", "CAGR", "MaxDD", "TotalReturn"]])
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(top.to_string(index=False))

    # ---- heatmap plot (Sharpe) ----
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(pv))))
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(pv.columns)), pv.columns)
    ax.set_yticks(range(len(pv.index)), pv.index)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v = pv.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("Sharpe — best (window, k) per strategy")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_heatmap.png", dpi=120)
    plt.close(fig)

    # ---- parameter-sensitivity heatmap for MR+HMM on each ticker ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, t in zip(axes, TICKERS):
        sub = df[(df["ticker"] == t) & (df["strategy"] == "MR+HMM")]
        pv2 = sub.pivot(index="window", columns="k_entry", values="Sharpe")
        im = ax.imshow(pv2.values, aspect="auto", cmap="RdYlGn",
                       vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(pv2.columns)), [f"{v:.1f}" for v in pv2.columns])
        ax.set_yticks(range(len(pv2.index)), pv2.index)
        ax.set_xlabel("k_entry (σ)")
        ax.set_ylabel("VWAP window")
        ax.set_title(f"{t} — MR+HMM Sharpe")
        for i in range(pv2.shape[0]):
            for j in range(pv2.shape[1]):
                ax.text(j, i, f"{pv2.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, label="Sharpe", shrink=0.8)
    fig.savefig(OUT / "param_sensitivity_MR_HMM.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("Running sweep …")
    df = run_sweep()
    print(f"  {len(df)} rows written to {OUT/'sweep_all.csv'}")
    summarize(df)
    print(f"\nArtifacts in {OUT}")


if __name__ == "__main__":
    main()

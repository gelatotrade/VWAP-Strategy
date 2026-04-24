"""Hidden Markov regime detection with 4 states.

Features used for state inference:
    f1 = log return (ewma 5d)
    f2 = realized volatility (rolling std 20d of log returns, annualized)
    f3 = return z-score (20d rolling z)

After fitting a 4-state GaussianHMM we LABEL states by (mean_return, mean_vol):
    HIGHVOL_BEAR  - lowest mean return
    LOWVOL_BEAR
    LOWVOL_BULL
    HIGHVOL_BULL  - highest mean return or highest vol
This gives a stable ordering regardless of the HMM's random init.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


REGIME_LABELS = ("HV_BEAR", "LV_BEAR", "LV_BULL", "HV_BULL")


@dataclass
class RegimeResult:
    states: pd.Series            # {0..3} labelled per REGIME_LABELS
    probs: pd.DataFrame          # 4-col posterior probs
    model: GaussianHMM
    label_map: dict[int, str]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))
    f1 = log_ret.ewm(span=5, adjust=False).mean()
    f2 = log_ret.rolling(20).std() * np.sqrt(252)
    mu = log_ret.rolling(20).mean()
    sd = log_ret.rolling(20).std().replace(0, np.nan)
    f3 = (log_ret - mu) / sd
    X = pd.concat([f1, f2, f3], axis=1, keys=["ret_ew", "vol20", "zret"]).dropna()
    return X


def _label_states(model: GaussianHMM) -> dict[int, str]:
    """Assign the 4 labels HV_BEAR..HV_BULL based on means/covariances."""
    means = model.means_                  # shape (4, n_features)
    var_return = means[:, 0]              # ewma return
    var_vol = means[:, 1]                 # realized vol
    # rank by return (ascending -> bearish to bullish)
    order_ret = np.argsort(var_return)
    # Among the two most negative: higher vol -> HV_BEAR, other -> LV_BEAR
    # Among the two most positive: higher vol -> HV_BULL, other -> LV_BULL
    bear1, bear2 = order_ret[0], order_ret[1]
    bull1, bull2 = order_ret[2], order_ret[3]
    hv_bear = bear1 if var_vol[bear1] >= var_vol[bear2] else bear2
    lv_bear = bear2 if hv_bear == bear1 else bear1
    hv_bull = bull2 if var_vol[bull2] >= var_vol[bull1] else bull1
    lv_bull = bull1 if hv_bull == bull2 else bull2
    return {int(hv_bear): "HV_BEAR", int(lv_bear): "LV_BEAR",
            int(lv_bull): "LV_BULL", int(hv_bull): "HV_BULL"}


def fit_regime(df: pd.DataFrame,
               n_states: int = 4,
               seed: int = 42,
               n_iter: int = 200) -> RegimeResult:
    X_df = _features(df)
    X = X_df.values

    # standardize features for numerical stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    best_model, best_ll = None, -np.inf
    for s in (seed, seed + 1, seed + 7):
        m = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=n_iter, random_state=s, tol=1e-4)
        try:
            m.fit(Xs)
            ll = m.score(Xs)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError("HMM failed to fit")

    # transform model.means_ back to original scale for labeling
    means_orig = best_model.means_ * sd + mu
    # cheap trick: overwrite a temp copy for labeling
    label_model = type("Dummy", (), {})()
    label_model.means_ = means_orig
    label_map = _label_states(label_model)

    states_int = best_model.predict(Xs)
    probs = best_model.predict_proba(Xs)

    states = pd.Series([label_map[s] for s in states_int],
                       index=X_df.index, name="regime")
    probs_df = pd.DataFrame(probs, index=X_df.index,
                            columns=[label_map[i] for i in range(n_states)])
    return RegimeResult(states=states, probs=probs_df,
                        model=best_model, label_map=label_map)


def regime_modulated_signal(vwap_sig: pd.Series,
                            regimes: pd.Series) -> pd.Series:
    """Combine VWAP trend signal with 4 regimes.

    Rule-set (inspired by common regime-switching practice):
      LV_BULL: take VWAP signal fully (trend-friendly, low vol)
      HV_BULL: take long side only, skip shorts (bull but choppy)
      LV_BEAR: take short side only, skip longs
      HV_BEAR: stay FLAT (too noisy, high risk of whipsaw)
    """
    out = vwap_sig.reindex(regimes.index).fillna(0.0).astype(float)
    regs = regimes.reindex(out.index)
    # HV_BULL: only longs
    mask = (regs == "HV_BULL") & (out < 0)
    out[mask] = 0.0
    # LV_BEAR: only shorts
    mask = (regs == "LV_BEAR") & (out > 0)
    out[mask] = 0.0
    # HV_BEAR: flat
    out[regs == "HV_BEAR"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Hybrid momentum + mean-reversion, switched per regime
# ---------------------------------------------------------------------------
# Default regime -> strategy map, grounded in Giner & Zakamulin (2023) and
# the practitioner rule "momentum in calm trends, mean-rev in chop".
DEFAULT_HYBRID_MAP = {
    "LV_BULL": "momentum",      # calm uptrend — ride it
    "HV_BULL": "mean_rev",      # bullish but choppy — fade extremes
    "LV_BEAR": "mean_rev",      # bearish drift, bounces to fade
    "HV_BEAR": "flat",          # crisis chop — step aside
}


def walk_forward_regimes(df: pd.DataFrame,
                         train_days: int = 504,
                         step_days: int = 63,
                         n_states: int = 4,
                         seed: int = 42) -> RegimeResult:
    """Walk-forward HMM inference.

    At each step the HMM is re-fit on the last `train_days` sessions and
    used to decode regimes for the next `step_days`. Avoids using future
    information to label past regimes. `train_days` default = 2 years;
    `step_days` default = 1 quarter.

    Returns a RegimeResult where `states` is only populated from
    sample `train_days` onwards.
    """
    X_all = _features(df)
    if len(X_all) <= train_days + 1:
        # not enough data for walk-forward — fall back to in-sample fit
        return fit_regime(df, n_states=n_states, seed=seed)

    mu = X_all.values.mean(axis=0)
    sd = X_all.values.std(axis=0)
    sd[sd == 0] = 1.0

    states_out = pd.Series(index=X_all.index, dtype="object")
    probs_out = pd.DataFrame(index=X_all.index,
                             columns=list(REGIME_LABELS), dtype=float)
    last_model: GaussianHMM | None = None
    last_label_map: dict[int, str] = {}

    start = train_days
    while start < len(X_all):
        end = min(start + step_days, len(X_all))
        train_X = X_all.iloc[start - train_days:start].values
        train_Xs = (train_X - mu) / sd
        best_m, best_ll = None, -np.inf
        for s in (seed, seed + 1, seed + 7):
            m = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=150, random_state=s, tol=1e-4)
            try:
                m.fit(train_Xs)
                ll = m.score(train_Xs)
                if ll > best_ll:
                    best_ll, best_m = ll, m
            except Exception:
                continue
        if best_m is None:
            best_m = last_model  # carry forward
        # label states using means on original scale
        means_orig = best_m.means_ * sd + mu
        lbl = _label_states(type("D", (), {"means_": means_orig})())
        test_X = X_all.iloc[start:end].values
        test_Xs = (test_X - mu) / sd
        s_int = best_m.predict(test_Xs)
        p = best_m.predict_proba(test_Xs)
        idx = X_all.index[start:end]
        states_out.loc[idx] = [lbl[s] for s in s_int]
        for j, name in enumerate(REGIME_LABELS):
            if name in lbl.values():
                col = [k for k, v in lbl.items() if v == name][0]
                probs_out.loc[idx, name] = p[:, col]
            else:
                probs_out.loc[idx, name] = 0.0
        last_model, last_label_map = best_m, lbl
        start = end

    return RegimeResult(states=states_out.dropna(),
                        probs=probs_out.loc[states_out.dropna().index],
                        model=last_model, label_map=last_label_map)


def hybrid_regime_signal(momentum_sig: pd.Series,
                         mean_rev_sig: pd.Series,
                         regimes: pd.Series,
                         mapping: dict[str, str] | None = None) -> pd.Series:
    """Select momentum or mean-reversion signal per regime.

    `mapping` values must be one of: 'momentum', 'mean_rev', 'both', 'flat'.
      - 'momentum' : use momentum_sig
      - 'mean_rev' : use mean_rev_sig
      - 'both'     : average of the two (implicit combination)
      - 'flat'     : 0
    """
    mapping = mapping or DEFAULT_HYBRID_MAP
    idx = regimes.index
    mom = momentum_sig.reindex(idx).fillna(0.0)
    mr = mean_rev_sig.reindex(idx).fillna(0.0)

    out = pd.Series(0.0, index=idx)
    for regime, action in mapping.items():
        mask = (regimes == regime)
        if not mask.any():
            continue
        if action == "momentum":
            out[mask] = mom[mask]
        elif action == "mean_rev":
            out[mask] = mr[mask]
        elif action == "both":
            out[mask] = 0.5 * (mom[mask] + mr[mask])
        elif action == "flat":
            out[mask] = 0.0
        else:
            raise ValueError(f"Unknown hybrid action: {action}")
    return out.clip(-1, 1)

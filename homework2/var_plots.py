"""Homework 2, Point 1 (comment) - evolution of the VaR over the backtesting sample for the market index, the EWP and
Responsible: member B (reviewer: C).
one stock (PLOT_ASSET), and analysis of the violations (local vs global shocks).
Outputs:
  figures/F1.1_var_<series>_<alpha>.png   returns, VaR of three groups of models, violations, main market events
  tables/var_dynamics/V1_dynamics_<series>.csv
        per model: mean and std of VaR, largest one-day change (spikes), share of days with unchanged VaR
        (staleness of HS/EVT), correlation of |VaR| with the realised 21-day volatility at lags 0..20, and the lag
        with the highest correlation (how many days the VaR needs to catch up with volatility)
  tables/var_dynamics/V2_violation_dates_<model>_<alpha>.csv
        days ranked by number of stocks violating the VaR at the same time; GLOBAL if the index (or EWP) also
        violates or at least GLOBAL_MIN stocks do, LOCAL if only 1-2 stocks violate and the index does not
  tables/var_dynamics/V3_local_global_<alpha>.csv  for each stock and model: violations split in local/global
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                                        # save figures without opening windows
import matplotlib.pyplot as plt
from settings_hw2 import *
from common import load_returns, load_var, ensure_dirs

warnings.filterwarnings("ignore")
PLOT_ASSET = "UBS"        # stock commented in the report (Credit Suisse crisis, March 2023)
GLOBAL_MIN = 8            # at least 8 of 16 stocks violating on the same day -> global shock
TAB = os.path.join(TABLE_DIR, "var_dynamics")
GROUPS = {"Unconditional (rolling / expanding)": ["M01_HS", "M03_T", "M08_GPD"],   # one panel per group of models
          "Conditional (GARCH, EWMA)": ["M09_AG_N", "M11_AG_HS", "M20_EWMA"],
          "Factor models": ["M17_DCC_N", "M19_DCC_FHS", "M21_X_SKT"]}
# Main market events of the backtesting sample: dates to be checked by the group before using them in the report
EVENTS = {"2021-11-26": "Omicron variant", "2022-02-24": "Russia invades Ukraine",
          "2022-06-13": "US inflation / Fed sell-off", "2023-03-15": "Credit Suisse crisis",
          "2023-03-20": "UBS-CS takeover", "2024-08-02": "Global sell-off (US jobs, yen carry)",
          "2024-08-05": "Yen carry-trade unwind", "2025-04-03": "US tariffs announced",
          "2025-04-04": "US tariff sell-off", "2025-04-07": "US tariff sell-off"}
REF_MODELS = ["M01_HS", "M11_AG_HS"]   # reference models for the local/global analysis (unconditional vs conditional)


def plot_series(name, d, a):
    """FIGURE F1.1: returns, VaR of each group of models, violations (dots) and market events (dotted lines)."""
    groups = {g: [m for m in ms if f"{m}_{a}" in d] for g, ms in GROUPS.items()}
    groups = {g: ms for g, ms in groups.items() if ms}       # the index has no factor models -> 2 panels
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 3.2 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (g, ms) in zip(axes, groups.items()):
        ax.bar(d.index, d.ret, width=1.5, color="0.65", label="return")   # daily returns
        for m in ms:
            v = d[f"{m}_{a}"]
            line, = ax.plot(d.index, v, lw=1, label=MODELS.get(m, m))     # VaR path
            hit = d.ret < v
            ax.scatter(d.index[hit], d.ret[hit], s=10, zorder=3, color=line.get_color())   # violations, same colour
        for t, lab in EVENTS.items():
            ax.axvline(pd.Timestamp(t), color="k", ls=":", lw=0.7)        # event dates
        ax.set_title(g, fontsize=10); ax.legend(fontsize=7, loc="lower left", ncol=4); ax.set_ylabel("%")
    for t, lab in EVENTS.items():
        axes[0].annotate(lab, (pd.Timestamp(t), axes[0].get_ylim()[1]), fontsize=6, rotation=90, va="top", ha="right")
    fig.suptitle(f"Fig F1.1 - {name}: daily returns and VaR {int(a * 100)}% (dots = violations)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"F1.1_var_{name}_{a}.png"), dpi=150); plt.close(fig)


def dynamics(name, d, r_full):
    """TABLE V1: spikes, staleness and reaction to volatility of each VaR series."""
    vol = r_full[name].rolling(21).std()                        # realised volatility over the last month
    rows = {}
    for c in [c for c in d.columns if c.startswith("M")]:
        v = d[c]
        dv = v.diff().abs()                                     # one-day changes of the VaR
        corr = {k: np.corrcoef(v.abs(), vol.shift(k).reindex(d.index))[0, 1] for k in range(0, 21)}   # corr with volatility k days earlier
        rows[c] = {"mean_VaR": v.mean(), "std_VaR": v.std(), "max_abs_change": dv.max(),
                   "date_max_change": dv.idxmax().date(), "share_unchanged_days": (dv < 1e-10).mean(),
                   "corr_vol_lag0": corr[0], "best_lag": max(corr, key=corr.get), "corr_vol_best": max(corr.values())}
    pd.DataFrame(rows).T.to_csv(os.path.join(TAB, f"V1_dynamics_{name}.csv"))


def local_global(series):
    """TABLES V2 and V3: are the violations caused by global (market-wide) or local (stock-specific) shocks?"""
    stocks = [s for s in RIC_TO_NAME.values() if s in series]
    D = {s: load_var(s) for s in stocks + [x for x in (EWP, INDEX_NAME) if x in series]}
    for a in ALPHAS:
        split = {}
        for m in REF_MODELS:
            H = pd.DataFrame({s: D[s].ret < D[s][f"{m}_{a}"] for s in stocks})   # hits of every stock (days x stocks)
            n = H.sum(axis=1)                                   # number of stocks violating on each day
            idx = D[INDEX_NAME].ret < D[INDEX_NAME][f"{m}_{a}"] if INDEX_NAME in D else pd.Series(False, H.index)   # index hit
            ewp = D[EWP].ret < D[EWP][f"{m}_{a}"]               # EWP hit
            kind = np.where(idx.reindex(H.index, fill_value=False) | ewp | (n >= GLOBAL_MIN), "GLOBAL",
                            np.where(n <= 2, "LOCAL", "SECTOR/MIXED"))      # classification of the day
            tab = pd.DataFrame({"n_stocks_violating": n, f"{INDEX_NAME}_violates": idx.reindex(H.index, fill_value=False),
                                "EWP_violates": ewp, "type": kind,
                                "stocks": H.apply(lambda x: ", ".join(x.index[x]), axis=1),
                                "event": [EVENTS.get(str(t.date()), "") for t in H.index]})
            tab = tab[tab.n_stocks_violating > 0].sort_values("n_stocks_violating", ascending=False)
            tab.to_csv(os.path.join(TAB, f"V2_violation_dates_{m}_{a}.csv"), index_label="Date")   # TABLE V2
            K = pd.Series(kind, H.index)
            split[m] = pd.DataFrame({t: H[K == t].sum() for t in ("LOCAL", "SECTOR/MIXED", "GLOBAL")})   # hits per stock and type
        pd.concat(split, axis=1).to_csv(os.path.join(TAB, f"V3_local_global_{a}.csv"))                 # TABLE V3


if __name__ == "__main__":
    ensure_dirs(); os.makedirs(TAB, exist_ok=True)
    r = load_returns()
    series = [c for c in r.columns if os.path.exists(os.path.join(VAR_DIR, "iid", f"{c}.csv"))]   # series with forecasts
    for name in [s for s in (INDEX_NAME, EWP, PLOT_ASSET) if s in series]:   # index, portfolio, one stock
        d = load_var(name, r)
        for a in ALPHAS:
            plot_series(name, d, a)                             # FIGURE F1.1
        dynamics(name, d, r)                                    # TABLE V1
        print(f"  {name}: figures and V1 table", flush=True)
    local_global(series)                                        # TABLES V2, V3
    print("  V2/V3 local vs global violations done")

"""Homework 2, Point 2 - backtesting of the VaR forecasts of all models, at 1% and 5%.
Responsible: member D (reviewer: A).
For every series and level:
  - hits, violation rate, Kupiec UC, Christoffersen IND and CC, Berkowitz (BCP 2011) Ljung-Box on the hits,
    Engle-Manganelli Dynamic Quantile test
  - Diebold-Mariano of every model against Historical Simulation (M01) with two losses: tick loss and Lopez loss
  - Model Confidence Set (Hansen-Lunde-Nason, T_max, 90%) on the tick loss, across all models
  - the same coverage tests, average tick loss and MCS repeated on each calendar year (2021 = Oct-Dec, 2026 = Jan-Sep)
Outputs in outputs/tables/backtest/:
  T2.1_full_<series>.csv   all tests, full sample (detailed comment: EWP and market index)
  T2.2_years_<series>.csv  tests by calendar year
  T2.3_assets_summary.csv  for each model: how many of the 16 stocks reject each test, are in the MCS, beat/lose to HS
  T2.4_ranking.csv         how often each model is "preferred" (no rejection at 5% and in the MCS), all series
  T2.5_years_summary.csv   share of series in which each model is in the MCS, by year
  T2.6_years_hit_rate.csv  average violation rate of each model, by year
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from settings_hw2 import *
from common import load_returns, load_var, ensure_dirs, run_parallel
import tests_lib as tl                                       # tests and loss functions

warnings.filterwarnings("ignore")
TAB = os.path.join(TABLE_DIR, "backtest")                    # output folder of Point 2
SIG = 0.05            # significance level of the tests
MCS_ALPHA, MCS_B, MCS_BLOCK = 0.10, 1000, 10                 # 90% MCS, bootstrap replications, block length
BENCH = "M01_HS"                                             # benchmark of the Diebold-Mariano tests


def models_in(d, a):
    """Model codes available for level a (e.g. 21 for stocks and EWP, 16 for the index)."""
    return sorted(c[: -len(f"_{a}")] for c in d.columns if c.endswith(f"_{a}"))


def test_block(r, d, a, mods):
    """All tests on one sample (full or one year). d: VaR columns <model>_<a>. Returns a DataFrame, one row per model."""
    rows = {}
    L_tick, L_lop = {}, {}                                   # losses of every model (for DM and MCS)
    for m in mods:
        v = d[f"{m}_{a}"].values
        ok = np.isfinite(v)                                  # days with a valid forecast
        rr, vv = r[ok], v[ok]
        I = (rr < vv).astype(int)                            # hit sequence
        uc = tl.kupiec_uc(I, a); ind = tl.christoffersen_ind(I); cc = tl.christoffersen_cc(I, a)   # coverage tests
        lb = tl.berkowitz_ind(I, a); dq = tl.dq_test(I, vv, a)   # independence tests
        L_tick[m] = np.where(ok, tl.tick_loss(r, v, a), np.nan)  # loss 1
        L_lop[m] = np.where(ok, tl.lopez_loss(r, v), np.nan)     # loss 2
        rows[m] = {"n": len(I), "hits": int(I.sum()), "hit_rate": I.mean(), "expected": a,
                   "UC_LR": uc[0], "UC_p": uc[1], "IND_LR": ind[0], "IND_p": ind[1], "consecutive_hits": ind[2],
                   "CC_LR": cc[0], "CC_p": cc[1], "BCP_LB5": lb[0], "BCP_p": lb[1], "DQ": dq[0], "DQ_p": dq[1],
                   "avg_VaR": vv.mean(), "tick_loss": np.nanmean(L_tick[m]), "lopez_loss": np.nanmean(L_lop[m])}
    out = pd.DataFrame(rows).T
    # Diebold-Mariano against Historical Simulation (common days)
    if BENCH in mods:
        for lname, LL in (("tick", L_tick), ("lopez", L_lop)):
            for m in mods:
                if m == BENCH:
                    continue
                ok = np.isfinite(LL[m]) & np.isfinite(LL[BENCH])
                s, p = tl.diebold_mariano(LL[m][ok], LL[BENCH][ok])   # stat < 0: model better than HS
                out.loc[m, f"DM_{lname}"] = s; out.loc[m, f"DM_{lname}_p"] = p
    # Model Confidence Set on the tick loss (days on which every model has a forecast)
    M = pd.DataFrame(L_tick).dropna()
    p = tl.mcs(M.values, list(M.columns), MCS_ALPHA, MCS_B, MCS_BLOCK, SEED)
    out["MCS_p"] = pd.Series(p)
    out["in_MCS"] = out["MCS_p"] >= MCS_ALPHA                # model in the 90% MCS
    out["no_rejection"] = (out[["UC_p", "CC_p", "BCP_p", "DQ_p"]] >= SIG).all(axis=1)   # passes every test at 5%
    return out


def run_series(name):
    d = load_var(name)                                       # all VaR forecasts + realised return
    r = d["ret"].values
    full, years = [], []
    for a in ALPHAS:
        mods = models_in(d, a)
        t = test_block(r, d, a, mods); t.insert(0, "alpha", a); full.append(t)   # full backtest sample
        for y, g in d.groupby(d.index.year):                 # each calendar year separately
            ty = test_block(g["ret"].values, g, a, mods)
            ty.insert(0, "year", y); ty.insert(0, "alpha", a); years.append(ty)
    full = pd.concat(full).rename_axis("model"); years = pd.concat(years).rename_axis("model")
    full.to_csv(os.path.join(TAB, f"T2.1_full_{name}.csv")); years.to_csv(os.path.join(TAB, f"T2.2_years_{name}.csv"))   # TABLES T2.1, T2.2
    print(f"  {name:16s} {len(models_in(d, ALPHAS[0]))} models", flush=True)
    return name


def summaries(series):
    stocks = list(RIC_TO_NAME.values())
    F = {s: pd.read_csv(os.path.join(TAB, f"T2.1_full_{s}.csv"), index_col=0) for s in series}   # full-sample results
    Y = {s: pd.read_csv(os.path.join(TAB, f"T2.2_years_{s}.csv"), index_col=0) for s in series}  # yearly results
    # TABLE T2.3: summary across the 16 stocks
    rows = []
    for a in ALPHAS:
        A = pd.concat({s: F[s][F[s].alpha == a] for s in stocks if s in F})
        g = A.groupby(level=1)                               # group by model
        rows.append(pd.DataFrame({
            "alpha": a, "n_series": g.size(), "avg_hit_rate": g.hit_rate.mean(),
            "reject_UC": g.UC_p.apply(lambda x: (x < SIG).sum()), "reject_IND": g.IND_p.apply(lambda x: (x < SIG).sum()),
            "reject_CC": g.CC_p.apply(lambda x: (x < SIG).sum()), "reject_BCP": g.BCP_p.apply(lambda x: (x < SIG).sum()),
            "reject_DQ": g.DQ_p.apply(lambda x: (x < SIG).sum()), "no_rejection": g.no_rejection.sum(),
            "in_MCS": g.in_MCS.sum(), "avg_tick_loss": g.tick_loss.mean(),
            "DM_tick_better_than_HS": g.apply(lambda x: ((x.DM_tick < 0) & (x.DM_tick_p < SIG)).sum()),
            "DM_tick_worse_than_HS": g.apply(lambda x: ((x.DM_tick > 0) & (x.DM_tick_p < SIG)).sum()),
            "DM_lopez_better_than_HS": g.apply(lambda x: ((x.DM_lopez < 0) & (x.DM_lopez_p < SIG)).sum()),
            "DM_lopez_worse_than_HS": g.apply(lambda x: ((x.DM_lopez > 0) & (x.DM_lopez_p < SIG)).sum())}))
    pd.concat(rows).rename_axis("model").round(4).to_csv(os.path.join(TAB, "T2.3_assets_summary.csv"))
    # TABLE T2.4: ranking over all series (stocks, EWP, index): preferred = no rejection at 5% AND in the 90% MCS
    allf = pd.concat(F).reset_index(level=0, names="series")
    allf["preferred"] = allf.no_rejection.astype(bool) & allf.in_MCS.astype(bool)
    rk = allf.groupby(["alpha", "model"]).agg(n_series=("series", "size"), preferred=("preferred", "sum"),
                                              no_rejection=("no_rejection", "sum"), in_MCS=("in_MCS", "sum"),
                                              avg_tick_loss=("tick_loss", "mean"))
    rk["share_preferred"] = rk.preferred / rk.n_series
    rk.sort_values(["alpha", "share_preferred"], ascending=[True, False]).round(4).to_csv(os.path.join(TAB, "T2.4_ranking.csv"))
    # TABLES T2.5 and T2.6: stability over time (share in the MCS and average hit rate by calendar year)
    ally = pd.concat(Y).reset_index(level=0, names="series")
    ys = ally.groupby(["alpha", "model", "year"]).in_MCS.mean().unstack("year")
    ys.round(3).to_csv(os.path.join(TAB, "T2.5_years_summary.csv"))
    hr = ally.groupby(["alpha", "model", "year"]).hit_rate.mean().unstack("year")
    hr.round(4).to_csv(os.path.join(TAB, "T2.6_years_hit_rate.csv"))
    print("\n=== T2.4 ranking (share of series where the model passes all tests and is in the MCS) ===")
    print(rk.sort_values(["alpha", "share_preferred"], ascending=[True, False]).round(3).to_string())


if __name__ == "__main__":
    ensure_dirs(); os.makedirs(TAB, exist_ok=True)
    series = sys.argv[1:] or [c for c in load_returns().columns if os.path.exists(os.path.join(VAR_DIR, "iid", f"{c}.csv"))]   # series with forecasts
    print(f"backtest_var: {len(series)} series, MCS {int((1 - MCS_ALPHA) * 100)}% with B={MCS_B}, block={MCS_BLOCK}")
    run_parallel(run_series, series)                         # T2.1, T2.2 for every series
    summaries(series)                                        # T2.3 - T2.6

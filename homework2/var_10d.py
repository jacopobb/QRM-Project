"""Homework 2, Point 3 - 10-day VaR for the EWP and the market index.
Model: AR(1)-GJR-GARCH(1,1), Gaussian QML, rolling window of W_ROLL days (same as M09/M11).
  FHS 10d : N_SIM paths of 10 days, innovations drawn from the standardised residuals of the window, mean and
            variance updated along each path; VaR = alpha-quantile of the 10-day cumulative log return.
  SRR     : square-root-of-time rule applied to the 1-day FHS VaR:  VaR_10 = sqrt(10) * VaR_1.
The VaR made at the close of day t-1 covers the 10-day return r_t + ... + r_{t+9}; the last 9 days of the backtest
have no complete 10-day return and are dropped from the evaluation.
10-day windows overlap, so hits and losses are autocorrelated (MA(9) under H0): the Diebold-Mariano test uses a
Newey-West variance with at least 10 lags; coverage tests are also run on the non-overlapping sub-sample (every 10th day).
Losses: tick loss and Lopez loss of Point 2, evaluated on the 10-day returns.
Outputs: outputs/var10/<series>.csv, tables/var10/T3.1_var10.csv, figures/F3.1_var10_<series>_<alpha>.png
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from settings_hw2 import *
from common import load_returns, backtest_dates, ensure_dirs, run_parallel
import garch_core as gc
import tests_lib as tl

warnings.filterwarnings("ignore")
H, N_SIM = 10, 10000                                         # horizon (days) and number of simulated paths
OUT = os.path.join(OUT_DIR, "var10")                         # 10-day forecasts
TAB = os.path.join(TABLE_DIR, "var10")


def run_series(name):
    """10-day FHS VaR, 1-day FHS VaR and square-root-rule VaR for every backtest day."""
    t0 = time.time()
    r = load_returns()[name].dropna()
    dates = backtest_dates(r.to_frame())
    pos = r.index.get_indexer(dates)
    x = r.values
    rows, th = [], None
    for k, i in enumerate(pos):
        w = x[i - W_ROLL:i]                                  # rolling window before day t
        if k % REFIT == 0:
            th = gc.fit(w, "normal", th)                     # Gaussian QML
        z, _, _ = gc.forecast(th, w)                         # standardised residuals = pool of the bootstrap
        rng = np.random.default_rng([SEED, k])               # reproducible draws for day k
        s10 = gc.simulate_paths(th, w, z, H, N_SIM, rng)     # simulated 10-day returns
        s1 = gc.simulate_paths(th, w, z, 1, N_SIM, rng)      # simulated 1-day returns
        row = {"R10": x[i:i + H].sum() if i + H <= len(x) else np.nan}   # realised 10-day return r_t + ... + r_{t+9}
        for a in ALPHAS:
            row[f"FHS10_{a}"] = np.quantile(s10, a)          # 10-day FHS VaR
            row[f"VaR1_{a}"] = np.quantile(s1, a)            # 1-day FHS VaR
            row[f"SRR10_{a}"] = np.sqrt(H) * row[f"VaR1_{a}"]   # square-root-of-time rule
        rows.append(row)
    d = pd.DataFrame(rows, index=pd.Index(dates, name="Date"))
    d = d[d.index <= BT_END]
    d.loc[d.index > dates[-H], "R10"] = np.nan                   # 10-day return not complete inside the sample
    os.makedirs(OUT, exist_ok=True); d.to_csv(os.path.join(OUT, f"{name}.csv"))
    print(f"  {name:16s} done in {time.time() - t0:6.1f}s", flush=True)
    return name


def evaluate(name):
    """TABLE T3.1 rows and FIGURE F3.1 for one series."""
    d = pd.read_csv(os.path.join(OUT, f"{name}.csv"), index_col=0, parse_dates=True).dropna(subset=["R10"])
    rows = []
    for a in ALPHAS:
        R = d.R10.values                                     # realised 10-day returns
        L = {}
        for m in ("FHS10", "SRR10"):
            v = d[f"{m}_{a}"].values
            I = (R < v).astype(int)                          # 10-day hits (overlapping)
            Inov = I[::H]                                    # non-overlapping 10-day periods
            uc = tl.kupiec_uc(Inov, a); cc = tl.christoffersen_cc(Inov, a)   # valid on the non-overlapping sample
            L[m] = {"tick": tl.tick_loss(R, v, a), "lopez": tl.lopez_loss(R, v)}   # 1-day losses applied to 10-day returns
            rows.append({"series": name, "alpha": a, "model": m, "n": len(I), "hit_rate": I.mean(),
                         "n_nonoverlap": len(Inov), "hits_nonoverlap": int(Inov.sum()), "UC_p_nonoverlap": uc[1],
                         "CC_p_nonoverlap": cc[1], "avg_VaR": v.mean(),
                         "tick_loss": L[m]["tick"].mean(), "lopez_loss": L[m]["lopez"].mean()})
        lags = max(H, int(np.floor(4 * (len(R) / 100) ** (2 / 9))))   # HAC lags >= 10 because of the overlap
        for lname in ("tick", "lopez"):
            s, p = tl.diebold_mariano(L["FHS10"][lname], L["SRR10"][lname], lags)         # all days, HAC
            s2, p2 = tl.diebold_mariano(L["FHS10"][lname][::H], L["SRR10"][lname][::H])   # non-overlapping days
            rows[-1][f"DM_{lname}_FHS_vs_SRR"] = s; rows[-1][f"DM_{lname}_p"] = p          # stat < 0: FHS better
            rows[-1][f"DM_{lname}_nonoverlap"] = s2; rows[-1][f"DM_{lname}_nonoverlap_p"] = p2
        # FIGURE F3.1: realised 10-day returns vs the two VaR
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(d.index, d.R10, width=1.5, color="0.7", label="realised 10-day return")
        for m, lab in (("FHS10", "FHS 10-day"), ("SRR10", "sqrt(10) x FHS 1-day")):
            line, = ax.plot(d.index, d[f"{m}_{a}"], lw=1, label=lab)
            hit = d.R10 < d[f"{m}_{a}"]
            ax.scatter(d.index[hit], d.R10[hit], s=8, color=line.get_color(), zorder=3)   # violations
        ax.legend(fontsize=8); ax.set_ylabel("%")
        ax.set_title(f"Fig F3.1 - {name}: 10-day VaR {int(a * 100)}%, FHS vs square-root rule (dots = violations)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, f"F3.1_var10_{name}_{a}.png"), dpi=150); plt.close(fig)
    return rows


if __name__ == "__main__":
    ensure_dirs(); os.makedirs(TAB, exist_ok=True)
    r = load_returns()
    series = sys.argv[1:] or [s for s in (EWP, INDEX_NAME) if s in r.columns]   # EWP and market index only
    print(f"var_10d: {series}, {N_SIM} paths of {H} days")
    run_parallel(run_series, series)
    tab = pd.DataFrame([row for s in series for row in evaluate(s)])
    tab.round(5).to_csv(os.path.join(TAB, "T3.1_var10.csv"), index=False)   # TABLE T3.1
    print(tab.round(4).to_string())

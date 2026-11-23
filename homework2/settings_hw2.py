"""Homework 2 - global settings. Change things ONLY here, so every script gives identical numbers.

This folder is independent of Homework 1: it only READS ../data/returns.csv (built by the HW1 pipeline) and
writes everything in homework2/outputs/. No file of Homework 1 is modified.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))           # .../QRM-Project/homework2
ROOT = os.path.dirname(HERE)                                # .../QRM-Project

SEED = 42                                                   # every simulation (FHS, bootstrap of the MCS) uses this seed

# ------------------------------------------------------------------ samples
BT_START, BT_END = "2021-10-01", "2026-09-30"              # backtesting sample (5 years), as in the assignment
W_ROLL = 1260                                               # rolling window: 5 years of trading days (5 x 252)
ALPHAS = [0.05, 0.01]                                       # VaR coverage levels

# ------------------------------------------------------------------ data
RET_FILE = os.path.join(ROOT, "data", "returns.csv")       # HW1 log returns in %, one column per RIC (read-only)
INDEX_FILE = os.path.join(HERE, "data", "index_prices.csv")  # market index: columns Date, Close (to be downloaded, LSEG)
INDEX_NAME = "SMI"                                          # label of the market index in tables and figures
FACTORS_FILE = os.path.join(HERE, "data", "Europe_5_Factors_Daily.csv")  # Kenneth French, used in models 16-21

# RIC -> company name (same 16 stocks as HW1; copied here so that HW2 does not depend on HW1 settings)
RIC_TO_NAME = {
    "ROPC.S": "Roche", "NOVN.S": "Novartis", "NESN.S": "Nestle", "UBSG.S": "UBS", "ABBN.S": "ABB",
    "ZURN.S": "Zurich", "CFR.S": "Richemont", "SRENH.S": "SwissRe", "LONN.S": "Lonza", "GIVN.S": "Givaudan",
    "HOLN.S": "Holcim", "SIKA.S": "Sika", "SLHN.S": "SwissLife", "GEBN.S": "Geberit", "SCMN.S": "Swisscom",
    "HBANC.S": "HelvetiaBaloise",
}
EWP = "EWP"                                                 # equally weighted portfolio of the 16 stocks

# ------------------------------------------------------------------ model choices
REFIT = 1            # parametric densities (t, GH, GEV, GPD) re-estimated every REFIT days (1 = every day)
BLOCK = 21           # GEV block size: one month of trading days
GPD_U_Q = 0.90       # GPD threshold = 90% quantile of the losses in the estimation sample
THETA_U_Q = 0.95     # threshold for the blocks estimator of the extremal index = 95% quantile of the losses
EWMA_LAMBDA = 0.94   # RiskMetrics decay for daily data
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # parallel processes (one series per process)

# ------------------------------------------------------------------ output folders
OUT_DIR = os.path.join(HERE, "outputs")
VAR_DIR = os.path.join(OUT_DIR, "var")                      # one sub-folder per script, one csv per series
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(OUT_DIR, "logs")

# Model codes used as column names: <code>_<alpha>, e.g. M04_GH_0.01. Numbers follow the assignment (points 1-21).
MODELS = {
    "M01_HS": "Historical Simulation (rolling 5y)",
    "M02_N": "i.i.d. Normal (rolling 5y)",
    "M03_T": "i.i.d. Student-t (rolling 5y)",
    "M04_GH": "i.i.d. Generalized Hyperbolic (rolling 5y)",
    "M05_CF": "Cornish-Fisher (rolling 5y)",
    "M06_GEV": "i.i.d. GEV, no extremal index (expanding)",
    "M07_GEVT": "i.i.d. GEV with extremal index (expanding)",
    "M08_GPD": "i.i.d. GPD (expanding)",
    "M09_AG_N": "ARMA-GARCH + Normal",
    "M10_AG_T": "ARMA-GARCH + Student-t",
    "M11_AG_HS": "ARMA-GARCH + HS (FHS)",
    "M12_AG_CF": "ARMA-GARCH + Cornish-Fisher",
    "M13_AG_GEV": "ARMA-GARCH + GEV",
    "M14_AG_GEVT": "ARMA-GARCH + GEV with extremal index",
    "M15_AG_GPD": "ARMA-GARCH + GPD",
    "M16_CCC_N": "Factor mapping + CCC, Gaussian",
    "M17_DCC_N": "Factor mapping + DCC, Gaussian",
    "M18_CCC_FHS": "Factor mapping + CCC, FHS",
    "M19_DCC_FHS": "Factor mapping + DCC, FHS",
    "M20_EWMA": "EWMA mean and variance, Gaussian",
    "M21_X_SKT": "ARMA-GARCH on factor combination X, Skew-t",
}

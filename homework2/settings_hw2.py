"""Homework 2 - global settings. Change things ONLY here, so every script gives identical numbers.
Agreed by the whole group (week 1); data part maintained by member A, the rest by member D.

This folder is independent of Homework 1: it only READS ../data/returns.csv (built by the HW1 pipeline) and
writes everything in homework2/outputs/. No file of Homework 1 is modified.
"""
import os                                                    # path handling

HERE = os.path.dirname(os.path.abspath(__file__))            # .../QRM-Project/homework2 (folder of this file)
ROOT = os.path.dirname(HERE)                                 # .../QRM-Project (repository root, HW1 files)

SEED = 42                                                    # seed of every simulation (FHS) and bootstrap (MCS, McNeil-Frey)

# ------------------------------------------------------------------ samples
BT_START, BT_END = "2021-10-01", "2026-09-30"               # backtesting sample (5 years), as required by the assignment
W_ROLL = 1260                                                # rolling estimation window: 5 years x 252 trading days
ALPHAS = [0.05, 0.01]                                        # VaR coverage levels: 5% and 1%

# ------------------------------------------------------------------ data
RET_FILE = os.path.join(ROOT, "data", "returns.csv")        # HW1 daily log returns in %, one column per RIC (read-only)
INDEX_FILE = os.path.join(HERE, "data", "index_prices.csv")  # market index closes: columns Date, Close (from LSEG)
INDEX_NAME = "SMI"                                           # label of the market index in tables and figures
FACTORS_FILE = os.path.join(HERE, "data", "Europe_5_Factors_Daily.csv")  # Kenneth French Europe 5 factors (models 16-21)

# RIC -> company name: same 16 stocks as HW1, copied here so that HW2 does not import the HW1 settings
RIC_TO_NAME = {
    "ROPC.S": "Roche", "NOVN.S": "Novartis", "NESN.S": "Nestle", "UBSG.S": "UBS", "ABBN.S": "ABB",
    "ZURN.S": "Zurich", "CFR.S": "Richemont", "SRENH.S": "SwissRe", "LONN.S": "Lonza", "GIVN.S": "Givaudan",
    "HOLN.S": "Holcim", "SIKA.S": "Sika", "SLHN.S": "SwissLife", "GEBN.S": "Geberit", "SCMN.S": "Swisscom",
    "HBANC.S": "HelvetiaBaloise",
}
EWP = "EWP"                                                  # name of the equally weighted portfolio of the 16 stocks

# ------------------------------------------------------------------ model choices
REFIT = 1            # parametric models re-estimated every REFIT days (1 = every day; 5 = faster, for quick tests only)
BLOCK = 21           # GEV block size: one month of trading days (assignment: "block size use one month")
GPD_U_Q = 0.90       # GPD threshold = 90% quantile of the losses in the estimation sample (assignment: "a quantile")
THETA_U_Q = 0.95     # threshold of the blocks estimator of the extremal index = 95% quantile of the losses
EWMA_LAMBDA = 0.94   # RiskMetrics decay factor for daily data
N_JOBS = max(1, (os.cpu_count() or 2) - 1)   # parallel processes (one series per process), leaves one core free

# ------------------------------------------------------------------ output folders
OUT_DIR = os.path.join(HERE, "outputs")                      # every result of HW2 goes below this folder
VAR_DIR = os.path.join(OUT_DIR, "var")                       # VaR forecasts: one sub-folder per script, one csv per series
TABLE_DIR = os.path.join(OUT_DIR, "tables")                  # tables of the report (csv)
FIG_DIR = os.path.join(OUT_DIR, "figures")                   # figures of the report (png)
LOG_DIR = os.path.join(OUT_DIR, "logs")                      # console output of each step (written by main_hw2.py)

# Model codes used as column names <code>_<alpha> (e.g. M04_GH_0.01); numbers follow the assignment (models 1-21)
MODELS = {
    "M01_HS": "Historical Simulation (rolling 5y)",                          # (1)
    "M02_N": "i.i.d. Normal (rolling 5y)",                                   # (2)
    "M03_T": "i.i.d. Student-t (rolling 5y)",                                # (3)
    "M04_GH": "i.i.d. Generalized Hyperbolic (rolling 5y)",                  # (4)
    "M05_CF": "Cornish-Fisher (rolling 5y)",                                 # (5)
    "M06_GEV": "i.i.d. GEV, no extremal index (expanding)",                  # (6)
    "M07_GEVT": "i.i.d. GEV with extremal index (expanding)",                # (7)
    "M08_GPD": "i.i.d. GPD (expanding)",                                     # (8)
    "M09_AG_N": "ARMA-GARCH + Normal",                                       # (9)
    "M10_AG_T": "ARMA-GARCH + Student-t",                                    # (10)
    "M11_AG_HS": "ARMA-GARCH + HS (FHS)",                                    # (11)
    "M12_AG_CF": "ARMA-GARCH + Cornish-Fisher",                              # (12)
    "M13_AG_GEV": "ARMA-GARCH + GEV",                                        # (13)
    "M14_AG_GEVT": "ARMA-GARCH + GEV with extremal index",                   # (14)
    "M15_AG_GPD": "ARMA-GARCH + GPD",                                        # (15)
    "M16_CCC_N": "Factor mapping + CCC, Gaussian",                           # (16)
    "M17_DCC_N": "Factor mapping + DCC, Gaussian",                           # (17)
    "M18_CCC_FHS": "Factor mapping + CCC, FHS",                              # (18)
    "M19_DCC_FHS": "Factor mapping + DCC, FHS",                              # (19)
    "M20_EWMA": "EWMA mean and variance, Gaussian",                          # (20)
    "M21_X_SKT": "ARMA-GARCH on factor combination X, Skew-t",               # (21)
}

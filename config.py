"""Global settings: change things here only, so every script gives identical numbers."""
SEED = 42                        # fixes portfolio weights and any simulation
START, END = "2016-10-01", "2026-09-30"
IN_SAMPLE_END = "2025-09-30"     # last in-sample day; out-of-sample = 1/10/2025 - 30/09/2026
TICKERS = ["NESN.S", "NOVN.S", "ROG.S", "UBSG.S", "ABBN.S", "ZURN.S",
           "CFR.S", "GIVN.S", "SIKA.S", "SCMN.S"]   # final choice of 10-20 RICs
ALPHAS = [0.05, 0.01]            # VaR levels
H_REFIT = 20                     # re-estimate GARCH every h days (1 = daily)
BLOCK = 21                       # GEV block size (about 1 month)
RAW_FILE = "data/prices_raw.csv" # exported from LSEG: Date + one column per RIC



"""Global settings: change things here only, so every script gives identical numbers."""
SEED = 42                          # fixes portfolio weights and any simulation
START, END = "2011-09-26", "2026-09-30"
IN_SAMPLE_END = "2025-09-30"       # last in-sample day; out-of-sample = 1/10/2025 - 30/09/2026

# company -> (RIC in LSEG Workspace, ISIN, sector label, weight in Solactive Swiss Large Cap Index on 23-Oct-2026 in %).
# Sector labels as in the report table. RIC: only ROPC.S was verified one by one - VERIFY the four new ones
# (HOLN.S, SLHN.S, GEBN.S, HBAN.S). ISINs of the new stocks are from memory: VERIFY too (no code depends on them).
STOCKS = {
    "Roche":           ("ROPC.S", "CH0012032048", "Healthcare",             18.40),
    "Novartis":        ("NOVN.S", "CH0012005267", "Healthcare",             15.33),
    "Nestle":          ("NESN.S", "CH0038863350", "Consumer Staples",        14.17),
    "UBS":             ("UBSG.S", "CH0244767585", "Financials",              7.76),
    "ABB":             ("ABBN.S", "CH0012221716", "Industrials",             7.07),
    "Zurich":          ("ZURN.S", "CH0011075394", "Financials",              6.09),
    "Richemont":       ("CFR.S",  "CH0210483332", "Consumer discretionary",  5.68),
    "SwissRe":         ("SRENH.S", "CH0126881561", "Financials",              3.09),
    "Lonza":           ("LONN.S", "CH0013841017", "Healthcare",              2.95),
    "Givaudan":        ("GIVN.S", "CH0010645932", "Materials",               2.47),
    "Holcim":          ("HOLN.S", "CH0012214059", "Materials",               2.24),
    "Sika":            ("SIKA.S", "CH0418792922", "Materials",               1.93),
    "SwissLife":       ("SLHN.S", "CH0014852781", "Financials",              1.73),
    "Geberit":         ("GEBN.S", "CH0030170408", "Industrials",             1.21),
    "Swisscom":        ("SCMN.S", "CH0008742519", "Communication services",  1.16),
    "HelvetiaBaloise": ("HBANC.S", "CH0466642201",       "Financials",       1.08),
}
TICKERS = [v[0] for v in STOCKS.values()]
SECTOR = {name: v[2] for name, v in STOCKS.items()}
INDEX_WEIGHT = {name: v[3] for name, v in STOCKS.items()}
assert round(sum(INDEX_WEIGHT.values()), 2) == 92.36, "weights should sum to 92.36%"

ALPHAS = [0.05, 0.01]              # VaR levels
H_REFIT = 20                       # re-estimate GARCH every h days
BLOCK = 21                         # GEV block size (about 1 month)

RAW_DIR   = "data/raw"             # the 12 LSEG exports, one file per RIC, e.g. NESN.S.xlsx
RAW_FILE  = "data/prices_raw.csv"  # merged raw prices (built by build_raw_csv.py)
CLEAN_FILE = "data/prices_clean.csv"
RET_FILE   = "data/returns.csv"

# company -> (RIC as shown in LSEG Workspace, ISIN). REPLACE each RIC with the one you verified by ISIN.
STOCKS = {
    "Nestle":    ("NESN.S", "CH0038863350"),
    "Novartis":  ("NOVN.S", "CH0012005267"),
    "Roche":     ("ROPC.S", "CH0012032048"),   # participation share, verified in Workspace
    "UBS":       ("UBSG.S", "CH0244767585"),
    "ABB":       ("ABBN.S", "CH0012221716"),
    "Zurich":    ("ZURN.S", "CH0011075394"),
    "Richemont": ("CFR.S",  "CH0210483332"),
    "Givaudan":  ("GIVN.S", "CH0010645932"),
    "Sika":      ("SIKA.S", "CH0418792922"),
    "Swisscom":  ("SCMN.S", "CH0008742519"),
    "Lonza":     ("LONN.S", "CH0013841017"),
    "SwissRe":   ("SRENH.S", "CH0126881561"),
}
TICKERS = [v[0] for v in STOCKS.values()]

ALPHAS = [0.05, 0.01]              # VaR levels
H_REFIT = 20                       # re-estimate GARCH every h days
BLOCK = 21                         # GEV block size (about 1 month)
SEED = 42

RAW_DIR   = "data/raw"             # the 12 LSEG exports, one file per RIC, e.g. NESN.S.xlsx
RAW_FILE  = "data/prices_raw.csv"  # merged raw prices (built by build_raw_csv.py)
CLEAN_FILE = "data/prices_clean.csv"
RET_FILE   = "data/returns.csv"

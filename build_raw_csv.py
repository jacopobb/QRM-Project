"""Point 1 (collection) - merge the 12 LSEG Excel exports into data/prices_raw.csv.

Run once:  python build_raw_csv.py     File names must be the RIC, e.g. data/raw/NESN.S.xlsx"""
import os, pandas as pd
from config import *

CLOSE_NAMES = ["close", "close price", "price close", "last", "last price"]

def read_one(path):
    raw = pd.read_excel(path, header=None)
    # find the header row: the first row containing a cell called "Date" (exports often have a title block)
    hdr = next(i for i in range(min(30, len(raw)))
               if raw.iloc[i].astype(str).str.strip().str.lower().isin(["date", "exchange date"]).any())
    df = raw.iloc[hdr + 1:].copy(); df.columns = raw.iloc[hdr].astype(str).str.strip()
    date_col  = next(c for c in df.columns if c.lower() in ("date", "exchange date"))
    close_col = next(c for c in df.columns if c.lower() in CLOSE_NAMES)   # fails loudly if not found
    s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col], errors="coerce"))
    return s[s.index.notna()].sort_index()

if __name__ == "__main__":
    series = {}
    for ric in TICKERS:
        # accept both "NESN.S.xlsx" and "NESN.xlsx" (file named without the ".S" suffix)
        candidates = [os.path.join(RAW_DIR, f"{n}.xlsx") for n in (ric, ric.split(".")[0])]
        path = next(c for c in candidates if os.path.exists(c))
        series[ric] = read_one(path)
        print(f"{ric}: {series[ric].index.min().date()} -> {series[ric].index.max().date()}, {series[ric].notna().sum()} prices")
    pd.DataFrame(series).sort_index().to_csv(RAW_FILE)
    print("saved", RAW_FILE)

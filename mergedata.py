# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:49:10 2025

@author: 高文萱
"""

# merge.py
# Created: 2025-08-28
# Author: Mengxiao Chen (高文萱)
# Purpose: Merge 2019–2023 CSVs under data/2019-2023, apply QC==1, output to model_outputs/
# Usage:  python merge.py
# Requires: Python 3.9+, pandas (pyarrow optional for parquet)

from pathlib import Path
import pandas as pd

# ---------------- 1) Paths (relative to this file) ----------------
BASE_DIR = Path(__file__).resolve().parent                 # EAN-11523777-am/
DATA_DIR = BASE_DIR / "data" / "2019-2023"                # EAN-11523777-am/data/2019-2023
OUT_DIR  = BASE_DIR / "model_outputs"                     # EAN-11523777-am/model_outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_qc_one(csv_path: Path) -> pd.DataFrame:
    """Load one CSV, standardise datetime, apply QC==1 where flags exist."""
    df = pd.read_csv(csv_path)
    # strip accidental spaces
    df.columns = [c.strip() for c in df.columns]

    # find a datetime-like column
    dt_candidates = [c for c in df.columns if c.lower() in {"datetime", "time", "timestamp"}]
    if not dt_candidates:
        raise KeyError(f"No datetime column in {csv_path.name}. Expected one of: datetime/time/timestamp")
    dt_col = dt_candidates[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    # target measurement columns (case-insensitive exact name match)
    targets = []
    for v in ["CH4", "H2O", "CO2", "CO"]:
        m = [c for c in df.columns if c.lower() == v.lower()]
        if m:
            targets.append(m[0])

    # apply QC flags if present: keep values where *_qc_flag(s) == 1, else set NA
    for mcol in targets:
        candidates = [
            f"{mcol}_qc_flags", f"{mcol}_qc_flag",
            f"{mcol.lower()}_qc_flags", f"{mcol.lower()}_qc_flag",
            f"{mcol.upper()}_qc_flags", f"{mcol.upper()}_qc_flag",
        ]
        qc = next((c for c in candidates if c in df.columns), None)
        if qc is not None:
            df.loc[df[qc] != 1, mcol] = pd.NA

    # keep datetime + targets + any qc columns; standardise datetime name
    keep = [dt_col] + targets + [c for c in df.columns if c.endswith("_qc_flags") or c.endswith("_qc_flag")]
    df = df[keep].rename(columns={dt_col: "datetime"})
    return df

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            f"Expected structure:\n"
            f"{BASE_DIR.name}/data/2019-2023/*.csv"
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    print(f"[ROOT] {BASE_DIR}")
    print(f"[DATA] {DATA_DIR}  ({len(csv_files)} files)")
    print(f"[OUT ] {OUT_DIR}")

    dfs = []
    for fp in csv_files:
        print(f"[LOAD] {fp.name}")
        dfs.append(load_and_qc_one(fp))

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.set_index("datetime").sort_index()

    print(f"[INFO] Combined shape: {df_all.shape}")
    print(df_all.head(3))

    parquet_path = OUT_DIR / "all_years_cleaned.parquet"
    csv_path     = OUT_DIR / "all_years_cleaned.csv"

    # Try Parquet (fast/compact); fallback to CSV if pyarrow/fastparquet missing
    try:
        df_all.to_parquet(parquet_path)
        print(f"[SAVE] Parquet: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}); saving CSV instead.")
        df_all.to_csv(csv_path)
        print(f"[SAVE] CSV: {csv_path}")

if __name__ == "__main__":
    main()

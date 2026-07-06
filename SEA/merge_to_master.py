"""
merge_sea_to_master.py
======================
Merges two SEA supplement files into the final master dataset:
  1. sea_all_startups_fixed.csv  — Failory scrape (all SEA countries)
  2. singapore_startups.csv      — GrowthList Singapore-only scrape

Rules:
  - Master dataset rows are NEVER modified or removed.
  - New rows are appended only.
  - All new rows conform to the master's 42-column schema.
  - Columns with no mapping from a source are left as NaN / empty.
  - Duplicate names already present in the master are skipped.

Usage:
    python merge_sea_to_master.py

Inputs  (edit paths below if needed):
    SEA_FAILORY_PATH   — sea_all_startups_fixed.csv
    SEA_SG_PATH        — singapore_startups.csv
    MASTER_PATH        — final_master_dataset_updated.csv

Output:
    final_master_dataset_sea_merged.csv
"""

import re
import pandas as pd
from pathlib import Path

SEA_FAILORY_PATH = "sea_all_startups_fixed.csv"
SEA_SG_PATH      = "singapore_startups.csv"
MASTER_PATH      = "final_master_dataset_updated.csv"
OUTPUT_PATH      = "final_master_dataset_sea_merged.csv"

COUNTRY_ISO = {
    "Singapore":   "SGP",
    "Malaysia":    "MYS",
    "Indonesia":   "IDN",
    "Vietnam":     "VNM",
    "Thailand":    "THA",
    "Philippines": "PHL",
}

# ── Funding string parser (handles $75M, $8.3B, $1,800,000 etc.) ─────────────
DOLLAR_ALIASES = {"$", "₱", "£", "€", "¥"}

def parse_funding_str(value) -> float | None:
    """Convert a funding string to a plain float (USD)."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    # Fix distorted currency symbols
    if v and v[0] in DOLLAR_ALIASES:
        v = "$" + v[1:]
    if not v.startswith("$"):
        return None
    number_str = v[1:].replace(",", "")
    suffixes = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    suffix = number_str[-1].upper() if number_str and number_str[-1].upper() in suffixes else None
    if suffix:
        number_str = number_str[:-1]
        magnitude = suffixes[suffix]
    else:
        magnitude = 1
    try:
        return round(float(number_str) * magnitude, 2)
    except ValueError:
        return None


def parse_funding_numeric(value) -> float | None:
    """Handle already-numeric or string-numeric funding values."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_funding_str(str(value))


# ── Map last_funding_status → funding round column ────────────────────────────
ROUND_COL_MAP = {
    "seed":          "funding_seed_usd",
    "pre-seed":      "funding_seed_usd",
    "series a":      "funding_round_A_usd",
    "series b":      "funding_round_B_usd",
    "series c":      "funding_round_C_usd",
    "series d":      "funding_round_D_usd",
    "series e":      "funding_round_E_usd",
    "series f":      "funding_round_F_usd",
    "series g":      "funding_round_G_usd",
    "series h":      "funding_round_H_usd",
    "angel":         "funding_angel_usd",
    "grant":         "funding_grant_usd",
    "debt":          "funding_debt_financing_usd",
    "convertible":   "funding_convertible_note_usd",
    "private equity":"funding_private_equity_usd",
    "post-ipo":      "funding_post_ipo_equity_usd",
    "crowdfunding":  "funding_equity_crowdfunding_usd",
    "venture":       "funding_venture_unclassified_usd",
    "undisclosed":   "funding_undisclosed_usd",
}

def round_to_col(round_label: str) -> str | None:
    if not isinstance(round_label, str):
        return None
    key = round_label.strip().lower()
    for pattern, col in ROUND_COL_MAP.items():
        if pattern in key:
            return col
    return None


# ── Build an empty master-schema row ─────────────────────────────────────────
def empty_row(master_cols: list) -> dict:
    return {col: None for col in master_cols}


# ── Convert Failory rows → master schema ─────────────────────────────────────
def map_failory_row(row: pd.Series, master_cols: list) -> dict:
    r = empty_row(master_cols)

    country_name = str(row.get("country", "")).strip()

    r["name"]          = row.get("name")
    r["homepage_url"]  = row.get("url")
    r["category_list"] = row.get("industry")
    r["city"]          = row.get("headquarters")
    r["investors"]     = row.get("top_investors")
    r["country_code"]  = COUNTRY_ISO.get(country_name)
    r["country"]       = country_name
    r["region_group"]  = "Southeast Asia"
    r["data_source"]   = "sea_failory_supplement"
    r["status"]        = "Active"

    # founded_at
    yr = row.get("year_founded")
    if pd.notna(yr):
        try:
            r["founded_at"] = f"{int(yr)}-01-01"
        except (ValueError, TypeError):
            pass

    # funding
    amount = parse_funding_numeric(row.get("funding_amount_usd") or row.get("funding_amount"))
    if amount is not None:
        r["funding_total_usd"] = amount
        round_col = round_to_col(str(row.get("last_funding_status", "")))
        if round_col:
            r[round_col] = amount

    return r


# ── Convert GrowthList Singapore rows → master schema ────────────────────────
def map_sg_row(row: pd.Series, master_cols: list) -> dict:
    """
    singapore_startups.csv columns (with confirmed swap):
      Name | Website | Industry | Country | Funding_Amount_USD | Funding_Type | Last_Funding_Date
    Note: Funding_Amount_USD and Funding_Type appear swapped in the raw file —
          the amount column contains dates like 'May 2026' and the type column
          contains dollar strings like '$6,800,000'. The script corrects for this.
    """
    r = empty_row(master_cols)

    r["name"]          = row.get("Name")
    r["homepage_url"]  = row.get("Website")
    r["category_list"] = row.get("Industry")
    r["country_code"]  = "SGP"
    r["country"]       = "Singapore"
    r["region_group"]  = "Southeast Asia"
    r["data_source"]   = "sea_sg_growthlist_supplement"
    r["status"]        = "Active"

    # Columns are swapped in source: Funding_Amount_USD holds the date,
    # Funding_Type holds the dollar amount string.
    raw_amount     = row.get("Funding_Type")        # e.g. "$6,800,000"
    raw_round      = row.get("Last_Funding_Date")   # e.g. "Seed"

    amount = parse_funding_numeric(raw_amount)
    if amount is not None:
        r["funding_total_usd"] = amount
        round_col = round_to_col(str(raw_round))
        if round_col:
            r[round_col] = amount

    return r


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading master dataset...")
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    master_cols = list(master.columns)
    print(f"  {len(master):,} rows, {len(master_cols)} columns")

    # Build a set of existing names (lowercase) to detect duplicates
    existing_names = set(master["name"].dropna().str.strip().str.lower())

    new_rows = []
    skipped  = {"failory": 0, "sg": 0}

    # ── Failory SEA ──
    print("\nProcessing sea_all_startups_fixed.csv (Failory)...")
    failory = pd.read_csv(SEA_FAILORY_PATH)
    print(f"  {len(failory):,} rows")
    for _, row in failory.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name.lower() in existing_names:
            skipped["failory"] += 1
            continue
        mapped = map_failory_row(row, master_cols)
        new_rows.append(mapped)
        existing_names.add(name.lower())
    print(f"  → {len(failory) - skipped['failory']} rows added, {skipped['failory']} skipped (duplicates/empty)")

    # ── GrowthList Singapore ──
    print("\nProcessing singapore_startups.csv (GrowthList)...")
    sg = pd.read_csv(SEA_SG_PATH)
    print(f"  {len(sg):,} rows")
    for _, row in sg.iterrows():
        name = str(row.get("Name", "")).strip()
        if not name or name.lower() in existing_names:
            skipped["sg"] += 1
            continue
        mapped = map_sg_row(row, master_cols)
        new_rows.append(mapped)
        existing_names.add(name.lower())
    print(f"  → {len(sg) - skipped['sg']} rows added, {skipped['sg']} skipped (duplicates/empty)")

    # ── Merge ──
    print(f"\nMerging {len(new_rows):,} new rows into master...")
    new_df = pd.DataFrame(new_rows, columns=master_cols)
    merged = pd.concat([master, new_df], ignore_index=True)
    print(f"  Final row count: {len(merged):,}")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── Quick sanity check ──
    print("\n── Sanity check ──────────────────────────────────────────")
    added = merged[merged["data_source"].isin(["sea_failory_supplement", "sea_sg_growthlist_supplement"])]
    print(added.groupby("data_source")["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
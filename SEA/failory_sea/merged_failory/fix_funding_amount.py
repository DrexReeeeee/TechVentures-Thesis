"""
fix_funding_amount.py
=====================
Fixes distorted currency symbols in the funding_amount column of a Failory
SEA CSV output, then parses the cleaned strings into a numeric USD float column.

Usage:
    python fix_funding_amount.py <input.csv> [output.csv]

If output.csv is omitted, the file is saved as <input>_fixed.csv.

Assumptions:
    - All amounts are in USD regardless of the currency symbol present.
    - The peso sign (₱) is a rendering artifact of the dollar sign ($).
    - Supported magnitude suffixes: K, M, B (case-insensitive).
    - Unparseable or blank values become NaN in funding_amount_usd.
"""

import sys
import re
import pandas as pd
from pathlib import Path


# Any symbol that should be treated as "$"
DOLLAR_ALIASES = {"$", "₱", "£", "€", "¥"}


def fix_symbol(value: str) -> str:
    """Replace any known distorted currency symbol with '$'."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value and value[0] in DOLLAR_ALIASES:
        return "$" + value[1:]
    return value


def parse_funding(value: str) -> float | None:
    """
    Convert a cleaned funding string like '$75M', '$8.3B', '$1.8M' to a float.
    Returns None for blanks or unparseable strings.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.strip()
    if not value.startswith("$"):
        return None

    number_str = value[1:]

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


def process(input_path: str, output_path: str | None = None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_fixed.csv")
    else:
        output_path = Path(output_path)

    df = pd.read_csv(input_path)

    if "funding_amount" not in df.columns:
        print(f"ERROR: 'funding_amount' column not found in {input_path.name}")
        sys.exit(1)

    # Show before state
    distorted = df["funding_amount"].dropna()
    distorted = distorted[distorted.str[0].isin(DOLLAR_ALIASES - {"$"})]
    if not distorted.empty:
        print(f"Found {len(distorted)} distorted symbol(s): {distorted.tolist()}")
    else:
        print("No distorted symbols found.")

    # Fix symbols
    df["funding_amount"] = df["funding_amount"].apply(fix_symbol)

    # Parse to numeric
    df["funding_amount_usd"] = df["funding_amount"].apply(parse_funding)

    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    print(f"Parsed {df['funding_amount_usd'].notna().sum()} / {len(df)} rows successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_funding_amount.py <input.csv> [output.csv]")
        sys.exit(1)

    input_csv  = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    process(input_csv, output_csv)
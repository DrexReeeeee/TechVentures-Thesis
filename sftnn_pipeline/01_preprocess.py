"""
01_preprocess.py
=================
Stage 1 of the six-stage development model (Chapter 4, Section 4.6).

Implements:
  - Section 4.3.2 (Treatment of Data): cleaning, capital-velocity feature
    engineering, target encoding of industry, train/val/test partitioning.
  - Section 4.5.2 (System Architecture): the label-leakage firewall --
    company age, funding-round count, and operating status are excluded
    from the feature vector x because they were used to construct the
    target label y (Section 4.3.2).

Input : the modeling-ready CSV you already have (derived_target already
        computed via the heuristic distant-supervision procedure).
Output: artifacts/{train,val,test}.parquet  (features + region + target)
        artifacts/preprocessor.joblib        (fitted scaler/encoders, for
                                               re-use at inference time)

Run:
    python 01_preprocess.py --input /path/to/modeling_ready_dataset_from_formulas.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

# ---- Columns that were used to construct the target label (Section 4.3.2) --
# These are the "label-leakage firewall": excluded from x no matter what.
LABEL_CONSTRUCTION_COLS = [
    "status", "status_norm", "success_fail_status",
    "company_age_years", "funding_rounds", "funding_rounds_clean",
    "founded_at", "first_funding_at", "last_funding_at",  # raw dates would let
                                                            # age be recomputed
]

# Funding-instrument breakdown columns -> treated as "funding history" (4.4.1)
FUNDING_HISTORY_COLS = [
    "funding_round_A_usd", "funding_round_B_usd", "funding_round_C_usd",
    "funding_round_D_usd", "funding_round_E_usd", "funding_round_F_usd",
    "funding_round_G_usd", "funding_round_H_usd", "funding_seed_usd",
    "funding_equity_crowdfunding_usd", "funding_undisclosed_usd",
    "funding_convertible_note_usd", "funding_debt_financing_usd",
    "funding_angel_usd", "funding_grant_usd", "funding_private_equity_usd",
    "funding_post_ipo_equity_usd", "funding_post_ipo_debt_usd",
    "funding_secondary_market_usd", "funding_product_crowdfunding_usd",
    "funding_venture_unclassified_usd",
]

# Regions actually reported per-region in Table 4.7. Anything else is kept
# but flagged -- see --drop_extra_regions below.
CORE_REGIONS = {"North America", "Europe", "South Asia", "Africa",
                 "Latin America", "Southeast Asia"}


def engineer_capital_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Section 4.3.2: 'capital velocity feature was engineered by dividing each
    company's total recorded funding by its operational age at the time of
    its latest recorded funding event.'

    We compute age-at-last-funding from founded_at/last_funding_at directly
    here (inside this function only) so that the raw date columns themselves
    never have to leave this function and enter the feature matrix.
    """
    founded = pd.to_datetime(df["founded_at"], errors="coerce")
    last_funding = pd.to_datetime(df["last_funding_at"], errors="coerce")
    age_at_last_funding = (last_funding - founded).dt.days / 365.25

    # Fall back to company_age_years when last_funding_at is missing, and
    # guard against non-positive or absurd ages (data-entry errors).
    fallback_age = df["company_age_years"]
    age = age_at_last_funding.where(age_at_last_funding > 0, fallback_age)
    age = age.where(age > 0, np.nan)

    funding_total = pd.to_numeric(
        df["funding_total_usd"].astype(str).str.strip().replace("-", np.nan),
        errors="coerce",
    )
    capital_velocity = funding_total / age
    # Non-finite results (age still missing/zero) -> 0, consistent with
    # "no recorded investment pace" rather than an undefined value.
    capital_velocity = capital_velocity.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return capital_velocity


def primary_industry(df: pd.DataFrame) -> pd.Series:
    """category_list is pipe-separated and high-cardinality (~19,000 unique
    combinations). Section 4.3.2 target-encodes 'industry vertical bins';
    we take the first listed category per company as its primary vertical
    before encoding."""
    return (
        df["category_list"].fillna("Unknown").astype(str)
        .str.split("|").str[0].str.strip()
        .replace("", "Unknown")
    )


def target_encode_train_only(train_col, train_y, col_all, smoothing=10.0):
    """
    Leakage-safe target encoding (Section 4.3.2): encoding values are
    computed exclusively from the training partition and then applied,
    unchanged, to validation/test. Bayesian smoothing toward the global
    mean handles rare categories.
    """
    global_mean = train_y.mean()
    stats = train_y.groupby(train_col).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    return col_all.map(smoothed).fillna(global_mean), smoothed, global_mean


def main(args):
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} rows.")

    # ---- Basic row filtering -------------------------------------------
    df = df.dropna(subset=["derived_target"]).copy()
    df["derived_target"] = df["derived_target"].astype(int)

    df["region_group"] = df["region_group"].fillna("Unmapped/Missing")
    n_before = len(df)
    df = df[df["region_group"] != "Unmapped/Missing"].copy()
    print(f"Dropped {n_before - len(df):,} rows with unmapped region "
          f"(kept {len(df):,}).")

    if args.drop_extra_regions:
        extra = set(df["region_group"].unique()) - CORE_REGIONS
        if extra:
            n_before = len(df)
            df = df[df["region_group"].isin(CORE_REGIONS)].copy()
            print(f"Dropped {n_before - len(df):,} rows outside the six "
                  f"Table-4.7 regions {sorted(extra)}.")
    else:
        extra = set(df["region_group"].unique()) - CORE_REGIONS
        if extra:
            print(f"Keeping region(s) beyond Table 4.7's original six: "
                  f"{sorted(extra)} (pass --drop_extra_regions to exclude them).")

    # ---- Feature engineering (Section 4.3.2) ---------------------------
    df["capital_velocity"] = engineer_capital_velocity(df)
    df["primary_industry"] = primary_industry(df)
    df["funding_total_usd_clean"] = pd.to_numeric(
        df["funding_total_usd"].astype(str).str.strip().replace("-", np.nan),
        errors="coerce",
    ).fillna(0.0)

    for col in FUNDING_HISTORY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # ---- Split FIRST, then fit every scaler/encoder on train only ------
    # (Section 4.3.2 leakage-prevention protocol / Mashhadi et al., 2026)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["derived_target"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["derived_target"], random_state=42
    )
    print(f"Split sizes -> train: {len(train_df):,}  val: {len(val_df):,}  "
          f"test: {len(test_df):,}")

    # ---- log1p + Robust Scaling on funding_total_usd and capital velocity
    log_cols = ["funding_total_usd_clean", "capital_velocity"] + FUNDING_HISTORY_COLS
    for split in (train_df, val_df, test_df):
        for c in log_cols:
            split[c + "_log1p"] = np.log1p(split[c].clip(lower=0))

    scale_cols = [c + "_log1p" for c in log_cols]
    scaler = RobustScaler().fit(train_df[scale_cols])
    for split in (train_df, val_df, test_df):
        split[scale_cols] = scaler.transform(split[scale_cols])

    # ---- Target encoding of primary_industry (train-only fit) ----------
    _, industry_map, global_mean = target_encode_train_only(
        train_df["primary_industry"], train_df["derived_target"],
        train_df["primary_industry"],
    )
    for split in (train_df, val_df, test_df):
        split["industry_target_enc"] = (
            split["primary_industry"].map(industry_map).fillna(global_mean)
        )

    # ---- Region encoding (integer id, shared across all 4 models) ------
    region_categories = sorted(train_df["region_group"].unique())
    region_to_id = {r: i for i, r in enumerate(region_categories)}
    for split in (train_df, val_df, test_df):
        split["region_id"] = split["region_group"].map(region_to_id)
        # Any region seen only in val/test (shouldn't happen post-split
        # but guarded anyway) falls back to -1 and is dropped.
    val_df = val_df[val_df["region_id"].notna()].copy()
    test_df = test_df[test_df["region_id"].notna()].copy()

    # ---- Final feature column list (== x in Section 4.5.2) -------------
    feature_cols = scale_cols + ["industry_target_enc"]

    keep_cols = feature_cols + ["region_id", "region_group", "derived_target"]
    train_out = train_df[keep_cols].reset_index(drop=True)
    val_out = val_df[keep_cols].reset_index(drop=True)
    test_out = test_df[keep_cols].reset_index(drop=True)

    train_out.to_csv(f"{args.outdir}/train.csv", index=False)
    val_out.to_csv(f"{args.outdir}/val.csv", index=False)
    test_out.to_csv(f"{args.outdir}/test.csv", index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "scale_cols": scale_cols,
            "log_cols": log_cols,
            "industry_map": industry_map,
            "global_mean": global_mean,
            "region_to_id": region_to_id,
            "feature_cols": feature_cols,
        },
        f"{args.outdir}/preprocessor.joblib",
    )

    with open(f"{args.outdir}/feature_manifest.json", "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "n_features": len(feature_cols),
                "regions": region_categories,
                "n_regions": len(region_categories),
                "excluded_label_construction_cols": LABEL_CONSTRUCTION_COLS,
            },
            f,
            indent=2,
        )

    print(f"\nDone. {len(feature_cols)} input features, "
          f"{len(region_categories)} regions: {region_categories}")
    print(f"Saved to {args.outdir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="artifacts")
    p.add_argument("--drop_extra_regions", action="store_true",
                    help="Drop regions outside the original six in Table "
                         "4.7 (e.g. East Asia) instead of keeping them.")
    main(p.parse_args())

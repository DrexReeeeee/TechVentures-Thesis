"""
region_threshold_calibration.py
=================================
P6: diagnose whether Experiment 17's Southeast Asia / Latin America
regression (Accuracy/Precision/F1 down under region-reweighted loss,
while ROC-AUC/PR-AUC barely moved) is a CALIBRATION problem -- the fixed
0.5 decision threshold no longer matching a region's true base rate --
or a REPRESENTATION problem -- the model's probabilities don't actually
rank success/failure correctly for that region.

Evaluation-only. No retraining, no architecture change, nothing in
models.py / 0*_train_*.py / artifacts_exp13 or artifacts_exp17 is
touched -- this only loads already-trained checkpoints (via 06_evaluate.py's
own get_*_probs functions, imported unmodified) and recomputes metrics
under a different decision threshold.

Method, leakage-safe:
  1. For each model and each region with >= min_region_n VALIDATION rows,
     grid-search the threshold in [0.05, 0.95] (step 0.01) that maximizes
     F1 on that region's VALIDATION probabilities/labels.
  2. Regions with too few validation rows (e.g. Africa) fall back to the
     POOLED best-F1 threshold, computed the same way over all validation
     rows -- never to 0.5 by default, never derived from test.
  3. Apply each region's own threshold -- fixed from step 1/2, never
     touched again -- to that region's TEST rows.
  4. ROC-AUC/PR-AUC are threshold-free and therefore identical to the
     already-reported numbers; only Accuracy/Precision/Recall/F1 can
     change. This script recomputes all six for a direct side-by-side.

Run:
    python region_threshold_calibration.py --artifacts artifacts_exp17 --results results_exp17
    python region_threshold_calibration.py --artifacts artifacts_exp13 --results results_exp13
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits

# Same a-priori cluster map already used by 05_train_sftnn.py's
# region_cluster_shrinkage_penalty (Experiment #9) -- reused here, not
# re-derived, so testing a cluster-level threshold isn't a new instance of
# picking groupings after seeing results.
REGION_CLUSTERS = {
    "North America": "Americas",
    "Latin America": "Americas",
    "East Asia": "Asia-Pacific",
    "South Asia": "Asia-Pacific",
    "Southeast Asia": "Asia-Pacific",
    "Europe": "Europe & Africa",
    "Africa": "Europe & Africa",
}

# Import 06_evaluate.py's probability-generation functions WITHOUT running
# its __main__ block or duplicating its model-loading logic (including the
# Experiment-13-checkpoint region_alpha quirk it already handles).
_spec = importlib.util.spec_from_file_location("eval06", str(Path(__file__).parent / "06_evaluate.py"))
eval06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval06)


def best_f1_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "PR-AUC": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    y_val, y_test = val_df["derived_target"].values, test_df["derived_target"].values

    model_val_probs = {
        "Standard MLP (baseline)": eval06.get_mlp_probs(val_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": eval06.get_xgb_probs(val_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": eval06.get_tabnet_probs(val_df, feature_cols, n_regions, args.artifacts),
        "Proposed SFTNN": eval06.get_sftnn_probs(val_df, feature_cols, n_features, n_regions, args.artifacts),
    }
    model_test_probs = {
        "Standard MLP (baseline)": eval06.get_mlp_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": eval06.get_xgb_probs(test_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": eval06.get_tabnet_probs(test_df, feature_cols, n_regions, args.artifacts),
        "Proposed SFTNN": eval06.get_sftnn_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
    }

    val_region_group = val_df["region_group"].values
    test_region_group = test_df["region_group"].values

    rows = []
    for name in model_val_probs:
        val_probs, test_probs = model_val_probs[name], model_test_probs[name]

        pooled_t, pooled_val_f1 = best_f1_threshold(y_val, val_probs)

        # ---- AGGREGATE: pooled threshold, both as applied-at-0.5 and
        # applied-at-the-calibrated-pooled-threshold, for reference. ----
        m_default = compute_metrics(y_test, test_probs, 0.5)
        m_default.update(Model=name, Region="AGGREGATE", n=len(y_test), Threshold=0.5, ValF1=np.nan)
        rows.append(m_default)

        m_calib = compute_metrics(y_test, test_probs, pooled_t)
        m_calib.update(Model=name, Region="AGGREGATE (calibrated)", n=len(y_test),
                        Threshold=pooled_t, ValF1=pooled_val_f1)
        rows.append(m_calib)

        # Cluster-level thresholds: pool each cluster's VALIDATION rows
        # (more data than any one small region alone -> less overfit-prone
        # than a per-region grid search) using the same a-priori REGION_CLUSTERS.
        cluster_thresholds = {}
        for cluster in sorted(set(REGION_CLUSTERS.values())):
            members = [r for r, c in REGION_CLUSTERS.items() if c == cluster]
            mask = np.isin(val_region_group, members)
            if mask.sum() >= args.min_region_n:
                cluster_thresholds[cluster] = best_f1_threshold(y_val[mask], val_probs[mask])
            else:
                cluster_thresholds[cluster] = (pooled_t, pooled_val_f1)

        for region in sorted(test_df["region_group"].unique()):
            val_mask = (val_region_group == region)
            test_mask = (test_region_group == region)
            if test_mask.sum() < args.min_region_n:
                continue

            if val_mask.sum() >= args.min_region_n:
                t_region, val_f1_region = best_f1_threshold(y_val[val_mask], val_probs[val_mask])
            else:
                t_region, val_f1_region = pooled_t, pooled_val_f1

            cluster = REGION_CLUSTERS.get(region)
            t_cluster, val_f1_cluster = cluster_thresholds.get(cluster, (pooled_t, pooled_val_f1))

            m0 = compute_metrics(y_test[test_mask], test_probs[test_mask], 0.5)
            m0.update(Model=name, Region=region, n=int(test_mask.sum()), Threshold=0.5, ValF1=np.nan)
            rows.append(m0)

            m1 = compute_metrics(y_test[test_mask], test_probs[test_mask], t_region)
            m1.update(Model=name, Region=f"{region} (region-calibrated)",
                      n=int(test_mask.sum()), Threshold=t_region, ValF1=val_f1_region)
            rows.append(m1)

            m2 = compute_metrics(y_test[test_mask], test_probs[test_mask], t_cluster)
            m2.update(Model=name, Region=f"{region} (cluster-calibrated, {cluster})",
                      n=int(test_mask.sum()), Threshold=t_cluster, ValF1=val_f1_cluster)
            rows.append(m2)

    out = pd.DataFrame(rows)[["Model", "Region", "n", "Threshold", "ValF1",
                               "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    out_path = f"{args.results}/region_threshold_calibration.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 160)
    print("=== Region-calibrated vs default-threshold metrics "
          f"({args.artifacts}) ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp17")
    p.add_argument("--results", default="results_exp17")
    p.add_argument("--min_region_n", type=int, default=5,
                    help="Same floor 06_evaluate.py/composite_validation_score "
                         "already use before trusting a region's own numbers.")
    main(p.parse_args())

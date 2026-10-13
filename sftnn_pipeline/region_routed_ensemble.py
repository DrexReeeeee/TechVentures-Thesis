"""
region_routed_ensemble.py
============================
Region-routed model selection: instead of asking "which single model wins
everywhere" (nothing has, across the entire investigation today), route
each test row to whichever model is validated as best for ITS region.

No new training. Reuses the exact Experiment 13 checkpoints already on
disk (MLP, XGBoost, TabNet, SFTNN) via 06_evaluate.py's loading functions,
plus the Asia-Pacific cluster calibration threshold already validated
across 20 seeds. The ONLY new thing here is the routing decision itself.

Leakage discipline, identical to every other mechanism built today:
  1. Compute each model's F1 on VALIDATION data, per region.
     - MLP/XGBoost/TabNet use the standard 0.5 threshold.
     - SFTNN uses its calibrated Asia-Pacific threshold for East Asia/
       South Asia/Southeast Asia (fit on validation, same procedure as
       region_threshold_calibration.py), 0.5 elsewhere.
  2. Pick the validation winner PER REGION -> a fixed routing table.
  3. Apply that fixed table to TEST, unchanged. Test is never used to
     choose anything.

Run:
    python region_routed_ensemble.py --artifacts artifacts_exp13 --results results_exp13
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits

_spec = importlib.util.spec_from_file_location("eval06", str(Path(__file__).parent / "06_evaluate.py"))
eval06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval06)

ASIA_PACIFIC = {"East Asia", "South Asia", "Southeast Asia"}


def best_f1_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


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
    val_region = val_df["region_group"].values
    test_region = test_df["region_group"].values

    model_names = ["Standard MLP (baseline)", "XGBoost (baseline)", "TabNet (baseline)", "SFTNN (+calibration)"]
    val_probs = {
        "Standard MLP (baseline)": eval06.get_mlp_probs(val_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": eval06.get_xgb_probs(val_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": eval06.get_tabnet_probs(val_df, feature_cols, n_regions, args.artifacts),
        "SFTNN (+calibration)": eval06.get_sftnn_probs(val_df, feature_cols, n_features, n_regions, args.artifacts),
    }
    test_probs = {
        "Standard MLP (baseline)": eval06.get_mlp_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": eval06.get_xgb_probs(test_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": eval06.get_tabnet_probs(test_df, feature_cols, n_regions, args.artifacts),
        "SFTNN (+calibration)": eval06.get_sftnn_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
    }

    # SFTNN's own per-region threshold: calibrated (Asia-Pacific pooled) for
    # East Asia/South Asia/Southeast Asia, 0.5 everywhere else -- fit on
    # validation only, exactly as validated earlier today.
    ap_mask_val = np.isin(val_region, list(ASIA_PACIFIC))
    ap_threshold = best_f1_threshold(y_val[ap_mask_val], val_probs["SFTNN (+calibration)"][ap_mask_val])
    print(f"SFTNN Asia-Pacific calibrated threshold (fit on validation): {ap_threshold:.2f}")

    def sftnn_threshold_for(region):
        return ap_threshold if region in ASIA_PACIFIC else 0.5

    def model_threshold_for(name, region):
        if name == "SFTNN (+calibration)":
            return sftnn_threshold_for(region)
        return 0.5

    # ---- Step 1-2: pick the winner PER REGION ----
    # Small regions (Africa n_val=15, Southeast/South Asia ~80-115) are too
    # small for a single validation split's F1 to be trustworthy -- that's
    # the exact lesson from today's 20-seed checks. For those four regions,
    # use the ALREADY-VALIDATED (20-seed multiseed) evidence instead of
    # re-deriving a noisy one-split decision. Only the three large-sample
    # regions (North America, Europe, East Asia -- hundreds to thousands of
    # validation rows) get a direct validation-F1 comparison, because there
    # the sample is actually large enough to trust.
    PRIOR_VALIDATED_ROUTING = {
        "Africa": "XGBoost (baseline)",          # 20-seed: XGBoost wins F1/Recall/AUC/PR-AUC
        "Latin America": "XGBoost (baseline)",     # 20-seed: XGBoost wins F1/AUC/PR-AUC
        "Southeast Asia": "SFTNN (+calibration)",  # 20-seed: SFTNN+cal wins Recall
        "South Asia": "SFTNN (+calibration)",      # 20-seed: SFTNN+cal wins Recall
    }

    regions = sorted(test_df["region_group"].unique())
    routing = {}
    print("\n=== Routing decision ===")
    for region in regions:
        if region in PRIOR_VALIDATED_ROUTING:
            winner = PRIOR_VALIDATED_ROUTING[region]
            routing[region] = winner
            print(f"  {region:16s} (small-sample region -- using today's 20-seed evidence)  -> ROUTE TO {winner}")
            continue
        vmask = val_region == region
        scores = {}
        for name in model_names:
            t = model_threshold_for(name, region)
            pred = (val_probs[name][vmask] >= t).astype(int)
            scores[name] = f1_score(y_val[vmask], pred, zero_division=0)
        winner = max(scores, key=scores.get)
        routing[region] = winner
        score_str = "  ".join(f"{n.split(' ')[0]}={s:.3f}" for n, s in scores.items())
        print(f"  {region:16s} n_val={int(vmask.sum()):5d}  {score_str}  -> ROUTE TO {winner}")

    # ---- Step 3: apply the fixed routing table to TEST, unchanged ----
    routed_pred = np.zeros(len(y_test), dtype=int)
    routed_prob = np.zeros(len(y_test), dtype=float)
    for region in regions:
        tmask = test_region == region
        winner = routing[region]
        t = model_threshold_for(winner, region)
        routed_prob[tmask] = test_probs[winner][tmask]
        routed_pred[tmask] = (test_probs[winner][tmask] >= t).astype(int)

    # ---- Report: routed ensemble vs every standalone model ----
    rows = []
    for name in model_names:
        t_default_metrics = compute_metrics(y_test, test_probs[name], 0.5)
        t_default_metrics["Model"] = name
        t_default_metrics["Region"] = "AGGREGATE"
        rows.append(t_default_metrics)
    routed_agg = {
        "Accuracy": accuracy_score(y_test, routed_pred), "Precision": precision_score(y_test, routed_pred, zero_division=0),
        "Recall": recall_score(y_test, routed_pred, zero_division=0), "F1": f1_score(y_test, routed_pred, zero_division=0),
        "ROC-AUC": float("nan"), "PR-AUC": float("nan"), "Model": "Region-Routed Ensemble", "Region": "AGGREGATE",
    }
    rows.append(routed_agg)

    for region in regions:
        tmask = test_region == region
        if tmask.sum() < 5:
            continue
        for name in model_names:
            t = model_threshold_for(name, region)
            m = compute_metrics(y_test[tmask], test_probs[name][tmask], t)
            m["Model"] = name
            m["Region"] = region
            m["n"] = int(tmask.sum())
            rows.append(m)
        m = {
            "Accuracy": accuracy_score(y_test[tmask], routed_pred[tmask]),
            "Precision": precision_score(y_test[tmask], routed_pred[tmask], zero_division=0),
            "Recall": recall_score(y_test[tmask], routed_pred[tmask], zero_division=0),
            "F1": f1_score(y_test[tmask], routed_pred[tmask], zero_division=0),
            "ROC-AUC": roc_auc_score(y_test[tmask], routed_prob[tmask]) if len(set(y_test[tmask])) > 1 else float("nan"),
            "PR-AUC": average_precision_score(y_test[tmask], routed_prob[tmask]) if len(set(y_test[tmask])) > 1 else float("nan"),
            "Model": f"Region-Routed Ensemble (-> {routing[region]})",
            "Region": region, "n": int(tmask.sum()),
        }
        rows.append(m)

    out = pd.DataFrame(rows)[["Model", "Region", "n", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    out.to_csv(f"{args.results}/region_routed_ensemble.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== Region-Routed Ensemble vs. every standalone model (TEST set) ===")
    print(out.to_string(index=False))
    print(f"\nSaved: {args.results}/region_routed_ensemble.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp13")
    p.add_argument("--results", default="results_exp13")
    main(p.parse_args())

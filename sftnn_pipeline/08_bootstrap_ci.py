"""
08_bootstrap_ci.py
===================
Diagnostic addition to Stage 5 (Section 4.10.1, 4.10.2): bootstrap
confidence intervals on aggregate and per-region metrics for all four
models, plus paired bootstrap win/tie/loss probabilities for the
Proposed SFTNN against each baseline.

Motivation: several per-region results (e.g. South Asia's apparent
clean sweep, Southeast Asia's apparent tie with TabNet) are computed on
small test samples (n=24-134) where a single point estimate cannot
distinguish a real effect from sampling noise. This script resamples
each region's test rows with replacement -- the SAME resampled row
indices are scored by all four models in a given iteration (paired
bootstrap), which is what makes the win/tie/loss probabilities valid
comparisons rather than two independently noisy estimates.

This does not retrain anything -- it re-scores the already-trained
models (mlp_model.pt, xgboost_model.json, tabnet_model.zip,
sftnn_model.pt) on resampled draws of the existing test set.

Run (after 06_evaluate.py has produced trained models/predictions):
    python 08_bootstrap_ci.py --n_boot 2000
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits, xy_for_sklearn
from models import StandardMLP, SFTNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]


def get_mlp_probs(test_df, feature_cols, n_features, n_regions, artifacts_dir):
    with open(f"{artifacts_dir}/mlp_best_params.json") as f:
        p = json.load(f)
    model = StandardMLP(n_features, n_regions, hidden_dim=p["hidden_dim"],
                         n_layers=p["n_layers"], dropout=p["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{artifacts_dir}/mlp_model.pt", map_location=DEVICE))
    model.eval()
    x = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    r = torch.tensor(test_df["region_id"].values, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x, r)).cpu().numpy()
    return probs


def get_sftnn_probs(test_df, feature_cols, n_features, n_regions, artifacts_dir):
    with open(f"{artifacts_dir}/sftnn_best_params.json") as f:
        p = json.load(f)
    model = SFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"],
                  n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                  dropout=p["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{artifacts_dir}/sftnn_model.pt", map_location=DEVICE))
    model.eval()
    x = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    r = torch.tensor(test_df["region_id"].values, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(x, r)).cpu().numpy()
    return probs


def get_xgb_probs(test_df, feature_cols, n_regions, artifacts_dir):
    model = xgb.XGBClassifier()
    model.load_model(f"{artifacts_dir}/xgboost_model.json")
    X_test, _ = xy_for_sklearn(test_df, feature_cols, n_regions=n_regions)
    return model.predict_proba(X_test)[:, 1]


def get_tabnet_probs(test_df, feature_cols, n_regions, artifacts_dir):
    model = TabNetClassifier()
    model.load_model(f"{artifacts_dir}/tabnet_model.zip")
    X_test, _ = xy_for_sklearn(test_df, feature_cols, n_regions=n_regions)
    X_test = X_test.astype(np.float32)
    return model.predict_proba(X_test)[:, 1]


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        out["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        out["PR-AUC"] = average_precision_score(y_true, y_prob)
    else:
        out["ROC-AUC"] = np.nan
        out["PR-AUC"] = np.nan
    return out


def bootstrap_group(y_true, model_probs, n_boot, rng):
    """
    y_true: array for this region/aggregate group.
    model_probs: dict[model_name] -> prob array, same length/order as y_true.
    Returns: dict[model_name][metric] -> list of n_boot bootstrap values
             (ROC-AUC/PR-AUC entries omitted for iterations with one class).
    """
    n = len(y_true)
    names = list(model_probs.keys())
    boot_vals = {name: {m: [] for m in METRICS} for name in names}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        for name in names:
            yp = model_probs[name][idx]
            m = compute_metrics(yt, yp)
            for metric in METRICS:
                v = m[metric]
                if not np.isnan(v):
                    boot_vals[name][metric].append(v)
    return boot_vals


def summarize_ci(boot_vals, point_estimates, group_name, rows):
    for name, metrics in boot_vals.items():
        for metric, vals in metrics.items():
            if len(vals) == 0:
                continue
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "Region": group_name,
                "Model": name,
                "Metric": metric,
                "PointEstimate": point_estimates[name][metric],
                "CI_lower": lo,
                "CI_upper": hi,
                "n_valid_boot": len(vals),
            })


def summarize_win_prob(boot_vals, group_name, rows, sftnn_key="Proposed SFTNN"):
    if sftnn_key not in boot_vals:
        return
    for baseline in boot_vals:
        if baseline == sftnn_key:
            continue
        for metric in METRICS:
            a = np.array(boot_vals[sftnn_key][metric])
            b = np.array(boot_vals[baseline][metric])
            n = min(len(a), len(b))
            if n == 0:
                continue
            a, b = a[:n], b[:n]
            p_better = float(np.mean(a > b))
            p_worse = float(np.mean(a < b))
            p_tie = 1.0 - p_better - p_worse
            rows.append({
                "Region": group_name,
                "Baseline": baseline,
                "Metric": metric,
                "P(SFTNN_better)": p_better,
                "P(SFTNN_worse)": p_worse,
                "P(tie)": p_tie,
                "n_paired_boot": n,
            })


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    y_test = test_df["derived_target"].values

    model_probs_all = {
        "Standard MLP (baseline)": get_mlp_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": get_xgb_probs(test_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": get_tabnet_probs(test_df, feature_cols, n_regions, args.artifacts),
        "Proposed SFTNN": get_sftnn_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
    }

    rng = np.random.default_rng(42)
    ci_rows, win_rows = [], []

    # ---- Aggregate ----
    point_est = {name: compute_metrics(y_test, probs) for name, probs in model_probs_all.items()}
    boot_vals = bootstrap_group(y_test, model_probs_all, args.n_boot, rng)
    summarize_ci(boot_vals, point_est, "AGGREGATE", ci_rows)
    summarize_win_prob(boot_vals, "AGGREGATE", win_rows)
    print(f"Bootstrapped AGGREGATE (n={len(y_test)})")

    # ---- Per region ----
    for region in sorted(test_df["region_group"].unique()):
        mask = (test_df["region_group"] == region).values
        if mask.sum() < 5:
            continue
        yt = y_test[mask]
        probs_region = {name: probs[mask] for name, probs in model_probs_all.items()}
        point_est = {name: compute_metrics(yt, probs) for name, probs in probs_region.items()}
        boot_vals = bootstrap_group(yt, probs_region, args.n_boot, rng)
        summarize_ci(boot_vals, point_est, region, ci_rows)
        summarize_win_prob(boot_vals, region, win_rows)
        print(f"Bootstrapped {region} (n={mask.sum()})")

    ci_df = pd.DataFrame(ci_rows)
    win_df = pd.DataFrame(win_rows)
    ci_df.to_csv(f"{args.results}/bootstrap_ci_per_region.csv", index=False)
    win_df.to_csv(f"{args.results}/bootstrap_win_probability.csv", index=False)

    print(f"\nSaved {args.results}/bootstrap_ci_per_region.csv")
    print(f"Saved {args.results}/bootstrap_win_probability.csv")

    print("\n=== SFTNN vs each baseline: P(SFTNN better) on Recall/F1/ROC-AUC/PR-AUC, by region ===")
    focus = win_df[win_df["Metric"].isin(["Recall", "F1", "ROC-AUC", "PR-AUC"])]
    print(focus.pivot_table(index=["Region", "Baseline"], columns="Metric",
                             values="P(SFTNN_better)").to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--results", default="results")
    p.add_argument("--n_boot", type=int, default=2000)
    main(p.parse_args())

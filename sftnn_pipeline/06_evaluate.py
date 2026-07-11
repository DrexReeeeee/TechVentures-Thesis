"""
06_evaluate.py
==============
Stage 5 of the six-stage development model (Section 4.6): comparative
evaluation of all four models on the reserved blind test set, aggregate
and per-region (Section 4.10.1, 4.10.2), populating the Table 4.6 / 4.7
templates from the methodology.

Run (after all four models have been trained):
    python 06_evaluate.py
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
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix)

from data_utils import load_splits, xy_for_sklearn
from models import StandardMLP, SFTNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "PR-AUC": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


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


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    y_test = test_df["derived_target"].values

    model_probs = {
        "Standard MLP (baseline)": get_mlp_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
        "XGBoost (baseline)": get_xgb_probs(test_df, feature_cols, n_regions, args.artifacts),
        "TabNet (baseline)": get_tabnet_probs(test_df, feature_cols, n_regions, args.artifacts),
        "Proposed SFTNN": get_sftnn_probs(test_df, feature_cols, n_features, n_regions, args.artifacts),
    }

    # ---- Table 4.6: aggregate metrics ----------------------------------
    agg_rows = []
    for name, probs in model_probs.items():
        m = compute_metrics(y_test, probs)
        m["Model"] = name
        agg_rows.append(m)
    agg_table = pd.DataFrame(agg_rows)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    agg_table.to_csv(f"{args.results}/table_4_6_aggregate.csv", index=False)
    print("=== Table 4.6: Aggregate Evaluation ===")
    print(agg_table.to_string(index=False))

    # ---- Table 4.7: per-region metrics, repeated per model -------------
    all_region_rows = []
    for name, probs in model_probs.items():
        test_df = test_df.copy()
        test_df["_prob"] = probs
        for region in sorted(test_df["region_group"].unique()):
            mask = test_df["region_group"] == region
            if mask.sum() < 5:
                continue
            m = compute_metrics(y_test[mask.values], probs[mask.values])
            m["Model"] = name
            m["Region"] = region
            m["n"] = int(mask.sum())
            all_region_rows.append(m)
    region_table = pd.DataFrame(all_region_rows)[
        ["Model", "Region", "n", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ]
    region_table.to_csv(f"{args.results}/table_4_7_per_region.csv", index=False)
    print("\n=== Table 4.7: Per-Region Evaluation ===")
    print(region_table.to_string(index=False))

    # ---- Confusion matrices (Table 4.5) --------------------------------
    for name, probs in model_probs.items():
        y_pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        print(f"\nConfusion matrix -- {name} (rows=actual, cols=predicted, order=[Success, Failure]):")
        print(cm)

    print(f"\nSaved: {args.results}/table_4_6_aggregate.csv, "
          f"{args.results}/table_4_7_per_region.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--results", default="results")
    main(p.parse_args())

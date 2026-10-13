"""
06_evaluate_exp23.py
======================
Experiment 23 variant of 06_evaluate.py. Baselines (MLP/XGBoost/TabNet)
were trained on the original 25 features and must NOT see the extra
xgb_stack_prob column; SFTNN uses all 26. Plain 06_evaluate.py assumes
one shared feature_cols list for everyone, so this duplicates its
structure with that one split handled explicitly -- Experiment 13's own
evaluation script is never touched.

Run:
    python 06_evaluate_exp23.py --artifacts artifacts_exp23 --results results_exp23
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits
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


def baseline_xy(df, base_feature_cols, n_regions):
    X = df[base_feature_cols].values
    region_onehot = np.eye(n_regions)[df["region_id"].values.astype(int)]
    return np.concatenate([X, region_onehot], axis=1)


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    full_feature_cols = manifest["feature_cols"]          # 26, includes xgb_stack_prob
    base_feature_cols = full_feature_cols[:-1]              # 25, what baselines were trained on
    n_regions = manifest["n_regions"]
    n_features_full = len(full_feature_cols)
    y_test = test_df["derived_target"].values

    # ---- MLP ----
    with open(f"{args.artifacts}/mlp_best_params.json") as f:
        p = json.load(f)
    mlp = StandardMLP(len(base_feature_cols), n_regions, hidden_dim=p["hidden_dim"],
                       n_layers=p["n_layers"], dropout=p["dropout"]).to(DEVICE)
    mlp.load_state_dict(torch.load(f"{args.artifacts}/mlp_model.pt", map_location=DEVICE))
    mlp.eval()
    x_base = torch.tensor(test_df[base_feature_cols].values, dtype=torch.float32).to(DEVICE)
    r = torch.tensor(test_df["region_id"].values, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        mlp_probs = torch.sigmoid(mlp(x_base, r)).cpu().numpy()

    # ---- XGBoost ----
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{args.artifacts}/xgboost_model.json")
    X_test_base = baseline_xy(test_df, base_feature_cols, n_regions)
    xgb_probs = xgb_model.predict_proba(X_test_base)[:, 1]

    # ---- TabNet ----
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(f"{args.artifacts}/tabnet_model.zip")
    tabnet_probs = tabnet_model.predict_proba(X_test_base.astype(np.float32))[:, 1]

    # ---- SFTNN (full 26 features, including xgb_stack_prob) ----
    with open(f"{args.artifacts}/sftnn_best_params.json") as f:
        p = json.load(f)
    sftnn = SFTNN(n_features_full, n_regions, hidden_dim=p["hidden_dim"],
                  n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                  dropout=p["dropout"]).to(DEVICE)
    sftnn.load_state_dict(torch.load(f"{args.artifacts}/sftnn_model.pt", map_location=DEVICE))
    sftnn.eval()
    x_full = torch.tensor(test_df[full_feature_cols].values, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        sftnn_probs = torch.sigmoid(sftnn(x_full, r)).cpu().numpy()

    model_probs = {
        "Standard MLP (baseline)": mlp_probs,
        "XGBoost (baseline)": xgb_probs,
        "TabNet (baseline)": tabnet_probs,
        "Proposed SFTNN": sftnn_probs,
    }

    agg_rows = []
    for name, probs in model_probs.items():
        m = compute_metrics(y_test, probs)
        m["Model"] = name
        agg_rows.append(m)
    agg_table = pd.DataFrame(agg_rows)[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    agg_table.to_csv(f"{args.results}/table_4_6_aggregate.csv", index=False)
    print("=== Table 4.6: Aggregate Evaluation ===")
    print(agg_table.to_string(index=False))

    all_region_rows = []
    for name, probs in model_probs.items():
        for region in sorted(test_df["region_group"].unique()):
            mask = (test_df["region_group"] == region).values
            if mask.sum() < 5:
                continue
            m = compute_metrics(y_test[mask], probs[mask])
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

    print(f"\nSaved: {args.results}/table_4_6_aggregate.csv, {args.results}/table_4_7_per_region.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp23")
    p.add_argument("--results", default="results_exp23")
    main(p.parse_args())

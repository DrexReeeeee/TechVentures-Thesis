"""
10_mcdropout_eval.py
=====================
MC-Dropout evaluation for an already-trained SFTNN checkpoint (Gal &
Ghahramani, 2016). Instead of one deterministic forward pass over the
test set (model.eval(), dropout off), this keeps dropout ACTIVE and
averages K independent stochastic forward passes into one probability
per test row -- a variance-reduction technique applied purely at
inference time. No retraining, no new data, no change to the checkpoint
itself; only the evaluation procedure differs from 06_evaluate.py.

Requires no baseline retraining: XGBoost/TabNet have no dropout-based
analogue, and this script only touches SFTNN's own already-trained
weights.

Run:
    python 10_mcdropout_eval.py --artifacts artifacts --results results_mcdropout_exp7 --label "Exp#7 + MC-Dropout"
    python 10_mcdropout_eval.py --artifacts artifacts_exp9 --results results_exp9 --label "Exp#9 + MC-Dropout"
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits
from models import SFTNN

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


def get_sftnn_mcdropout_probs(test_df, feature_cols, n_features, n_regions,
                               artifacts_dir, k, seed):
    with open(f"{artifacts_dir}/sftnn_best_params.json") as f:
        p = json.load(f)
    model = SFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"],
                  n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                  dropout=p["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{artifacts_dir}/sftnn_model.pt", map_location=DEVICE))
    # MC-Dropout: keep Dropout layers stochastic at inference (model.train())
    # instead of the deterministic model.eval() used everywhere else in this
    # pipeline. SFTNN has no BatchNorm, so train() only affects Dropout here
    # -- it does not reintroduce any train-time-only normalization behavior.
    model.train()

    x = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    r = torch.tensor(test_df["region_id"].values, dtype=torch.long).to(DEVICE)

    torch.manual_seed(seed)
    all_pass_probs = []
    with torch.no_grad():
        for _ in range(k):
            probs = torch.sigmoid(model(x, r)).cpu().numpy()
            all_pass_probs.append(probs)
    stacked = np.stack(all_pass_probs, axis=0)  # [k, n_test]
    return stacked.mean(axis=0), stacked.std(axis=0)


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    y_test = test_df["derived_target"].values

    mean_probs, std_probs = get_sftnn_mcdropout_probs(
        test_df, feature_cols, n_features, n_regions, args.artifacts,
        k=args.k, seed=args.seed,
    )
    print(f"Mean per-row predictive std across {args.k} stochastic passes: "
          f"{std_probs.mean():.4f} (higher = less stable single-pass prediction)")

    # ---- Aggregate metrics ----------------------------------------------
    agg = compute_metrics(y_test, mean_probs)
    agg["Model"] = args.label
    agg_table = pd.DataFrame([agg])[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
    agg_table.to_csv(f"{args.results}/mcdropout_aggregate.csv", index=False)
    print(f"\n=== Aggregate ({args.label}, K={args.k}) ===")
    print(agg_table.to_string(index=False))

    # ---- Per-region metrics ----------------------------------------------
    region_rows = []
    for region in sorted(test_df["region_group"].unique()):
        mask = (test_df["region_group"] == region).values
        if mask.sum() < 5:
            continue
        m = compute_metrics(y_test[mask], mean_probs[mask])
        m["Model"] = args.label
        m["Region"] = region
        m["n"] = int(mask.sum())
        region_rows.append(m)
    region_table = pd.DataFrame(region_rows)[
        ["Model", "Region", "n", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ]
    region_table.to_csv(f"{args.results}/mcdropout_per_region.csv", index=False)
    print(f"\n=== Per-Region ({args.label}, K={args.k}) ===")
    print(region_table.to_string(index=False))

    print(f"\nSaved: {args.results}/mcdropout_aggregate.csv, "
          f"{args.results}/mcdropout_per_region.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--results", default="results")
    p.add_argument("--k", type=int, default=30, help="Number of stochastic forward passes to average.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="SFTNN + MC-Dropout")
    main(p.parse_args())

"""
evaluate_gated_exp13.py
=========================
Correctly re-evaluates the ACTUAL saved artifacts_exp13/sftnn_model.pt
checkpoint using GatedSFTNN (multiseed_check_9_vs_13.py's class, which
properly applies the saved region_alpha dampening buffer) instead of the
plain SFTNN class that 06_evaluate.py uses. This fixes the evaluation
mismatch: the checkpoint was trained WITH the gate's influence, so it
should be evaluated WITH the gate, not without it.

Evaluation-only, no retraining, no changes to any existing file.

Run:
    python evaluate_gated_exp13.py
"""
import json
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits
from multiseed_check_9_vs_13 import GatedSFTNN

DEVICE = "cpu"


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


def main():
    train_df, val_df, test_df, manifest = load_splits("artifacts_exp13")
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    with open("artifacts_exp13/sftnn_best_params.json") as f:
        p = json.load(f)

    model = GatedSFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"],
                       n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                       dropout=p["dropout"]).to(DEVICE)

    state_dict = torch.load("artifacts_exp13/sftnn_model.pt", map_location=DEVICE)
    assert "region_alpha" in state_dict, "Checkpoint has no region_alpha -- gate not actually present."
    model.register_buffer("region_alpha", state_dict["region_alpha"].clone())
    model.load_state_dict(state_dict)
    model.eval()

    print("region_alpha (gate strength per region, id order):", state_dict["region_alpha"].tolist())

    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    y_test = test_df["derived_target"].values
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test, r_test)).numpy()

    print("\n=== AGGREGATE (gated, correctly evaluated) ===")
    agg = compute_metrics(y_test, probs)
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== PER REGION (gated, correctly evaluated) ===")
    test_df = test_df.reset_index(drop=True)
    for region in sorted(test_df["region_group"].unique()):
        mask = (test_df["region_group"] == region).values
        if mask.sum() < 5:
            continue
        m = compute_metrics(y_test[mask], probs[mask])
        print(f"  {region:16s} n={int(mask.sum()):5d}  "
              f"Acc={m['Accuracy']:.4f} Prec={m['Precision']:.4f} Rec={m['Recall']:.4f} "
              f"F1={m['F1']:.4f} AUC={m['ROC-AUC']:.4f} PRAUC={m['PR-AUC']:.4f}")


if __name__ == "__main__":
    main()

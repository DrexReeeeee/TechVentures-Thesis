"""
save_final_calibration.py
============================
Computes and PERSISTS the Asia-Pacific cluster calibration threshold for
the exact, final deployed checkpoint (artifacts_exp23/sftnn_model.pt) --
fit on validation data only, same leakage discipline as every other
calibration step today. Until now this was only ever computed on-the-fly
inside diagnostic scripts; the live demo needs a saved, fixed value to
load at inference time rather than recomputing it.

Run:
    python save_final_calibration.py
"""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score

from data_utils import load_splits
from models import SFTNN

ARTIFACTS = "artifacts_exp23"
ASIA_PACIFIC = {"East Asia", "South Asia", "Southeast Asia"}


def best_f1_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main():
    train_df, val_df, test_df, manifest = load_splits(ARTIFACTS)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    with open(f"{ARTIFACTS}/sftnn_best_params.json") as f:
        p = json.load(f)
    model = SFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"],
                  n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                  dropout=p["dropout"])
    model.load_state_dict(torch.load(f"{ARTIFACTS}/sftnn_model.pt", map_location="cpu"))
    model.eval()

    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    r_val = torch.tensor(val_df["region_id"].values, dtype=torch.long)
    y_val = val_df["derived_target"].values
    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val, r_val)).numpy()

    ap_mask = val_df["region_group"].isin(ASIA_PACIFIC).values
    threshold, val_f1 = best_f1_threshold(y_val[ap_mask], val_probs[ap_mask])

    config = {
        "description": "Decision threshold for the final locked model (artifacts_exp23/sftnn_model.pt). "
                        "Apply this threshold to predictions for East Asia, South Asia, and Southeast Asia. "
                        "Use 0.5 for all other regions.",
        "asia_pacific_regions": sorted(ASIA_PACIFIC),
        "asia_pacific_threshold": threshold,
        "default_threshold": 0.5,
        "fit_on": "validation split only (never test) -- same leakage discipline as calibration elsewhere in this study",
        "validation_f1_at_threshold": val_f1,
    }
    with open(f"{ARTIFACTS}/calibration_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Asia-Pacific threshold for the FINAL checkpoint: {threshold:.4f} (val F1={val_f1:.4f})")
    print(f"Saved to {ARTIFACTS}/calibration_config.json")


if __name__ == "__main__":
    main()

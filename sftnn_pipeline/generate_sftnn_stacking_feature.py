"""
generate_sftnn_stacking_feature.py
======================================
The control check for Experiment 23: if giving XGBoost SFTNN's own
prediction as a feature produces a similar jump to what Experiment 23
showed (SFTNN + XGBoost's prediction), that proves the earlier result is
a generic "stacking helps" effect, not something specific to SFTNN's
region-conditioning. If XGBoost does NOT improve nearly as much, that's
real evidence SFTNN's mechanism is doing something specific with the
combined signal.

Same leakage-safe construction as generate_xgb_stacking_feature.py,
roles reversed:
  - TRAIN rows: 5-fold CV. Each fold, train a fresh SFTNN (Experiment 13's
    exact already-chosen hyperparameters, no new search) on the other 4
    folds, predict on the held-out fold.
  - VAL/TEST rows: use the existing, already-trained
    artifacts_exp13/sftnn_model.pt directly.

Output: artifacts_exp24/{train,val,test}.csv with one new feature column
(sftnn_stack_prob) added to the ORIGINAL 25 features (not Experiment 23's
26 -- this is a fresh baseline-feature-set augmentation, mirroring
Experiment 13's feature set, not Experiment 23's).

Run:
    python generate_sftnn_stacking_feature.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from data_utils import load_splits, StartupDataset, set_global_seed, seeded_generator, composite_validation_score
from models import SFTNN

SRC = "artifacts_exp13"
DST = "artifacts_exp24"
N_FOLDS = 5
SEED = 42
DEVICE = "cpu"

REGION_CLUSTERS = {
    "North America": "Americas", "Latin America": "Americas",
    "East Asia": "Asia-Pacific", "South Asia": "Asia-Pacific", "Southeast Asia": "Asia-Pacific",
    "Europe": "Europe & Africa", "Africa": "Europe & Africa",
}


def build_optimizer(model, lr, weight_decay):
    no_decay = (list(model.region_embedding.parameters())
                + list(model.affine_generator.parameters())
                + [model.gamma_scale, model.beta_scale])
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def train_one_fold(params, train_ds_fold, n_features, n_regions, region_counts, cluster_ids,
                    max_epochs=60, patience=5, seed=42):
    set_global_seed(seed)
    model = SFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                  n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                  dropout=params["dropout"]).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params["cluster_shrink_lambda"]
    loader = DataLoader(train_ds_fold, batch_size=256, shuffle=True, generator=seeded_generator(seed))

    # No held-out val split inside the fold -- just run the fixed epoch
    # budget used throughout today's checks for a FIXED config (no
    # early-stopping selection needed here, this isn't an Optuna search).
    for epoch in range(max_epochs):
        model.train()
        for x, r, y in loader:
            x, r, y = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = (loss_fn(logits, y)
                    + shrink_lambda * model.region_shrinkage_penalty(region_counts)
                    + cluster_shrink_lambda * model.region_cluster_shrinkage_penalty(region_counts, cluster_ids))
            loss.backward()
            opt.step()
    return model


def main():
    os.makedirs(DST, exist_ok=True)
    train_df, val_df, test_df, manifest = load_splits(SRC)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    region_names = manifest["regions"]

    with open(f"{SRC}/sftnn_best_params.json") as f:
        best_params = json.load(f)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)], dtype=torch.float32)
    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cname_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor([cname_to_id[REGION_CLUSTERS[r]] for r in region_names], dtype=torch.long)

    full_train_ds = StartupDataset(train_df, feature_cols)
    y_train = train_df["derived_target"].values

    oof_prob = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
        fold_ds = Subset(full_train_ds, tr_idx)
        # Fold-specific region_counts (reflects only this fold's training rows).
        fold_region_ids = train_df["region_id"].values[tr_idx]
        fold_region_counts = torch.tensor(
            [max((fold_region_ids == r).sum(), 1) for r in range(n_regions)], dtype=torch.float32)

        model = train_one_fold(best_params, fold_ds, n_features, n_regions,
                                fold_region_counts, cluster_ids, seed=SEED)
        model.eval()
        x_ho = torch.tensor(train_df[feature_cols].values[ho_idx], dtype=torch.float32)
        r_ho = torch.tensor(train_df["region_id"].values[ho_idx], dtype=torch.long)
        with torch.no_grad():
            oof_prob[ho_idx] = torch.sigmoid(model(x_ho, r_ho)).numpy()
        print(f"  fold {fold}: trained on {len(tr_idx)}, predicted {len(ho_idx)} held-out rows")

    # ---- VAL/TEST: use the existing, already-trained SFTNN model ----
    final_model = SFTNN(n_features, n_regions, hidden_dim=best_params["hidden_dim"],
                         n_layers=best_params["n_layers"], embed_dim=best_params["embed_dim"],
                         dropout=best_params["dropout"]).to(DEVICE)
    state_dict = torch.load(f"{SRC}/sftnn_model.pt", map_location=DEVICE)
    if "region_alpha" in state_dict:
        # Experiment 13's checkpoint was trained with the fixed per-region
        # dampening buffer -- register it so strict loading succeeds,
        # same handling as 06_evaluate.py / evaluate_gated_exp13.py.
        final_model.register_buffer("region_alpha", state_dict["region_alpha"].clone())
    final_model.load_state_dict(state_dict)
    final_model.eval()

    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    r_val = torch.tensor(val_df["region_id"].values, dtype=torch.long)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    with torch.no_grad():
        val_prob = torch.sigmoid(final_model(x_val, r_val)).numpy()
        test_prob = torch.sigmoid(final_model(x_test, r_test)).numpy()

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    train_out["sftnn_stack_prob"] = oof_prob
    val_out["sftnn_stack_prob"] = val_prob
    test_out["sftnn_stack_prob"] = test_prob

    train_out.to_csv(f"{DST}/train.csv", index=False)
    val_out.to_csv(f"{DST}/val.csv", index=False)
    test_out.to_csv(f"{DST}/test.csv", index=False)

    new_manifest = dict(manifest)
    new_manifest["feature_cols"] = feature_cols + ["sftnn_stack_prob"]
    new_manifest["n_features"] = len(new_manifest["feature_cols"])
    with open(f"{DST}/feature_manifest.json", "w") as f:
        json.dump(new_manifest, f, indent=2)

    y_val, y_test = val_df["derived_target"].values, test_df["derived_target"].values
    print(f"\nAUC check -- does the stacking feature itself look sane:")
    print(f"  train (out-of-fold) AUC: {roc_auc_score(y_train, oof_prob):.4f}")
    print(f"  val AUC: {roc_auc_score(y_val, val_prob):.4f}")
    print(f"  test AUC: {roc_auc_score(y_test, test_prob):.4f}")
    print(f"\nSaved to {DST}/ -- {len(new_manifest['feature_cols'])} features "
          f"(was {len(feature_cols)}, +1: sftnn_stack_prob)")


if __name__ == "__main__":
    main()

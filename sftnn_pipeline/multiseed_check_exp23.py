"""
multiseed_check_exp23.py
===========================
Multiseed validation of Experiment 23 (SFTNN + XGBoost-stacking-feature)
against Experiment 13 (plain SFTNN), both with and without the
Asia-Pacific calibration threshold -- same discipline as every other
mechanism today: no single-seed conclusions trusted.

Run:
    python multiseed_check_exp23.py --seeds 1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits, StartupDataset, set_global_seed, seeded_generator, composite_validation_score
from models import SFTNN

DEVICE = "cpu"
ASIA_PACIFIC = {"East Asia", "South Asia", "Southeast Asia"}

PARAMS_13 = json.load(open("artifacts_exp13/sftnn_best_params.json"))
PARAMS_23 = json.load(open("artifacts_exp23/sftnn_best_params.json"))

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


def train_fixed(params, seed, train_ds, val_ds, n_features, n_regions,
                 region_counts, cluster_ids, max_epochs=60, patience=5):
    set_global_seed(seed)
    model = SFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                  n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                  dropout=params["dropout"]).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params["cluster_shrink_lambda"]

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=seeded_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_score, best_state, no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, r, y in train_loader:
            x, r, y = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = (loss_fn(logits, y)
                    + shrink_lambda * model.region_shrinkage_penalty(region_counts)
                    + cluster_shrink_lambda * model.region_cluster_shrinkage_penalty(region_counts, cluster_ids))
            loss.backward()
            opt.step()

        model.eval()
        all_logits, all_y, all_r = [], [], []
        with torch.no_grad():
            for x, r, y in val_loader:
                x, r = x.to(DEVICE), r.to(DEVICE)
                all_logits.append(model(x, r).cpu())
                all_y.append(y)
                all_r.append(r.cpu())
        val_probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        val_y = torch.cat(all_y).numpy()
        val_r = torch.cat(all_r).numpy()
        val_score, _ = composite_validation_score(val_y, val_probs, val_r)

        if val_score > best_score:
            best_score, best_state, no_improve = val_score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def best_f1_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def metrics_from_pred(yt, pred, yp):
    out = dict(Accuracy=accuracy_score(yt, pred), Precision=precision_score(yt, pred, zero_division=0),
               Recall=recall_score(yt, pred, zero_division=0), F1=f1_score(yt, pred, zero_division=0))
    if len(set(yt)) > 1:
        out["ROC-AUC"] = roc_auc_score(yt, yp)
        out["PR-AUC"] = average_precision_score(yt, yp)
    else:
        out["ROC-AUC"] = out["PR-AUC"] = float("nan")
    return out


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]

    # Two different data sources: 13 uses 25 features, 23 uses 26 (xgb_stack_prob added)
    data_13 = load_splits("artifacts_exp13")
    data_23 = load_splits("artifacts_exp23")

    configs = [("Exp13-config", PARAMS_13, data_13), ("Exp23-config", PARAMS_23, data_23)]
    regions_to_report = ["Africa", "Southeast Asia", "South Asia", "Latin America"]

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        for label, params, (train_df, val_df, test_df, manifest) in configs:
            feature_cols = manifest["feature_cols"]
            n_features, n_regions = len(feature_cols), manifest["n_regions"]
            region_names = manifest["regions"]

            train_ds = StartupDataset(train_df, feature_cols)
            val_ds = StartupDataset(val_df, feature_cols)
            region_counts = torch.tensor(
                [(train_df["region_id"] == r).sum() for r in range(n_regions)], dtype=torch.float32)
            cluster_names = sorted(set(REGION_CLUSTERS.values()))
            cname_to_id = {c: i for i, c in enumerate(cluster_names)}
            cluster_ids = torch.tensor([cname_to_id[REGION_CLUSTERS[r]] for r in region_names], dtype=torch.long)

            x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
            r_val = torch.tensor(val_df["region_id"].values, dtype=torch.long)
            y_val = val_df["derived_target"].values
            val_region = val_df["region_group"].values

            x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
            r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
            y_test = test_df["derived_target"].values
            test_df_reset = test_df.reset_index(drop=True)

            print(f"  training {label} ...", flush=True)
            model = train_fixed(params, seed, train_ds, val_ds, n_features, n_regions,
                                 region_counts, cluster_ids)
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(x_val, r_val)).numpy()
                test_probs = torch.sigmoid(model(x_test, r_test)).numpy()

            ap_mask_val = np.isin(val_region, list(ASIA_PACIFIC))
            ap_threshold = best_f1_threshold(y_val[ap_mask_val], val_probs[ap_mask_val])
            ap_mask_test = test_df_reset["region_group"].isin(ASIA_PACIFIC).values
            pred_default = (test_probs >= 0.5).astype(int)
            pred_calibrated = np.where(ap_mask_test, (test_probs >= ap_threshold).astype(int), pred_default)

            scopes = [("AGGREGATE", np.ones(len(y_test), dtype=bool))] + \
                     [(r, (test_df_reset.region_group == r).values) for r in regions_to_report]
            for scope_name, mask in scopes:
                m_def = metrics_from_pred(y_test[mask], pred_default[mask], test_probs[mask])
                rows.append({"seed": seed, "config": label, "scope": scope_name, "variant": "default", **m_def})
                m_cal = metrics_from_pred(y_test[mask], pred_calibrated[mask], test_probs[mask])
                rows.append({"seed": seed, "config": label, "scope": scope_name, "variant": "cluster_calibrated", **m_cal})

            print(f"    ap_threshold={ap_threshold:.2f}")
            for region in ["Southeast Asia", "South Asia", "Africa"]:
                mask = (test_df_reset.region_group == region).values
                f1_def = f1_score(y_test[mask], pred_default[mask], zero_division=0)
                f1_cal = f1_score(y_test[mask], pred_calibrated[mask], zero_division=0)
                print(f"    {region:15s} F1 default={f1_def:.4f} calibrated={f1_cal:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds\n{'='*70}")
    for scope in ["AGGREGATE"] + regions_to_report:
        print(f"\n--- {scope} ---")
        for label, _, _ in configs:
            for variant in ["default", "cluster_calibrated"]:
                sub = df[(df.scope == scope) & (df.config == label) & (df.variant == variant)]
                line = f"  {label:14s} {variant:20s}"
                for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
                    line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
                print(line)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12")
    p.add_argument("--out", default="multiseed_exp23_raw.csv")
    main(p.parse_args())

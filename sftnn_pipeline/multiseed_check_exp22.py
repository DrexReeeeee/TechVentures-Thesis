"""
multiseed_check_exp22.py
===========================
Multiseed validation of Experiment 22 (Deep FiLM) against Experiment 13
(original, single-point FiLM), same discipline as every other check
today: takes Exp22's exact best hyperparameters (once found by Optuna)
and Exp13's exact best hyperparameters, retrains both under many seeds,
and reports mean +/- std -- no single-seed conclusions trusted.

Run:
    python multiseed_check_exp22.py --seeds 1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12
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
from deep_film_sftnn import DeepFiLMSFTNN

DEVICE = "cpu"
PARAMS_13 = json.load(open("artifacts_exp13/sftnn_best_params.json"))
PARAMS_22 = json.load(open("artifacts_exp22/sftnn_best_params.json"))


def build_optimizer_13(model, lr, weight_decay):
    no_decay = (list(model.region_embedding.parameters())
                + list(model.affine_generator.parameters())
                + [model.gamma_scale, model.beta_scale])
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def build_optimizer_22(model, lr, weight_decay):
    no_decay = (list(model.region_embedding.parameters())
                + list(model.film_readouts.parameters())
                + [model.gamma_scale, model.beta_scale])
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def train_fixed(params, seed, variant, train_ds, val_ds, n_features, n_regions,
                 region_counts, cluster_ids, max_epochs=60, patience=5):
    set_global_seed(seed)
    if variant == "exp22":
        model = DeepFiLMSFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                               n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                               dropout=params["dropout"]).to(DEVICE)
        opt = build_optimizer_22(model, lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        model = SFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                      n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                      dropout=params["dropout"]).to(DEVICE)
        opt = build_optimizer_13(model, lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params.get("cluster_shrink_lambda", 0.0)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=seeded_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_score, best_state, no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, r, y in train_loader:
            x, r, y = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = loss_fn(logits, y) + shrink_lambda * model.region_shrinkage_penalty(region_counts)
            if variant != "exp22" and cluster_shrink_lambda:
                loss = loss + cluster_shrink_lambda * model.region_cluster_shrinkage_penalty(region_counts, cluster_ids)
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


def metrics(yt, yp):
    pred = (yp >= 0.5).astype(int)
    out = dict(Accuracy=accuracy_score(yt, pred), Precision=precision_score(yt, pred, zero_division=0),
               Recall=recall_score(yt, pred, zero_division=0), F1=f1_score(yt, pred, zero_division=0))
    if len(set(yt)) > 1:
        out["ROC-AUC"] = roc_auc_score(yt, yp)
        out["PR-AUC"] = average_precision_score(yt, yp)
    else:
        out["ROC-AUC"] = out["PR-AUC"] = float("nan")
    return out


REGION_CLUSTERS = {
    "North America": "Americas", "Latin America": "Americas",
    "East Asia": "Asia-Pacific", "South Asia": "Asia-Pacific", "Southeast Asia": "Asia-Pacific",
    "Europe": "Europe & Africa", "Africa": "Europe & Africa",
}


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    train_df, val_df, test_df, manifest = load_splits("artifacts_exp13")
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    region_names = manifest["regions"]

    train_ds = StartupDataset(train_df, feature_cols)
    val_ds = StartupDataset(val_df, feature_cols)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)], dtype=torch.float32).to(DEVICE)
    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cname_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor([cname_to_id[REGION_CLUSTERS[r]] for r in region_names], dtype=torch.long).to(DEVICE)

    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    y_test = test_df["derived_target"].values
    test_df_reset = test_df.reset_index(drop=True)

    regions_to_report = ["Africa", "Southeast Asia", "South Asia", "Latin America"]
    configs = [("Exp13-config", PARAMS_13, "exp13"), ("Exp22-config", PARAMS_22, "exp22")]

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        for label, params, variant in configs:
            print(f"  training {label} ...", flush=True)
            model = train_fixed(params, seed, variant, train_ds, val_ds, n_features, n_regions,
                                 region_counts, cluster_ids)
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x_test, r_test)).numpy()

            agg_m = metrics(y_test, probs)
            rows.append({"seed": seed, "config": label, "scope": "AGGREGATE", **agg_m})
            print(f"    AGGREGATE F1={agg_m['F1']:.4f} Acc={agg_m['Accuracy']:.4f} AUC={agg_m['ROC-AUC']:.4f}")

            for region in regions_to_report:
                mask = (test_df_reset.region_group == region).values
                m = metrics(y_test[mask], probs[mask])
                rows.append({"seed": seed, "config": label, "scope": region, **m})
                print(f"    {region:15s} F1={m['F1']:.4f} Recall={m['Recall']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds\n{'='*70}")
    for scope in ["AGGREGATE"] + regions_to_report:
        print(f"\n--- {scope} ---")
        for label, _, _ in configs:
            sub = df[(df.scope == scope) & (df.config == label)]
            line = f"  {label:14s}"
            for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
                line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
            print(line)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12")
    p.add_argument("--out", default="multiseed_exp22_raw.csv")
    main(p.parse_args())

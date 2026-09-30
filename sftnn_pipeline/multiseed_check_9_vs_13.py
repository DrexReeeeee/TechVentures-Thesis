"""
multiseed_check_9_vs_13.py
===========================
Targeted training-variance check: are Experiment #9 and #13's exact,
already-chosen hyperparameters stable across random seeds, or is the
9-vs-13 comparison sensitive to the single seed=42 used everywhere else
this study?

Deliberately NOT a re-run of 09_multiseed.py: this does not re-run
Optuna (which would conflate "did the search find a different config"
with "is this specific config stable") and does not retrain baselines
(irrelevant to this question -- baselines are frozen and not part of
the #9-vs-#13 comparison). It takes #9's and #13's exact best
hyperparameters (pulled from artifacts_exp9/ and artifacts_exp13/ on
disk, not memory) and retrains ONLY those two fixed configs under
several new seeds, self-contained (the #13 gate is implemented as a
local subclass here, so this script does not touch models.py or
05_train_sftnn.py at all -- nothing to revert).

Run:
    python multiseed_check_9_vs_13.py --seeds 1,7,123,2024
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

from data_utils import (load_splits, StartupDataset, set_global_seed,
                         seeded_generator, composite_validation_score)
from models import SFTNN

DEVICE = "cpu"

REGION_CLUSTERS = {
    "North America": "Americas", "Latin America": "Americas",
    "East Asia": "Asia-Pacific", "South Asia": "Asia-Pacific", "Southeast Asia": "Asia-Pacific",
    "Europe": "Europe & Africa", "Africa": "Europe & Africa",
}

# Exact hyperparameters, pulled from disk (not memory) before this check.
PARAMS_9 = json.load(open("artifacts_exp9/sftnn_best_params.json"))
PARAMS_13 = json.load(open("artifacts_exp13/sftnn_best_params.json"))


class GatedSFTNN(SFTNN):
    """Experiment #13's fixed dampening gate, as a local subclass so this
    script never touches models.py -- nothing to revert afterward."""
    def set_gate(self, region_counts, c=500.0):
        alpha = region_counts.float() / (region_counts.float() + c)
        self.register_buffer("region_alpha", alpha)

    def forward(self, x, region_id, return_affine=False):
        h = self.trunk(x)
        e_r = self.region_embedding(region_id)
        gamma_beta = self.affine_generator(e_r)
        gamma_raw, beta_raw = gamma_beta.chunk(2, dim=-1)
        gamma_delta = self.gamma_scale * torch.tanh(gamma_raw)
        beta_val = self.beta_scale * torch.tanh(beta_raw)
        a = self.region_alpha[region_id].unsqueeze(-1)
        gamma_delta = a * gamma_delta
        beta_val = a * beta_val
        gamma = 1.0 + gamma_delta
        beta = beta_val
        y_modulated = gamma * h + beta
        logits = self.head(y_modulated).squeeze(-1)
        if return_affine:
            return logits, gamma, beta
        return logits


def train_fixed(params, seed, use_gate, train_ds, val_ds, n_features, n_regions,
                 region_counts, cluster_ids, max_epochs=60, patience=5):
    set_global_seed(seed)
    cls = GatedSFTNN if use_gate else SFTNN
    model = cls(n_features, n_regions, hidden_dim=params["hidden_dim"],
                n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                dropout=params["dropout"]).to(DEVICE)
    if use_gate:
        model.set_gate(region_counts)

    no_decay = (list(model.region_embedding.parameters())
                + list(model.affine_generator.parameters())
                + [model.gamma_scale, model.beta_scale])
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": params["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}], lr=params["lr"])
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


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    train_df, val_df, test_df, manifest = load_splits("artifacts_exp9")
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
    test_df = test_df.reset_index(drop=True)

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        for label, params, gate in [("Exp9-config", PARAMS_9, False), ("Exp13-config", PARAMS_13, True)]:
            print(f"  training {label} ...", flush=True)
            model = train_fixed(params, seed, gate, train_ds, val_ds, n_features, n_regions,
                                 region_counts, cluster_ids)
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x_test, r_test)).numpy()

            agg_m = metrics(y_test, probs)
            row = {"seed": seed, "config": label, "scope": "AGGREGATE", **agg_m}
            rows.append(row)
            print(f"    AGGREGATE  F1={agg_m['F1']:.4f} Recall={agg_m['Recall']:.4f} ROC-AUC={agg_m['ROC-AUC']:.4f} PR-AUC={agg_m['PR-AUC']:.4f}")

            for region in ["South Asia", "Africa"]:
                mask = (test_df.region_group == region).values
                m = metrics(y_test[mask], probs[mask])
                rows.append({"seed": seed, "config": label, "scope": region, **m})
                print(f"    {region:15s} F1={m['F1']:.4f} Recall={m['Recall']:.4f} ROC-AUC={m['ROC-AUC']:.4f} PR-AUC={m['PR-AUC']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("multiseed_9_vs_13_raw.csv", index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds (seed=42 excluded -- new seeds only)\n{'='*70}")
    for scope in ["AGGREGATE", "South Asia", "Africa"]:
        print(f"\n--- {scope} ---")
        for config in ["Exp9-config", "Exp13-config"]:
            sub = df[(df.scope == scope) & (df.config == config)]
            line = f"  {config:14s}"
            for m in ["F1", "Recall", "ROC-AUC", "PR-AUC"]:
                line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
            print(line)

    print("\nSaved: multiseed_9_vs_13_raw.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024")
    main(p.parse_args())

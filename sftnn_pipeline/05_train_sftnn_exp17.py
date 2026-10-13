"""
05_train_sftnn_exp17.py
=========================
Experiment 17 variant of 05_train_sftnn.py (NOT a modification of the
original -- 05_train_sftnn.py / models.py / Experiment 13's artifacts are
untouched). This is Experiment 13's exact SFTNN recipe (identity-centered
FiLM init, decoupled weight decay on the region-conditioning branch,
two-level global+cluster embedding shrinkage) with exactly ONE change
(P1): the classification loss is a per-sample, inverse-region-frequency-
weighted BCE instead of plain pooled BCE.

Why this is the first thing to test: North America is 67.8% of train.csv
(16,799 / 24,767 rows). Pooled BCE means the shared trunk's gradient is
overwhelmingly shaped by North America regardless of SFTNN's per-region
gamma/beta mechanism -- the shrinkage penalties in models.py regularize
the region EMBEDDING, but nothing in Experiment 13 rebalances which
samples drive the trunk/head's learning in the first place. See
region_reweighting.py for the full rationale and region_weight_table()
printout below for the exact per-region multipliers used.

Run:
    python 05_train_sftnn_exp17.py --trials 30 --max_epochs 60 --artifacts artifacts_exp17
"""
import argparse
import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from data_utils import (load_splits, StartupDataset, set_global_seed,
                         seeded_generator, composite_validation_score)
from models import SFTNN
from region_reweighting import WeightedStartupDataset, region_weight_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Identical, a-priori cluster map to 05_train_sftnn.py -- unchanged by P1.
REGION_CLUSTERS = {
    "North America": "Americas",
    "Latin America": "Americas",
    "East Asia": "Asia-Pacific",
    "South Asia": "Asia-Pacific",
    "Southeast Asia": "Asia-Pacific",
    "Europe": "Europe & Africa",
    "Africa": "Europe & Africa",
}


def build_optimizer(model, lr, weight_decay):
    """Identical to 05_train_sftnn.py's fix #2: the region-embedding /
    affine-generator branch gets its own weight_decay=0 param group so
    weight decay doesn't fight the identity-centered init."""
    no_decay_params = (
        list(model.region_embedding.parameters())
        + list(model.affine_generator.parameters())
        + [model.gamma_scale, model.beta_scale]
    )
    no_decay_ids = {id(p) for p in no_decay_params}
    decay_params = [p for p in model.parameters() if id(p) not in no_decay_ids]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_one_config(params, train_ds, val_ds, n_features, n_regions,
                      region_counts, cluster_ids, max_epochs,
                      patience=5, verbose=False, seed=42):
    set_global_seed(seed)
    model = SFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                  n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                  dropout=params["dropout"]).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    # Experiment 17 (P1): reduction="none" so each sample's BCE term can be
    # scaled by its region weight before the shrinkage penalties (which
    # are added AFTER averaging, exactly as in 05_train_sftnn.py) are
    # applied. The shrinkage-penalty math itself is untouched.
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params["cluster_shrink_lambda"]

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                               generator=seeded_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_auc, best_state, epochs_no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, r, y, w in train_loader:
            x, r, y, w = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = ((loss_fn(logits, y) * w).mean()
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
        # Composite selection score is UNCHANGED/unweighted (same function
        # as 05_train_sftnn.py) -- only the training loss above differs,
        # so what Optuna selects for and how 06_evaluate.py scores the
        # result both stay directly comparable to Experiment 13.
        val_score, parts = composite_validation_score(val_y, val_probs, val_r)

        if verbose:
            print(f"  epoch {epoch:02d}  score={val_score:.4f}  "
                  f"(auc={parts['auc']:.4f} pr_auc={parts['pr_auc']:.4f} "
                  f"f1={parts['f1']:.4f} macro_recall={parts['macro_recall']:.4f})")

        if val_score > best_auc:
            best_auc, best_state, epochs_no_improve = val_score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def main(args):
    set_global_seed(args.seed)
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    train_ds = WeightedStartupDataset(train_df, feature_cols, n_regions, power=args.weight_power)
    val_ds = StartupDataset(val_df, feature_cols)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"Region training counts (id order): {region_counts.cpu().tolist()}")

    print(f"Experiment 17 region weights (train split, power={args.weight_power}):")
    for name, info in region_weight_table(train_df["region_id"].values, n_regions,
                                           manifest["regions"], power=args.weight_power).items():
        print(f"  {name:16s} n={info['count']:6d}  weight={info['weight']:.3f}")

    region_names = manifest["regions"]
    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cluster_name_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor(
        [cluster_name_to_id[REGION_CLUSTERS[r]] for r in region_names],
        dtype=torch.long,
    ).to(DEVICE)
    print(f"Region clusters (id order, {region_names}): {cluster_ids.cpu().tolist()} "
          f"-> {cluster_names}")

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [2, 3]),
            "embed_dim": trial.suggest_categorical("embed_dim", [4, 8, 16]),
            "shrink_lambda": trial.suggest_float("shrink_lambda", 1.0, 50.0, log=True),
            "cluster_shrink_lambda": trial.suggest_float("cluster_shrink_lambda", 1.0, 50.0, log=True),
        }
        _, val_score = train_one_config(params, train_ds, val_ds, n_features,
                                         n_regions, region_counts, cluster_ids,
                                         max_epochs=args.max_epochs,
                                         seed=args.seed)
        return val_score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val composite score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    final_model, final_val_auc = train_one_config(
        study.best_params, train_ds, val_ds, n_features, n_regions,
        region_counts, cluster_ids, max_epochs=args.max_epochs,
        verbose=True, seed=args.seed,
    )
    torch.save(final_model.state_dict(), f"{args.artifacts}/sftnn_model.pt")
    with open(f"{args.artifacts}/sftnn_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved trained SFTNN to {args.artifacts}/sftnn_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp17")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--weight_power", type=float, default=0.5,
                   help="Region reweighting exponent: 0.5=sqrt-inverse-frequency "
                        "(default), 1.0=raw inverse frequency, 0.0=disabled "
                        "(reduces to the original unweighted 05_train_sftnn.py).")
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())

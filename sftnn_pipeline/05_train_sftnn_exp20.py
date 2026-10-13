"""
05_train_sftnn_exp20.py
=========================
Experiment 20 variant of 05_train_sftnn.py (NOT a modification of the
original -- 05_train_sftnn.py / models.py / Experiment 13 and 19's
artifacts are untouched). Experiment 13's exact recipe (plain, unweighted
BCE) with exactly ONE change (P3-v2): region_cluster_shrinkage_penalty is
replaced by the FIXED-reference evidence-aware version
(evidence_aware_shrinkage_v2.py) -- no tunable cluster_evidence_k, no
Optuna search over it. Americas and Europe & Africa land at the rho_c=1.0
ceiling (mathematically identical to Experiment 13's original penalty);
only Asia-Pacific (East Asia/South Asia/Southeast Asia) is damped, by a
fixed amount derived from Europe's own (pre-existing, not tuned) training
row count. See evidence_aware_shrinkage_v2.py for the full rationale --
this directly targets why Experiment 19's single tunable k let Africa
regress (it damped Europe & Africa too, as a side effect of damping
Asia-Pacific enough to help Southeast Asia).

Baselines (MLP/XGBoost/TabNet) are NOT retrained -- reused from
Experiment 13, same as Experiment 19.

Run:
    python 05_train_sftnn_exp20.py --trials 30 --max_epochs 60 --artifacts artifacts_exp20
"""
import argparse
import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from data_utils import (load_splits, StartupDataset, set_global_seed,
                         seeded_generator, composite_validation_score)
from evidence_aware_shrinkage_v2 import SFTNNEvidenceAwareClusterFixed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    model = SFTNNEvidenceAwareClusterFixed(
        n_features, n_regions, hidden_dim=params["hidden_dim"],
        n_layers=params["n_layers"], embed_dim=params["embed_dim"],
        dropout=params["dropout"],
    ).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params["cluster_shrink_lambda"]

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                               generator=seeded_generator(seed))
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_auc, best_state, epochs_no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, r, y in train_loader:
            x, r, y = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = (loss_fn(logits, y)
                    + shrink_lambda * model.region_shrinkage_penalty(region_counts)
                    # P3-v2: fixed-reference evidence-aware cluster penalty.
                    + cluster_shrink_lambda * model.region_cluster_shrinkage_penalty_evidence_aware_fixed(
                        region_counts, cluster_ids))
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

    train_ds = StartupDataset(train_df, feature_cols)
    val_ds = StartupDataset(val_df, feature_cols)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"Region training counts (id order): {region_counts.cpu().tolist()}")

    region_names = manifest["regions"]
    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cluster_name_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor(
        [cluster_name_to_id[REGION_CLUSTERS[r]] for r in region_names],
        dtype=torch.long,
    ).to(DEVICE)
    print(f"Region clusters (id order, {region_names}): {cluster_ids.cpu().tolist()} "
          f"-> {cluster_names}")

    cluster_totals = {}
    for c_name, c_id in cluster_name_to_id.items():
        cluster_totals[c_name] = float(region_counts[cluster_ids == c_id].sum().item())
    print(f"Cluster training totals: {cluster_totals}")
    print(f"rho_c (fixed reference, EUROPE_ANCHOR_N=5342): "
          f"{ {c: round(min(1.0, n/5342.0), 3) for c, n in cluster_totals.items()} }")

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
            # No cluster_evidence_k here -- P3-v2 has no tunable knob for
            # the evidence-awareness itself, only the existing
            # cluster_shrink_lambda (overall strength), same search range
            # as Experiment 13/19.
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
    p.add_argument("--artifacts", default="artifacts_exp20")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())

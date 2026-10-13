"""
05_train_sftnn_exp21.py
=========================
Experiment 21 variant of 05_train_sftnn.py (NOT a modification of the
original -- 05_train_sftnn.py / models.py / Experiment 13/19/20's
artifacts are untouched). Experiment 13's exact base recipe (plain,
unweighted BCE) with TWO changes combined:

  1. Independent per-cluster shrinkage strength (per_cluster_lambda_
     shrinkage.py) instead of one shared cluster_shrink_lambda -- removes
     the confound that let Experiment 20's single shared lambda (39.1)
     still over-pull Africa/Latin America even after their cluster's
     damping formula was made identical to Experiment 13's.
  2. Target-region-weighted validation selection score
     (target_region_validation_score.py) -- Optuna and early-stopping
     now select against a score that explicitly rewards Africa, South
     Asia, AND Southeast Asia's own F1, not just a 7-region-diluted
     macro_recall at weight 0.15.

Baselines (MLP/XGBoost/TabNet) are NOT retrained -- reused from
Experiment 13, same as Experiments 19/20.

Run:
    python 05_train_sftnn_exp21.py --trials 30 --max_epochs 60 --artifacts artifacts_exp21
"""
import argparse
import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from data_utils import load_splits, StartupDataset, set_global_seed, seeded_generator
from per_cluster_lambda_shrinkage import SFTNNPerClusterLambda
from target_region_validation_score import composite_validation_score_target_regions

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
                      region_counts, cluster_ids, region_id_to_name, max_epochs,
                      patience=5, verbose=False, seed=42):
    set_global_seed(seed)
    model = SFTNNPerClusterLambda(
        n_features, n_regions, hidden_dim=params["hidden_dim"],
        n_layers=params["n_layers"], embed_dim=params["embed_dim"],
        dropout=params["dropout"],
    ).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    # Cluster id order is sorted(set(REGION_CLUSTERS.values())) ->
    # ['Americas', 'Asia-Pacific', 'Europe & Africa'] (ids 0,1,2).
    cluster_lambdas = torch.tensor(
        [params["lambda_americas"], params["lambda_asia_pacific"], params["lambda_europe_africa"]],
        dtype=torch.float32,
    ).to(DEVICE)

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
                    # Experiment 21, change #1: independent per-cluster
                    # lambdas (already weighted inside this call).
                    + model.region_cluster_shrinkage_penalty_per_cluster(
                        region_counts, cluster_ids, cluster_lambdas))
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
        # Experiment 21, change #2: target-region-weighted selection score.
        val_score, parts = composite_validation_score_target_regions(
            val_y, val_probs, val_r, region_id_to_name)

        if verbose:
            print(f"  epoch {epoch:02d}  score={val_score:.4f}  "
                  f"(auc={parts['auc']:.4f} pr_auc={parts['pr_auc']:.4f} "
                  f"f1={parts['f1']:.4f} target_f1={parts['target_f1']:.4f})")

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
    region_id_to_name = manifest["regions"]

    train_ds = StartupDataset(train_df, feature_cols)
    val_ds = StartupDataset(val_df, feature_cols)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"Region training counts (id order): {region_counts.cpu().tolist()}")

    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cluster_name_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor(
        [cluster_name_to_id[REGION_CLUSTERS[r]] for r in region_id_to_name],
        dtype=torch.long,
    ).to(DEVICE)
    print(f"Region clusters (id order, {region_id_to_name}): {cluster_ids.cpu().tolist()} "
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
            # Experiment 21, change #1: three independent cluster
            # strengths instead of one shared cluster_shrink_lambda.
            "lambda_americas": trial.suggest_float("lambda_americas", 1.0, 50.0, log=True),
            "lambda_asia_pacific": trial.suggest_float("lambda_asia_pacific", 1.0, 50.0, log=True),
            "lambda_europe_africa": trial.suggest_float("lambda_europe_africa", 1.0, 50.0, log=True),
        }
        _, val_score = train_one_config(params, train_ds, val_ds, n_features,
                                         n_regions, region_counts, cluster_ids,
                                         region_id_to_name, max_epochs=args.max_epochs,
                                         seed=args.seed)
        return val_score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val composite score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    final_model, final_val_auc = train_one_config(
        study.best_params, train_ds, val_ds, n_features, n_regions,
        region_counts, cluster_ids, region_id_to_name, max_epochs=args.max_epochs,
        verbose=True, seed=args.seed,
    )
    torch.save(final_model.state_dict(), f"{args.artifacts}/sftnn_model.pt")
    with open(f"{args.artifacts}/sftnn_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved trained SFTNN to {args.artifacts}/sftnn_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp21")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())

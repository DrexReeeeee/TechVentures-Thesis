"""
05_train_sftnn_exp19.py
=========================
Experiment 19 variant of 05_train_sftnn.py (NOT a modification of the
original -- 05_train_sftnn.py / models.py / Experiment 13's artifacts are
untouched). This is Experiment 13's EXACT recipe -- plain pooled,
unweighted BCE loss (Experiment 13 remains the stronger baseline to build
from; the region-reweighted-loss variants in Experiment 17/18 regressed
Southeast Asia and didn't show a consistent dose-response, so this
experiment does NOT carry that change forward) -- with exactly ONE change
(P3): region_cluster_shrinkage_penalty is replaced by an evidence-aware
version (evidence_aware_shrinkage.py) that damps each region's pull
toward its cluster mean by how much total training evidence that
CLUSTER carries, not just the region's own n_r.

Why: Africa's cluster (Europe & Africa) is anchored by Europe (5,342
rows) -- pulling Africa toward that mean is pooling toward a reliable
target, and Africa is Experiment 13's best region. Southeast Asia's
cluster (Asia-Pacific = East Asia + South Asia + Southeast Asia) has no
such anchor -- its largest member is East Asia at 1,086 rows, 5x smaller
than Europe -- so pulling Southeast Asia toward that cluster mean with
the same flat strength is pooling noise with noise. This experiment lets
Optuna find how strongly to damp an unreliable cluster's pull (via the
new cluster_evidence_k hyperparameter) rather than hand-picking it.

Baselines (MLP/XGBoost/TabNet) are NOT retrained here -- this is purely
an SFTNN-internal regularization change (the region-embedding/cluster
mechanism is SFTNN-specific; the baselines have no analogous component to
give it to), so Experiment 13's existing baseline checkpoints are reused
as the fixed comparison point, matching how Experiments 8-16 already
treated baselines as frozen while only SFTNN was iterated on.

Run:
    python 05_train_sftnn_exp19.py --trials 30 --max_epochs 60 --artifacts artifacts_exp19
"""
import argparse
import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from data_utils import (load_splits, StartupDataset, set_global_seed,
                         seeded_generator, composite_validation_score)
from evidence_aware_shrinkage import SFTNNEvidenceAwareCluster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Identical, a-priori cluster map to 05_train_sftnn.py -- unchanged by P3.
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
    """Identical to 05_train_sftnn.py's fix #2."""
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
    model = SFTNNEvidenceAwareCluster(
        n_features, n_regions, hidden_dim=params["hidden_dim"],
        n_layers=params["n_layers"], embed_dim=params["embed_dim"],
        dropout=params["dropout"],
    ).to(DEVICE)
    opt = build_optimizer(model, lr=params["lr"], weight_decay=params["weight_decay"])
    # Plain, unweighted BCE -- same as 05_train_sftnn.py (Experiment 13),
    # deliberately NOT the region-reweighted loss from Experiment 17/18.
    loss_fn = nn.BCEWithLogitsLoss()
    shrink_lambda = params["shrink_lambda"]
    cluster_shrink_lambda = params["cluster_shrink_lambda"]
    cluster_evidence_k = params["cluster_evidence_k"]

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
                    # P3: evidence-aware cluster penalty replaces the
                    # original flat-strength cluster penalty.
                    + cluster_shrink_lambda * model.region_cluster_shrinkage_penalty_evidence_aware(
                        region_counts, cluster_ids, cluster_evidence_k))
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

    # Print cluster totals / rho_c at a couple of reference k values so
    # the run log shows exactly what evidence-weighting is in play,
    # independent of whatever k Optuna ends up picking.
    cluster_totals = {}
    for c_name, c_id in cluster_name_to_id.items():
        cluster_totals[c_name] = float(region_counts[cluster_ids == c_id].sum().item())
    print(f"Cluster training totals: {cluster_totals}")
    for k_ref in (500, 2000, 8000):
        rhos = {c: round(n / (n + k_ref), 3) for c, n in cluster_totals.items()}
        print(f"  rho_c at k={k_ref}: {rhos}")

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
            # P3: how much total cluster evidence it takes before a
            # cluster's mean is treated as a fully reliable shrinkage
            # target. Searched, not hand-picked -- see
            # evidence_aware_shrinkage.py for the rho_c = n_c/(n_c+k)
            # formula and why this range spans "almost like the original
            # flat penalty" (k small) to "most clusters heavily damped"
            # (k large) given this dataset's actual cluster totals
            # (Asia-Pacific ~1,987; Europe & Africa ~5,448; Americas ~17,331).
            "cluster_evidence_k": trial.suggest_float("cluster_evidence_k", 200.0, 20000.0, log=True),
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
    p.add_argument("--artifacts", default="artifacts_exp19")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())

"""
05_train_sftnn.py
==================
Stage 4 of the six-stage development model (Section 4.6): the proposed
Spatial Feature Transformation Neural Network (Section 4.5.2, 4.7).

Unlike the three baselines, region is NOT one-hot-concatenated into x.
Instead, region drives a separate embedding + affine-parameter-generator
branch that modulates the shared trunk's hidden representation
(Equations 2-3), which is exactly the mechanism the per-region evaluation
in Section 4.10.2 is designed to test against the baselines.

Search space (Section 4.7):
  - AdamW optimizer, same lr/dropout/weight_decay ranges as the MLP
  - ~30 Optuna trials (reflecting the added embedding + generator branch)
  - early stopping on validation AUC-ROC, patience = 5 epochs

Run:
    python 05_train_sftnn.py --trials 30 --max_epochs 60
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Experiment #9: fixed, a-priori regional clustering (Americas /
# Asia-Pacific / Europe & Africa -- the standard three-way split used
# across VC/startup industry reporting). Chosen before looking at any
# results; not derived from observed regional similarity.
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
    """
    Fix #2: weight decay pulls parameters toward 0. After fix #1's
    identity-centered reparameterization, gamma_raw/beta_raw = 0 is exactly
    the no-op state (gamma=1, beta=0) -- so applying weight decay to the
    region-embedding/affine-generator branch actively pushes SFTNN's
    region-conditioning mechanism toward doing nothing, rather than
    regularizing it the way weight decay regularizes an ordinary layer.
    This matters most for minority regions, which already have the least
    data/gradient signal to fight that shrinkage pressure.

    Solution: give the region-conditioning branch its own AdamW param
    group with weight_decay=0, and keep weight decay only on the shared
    trunk + classifier head, where it's doing its normal job.
    """
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
    # Re-seed per config so each Optuna trial starts from the same weight
    # initialization, making trials comparable to each other rather than
    # confounded with leftover RNG state from the previous trial.
    set_global_seed(seed)
    model = SFTNN(n_features, n_regions, hidden_dim=params["hidden_dim"],
                  n_layers=params["n_layers"], embed_dim=params["embed_dim"],
                  dropout=params["dropout"]).to(DEVICE)
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
            # Fix #3: pull each region's embedding toward the population
            # mean, weighted by 1/n_r, so small-n regions (South/Southeast
            # Asia, Africa) can't drift as far from the pack as large-n
            # regions (North America, which has earned the right to, per
            # the ablation results) can.
            loss = (loss_fn(logits, y)
                    + shrink_lambda * model.region_shrinkage_penalty(region_counts)
                    # Experiment #9: additional pull toward each region's
                    # CLUSTER mean (see REGION_CLUSTERS above), so small
                    # regions can also borrow strength from geographically/
                    # economically similar regions, not only the global mean.
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
        # Composite selection score (Section 4.7): replaces pooled
        # validation ROC-AUC alone, which is implicitly NA/Europe-weighted
        # at ~63% of the data. See data_utils.composite_validation_score.
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

    # Fix #3: how much training data each region actually has, in
    # region-id order (0..n_regions-1) so it lines up with
    # region_embedding.weight's row order. Regions with fewer training
    # rows get shrunk harder toward the population-mean embedding.
    region_counts = torch.tensor(
        [ (train_df["region_id"] == r).sum() for r in range(n_regions) ],
        dtype=torch.float32,
    ).to(DEVICE)
    print(f"Region training counts (id order): {region_counts.cpu().tolist()}")

    # Experiment #9: map each region_id to its fixed, a-priori cluster id
    # (see REGION_CLUSTERS above), in region_id order so it lines up with
    # region_embedding.weight's row order.
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
            # Experiment #7: restricted to the larger-capacity choices
            # explored in #4 (32 and n_layers=1 dropped from the search).
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [2, 3]),
            "embed_dim": trial.suggest_categorical("embed_dim", [4, 8, 16]),
            # Fix #3: strength of the shrinkage penalty. Let Optuna find
            # the right amount rather than hand-picking one value -- too
            # weak and small-n regions keep drifting to bad places, too
            # strong and every region gets flattened toward the mean
            # (which would also erase Latin America/North America's
            # genuine, ablation-confirmed gains).
            #
            # Range rescaled to 1.0-50.0 (Experiment #6), the mandatory
            # companion to switching region_shrinkage_penalty from sum(dim=1)
            # to mean(dim=1) in models.py -- NOT an independent change.
            # Averaging divides the penalty by embed_dim, so a numerically
            # equal shrink_lambda now applies ~embed_dim times LESS pressure
            # than it did under the old sum-based formula. Converting the
            # three prior runs to mean-equivalent units (lambda * embed_dim
            # at each run's own embed_dim) makes this concrete: the
            # confirmed collapse zone (Experiment #4: 0.00625@D=16; fix #4:
            # 0.0013@D=4) sits at ~0.10 and ~0.005 respectively, while the
            # one run with genuinely good regional behavior (pre-Experiment-4,
            # plain-AUC objective) sits at ~24.98 (6.245@D=4). The OLD
            # 0.3-10 range (Experiment #5) could not even reach 24.98 under
            # the new mean-based formula -- its ceiling of 10 sits below the
            # one known-good value. 1.0 is ~10x above the confirmed collapse
            # zone; 50.0 is ~2x above the known-good value, leaving room to
            # find something better rather than just recovering the old
            # optimum. Derived entirely from these prior VALIDATION-time
            # observations, not from this run's test results.
            "shrink_lambda": trial.suggest_float("shrink_lambda", 1.0, 50.0, log=True),
            # Experiment #9: strength of the new cluster-level shrinkage
            # term. Same mathematical form and 1/n_r weighting as
            # shrink_lambda, just pulling toward a cluster mean (2-3
            # regions) instead of the global mean (7 regions) -- so the
            # natural distance scale is comparable, and the existing
            # 1.0-50.0 range is reused rather than inventing a new,
            # unjustified range for it.
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
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())
"""
02_train_mlp_exp17.py
======================
Experiment 17 variant of 02_train_mlp.py (NOT a modification of the
original -- 02_train_mlp.py / models.py / Experiment 13's artifacts are
untouched). Only change from the original: the training loss is a
per-sample, inverse-region-frequency-weighted BCE instead of plain pooled
BCE (see region_reweighting.py for the "why"). Search space, budget,
early-stopping rule, and the composite validation-selection score are all
identical to 02_train_mlp.py so Experiment 17 isolates exactly one
variable against Experiment 13.

Run:
    python 02_train_mlp_exp17.py --trials 20 --max_epochs 60 --artifacts artifacts_exp17
"""
import argparse
import json
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from data_utils import (load_splits, StartupDataset, set_global_seed,
                         seeded_generator, composite_validation_score)
from models import StandardMLP
from region_reweighting import WeightedStartupDataset, region_weight_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_config(params, train_ds, val_ds, n_features, n_regions,
                      max_epochs, patience=5, verbose=False, seed=42):
    set_global_seed(seed)
    model = StandardMLP(n_features, n_regions, hidden_dim=params["hidden_dim"],
                         n_layers=params["n_layers"], dropout=params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                             weight_decay=params["weight_decay"])
    # Experiment 17 (P1): reduction="none" so each sample's loss can be
    # scaled by its region weight before averaging -- see train_loader
    # below. Everything else about the optimizer/model is unchanged from
    # 02_train_mlp.py.
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

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
            loss = (loss_fn(logits, y) * w).mean()
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
        # Composite selection score is UNCHANGED/unweighted -- only the
        # training loss above is reweighted, so selection and evaluation
        # stay directly comparable to Experiment 13.
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

    print("Experiment 17 region weights (train split, power="
          f"{args.weight_power}):")
    for name, info in region_weight_table(train_df["region_id"].values, n_regions,
                                           manifest["regions"], power=args.weight_power).items():
        print(f"  {name:16s} n={info['count']:6d}  weight={info['weight']:.3f}")

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [1, 2, 3]),
        }
        _, val_score = train_one_config(params, train_ds, val_ds, n_features,
                                         n_regions, max_epochs=args.max_epochs,
                                         seed=args.seed)
        return val_score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val composite score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    final_model, final_val_score = train_one_config(
        study.best_params, train_ds, val_ds, n_features, n_regions,
        max_epochs=args.max_epochs, verbose=True, seed=args.seed,
    )
    torch.save(final_model.state_dict(), f"{args.artifacts}/mlp_model.pt")
    with open(f"{args.artifacts}/mlp_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved trained MLP to {args.artifacts}/mlp_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp17")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--weight_power", type=float, default=0.5,
                   help="Region reweighting exponent: 0.5=sqrt-inverse-frequency "
                        "(default), 1.0=raw inverse frequency, 0.0=disabled "
                        "(reduces to the original unweighted 02_train_mlp.py).")
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/numpy/random AND the Optuna sampler. "
                         "Vary via 09_multiseed.py to measure training variance; "
                         "do not hand-pick a favourable value.")
    main(p.parse_args())

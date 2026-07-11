"""
02_train_mlp.py
================
Stage 2 of the six-stage development model (Section 4.6): the region-blind
base control MLP that establishes the performance floor (Section 4.7).

Hyperparameter search space and budget follow Section 4.7 exactly:
  - AdamW optimizer
  - learning rate: log-uniform 1e-4 to 1e-2
  - dropout: {0.1, 0.2, 0.3}
  - weight decay: {1e-5, 1e-4, 1e-3}
  - ~20 Optuna TPE trials
  - early stopping on validation AUC-ROC, patience = 5 epochs

Run:
    python 02_train_mlp.py --trials 20 --max_epochs 60
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data_utils import load_splits, StartupDataset
from models import StandardMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_config(params, train_ds, val_ds, n_features, n_regions,
                      max_epochs, patience=5, verbose=False):
    model = StandardMLP(n_features, n_regions, hidden_dim=params["hidden_dim"],
                         n_layers=params["n_layers"], dropout=params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"],
                             weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    best_auc, best_state, epochs_no_improve = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, r, y in train_loader:
            x, r, y = x.to(DEVICE), r.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x, r)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, r, y in val_loader:
                x, r = x.to(DEVICE), r.to(DEVICE)
                all_logits.append(model(x, r).cpu())
                all_y.append(y)
        val_auc = roc_auc_score(torch.cat(all_y), torch.sigmoid(torch.cat(all_logits)))

        if verbose:
            print(f"  epoch {epoch:02d}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_state, epochs_no_improve = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    train_ds = StartupDataset(train_df, feature_cols)
    val_ds = StartupDataset(val_df, feature_cols)

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [1, 2, 3]),
        }
        _, val_auc = train_one_config(params, train_ds, val_ds, n_features,
                                       n_regions, max_epochs=args.max_epochs)
        return val_auc

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Retrain final model on train, using the best config found.
    final_model, final_val_auc = train_one_config(
        study.best_params, train_ds, val_ds, n_features, n_regions,
        max_epochs=args.max_epochs, verbose=True,
    )
    torch.save(final_model.state_dict(), f"{args.artifacts}/mlp_model.pt")
    with open(f"{args.artifacts}/mlp_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved trained MLP to {args.artifacts}/mlp_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max_epochs", type=int, default=60)
    main(p.parse_args())

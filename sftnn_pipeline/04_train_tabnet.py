"""
04_train_tabnet.py
====================
Stage 3 of the six-stage development model (Section 4.6): TabNet
non-neural-network-style baseline (Section 4.7). Region is one-hot
encoded and appended as an input feature, exactly as with the MLP and
XGBoost baselines.

Search space (Section 4.7):
  - n_d, n_a (jointly): {8, 16, 24}
  - n_steps: {3, 5, 7}
  - gamma: {1.0, 1.3, 1.5}
  - ~20 Optuna trials, early stopping patience = 5 epochs

Run:
    python 04_train_tabnet.py --trials 20 --max_epochs 60
"""
import argparse
import json
import numpy as np
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from data_utils import load_splits, xy_for_sklearn, composite_validation_score


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_regions = manifest["n_regions"]

    X_train, y_train = xy_for_sklearn(train_df, feature_cols, n_regions=n_regions)
    X_val, y_val = xy_for_sklearn(val_df, feature_cols, n_regions=n_regions)
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    val_region_ids = val_df["region_id"].values

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial):
        nd_na = trial.suggest_categorical("n_d_n_a", [8, 16, 24])
        params = dict(
            n_d=nd_na, n_a=nd_na,
            n_steps=trial.suggest_categorical("n_steps", [3, 5, 7]),
            gamma=trial.suggest_categorical("gamma", [1.0, 1.3, 1.5]),
            seed=args.seed, device_name=device, verbose=0,
        )
        model = TabNetClassifier(**params)
        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=["auc"],
            max_epochs=args.max_epochs, patience=5, batch_size=1024,
        )
        preds = model.predict_proba(X_val)[:, 1]
        # Composite selection score (Section 4.7) -- see
        # data_utils.composite_validation_score; applied identically across
        # all four training scripts so switching from pooled validation AUC
        # alone doesn't introduce a new, undisclosed asymmetry.
        score, _ = composite_validation_score(y_val, preds, val_region_ids)
        return score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val composite score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    nd_na = study.best_params["n_d_n_a"]
    final_params = dict(
        n_d=nd_na, n_a=nd_na,
        n_steps=study.best_params["n_steps"],
        gamma=study.best_params["gamma"],
        seed=args.seed, device_name=device, verbose=0,
    )
    final_model = TabNetClassifier(**final_params)
    final_model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=["auc"],
        max_epochs=args.max_epochs, patience=5, batch_size=1024,
    )
    final_model.save_model(f"{args.artifacts}/tabnet_model")
    with open(f"{args.artifacts}/tabnet_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved trained TabNet model to {args.artifacts}/tabnet_model.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds the Optuna sampler and TabNet's internal seed. "
                         "Vary via 09_multiseed.py; do not hand-pick.")
    main(p.parse_args())

"""
03_train_xgboost.py
====================
Stage 3 of the six-stage development model (Section 4.6): XGBoost
non-neural baseline (Section 4.7). Region is one-hot encoded and appended
to the feature matrix, exactly mirroring the MLP baseline's treatment of
region, so the comparison isolates architecture rather than feature
representation.

Search space (Section 4.7):
  - max_depth: {4, 6, 8}
  - learning_rate: log-uniform 1e-3 to 3e-1
  - n_estimators: {200, 400, 600}
  - subsample: {0.7, 0.85, 1.0}
  - ~20 Optuna trials, early stopping = 5 rounds without improvement

Run:
    python 03_train_xgboost.py --trials 20
"""
import argparse
import json
import joblib
import numpy as np
import optuna
import xgboost as xgb

from data_utils import load_splits, xy_for_sklearn, composite_validation_score


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_regions = manifest["n_regions"]

    X_train, y_train = xy_for_sklearn(train_df, feature_cols, n_regions=n_regions)
    X_val, y_val = xy_for_sklearn(val_df, feature_cols, n_regions=n_regions)
    val_region_ids = val_df["region_id"].values

    def objective(trial):
        params = {
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.85, 1.0]),
            "eval_metric": "auc",
            "early_stopping_rounds": 5,
            # Seeded alongside the neural models so all four vary together
            # under 09_multiseed.py -- otherwise XGBoost would stay fixed
            # while MLP/SFTNN moved, making the comparison asymmetric.
            "random_state": args.seed,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        # Composite selection score (Section 4.7) -- see
        # data_utils.composite_validation_score; applied identically across
        # all four training scripts (MLP/XGBoost/TabNet/SFTNN) so switching
        # from pooled validation AUC alone doesn't introduce a new,
        # undisclosed asymmetry favoring any one model.
        score, _ = composite_validation_score(y_val, preds, val_region_ids)
        return score

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\nBest val composite score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = dict(study.best_params)
    best_params["eval_metric"] = "auc"
    best_params["early_stopping_rounds"] = 5
    best_params["random_state"] = args.seed
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    final_model.save_model(f"{args.artifacts}/xgboost_model.json")
    with open(f"{args.artifacts}/xgboost_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved trained XGBoost model to {args.artifacts}/xgboost_model.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds the Optuna sampler and XGBoost's random_state. "
                         "Vary via 09_multiseed.py; do not hand-pick.")
    main(p.parse_args())

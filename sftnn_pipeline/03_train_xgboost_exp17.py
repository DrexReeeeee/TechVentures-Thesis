"""
03_train_xgboost_exp17.py
===========================
Experiment 17 variant of 03_train_xgboost.py (NOT a modification of the
original -- 03_train_xgboost.py and Experiment 13's artifacts are
untouched). Only change: XGBoost's .fit() receives the SAME per-sample
region weights used by the MLP/SFTNN exp17 variants (sample_weight=),
applied to both the Optuna objective's fit and the final retrain, and to
both train and the eval_set's weight (so early-stopping/val AUC inside
XGBoost is scored on the same weighting its own training used). Search
space and budget are otherwise identical to 03_train_xgboost.py.

This exists so the comparison stays fair across all four models (see
region_reweighting.py): if only SFTNN got reweighted training, an
improvement wouldn't tell you whether SFTNN's architecture is better or
just that reweighting helps any model on this imbalanced-by-region data.

Run:
    python 03_train_xgboost_exp17.py --trials 20 --artifacts artifacts_exp17
"""
import argparse
import json
import optuna
import xgboost as xgb

from data_utils import load_splits, xy_for_sklearn, composite_validation_score
from region_reweighting import region_sample_weights, region_weight_table


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_regions = manifest["n_regions"]

    X_train, y_train = xy_for_sklearn(train_df, feature_cols, n_regions=n_regions)
    X_val, y_val = xy_for_sklearn(val_df, feature_cols, n_regions=n_regions)
    val_region_ids = val_df["region_id"].values

    train_weights = region_sample_weights(train_df["region_id"].values, n_regions,
                                           power=args.weight_power)
    print(f"Experiment 17 region weights (train split, power={args.weight_power}):")
    for name, info in region_weight_table(train_df["region_id"].values, n_regions,
                                           manifest["regions"], power=args.weight_power).items():
        print(f"  {name:16s} n={info['count']:6d}  weight={info['weight']:.3f}")

    def objective(trial):
        params = {
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600]),
            "subsample": trial.suggest_categorical("subsample", [0.7, 0.85, 1.0]),
            "eval_metric": "auc",
            "early_stopping_rounds": 5,
            "random_state": args.seed,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=train_weights,
                  eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
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
    final_model.fit(X_train, y_train, sample_weight=train_weights,
                     eval_set=[(X_val, y_val)], verbose=False)

    final_model.save_model(f"{args.artifacts}/xgboost_model.json")
    with open(f"{args.artifacts}/xgboost_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved trained XGBoost model to {args.artifacts}/xgboost_model.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp17")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--weight_power", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42,
                    help="Seeds the Optuna sampler and XGBoost's random_state. "
                         "Vary via 09_multiseed.py; do not hand-pick.")
    main(p.parse_args())

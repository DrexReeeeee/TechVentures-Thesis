"""
03_train_xgboost_exp24.py
============================
Control check for Experiment 23: XGBoost + SFTNN's prediction as a
feature (generate_sftnn_stacking_feature.py), roles reversed from
Experiment 23. Same search space/budget as 03_train_xgboost.py.

Run:
    python 03_train_xgboost_exp24.py --trials 20 --artifacts artifacts_exp24
"""
import argparse
import json
import numpy as np
import optuna
import xgboost as xgb

from data_utils import load_splits, composite_validation_score


def xy(df, feature_cols, n_regions):
    X = df[feature_cols].values
    region_onehot = np.eye(n_regions)[df["region_id"].values.astype(int)]
    return np.concatenate([X, region_onehot], axis=1), df["derived_target"].values


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]  # 26: 25 original + sftnn_stack_prob
    n_regions = manifest["n_regions"]

    X_train, y_train = xy(train_df, feature_cols, n_regions)
    X_val, y_val = xy(val_df, feature_cols, n_regions)
    val_region_ids = val_df["region_id"].values

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
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
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
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    final_model.save_model(f"{args.artifacts}/xgboost_model.json")
    with open(f"{args.artifacts}/xgboost_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Saved trained XGBoost to {args.artifacts}/xgboost_model.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts_exp24")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

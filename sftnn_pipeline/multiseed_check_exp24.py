"""
multiseed_check_exp24.py
===========================
The actual control check: does XGBoost + SFTNN-as-a-feature improve by a
similar margin to Experiment 23 (SFTNN + XGBoost-as-a-feature)? If yes,
Experiment 23's result is a generic stacking effect, not SFTNN-specific.
If XGBoost barely moves, SFTNN's region-conditioning is doing something
the combination doesn't just hand to any model for free.

XGBoost trains in seconds, so this is cheap: 20 seeds, same fixed
best-hyperparameters, only random_state varies -- no retraining of the
expensive SFTNN-based stacking feature itself (generated once, already
leakage-safe).

Run:
    python multiseed_check_exp24.py --seeds 1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12
"""
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits

PARAMS_13_XGB = json.load(open("artifacts_exp13/xgboost_best_params.json"))
PARAMS_24_XGB = json.load(open("artifacts_exp24/xgboost_best_params.json"))


def xy(df, feature_cols, n_regions):
    X = df[feature_cols].values
    region_onehot = np.eye(n_regions)[df["region_id"].values.astype(int)]
    return np.concatenate([X, region_onehot], axis=1), df["derived_target"].values


def metrics(yt, yp):
    pred = (yp >= 0.5).astype(int)
    out = dict(Accuracy=accuracy_score(yt, pred), Precision=precision_score(yt, pred, zero_division=0),
               Recall=recall_score(yt, pred, zero_division=0), F1=f1_score(yt, pred, zero_division=0))
    if len(set(yt)) > 1:
        out["ROC-AUC"] = roc_auc_score(yt, yp)
        out["PR-AUC"] = average_precision_score(yt, yp)
    else:
        out["ROC-AUC"] = out["PR-AUC"] = float("nan")
    return out


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    data_13 = load_splits("artifacts_exp13")   # plain XGBoost, 25 features
    data_24 = load_splits("artifacts_exp24")   # XGBoost + sftnn_stack_prob, 26 features

    configs = [("XGBoost-plain", PARAMS_13_XGB, data_13), ("XGBoost+SFTNN-feature", PARAMS_24_XGB, data_24)]
    regions_to_report = ["Africa", "Southeast Asia", "South Asia", "Latin America"]

    rows = []
    for seed in seeds:
        for label, params, (train_df, val_df, test_df, manifest) in configs:
            feature_cols = manifest["feature_cols"]
            n_regions = manifest["n_regions"]
            X_train, y_train = xy(train_df, feature_cols, n_regions)
            X_val, y_val = xy(val_df, feature_cols, n_regions)
            X_test, y_test = xy(test_df, feature_cols, n_regions)
            test_region = test_df["region_group"].values

            p = dict(params)
            p["random_state"] = seed
            p["eval_metric"] = "auc"
            model = xgb.XGBClassifier(**p)
            model.fit(X_train, y_train, verbose=False)
            probs = model.predict_proba(X_test)[:, 1]

            scopes = [("AGGREGATE", np.ones(len(y_test), dtype=bool))] + \
                     [(r, (test_region == r)) for r in regions_to_report]
            for scope_name, mask in scopes:
                m = metrics(y_test[mask], probs[mask])
                rows.append({"seed": seed, "config": label, "scope": scope_name, **m})

        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds\n{'='*70}")
    for scope in ["AGGREGATE"] + regions_to_report:
        print(f"\n--- {scope} ---")
        for label, _, _ in configs:
            sub = df[(df.scope == scope) & (df.config == label)]
            line = f"  {label:24s}"
            for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
                line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
            print(line)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12")
    p.add_argument("--out", default="multiseed_exp24_raw.csv")
    main(p.parse_args())

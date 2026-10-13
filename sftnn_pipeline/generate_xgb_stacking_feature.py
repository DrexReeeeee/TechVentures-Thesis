"""
generate_xgb_stacking_feature.py
===================================
Experiment 23: builds the leakage-safe input SFTNN needs to use XGBoost's
opinion as a feature, so SFTNN -- through its own region-conditioned
gamma/beta -- can learn how much to trust that opinion per region,
rather than a routing layer replacing SFTNN's output outright.

Leakage trap this avoids: naively adding the existing
artifacts_exp13/xgboost_model.json's predictions on the TRAIN rows it was
itself trained on would hand SFTNN an artificially perfect signal (the
model has already seen those labels), teaching it to over-trust something
that won't be as reliable on new data. Standard fix -- K-fold stacking:

  - TRAIN rows: 5-fold CV. For each fold, train a fresh XGBoost (same
    already-chosen hyperparameters) on the OTHER 4 folds, predict on the
    held-out fold. Every train row's "xgb_stack_prob" comes from a model
    that never saw that row during training.
  - VAL/TEST rows: use the existing, already-trained
    artifacts_exp13/xgboost_model.json directly -- no leakage concern
    there, since val/test were never part of its training data.

Output: artifacts_exp23/{train,val,test}.csv with one new feature column
(xgb_stack_prob), plus an updated feature_manifest.json. Everything else
(preprocessing, splits, other features) is an exact, untouched copy of
artifacts_exp13 -- this experiment isolates adding ONE new feature.

Run:
    python generate_xgb_stacking_feature.py
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from data_utils import load_splits, xy_for_sklearn

SRC = "artifacts_exp13"
DST = "artifacts_exp23"
N_FOLDS = 5
SEED = 42


def main():
    os.makedirs(DST, exist_ok=True)
    train_df, val_df, test_df, manifest = load_splits(SRC)
    feature_cols = manifest["feature_cols"]
    n_regions = manifest["n_regions"]

    with open(f"{SRC}/xgboost_best_params.json") as f:
        best_params = json.load(f)

    X_train, y_train = xy_for_sklearn(train_df, feature_cols, n_regions=n_regions)
    X_val, y_val = xy_for_sklearn(val_df, feature_cols, n_regions=n_regions)
    X_test, y_test = xy_for_sklearn(test_df, feature_cols, n_regions=n_regions)

    # ---- TRAIN: 5-fold out-of-fold predictions (leakage-safe) ----
    oof_prob = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(X_train, y_train)):
        params = dict(best_params)
        params["random_state"] = SEED
        model = xgb.XGBClassifier(**params)
        model.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        oof_prob[ho_idx] = model.predict_proba(X_train[ho_idx])[:, 1]
        print(f"  fold {fold}: trained on {len(tr_idx)}, predicted {len(ho_idx)} held-out rows")

    # ---- VAL/TEST: use the existing, already-trained XGBoost model ----
    final_model = xgb.XGBClassifier()
    final_model.load_model(f"{SRC}/xgboost_model.json")
    val_prob = final_model.predict_proba(X_val)[:, 1]
    test_prob = final_model.predict_proba(X_test)[:, 1]

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    train_out["xgb_stack_prob"] = oof_prob
    val_out["xgb_stack_prob"] = val_prob
    test_out["xgb_stack_prob"] = test_prob

    train_out.to_csv(f"{DST}/train.csv", index=False)
    val_out.to_csv(f"{DST}/val.csv", index=False)
    test_out.to_csv(f"{DST}/test.csv", index=False)

    new_manifest = dict(manifest)
    new_manifest["feature_cols"] = feature_cols + ["xgb_stack_prob"]
    new_manifest["n_features"] = len(new_manifest["feature_cols"])
    with open(f"{DST}/feature_manifest.json", "w") as f:
        json.dump(new_manifest, f, indent=2)

    print(f"\nAUC check -- does the stacking feature itself look sane:")
    from sklearn.metrics import roc_auc_score
    print(f"  train (out-of-fold) AUC: {roc_auc_score(y_train, oof_prob):.4f}")
    print(f"  val AUC: {roc_auc_score(y_val, val_prob):.4f}")
    print(f"  test AUC: {roc_auc_score(y_test, test_prob):.4f}")
    print(f"\nSaved to {DST}/ -- {len(new_manifest['feature_cols'])} features "
          f"(was {len(feature_cols)}, +1: xgb_stack_prob)")


if __name__ == "__main__":
    main()

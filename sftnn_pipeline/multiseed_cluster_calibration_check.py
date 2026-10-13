"""
multiseed_cluster_calibration_check.py
=========================================
Validates, across multiple seeds, the post-hoc Asia-Pacific-cluster
threshold calibration found by region_threshold_calibration.py at the
single seed=42 snapshot (Southeast Asia F1 .567->.613, South Asia F1
.537->.592, Africa left untouched). That snapshot is one seed -- this
checks whether the effect is real and consistent, the same discipline
applied to every other mechanism tested today.

Evaluation-only on top of an unmodified training recipe: reuses
train_fixed() from multiseed_check_13_19_20_21.py unchanged (no new
training mechanism). After training each (seed, config) pair, fits ONE
threshold on VALIDATION probabilities pooled across the Asia-Pacific
cluster (East Asia + South Asia + Southeast Asia) via F1-maximizing grid
search, then applies that threshold ONLY to Asia-Pacific TEST rows --
Africa, Latin America, Europe, North America stay at the default 0.5,
since calibration was already shown (region_threshold_calibration.py,
checked against Exp13/17 earlier) to not help those regions.

Run:
    python multiseed_cluster_calibration_check.py --seeds 1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12
"""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits, StartupDataset
from multiseed_check_13_19_20_21 import train_fixed, REGION_CLUSTERS, PARAMS_13, PARAMS_21

DEVICE = "cpu"
ASIA_PACIFIC = {"East Asia", "South Asia", "Southeast Asia"}


def best_f1_threshold(y_true, y_prob, grid=np.arange(0.05, 0.96, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def metrics_from_pred(yt, pred, yp):
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
    train_df, val_df, test_df, manifest = load_splits("artifacts_exp13")
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]
    region_names = manifest["regions"]

    train_ds = StartupDataset(train_df, feature_cols)
    val_ds = StartupDataset(val_df, feature_cols)

    region_counts = torch.tensor(
        [(train_df["region_id"] == r).sum() for r in range(n_regions)], dtype=torch.float32).to(DEVICE)
    cluster_names = sorted(set(REGION_CLUSTERS.values()))
    cname_to_id = {c: i for i, c in enumerate(cluster_names)}
    cluster_ids = torch.tensor([cname_to_id[REGION_CLUSTERS[r]] for r in region_names], dtype=torch.long).to(DEVICE)

    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
    r_val = torch.tensor(val_df["region_id"].values, dtype=torch.long)
    y_val = val_df["derived_target"].values
    val_df_reset = val_df.reset_index(drop=True)
    val_ap_mask = val_df_reset["region_group"].isin(ASIA_PACIFIC).values

    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    y_test = test_df["derived_target"].values
    test_df_reset = test_df.reset_index(drop=True)
    test_ap_mask = test_df_reset["region_group"].isin(ASIA_PACIFIC).values

    regions_to_report = ["Africa", "Southeast Asia", "South Asia", "Latin America"]
    configs = [("Exp13-config", PARAMS_13, "exp13"), ("Exp21-config", PARAMS_21, "exp21")]

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        for label, params, variant in configs:
            print(f"  training {label} ...", flush=True)
            model = train_fixed(params, seed, variant, train_ds, val_ds, n_features, n_regions,
                                 region_counts, cluster_ids, region_names)
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(x_val, r_val)).numpy()
                test_probs = torch.sigmoid(model(x_test, r_test)).numpy()

            ap_threshold = best_f1_threshold(y_val[val_ap_mask], val_probs[val_ap_mask])

            pred_default = (test_probs >= 0.5).astype(int)
            pred_calibrated = np.where(test_ap_mask, (test_probs >= ap_threshold).astype(int), pred_default)

            scopes = [("AGGREGATE", np.ones(len(y_test), dtype=bool))] + \
                     [(r, (test_df_reset.region_group == r).values) for r in regions_to_report]
            for scope_name, mask in scopes:
                m_def = metrics_from_pred(y_test[mask], pred_default[mask], test_probs[mask])
                rows.append({"seed": seed, "config": label, "scope": scope_name, "variant": "default", **m_def})
                m_cal = metrics_from_pred(y_test[mask], pred_calibrated[mask], test_probs[mask])
                rows.append({"seed": seed, "config": label, "scope": scope_name, "variant": "cluster_calibrated",
                             "ap_threshold": ap_threshold, **m_cal})

            print(f"    ap_threshold={ap_threshold:.2f}")
            for region in ["Southeast Asia", "South Asia", "Africa"]:
                mask = (test_df_reset.region_group == region).values
                f1_def = f1_score(y_test[mask], pred_default[mask], zero_division=0)
                f1_cal = f1_score(y_test[mask], pred_calibrated[mask], zero_division=0)
                print(f"    {region:15s} F1 default={f1_def:.4f} calibrated={f1_cal:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds\n{'='*70}")
    for scope in ["AGGREGATE"] + regions_to_report:
        print(f"\n--- {scope} ---")
        for label, _, _ in configs:
            for variant in ["default", "cluster_calibrated"]:
                sub = df[(df.scope == scope) & (df.config == label) & (df.variant == variant)]
                line = f"  {label:14s} {variant:20s}"
                for m in ["F1", "Recall", "ROC-AUC", "PR-AUC"]:
                    line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
                print(line)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12")
    p.add_argument("--out", default="multiseed_cluster_calibration_raw.csv")
    main(p.parse_args())

"""
multiseed_check_gate.py
=========================
Resolves the open question: is Experiment 13's gated (region_alpha
dampening) mechanism's strong Africa/Aggregate result at seed=42 real,
or is it the same seed-42 luck that the non-gated Exp13 config turned
out to have? Reuses train_fixed(use_gate=True/False) from
multiseed_check_9_vs_13.py unchanged -- no new training mechanism.

Run:
    python multiseed_check_gate.py --seeds 1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)

from data_utils import load_splits, StartupDataset, set_global_seed
from multiseed_check_9_vs_13 import train_fixed, REGION_CLUSTERS

DEVICE = "cpu"
PARAMS_13 = json.load(open("artifacts_exp13/sftnn_best_params.json"))


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

    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_test = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    y_test = test_df["derived_target"].values
    test_df_reset = test_df.reset_index(drop=True)

    regions_to_report = ["Africa", "Southeast Asia", "South Asia", "Latin America"]

    rows = []
    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)
        for label, use_gate in [("Exp13-nogate", False), ("Exp13-gated", True)]:
            print(f"  training {label} ...", flush=True)
            model = train_fixed(PARAMS_13, seed, use_gate, train_ds, val_ds, n_features, n_regions,
                                 region_counts, cluster_ids)
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x_test, r_test)).numpy()

            agg_m = metrics(y_test, probs)
            rows.append({"seed": seed, "config": label, "scope": "AGGREGATE", **agg_m})
            print(f"    AGGREGATE F1={agg_m['F1']:.4f} Recall={agg_m['Recall']:.4f}")

            for region in regions_to_report:
                mask = (test_df_reset.region_group == region).values
                m = metrics(y_test[mask], probs[mask])
                rows.append({"seed": seed, "config": label, "scope": region, **m})
                print(f"    {region:15s} F1={m['F1']:.4f} Recall={m['Recall']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*70}\nSUMMARY: mean +/- std across {len(seeds)} seeds\n{'='*70}")
    for scope in ["AGGREGATE"] + regions_to_report:
        print(f"\n--- {scope} ---")
        for label in ["Exp13-nogate", "Exp13-gated"]:
            sub = df[(df.scope == scope) & (df.config == label)]
            line = f"  {label:14s}"
            for m in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
                line += f"  {m}={sub[m].mean():.4f}+/-{sub[m].std():.4f}"
            print(line)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,7,123,2024,777,999,55,314,2718,31,2,3,4,5,6,8,9,10,11,12")
    p.add_argument("--out", default="multiseed_gate_raw.csv")
    main(p.parse_args())

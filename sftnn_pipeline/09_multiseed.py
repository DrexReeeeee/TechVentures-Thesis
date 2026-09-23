"""
09_multiseed.py
================
Training-variance protocol: retrain all four models end-to-end under several
seeds and report each metric as mean +/- std ACROSS seeds, plus how often each
model wins each metric.

Why this exists
---------------
Until the seeding fix landed in data_utils.set_global_seed(), StandardMLP and
SFTNN drew weight initialization and DataLoader batch order from an unseeded
global RNG, so their results moved run to run while XGBoost and TabNet (which
seeds itself) stayed effectively fixed. That made any single run's ranking
partly an accident of RNG state -- and made it tempting to read whichever run
looked best as "the" result.

Reporting the max over N runs does not estimate a model's performance; it
estimates the maximum of N draws, which is biased upward and gets more biased
the more runs you look at. This script reports the DISTRIBUTION instead, which
is the honest object: if SFTNN wins a metric on 2 of 5 seeds, the finding is
"within seed variance," and that is a legitimate, reportable result.

Relationship to 08_bootstrap_ci.py
----------------------------------
They measure different, complementary things and neither substitutes for the
other:
  - 08 resamples the TEST SET with the trained models held fixed
    -> test-set sampling variance.
  - 09 retrains from scratch under different seeds on the SAME split
    -> training variance (init, batch order, Optuna trajectory).
Neither captures train/val/test PARTITION variance, which would require
re-running 01_preprocess.py with different split seeds. That remains a
disclosed limitation.

Run:
    python 09_multiseed.py --seeds 1,2,3,4,5
    python 09_multiseed.py --seeds 1,2,3 --trials_nn 15 --trials_base 10   # faster
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
MODELS = ["Standard MLP (baseline)", "XGBoost (baseline)",
          "TabNet (baseline)", "Proposed SFTNN"]


def run(cmd):
    print(f"    $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res.stdout


def main(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    results_dir = Path(args.results)
    artifacts = Path(args.artifacts)
    per_seed_dir = results_dir / "multiseed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    agg_frames, region_frames = [], []

    for seed in seeds:
        print(f"\n{'='*66}\n  SEED {seed}\n{'='*66}", flush=True)
        py = sys.executable
        run([py, "02_train_mlp.py", "--trials", str(args.trials_base),
             "--max_epochs", str(args.max_epochs), "--seed", str(seed),
             "--artifacts", str(artifacts)])
        run([py, "03_train_xgboost.py", "--trials", str(args.trials_base),
             "--seed", str(seed), "--artifacts", str(artifacts)])
        run([py, "04_train_tabnet.py", "--trials", str(args.trials_base),
             "--max_epochs", str(args.max_epochs), "--seed", str(seed),
             "--artifacts", str(artifacts)])
        run([py, "05_train_sftnn.py", "--trials", str(args.trials_nn),
             "--max_epochs", str(args.max_epochs), "--seed", str(seed),
             "--artifacts", str(artifacts)])
        run([py, "06_evaluate.py", "--artifacts", str(artifacts),
             "--results", str(results_dir)])

        agg = pd.read_csv(results_dir / "table_4_6_aggregate.csv")
        reg = pd.read_csv(results_dir / "table_4_7_per_region.csv")
        agg["seed"], reg["seed"] = seed, seed
        agg_frames.append(agg)
        region_frames.append(reg)

        # Preserve this seed's tables so any single run can be re-inspected.
        agg.to_csv(per_seed_dir / f"aggregate_seed{seed}.csv", index=False)
        reg.to_csv(per_seed_dir / f"per_region_seed{seed}.csv", index=False)
        for f in ("sftnn_model.pt", "sftnn_best_params.json"):
            src = artifacts / f
            if src.exists():
                shutil.copy(src, per_seed_dir / f"{Path(f).stem}_seed{seed}{Path(f).suffix}")

    all_agg = pd.concat(agg_frames, ignore_index=True)
    all_reg = pd.concat(region_frames, ignore_index=True)
    all_agg.to_csv(results_dir / "multiseed_aggregate_raw.csv", index=False)
    all_reg.to_csv(results_dir / "multiseed_per_region_raw.csv", index=False)

    # ---- Aggregate: mean +/- std across seeds -------------------------
    summary = all_agg.groupby("Model")[METRICS].agg(["mean", "std"])
    print(f"\n{'='*66}\n  AGGREGATE across {len(seeds)} seeds (mean +/- std)\n{'='*66}")
    rows = []
    for m in MODELS:
        row = {"Model": m}
        for metric in METRICS:
            row[metric] = f"{summary.loc[m, (metric, 'mean')]:.4f} +/- {summary.loc[m, (metric, 'std')]:.4f}"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- Win counts: how often does each model top each metric? -------
    print(f"\n{'='*66}\n  WIN COUNTS -- times each model was best (out of {len(seeds)} seeds)\n{'='*66}")
    win_rows = []
    for metric in METRICS:
        counts = all_agg.loc[all_agg.groupby("seed")[metric].idxmax(), "Model"].value_counts()
        win_rows.append({"Metric": metric, **{m: int(counts.get(m, 0)) for m in MODELS}})
    win_df = pd.DataFrame(win_rows)
    print(win_df.to_string(index=False))
    win_df.to_csv(results_dir / "multiseed_win_counts.csv", index=False)

    # ---- Is SFTNN's gap to the best baseline inside seed noise? -------
    print(f"\n{'='*66}\n  SFTNN vs BEST BASELINE per seed\n{'='*66}")
    gap_rows = []
    for metric in METRICS:
        per_seed_gap = []
        for seed in seeds:
            sub = all_agg[all_agg.seed == seed]
            sftnn = sub.loc[sub.Model == "Proposed SFTNN", metric].iloc[0]
            best_base = sub.loc[sub.Model != "Proposed SFTNN", metric].max()
            per_seed_gap.append(sftnn - best_base)
        per_seed_gap = np.array(per_seed_gap)
        gap_rows.append({
            "Metric": metric,
            "mean_gap": per_seed_gap.mean(),
            "std_gap": per_seed_gap.std(ddof=1) if len(seeds) > 1 else np.nan,
            "seeds_SFTNN_best": int((per_seed_gap > 0).sum()),
            "n_seeds": len(seeds),
        })
    gap_df = pd.DataFrame(gap_rows)
    print(gap_df.round(4).to_string(index=False))
    gap_df.to_csv(results_dir / "multiseed_sftnn_vs_best.csv", index=False)

    # ---- Per-region stability, focused on the minority regions --------
    print(f"\n{'='*66}\n  PER-REGION Recall/F1 across seeds (mean +/- std)\n{'='*66}")
    reg_rows = []
    for region in sorted(all_reg["Region"].unique()):
        for m in MODELS:
            sub = all_reg[(all_reg.Region == region) & (all_reg.Model == m)]
            if sub.empty:
                continue
            entry = {"Region": region, "Model": m, "n": int(sub["n"].iloc[0])}
            for metric in METRICS:
                entry[f"{metric}_mean"] = sub[metric].mean()
                entry[f"{metric}_std"] = sub[metric].std(ddof=1) if len(sub) > 1 else np.nan
            reg_rows.append(entry)
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(results_dir / "multiseed_per_region_summary.csv", index=False)
    show = reg_df[["Region", "Model", "n", "Recall_mean", "Recall_std", "F1_mean", "F1_std"]]
    print(show.round(4).to_string(index=False))

    print(f"\nSaved:")
    for f in ["multiseed_aggregate_raw.csv", "multiseed_per_region_raw.csv",
              "multiseed_win_counts.csv", "multiseed_sftnn_vs_best.csv",
              "multiseed_per_region_summary.csv"]:
        print(f"  {results_dir / f}")
    print(f"  {per_seed_dir}/  (per-seed tables and SFTNN checkpoints)")
    print("\nReport the mean +/- std and the win counts. Citing the single "
          "best seed would report the maximum of N draws, not the model's "
          "performance.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1,2,3,4,5",
                    help="Comma-separated seeds, e.g. 1,2,3,4,5")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--results", default="results")
    p.add_argument("--trials_nn", type=int, default=30,
                    help="Optuna trials for SFTNN (methodology default 30)")
    p.add_argument("--trials_base", type=int, default=20,
                    help="Optuna trials for each baseline (methodology default 20)")
    p.add_argument("--max_epochs", type=int, default=60)
    main(p.parse_args())

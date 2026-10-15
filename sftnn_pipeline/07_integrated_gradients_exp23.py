"""
07_integrated_gradients_exp23.py
====================================
Stage 6 extension (Section 4.10.2 / Specific Objective 4): Integrated
Gradients feature attribution (Sundararajan et al., 2017) for the final
locked model (Experiment 23 -- SFTNN + XGBoost-derived stacking feature,
Section 5.5.4). Unlike 07_interpretability.py, which runs on an
un-augmented SFTNN checkpoint to isolate the FiLM region-conditioning
mechanism on its own (Section 5.4.1-5.4.2), this script runs on the final
model itself, since its purpose is to attribute the auxiliary
xgb_stack_prob feature that only exists in that configuration's
26-feature input.

SFTNN takes two inputs -- the continuous feature vector and a discrete
region_id used for the embedding lookup. Captum only attributes over
continuous tensors, so region_id is passed as an additional_forward_arg
and processed one region at a time (all rows sharing the same region_id,
so Captum's internal n_steps expansion of region_id stays aligned with
its expansion of the feature tensor).

Baseline: the all-zero feature vector. For the log1p-and-Robust-Scaled
funding features (Section 4.3.2.3) this approximates the training-median
venture, since Robust Scaling centres on the median; for the binary
funding_total_usd_disclosed flag it is the natural "undisclosed" state;
for xgb_stack_prob it is XGBoost assigning a 0% probability of success.
This mixed interpretation is a known limitation of a single baseline
across heterogeneous feature types and is disclosed in Section 5.4.3
rather than smoothed over.

Produces (in --results):
  1. ig_attributions_by_region_feature.csv -- mean/std |attribution| and
     mean signed attribution for every (region, feature) pair.
  2. ig_validity_checks.csv -- per-region completeness/convergence
     checks (should be small relative to each region's mean output
     range; this is the standard numerical check that the attribution
     run converged rather than silently producing meaningless values).
  3. ig_xgb_stack_prob_vs_baselines.png -- bar chart comparing
     xgb_stack_prob's mean |attribution| against the next-highest
     feature, per region (Figure 19 in Chapter 5).

Run (after generate_xgb_stacking_feature.py and 05_train_sftnn_exp23.py
have produced artifacts_exp23/sftnn_model.pt):
    pip install captum
    python 07_integrated_gradients_exp23.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from data_utils import load_splits
from models import SFTNN

DEVICE = "cpu"
N_STEPS = 50


class ProbWrapper(nn.Module):
    """Wraps SFTNN's logit output in a sigmoid so Integrated Gradients
    attributes P(Success) directly, matching every other probability-based
    result in this study. region_id is accepted as a second forward
    argument (not attributed over) rather than baked into the module as a
    fixed buffer, so Captum can expand it in lockstep with the
    n_steps-interpolated copies of x it builds internally."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, region_id):
        logits = self.model(x, region_id)
        return torch.sigmoid(logits)


def main(args):
    os.makedirs(args.results, exist_ok=True)

    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    with open(f"{args.artifacts}/sftnn_best_params.json") as f:
        p = json.load(f)
    model = SFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"], n_layers=p["n_layers"],
                  embed_dim=p["embed_dim"], dropout=p["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{args.artifacts}/sftnn_model.pt", map_location=DEVICE))
    model.eval()  # critical: disable Dropout so attributions are deterministic

    wrapper = ProbWrapper(model)
    ig = IntegratedGradients(wrapper)

    x_all = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)
    r_all = torch.tensor(test_df["region_id"].values, dtype=torch.long)
    region_group = test_df["region_group"].values

    results_rows, delta_rows = [], []
    for region in sorted(set(region_group)):
        mask = region_group == region
        if mask.sum() < 5:
            continue
        x_region = x_all[mask]
        rid = int(r_all[mask][0].item())
        region_id_tensor = torch.full((x_region.shape[0],), rid, dtype=torch.long)

        baseline = torch.zeros_like(x_region)
        attributions, delta = ig.attribute(x_region, baselines=baseline, n_steps=N_STEPS,
                                            additional_forward_args=(region_id_tensor,),
                                            return_convergence_delta=True)
        attributions = attributions.detach().numpy()
        delta = delta.detach().numpy()

        with torch.no_grad():
            actual_out = wrapper(x_region, region_id_tensor).numpy()
            baseline_out = wrapper(baseline, region_id_tensor).numpy()
        completeness_gap = (actual_out - baseline_out) - attributions.sum(axis=1)

        delta_rows.append({
            "Region": region, "n": int(mask.sum()),
            "mean_abs_convergence_delta": float(np.mean(np.abs(delta))),
            "mean_abs_completeness_gap": float(np.mean(np.abs(completeness_gap))),
            "mean_output_range": float(np.mean(np.abs(actual_out - baseline_out))),
        })

        mean_abs_attr = np.mean(np.abs(attributions), axis=0)
        mean_signed_attr = np.mean(attributions, axis=0)
        std_abs_attr = np.std(np.abs(attributions), axis=0)
        for i, feat in enumerate(feature_cols):
            results_rows.append({
                "Region": region, "Feature": feat,
                "mean_abs_attribution": mean_abs_attr[i],
                "mean_signed_attribution": mean_signed_attr[i],
                "std_abs_attribution": std_abs_attr[i],
            })
        print(f"{region}: n={mask.sum()}, mean|delta|={np.mean(np.abs(delta)):.5f}, "
              f"mean|completeness_gap|={np.mean(np.abs(completeness_gap)):.5f}")

    attr_df = pd.DataFrame(results_rows)
    delta_df = pd.DataFrame(delta_rows)
    attr_df.to_csv(f"{args.results}/ig_attributions_by_region_feature.csv", index=False)
    delta_df.to_csv(f"{args.results}/ig_validity_checks.csv", index=False)

    print("\n=== Validity checks (completeness gap should be small relative to output range) ===")
    print(delta_df.to_string(index=False))

    stack_rows = attr_df[attr_df.Feature == "xgb_stack_prob"].copy()
    ranks = []
    for region in delta_df["Region"]:
        region_sorted = attr_df[attr_df.Region == region].sort_values("mean_abs_attribution", ascending=False)
        ranks.append(list(region_sorted.Feature).index("xgb_stack_prob") + 1)
    stack_rows["rank_of_n_features"] = ranks
    print("\n=== xgb_stack_prob attribution rank, by region ===")
    print(stack_rows[["Region", "mean_abs_attribution", "rank_of_n_features"]].to_string(index=False))

    # ---- Figure: xgb_stack_prob vs. next-highest-attributed feature, per region ----
    region_order = list(delta_df.sort_values("n", ascending=False)["Region"])
    stack = attr_df[attr_df.Feature == "xgb_stack_prob"].set_index("Region").loc[region_order]
    second = (attr_df[attr_df.Feature != "xgb_stack_prob"]
              .sort_values("mean_abs_attribution", ascending=False)
              .groupby("Region").first().loc[region_order])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(region_order))
    width = 0.38
    ax.bar([i - width / 2 for i in x], stack["mean_abs_attribution"], width,
           label="xgb_stack_prob", color="#1F6F5C")
    ax.bar([i + width / 2 for i in x], second["mean_abs_attribution"], width,
           label="Next-highest feature", color="#9CA3AF")
    ax.set_xticks(list(x))
    ax.set_xticklabels(region_order, rotation=30, ha="right")
    ax.set_ylabel("Mean |Integrated Gradients attribution|")
    ax.set_title("xgb_stack_prob attribution vs. next-highest feature, by region")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.results}/ig_xgb_stack_prob_vs_baselines.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {args.results}/ig_attributions_by_region_feature.csv, "
          f"{args.results}/ig_validity_checks.csv, "
          f"{args.results}/ig_xgb_stack_prob_vs_baselines.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_exp23")
    ap.add_argument("--results", default="results_exp23")
    main(ap.parse_args())

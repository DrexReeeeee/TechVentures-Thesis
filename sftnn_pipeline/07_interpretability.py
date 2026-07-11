"""
07_interpretability.py
=======================
Stage 6 of the six-stage development model (Section 4.6): SFTNN-specific
interpretability analysis (Section 4.10.2).

Produces:
  1. gamma_beta_heatmap.png    -- learned gamma_r / beta_r per region and
                                   hidden dimension (direct interpretation,
                                   not post-hoc attribution).
  2. ablation_results.csv       -- counterfactual check: replace each
                                   region's embedding with the mean
                                   embedding and re-score Recall/F1 per
                                   region. A meaningful drop indicates
                                   regional conditioning is doing real work.
  3. region_embedding_pca.png  -- optional PCA projection of the learned
                                   region embedding table (Guo & Berkhahn,
                                   2016 convention).

Run (after 05_train_sftnn.py has produced sftnn_model.pt):
    python 07_interpretability.py
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, f1_score

from data_utils import load_splits
from models import SFTNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_sftnn(n_features, n_regions, artifacts_dir):
    with open(f"{artifacts_dir}/sftnn_best_params.json") as f:
        p = json.load(f)
    model = SFTNN(n_features, n_regions, hidden_dim=p["hidden_dim"],
                  n_layers=p["n_layers"], embed_dim=p["embed_dim"],
                  dropout=p["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(f"{artifacts_dir}/sftnn_model.pt", map_location=DEVICE))
    model.eval()
    return model


def gamma_beta_heatmaps(model, region_names, results_dir):
    gamma, beta = model.get_region_gamma_beta(device=DEVICE)  # [n_regions, hidden_dim]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, 0.4 * len(region_names)) + 2))
    sns.heatmap(gamma, ax=axes[0], cmap="RdBu_r", center=1.0,
                yticklabels=region_names, cbar_kws={"label": "gamma value"})
    axes[0].set_title("Learned scale (\u03b3) per region")
    axes[0].set_xlabel("hidden dimension")

    sns.heatmap(beta, ax=axes[1], cmap="RdBu_r", center=0.0,
                yticklabels=region_names, cbar_kws={"label": "beta value"})
    axes[1].set_title("Learned shift (\u03b2) per region")
    axes[1].set_xlabel("hidden dimension")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/gamma_beta_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved {results_dir}/gamma_beta_heatmap.png")
    return gamma, beta


def counterfactual_ablation(model, test_df, feature_cols, region_to_id, results_dir):
    """Section 4.10.2: replace each startup's learned region embedding with
    the mean embedding across all regions, holding the rest of the network
    fixed, and compare Recall/F1 to the true, region-specific embedding."""
    x = torch.tensor(test_df[feature_cols].values, dtype=torch.float32).to(DEVICE)
    r_true = torch.tensor(test_df["region_id"].values, dtype=torch.long).to(DEVICE)
    y_true = test_df["derived_target"].values

    with torch.no_grad():
        mean_embedding = model.region_embedding.weight.mean(dim=0, keepdim=True)  # [1, embed_dim]

        # True (region-specific) predictions
        h = model.trunk(x)
        e_r_true = model.region_embedding(r_true)
        gb_true = model.affine_generator(e_r_true)
        gamma_true, beta_true = gb_true.chunk(2, dim=-1)
        probs_true = torch.sigmoid(model.head(gamma_true * h + beta_true).squeeze(-1)).cpu().numpy()

        # Counterfactual (mean-embedding) predictions
        e_r_mean = mean_embedding.expand(x.shape[0], -1)
        gb_mean = model.affine_generator(e_r_mean)
        gamma_mean, beta_mean = gb_mean.chunk(2, dim=-1)
        probs_mean = torch.sigmoid(model.head(gamma_mean * h + beta_mean).squeeze(-1)).cpu().numpy()

    rows = []
    id_to_region = {v: k for k, v in region_to_id.items()}
    for region_id, region_name in id_to_region.items():
        mask = (test_df["region_id"] == region_id).values
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        pred_true = (probs_true[mask] >= 0.5).astype(int)
        pred_mean = (probs_mean[mask] >= 0.5).astype(int)
        rows.append({
            "Region": region_name,
            "n": int(mask.sum()),
            "Recall (region-specific)": recall_score(yt, pred_true, zero_division=0),
            "Recall (mean embedding)": recall_score(yt, pred_mean, zero_division=0),
            "F1 (region-specific)": f1_score(yt, pred_true, zero_division=0),
            "F1 (mean embedding)": f1_score(yt, pred_mean, zero_division=0),
        })
    ablation_df = pd.DataFrame(rows)
    ablation_df["Recall_drop"] = ablation_df["Recall (region-specific)"] - ablation_df["Recall (mean embedding)"]
    ablation_df["F1_drop"] = ablation_df["F1 (region-specific)"] - ablation_df["F1 (mean embedding)"]
    ablation_df.to_csv(f"{results_dir}/ablation_results.csv", index=False)
    print(f"\n=== Counterfactual Region-Embedding Ablation (Section 4.10.2) ===")
    print(ablation_df.to_string(index=False))
    print(f"\nSaved {results_dir}/ablation_results.csv")
    return ablation_df


def pca_embedding_plot(model, region_names, results_dir):
    with torch.no_grad():
        emb = model.region_embedding.weight.cpu().numpy()
    if emb.shape[1] < 2:
        print("Embedding dimension < 2, skipping PCA plot.")
        return
    coords = PCA(n_components=2, random_state=42).fit_transform(emb)
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], s=80)
    for i, name in enumerate(region_names):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=9,
                     xytext=(5, 5), textcoords="offset points")
    plt.title("PCA Projection of Learned Region Embeddings")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/region_embedding_pca.png", dpi=150)
    plt.close()
    print(f"Saved {results_dir}/region_embedding_pca.png")


def main(args):
    train_df, val_df, test_df, manifest = load_splits(args.artifacts)
    feature_cols = manifest["feature_cols"]
    n_features, n_regions = len(feature_cols), manifest["n_regions"]

    region_names = manifest["regions"]

    import joblib
    pre = joblib.load(f"{args.artifacts}/preprocessor.joblib")
    region_to_id = pre["region_to_id"]

    model = load_trained_sftnn(n_features, n_regions, args.artifacts)

    gamma_beta_heatmaps(model, region_names, args.results)
    counterfactual_ablation(model, test_df, feature_cols, region_to_id, args.results)
    pca_embedding_plot(model, region_names, args.results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--results", default="results")
    main(p.parse_args())

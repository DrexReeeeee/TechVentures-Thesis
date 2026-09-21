"""
data_utils.py
=============
Shared helpers for loading the Stage-1 preprocessed splits and exposing
them the way each of the four models expects (Section 4.7).
"""
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, recall_score)


def set_global_seed(seed: int):
    """
    Seed every RNG the training scripts touch, so a given --seed produces a
    reproducible run (Section 4.5.1, Reproducibility).

    Prior to this, the Optuna SAMPLER was seeded (TPESampler(seed=42)) but
    PyTorch was not seeded anywhere: model weight initialization and
    DataLoader(shuffle=True) batch ordering both drew from an unseeded global
    RNG. That made StandardMLP and SFTNN non-deterministic run-to-run while
    XGBoost and TabNet (which seeds itself internally) stayed effectively
    fixed -- an asymmetry that made cross-model comparison unreliable.

    IMPORTANT: determinism is not the same thing as validity. Fixing a seed
    makes one arbitrary draw reproducible; it does not make that draw
    representative. Choosing a seed because it flatters a particular model is
    still selection bias, merely reproducible selection bias. Use
    09_multiseed.py to report performance ACROSS seeds rather than citing any
    single one.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int):
    """Generator for DataLoader(shuffle=True) so batch ORDER is reproducible
    too -- torch.manual_seed alone does not pin a DataLoader's shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def composite_validation_score(y_true, y_prob, region_ids, min_region_n=5,
                                w_auc=0.35, w_prauc=0.25, w_f1=0.25,
                                w_macro_recall=0.15):
    """
    Composite Optuna/early-stopping selection score, applied IDENTICALLY in
    02/03/04/05_train_*.py so that changing what gets selected does not
    introduce a new, undisclosed asymmetry between SFTNN and the baselines.
    Only which validation number is maximized changes -- train still fits
    parameters, validation still selects hyperparameters/epoch, test is
    still untouched until 06_evaluate.py, exactly as before.

    Motivation: every script previously maximized pooled validation ROC-AUC
    alone. With North America at ~63% of the data, that is implicitly an
    NA/Europe-optimized criterion, even though Section 4.10.2's stated
    decision criterion is Recall/F1, especially in minority regions. This
    does not change the training loss or the architecture -- it only makes
    the number Optuna maximizes (and the checkpoint early-stopping keeps)
    closer to what the thesis actually claims to evaluate on, so the search
    stops rewarding configurations that look good only on pooled ranking.

    macro_recall is the unweighted mean of per-region Recall (at the same
    0.5 threshold 06_evaluate.py uses), computed only over regions with at
    least `min_region_n` validation rows -- the same size floor
    06_evaluate.py already applies per-region, so a region isn't included
    on the strength of a handful of rows.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    region_ids = np.asarray(region_ids)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    prauc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    recalls = []
    for r in np.unique(region_ids):
        mask = region_ids == r
        if mask.sum() < min_region_n:
            continue
        recalls.append(recall_score(y_true[mask], y_pred[mask], zero_division=0))
    macro_recall = float(np.mean(recalls)) if recalls else recall_score(y_true, y_pred, zero_division=0)

    score = w_auc * auc + w_prauc * prauc + w_f1 * f1 + w_macro_recall * macro_recall
    parts = {"auc": auc, "pr_auc": prauc, "f1": f1, "macro_recall": macro_recall}
    return score, parts


def load_splits(artifacts_dir="artifacts"):
    train = pd.read_csv(f"{artifacts_dir}/train.csv")
    val = pd.read_csv(f"{artifacts_dir}/val.csv")
    test = pd.read_csv(f"{artifacts_dir}/test.csv")
    with open(f"{artifacts_dir}/feature_manifest.json") as f:
        manifest = json.load(f)
    return train, val, test, manifest


class StartupDataset(Dataset):
    """x, region_id, y as tensors -- used by both the MLP and SFTNN."""

    def __init__(self, df, feature_cols):
        self.x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.region_id = torch.tensor(df["region_id"].values, dtype=torch.long)
        self.y = torch.tensor(df["derived_target"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.region_id[idx], self.y[idx]


def xy_for_sklearn(df, feature_cols, onehot_region=True, n_regions=None):
    """Feature matrix for XGBoost (Section 4.7: region supplied one-hot,
    mirroring the MLP baseline's treatment of region)."""
    X = df[feature_cols].values
    if onehot_region:
        region_onehot = np.eye(n_regions)[df["region_id"].values.astype(int)]
        X = np.concatenate([X, region_onehot], axis=1)
    y = df["derived_target"].values
    return X, y

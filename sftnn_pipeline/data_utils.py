"""
data_utils.py
=============
Shared helpers for loading the Stage-1 preprocessed splits and exposing
them the way each of the four models expects (Section 4.7).
"""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

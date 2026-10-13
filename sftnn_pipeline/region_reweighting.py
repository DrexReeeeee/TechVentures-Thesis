"""
region_reweighting.py
======================
Experiment 17 (P1): inverse-region-frequency per-sample loss weighting.

Diagnosis this addresses (see Experiment 13 analysis): every one of
02/03/04/05_train_*.py minimizes POOLED, unweighted BCE (or an unweighted
tree loss). North America is 67.8% of train.csv (16,799 / 24,767 rows), so
the shared trunk's gradient -- and, for the tree models, split selection --
is overwhelmingly shaped by North America regardless of SFTNN's per-region
gamma/beta mechanism sitting on top of it. The existing shrinkage penalties
in models.py only regularize the region EMBEDDING; they do nothing to
rebalance which samples actually drive training in the first place.

This module computes ONE set of per-sample weights from train-split region
counts only (no val/test leakage) and is imported, unchanged, by all four
*_exp17.py training scripts -- MLP, XGBoost, TabNet, AND SFTNN -- so the
reweighting is a fair, identical protocol change, not a mechanism handed
only to the proposed model. If SFTNN still improves more than the
baselines under the same reweighting, that is because SFTNN alone has a
dedicated per-region parameter branch able to exploit the rebalanced
signal -- a genuine architectural claim, not an artifact of an uneven
training recipe.

sqrt-inverse-frequency (power=0.5), not raw inverse (power=1.0): raw
inverse frequency on a region with 106 training rows (Africa) vs North
America's 16,799 would weight a single Africa row ~158x a single NA row,
which risks the opposite failure mode (collapsing onto a handful of
Africa rows / overfitting). sqrt-inverse caps that same ratio at ~12.6x --
still a large, deliberate correction, but bounded.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def region_sample_weights(region_ids, n_regions, power=0.5):
    """
    Per-sample weight w_r = 1 / count(r)^power, renormalized to mean 1 over
    the given array so the overall loss scale (and therefore effective
    learning rate) is unchanged -- only the RELATIVE contribution of each
    region's rows shifts.

    region_ids : array-like of int region ids (0..n_regions-1), e.g.
                 train_df["region_id"].values. Counts are computed from
                 whatever array is passed in here -- always call this with
                 TRAIN-split region ids only.
    n_regions  : total number of region ids (manifest["n_regions"]).
    power      : 0.5 (sqrt-inverse-frequency, default) or 1.0 (raw inverse
                 frequency / effective "balanced" weighting).
    """
    region_ids = np.asarray(region_ids)
    counts = np.array(
        [max((region_ids == r).sum(), 1) for r in range(n_regions)],
        dtype=np.float64,
    )
    inv = 1.0 / np.power(counts, power)
    raw_weights = inv[region_ids]
    return raw_weights / raw_weights.mean()


class WeightedStartupDataset(Dataset):
    """Same (x, region_id, y) contract as data_utils.StartupDataset, plus a
    4th per-sample region weight tensor from region_sample_weights() above.
    Only ever wrap the TRAIN split with this -- validation/test still use
    the plain, unweighted StartupDataset from data_utils.py so model
    SELECTION (composite_validation_score) and final evaluation are
    untouched by the training-loss reweighting."""

    def __init__(self, df, feature_cols, n_regions, power=0.5):
        self.x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.region_id = torch.tensor(df["region_id"].values, dtype=torch.long)
        self.y = torch.tensor(df["derived_target"].values, dtype=torch.float32)
        weights = region_sample_weights(df["region_id"].values, n_regions, power=power)
        self.weight = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.region_id[idx], self.y[idx], self.weight[idx]


def region_weight_table(region_ids, n_regions, region_names, power=0.5):
    """Human-readable {region_name: (count, weight)} for logging, so a run
    log shows exactly what correction was applied before any training
    happens."""
    region_ids = np.asarray(region_ids)
    weights = region_sample_weights(region_ids, n_regions, power=power)
    table = {}
    for r in range(n_regions):
        mask = region_ids == r
        table[region_names[r]] = {
            "count": int(mask.sum()),
            "weight": float(weights[mask][0]) if mask.any() else float("nan"),
        }
    return table

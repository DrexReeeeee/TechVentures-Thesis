"""
target_region_validation_score.py
=====================================
Experiment 21: validation-selection score reweighted toward the three
regions this round of experiments specifically targets (Africa, South
Asia, Southeast Asia), instead of data_utils.composite_validation_score's
generic macro_recall averaged across all 7 regions.

Why: that generic macro_recall term is weighted only 0.15 of the total
score, and even within that 0.15 it is diluted by North America and
Europe's own recall (both included in the all-7-region average). Optuna
has had very little real incentive to find a configuration that serves
Africa, South Asia, AND Southeast Asia simultaneously -- it has mostly
been free to trade them off against each other and against pooled
metrics. Section 4.10.2 already states the evaluation priority is
"Recall/F1, especially in minority regions" -- this operationalizes that
priority for the three specific regions in scope for this investigation,
rather than inventing a new criterion after seeing results.

Fairness note: this is used ONLY for Experiment 21's own SFTNN Optuna
search and early-stopping checkpoint selection -- it does not retrain or
reselect the baselines (MLP/XGBoost/TabNet remain Experiment 13's frozen
checkpoints, selected under the original composite_validation_score, as
in Experiments 19/20). This is a refinement of how the PROPOSED model's
own hyperparameters are searched -- comparable to tuning a tree's
max_depth against validation data -- not a different rulebook applied to
only one side of a cross-model race.

Uses F1 (not Recall) for the target-region term, deliberately: Experiment
17/18 showed that rewarding recall alone lets a model satisfy the metric
by shifting its decision threshold toward predicting "success" broadly
(which also lifted North America's recall, a region that didn't need
it) rather than improving minority-region discrimination. F1 penalizes
that shortcut by requiring precision not to collapse.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

TARGET_REGIONS = ("Africa", "South Asia", "Southeast Asia")


def composite_validation_score_target_regions(y_true, y_prob, region_ids, region_id_to_name,
                                               target_regions=TARGET_REGIONS, min_region_n=5,
                                               w_auc=0.25, w_prauc=0.20, w_f1=0.20, w_target_f1=0.35):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    region_ids = np.asarray(region_ids)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    prauc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    target_f1s = []
    for r in np.unique(region_ids):
        name = region_id_to_name[r]
        if name not in target_regions:
            continue
        mask = region_ids == r
        if mask.sum() < min_region_n:
            continue
        target_f1s.append(f1_score(y_true[mask], y_pred[mask], zero_division=0))
    target_f1 = float(np.mean(target_f1s)) if target_f1s else f1

    score = w_auc * auc + w_prauc * prauc + w_f1 * f1 + w_target_f1 * target_f1
    parts = {"auc": auc, "pr_auc": prauc, "f1": f1, "target_f1": target_f1}
    return score, parts

# TechVenture-SFTNN: Modeling Pipeline

This implements Chapter 4 (Sections 4.3.2 through 4.10.2) of the
methodology end to end, against `modeling_ready_dataset_from_formulas.csv`.
Every script is commented with the exact section number it implements.

**This is a revised version of the original handoff.** The proposed
SFTNN model initially failed to outperform the region-blind baselines on
the regions the thesis is about (Africa, Southeast Asia, South Asia) —
`05_train_sftnn.py`, `models.py`, and `07_interpretability.py` went
through three rounds of diagnosis and fixes before reaching the version
here. That history is documented in full in the "SFTNN fix history"
section below, since it's directly relevant to how Chapter 4's
methodology and equations should be written up, not just implementation
trivia.

## 0. Setup

```bash
pip install -r requirements.txt
```

(If you're on Google Colab per Section 4.1's compute fallback, just `pip
install` the same file in a cell -- Colab already has most of these.)

Put `modeling_ready_dataset_from_formulas.csv` somewhere accessible and
note its path.

## 1. Stage 1 -- Preprocessing (Section 4.3.2)

```bash
python 01_preprocess.py --input /path/to/modeling_ready_dataset_from_formulas.csv
```

What it does:
- Drops rows with unmapped region only. Your file has one region group
  outside Table 4.7's original six -- "East Asia" (1,556 rows) -- which
  is **kept by default** as a 7th region rather than dropped. Every
  downstream script (models, evaluation, interpretability) is
  region-count-agnostic, so East Asia flows through Table 4.7, the
  gamma/beta heatmap, and the counterfactual ablation check automatically
  as its own row. Pass `--drop_extra_regions` if you'd rather match
  Table 4.7 exactly as six regions instead. **If you keep East Asia,
  Table 4.7 and any Chapter 5 regional discussion need a 7th row.**
- Engineers `capital_velocity` from `founded_at` / `last_funding_at`.
- log1p + Robust-Scales `funding_total_usd` and all funding-instrument
  columns (Series A-H, seed, debt, grant, etc.).
- Target-encodes each startup's primary industry (first entry in
  `category_list`), fit on the training partition only.
- Splits 70/15/15, stratified on `derived_target`.
- **Enforces the label-leakage firewall**: `status`, `company_age_years`,
  `funding_rounds`/`funding_rounds_clean`, and the raw date columns are
  never exposed to any model, per Section 4.5.2.

Output: `artifacts/{train,val,test}.csv`, `artifacts/preprocessor.joblib`,
`artifacts/feature_manifest.json`.

## 2. Stage 2 -- Region-blind MLP baseline (Section 4.7)

```bash
python 02_train_mlp.py --trials 20 --max_epochs 60
```

Unchanged from the original handoff.

## 3. Stage 3 -- XGBoost and TabNet baselines (Section 4.7)

```bash
python 03_train_xgboost.py --trials 20
python 04_train_tabnet.py --trials 20 --max_epochs 60
```

Unchanged from the original handoff.

## 4. Stage 4 -- Proposed SFTNN (Section 4.5.2, 4.7)

```bash
python 05_train_sftnn.py --trials 30 --max_epochs 60
```

**This script now differs from the original handoff in three ways** (see
"SFTNN fix history" below for why):

1. `models.py`'s `SFTNN` module uses an identity-centered
   reparameterization for gamma/beta, not the raw linear output the
   methodology originally specified.
2. The region-embedding/affine-generator branch is exempt from weight
   decay (its own AdamW param group, via `build_optimizer()`).
3. Training adds a shrinkage penalty (`model.region_shrinkage_penalty()`)
   that pulls each region's embedding toward the population mean,
   weighted by `1/n_r`. Its strength, `shrink_lambda`, is now a 7th
   Optuna-tunable hyperparameter alongside lr/dropout/weight_decay/
   hidden_dim/n_layers/embed_dim.

A per-region variant of `gamma_scale`/`beta_scale` (the modulation
ceiling) was also tried and reverted -- see "SFTNN fix history" for why;
it is not in this script's active code path.

## 5. Stage 5 -- Comparative evaluation (Section 4.9, 4.10)

```bash
python 06_evaluate.py
```

Produces:
- `results/table_4_6_aggregate.csv` -- fills in Table 4.6.
- `results/table_4_7_per_region.csv` -- fills in Table 4.7, for all four
  models.
- Confusion matrices printed to console (Table 4.5 structure).

Uses a flat 0.5 decision threshold for all four models. **A per-region,
validation-tuned threshold variant (Youden's J, fit per Hardt et al.,
2016's post-hoc group-threshold framework) was also built and tested**
-- it is not used here because, for the final SFTNN checkpoint, it
*reduced* aggregate Recall and F1 relative to the flat threshold (see
fix history below). If you want to reproduce or discuss that comparison
in Chapter 4/5, ask for the extended `06_evaluate.py` version that
outputs both `*_flat_threshold.csv` and tuned versions plus
`per_region_thresholds.json` -- it isn't included in this bundle since
it isn't the primary reported result.

Look specifically at Recall and F1 on the minority regions (Africa,
Latin America, South Asia, Southeast Asia), per Section 4.10.2's stated
decision criterion, not just the aggregate row.

## 6. Stage 6 -- Interpretability (Section 4.10.2)

```bash
python 07_interpretability.py
```

Produces:
- `results/gamma_beta_heatmap.png` -- learned gamma_r/beta_r per region.
- `results/ablation_results.csv` -- the counterfactual mean-embedding
  ablation check (replace each region's learned embedding with the
  population-mean embedding, holding everything else fixed, and compare
  Recall/F1). **This is the single most diagnostic output in the whole
  pipeline** -- a region whose Recall_drop is negative is one where the
  model's region-specific behavior is actively worse than generic
  behavior, not just unhelpful.
- `results/region_embedding_pca.png` -- PCA projection of the learned
  region embedding table (Guo & Berkhahn, 2016 convention).

`counterfactual_ablation()` reimplements SFTNN's forward pass manually
(so it can swap in the mean embedding partway through) -- **it must be
kept in sync with `SFTNN.forward()`'s reparameterization by hand**. If
you ever change how gamma/beta are computed in `models.py` again, update
this function too, or the ablation numbers will silently stop matching
`06_evaluate.py`'s numbers (this happened once already; see fix history).

Take the gamma/beta heatmap and ablation table to your adviser and
(where accessible) someone with direct exposure to Southeast Asian
startup ecosystems, per Section 4.10.2's qualitative-review step.

## Files

| File | Methodology section | Status |
|---|---|---|
| `01_preprocess.py` | 4.3.2 | unchanged |
| `models.py` | 4.5.2 (Equations 1-4) | **revised -- Equations 2/3 changed, see below** |
| `data_utils.py` | shared utility, no section | unchanged |
| `02_train_mlp.py` | 4.7, "Standard MLP" | unchanged |
| `03_train_xgboost.py` | 4.7, "XGBoost" | unchanged |
| `04_train_tabnet.py` | 4.7, "TabNet" | unchanged |
| `05_train_sftnn.py` | 4.5.2 + 4.7, "Proposed: SFTNN" | **revised -- see fix history** |
| `06_evaluate.py` | 4.9, 4.10.1 | unchanged (flat threshold; see Stage 5 note) |
| `07_interpretability.py` | 4.10.2 | **revised -- bug fix, see fix history** |

## SFTNN fix history (relevant to your Chapter 4 rewrite)

The originally-specified SFTNN (raw linear gamma/beta, no shrinkage) did
not outperform the region-blind baselines, and specifically underserved
the minority regions it was designed to help -- e.g. an initial full run
showed Africa Recall (0.143) matching the plain MLP's unconditioned
floor and losing to both XGBoost and TabNet (0.29). Three changes were
made in response, in this order:

1. **Identity-centered gamma/beta reparameterization** (`models.py`).
   The original formula, `[gamma_r; beta_r] = W . ReLU(W_e e_r + b_e) + b`,
   initializes gamma near 0 -- i.e. the modulation branch starts by
   *erasing* the trunk's representation, and has to learn its way back
   out using gradient signal that's scarcest for the smallest regions.
   Changed to `gamma_r = 1 + tanh(gamma_raw)`, `beta_r = tanh(beta_raw)`,
   with the final affine-generator layer zero-initialized, so every
   region starts as an exact no-op (gamma=1, beta=0) and only deviates
   where training earns it. **This changes Equations 2 and 3 in Section
   4.5.2** -- the equations as currently written no longer match the
   code. Citable precedent: de Vries et al. (2017) on conditional batch
   normalization's delta-from-identity formulation (the direct precursor
   to the FiLM paper, Perez et al. 2018, already cited in Chapter 3), and
   Bachlechner et al. (2021, "ReZero") for the zero-initialization of the
   branch's final layer specifically.
2. **Weight decay decoupled from the region-conditioning branch**
   (`05_train_sftnn.py`'s `build_optimizer()`). Weight decay pulls
   parameters toward 0 -- which, after fix #1, is exactly the no-op
   state -- so ordinary weight decay was fighting the mechanism instead
   of regularizing it. The region embedding and affine generator now get
   their own zero-weight-decay AdamW param group.
3. **Shrinkage penalty on region embeddings** (`models.py`'s
   `region_shrinkage_penalty()`). The counterfactual ablation after fixes
   #1-2 showed region-specific behavior was actively *harmful* (not just
   unhelpful) for Southeast Asia, East Asia, and South Asia -- swapping
   in the generic mean embedding improved their Recall. A penalty
   pulling each region's embedding toward the population mean, weighted
   by `1/n_r`, was added to the training loss to correct this while
   preserving North America/Latin America's genuine, ablation-confirmed
   gains.

**A fourth change, Group DRO (Sagawa et al., 2020), was implemented and
tested but not adopted.** It regressed aggregate ROC-AUC, F1, and Recall
relative to the shrinkage-only checkpoint, and moved Southeast Asia's
AUC further from XGBoost's rather than closer -- the opposite of its
intent. This is a legitimate, disclosable negative result (worth one or
two sentences in Chapter 4/5: "Group DRO was evaluated and did not
improve on the shrinkage-only configuration within the trial budget
used") but is not in this pipeline's active code path.

**A fifth change, per-region post-hoc threshold tuning** (Hardt et al.,
2016; Youden, 1950), was also implemented and tested. It is not the
primary reported result because it reduced SFTNN's aggregate Recall and
F1 relative to the flat 0.5 threshold, even though it fairly improved
Southeast Asia specifically at the cost of erasing a South Asia
advantage the flat threshold already had. Worth a paragraph in Chapter
4/5 as a disclosed, tested-but-not-adopted refinement, same as Group DRO.
A related but distinct diagnostic -- a single *global* (not per-region)
threshold, re-derived from the validation set for each model separately
and applied fairly to both SFTNN and XGBoost -- was also checked. It
recovers roughly half of SFTNN's aggregate F1 gap with XGBoost (both
models improve, since both were running below their own optimum at flat
0.5: SFTNN 0.7345->0.7553, XGBoost 0.7466->0.7617), but Southeast Asia is
the one region that gets *worse* for both models at each model's own
global-optimal threshold (SFTNN 0.516->0.490 F1, XGBoost 0.571->0.491 F1)
-- the only region where the globally-optimal cutoff moves in the wrong
direction for it specifically.

**A sixth change, per-region gamma_scale/beta_scale, was implemented,
tested, and reverted.** Motivated directly by the threshold diagnostic
above -- a per-region calibration difference is exactly what gamma_r/beta_r
should be able to express, but gamma_scale/beta_scale were single global
scalars shared by every region, capping every region's modulation ceiling
at whatever was optimal in aggregate (implicitly dominated by North
America's 63% share of training rows). Promoted both to per-region
parameters (`nn.Parameter(torch.ones(n_regions))`, indexed by `region_id`),
with an analogous `scale_shrinkage_penalty()` and an independently
Optuna-searched `shrink_lambda_scale` (8 tunable hyperparameters total, up
from 7). **Result: a clean regression, not an improvement.** Aggregate
Recall fell 0.6702->0.6273, F1 fell 0.7345->0.7178, and Southeast Asia's
AUC collapsed from 0.814 to 0.710 -- the exact region this change targeted
got measurably worse, not better. Latin America was the only region that
improved. Inspecting the learned parameters explained why: the winning
Optuna trial set both `shrink_lambda` (0.0013) and the new
`shrink_lambda_scale` (0.0011) to nearly zero -- three orders of magnitude
below the original run's `shrink_lambda=7.29` -- and Southeast Asia ended
up with the *largest* gamma_scale (1.168) and beta_scale (0.494) of any
region, i.e. the least-regularized, not the most. Adding an 8th search
dimension gave the same 30-trial Optuna budget a harder space to cover,
and this run landed on a corner where both shrinkage mechanisms were
effectively disabled -- undoing fix #3's protection for the smallest-n
region rather than adding useful per-region flexibility on top of it.
This is not necessarily proof the underlying idea (per-region calibration
capacity) is wrong, only that a fair test of it needs a substantially
larger trial budget than 30 to reliably search 8 dimensions instead of 7
-- reverted rather than pursued further, since re-running with a larger
budget wasn't justified without stronger prior evidence it would help.
Disclosable the same way as Group DRO and per-region threshold tuning:
"per-region modulation-ceiling parameters were evaluated and regressed
performance within the trial budget used, most notably on the region
they were intended to help."

**Net result**, at flat threshold, on the CORRECTED dataset (after the
Southeast Asia label-completeness fix and the funding-disclosure
missingness indicator; see "Dataset corrections" below). These are the
numbers in `results/table_4_6_aggregate.csv` and
`results/table_4_7_per_region.csv` as currently saved, and they are the
only ones that should be cited in Chapter 5:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Standard MLP (one-hot region) | 0.750 | 0.802 | 0.658 | 0.723 | 0.825 | 0.838 |
| XGBoost | 0.768 | 0.799 | 0.710 | **0.752** | **0.844** | **0.856** |
| TabNet | 0.758 | 0.768 | **0.732** | 0.750 | 0.834 | 0.849 |
| Proposed SFTNN | 0.758 | 0.775 | 0.721 | 0.747 | 0.838 | 0.845 |

SFTNN is 2nd of 4 on Recall/F1/ROC-AUC and 3rd of 4 on PR-AUC in
aggregate. Africa AUC is 0.852 (vs XGBoost's 0.833), but on n=24 -- too
small to lean on. **Do not cite the pre-correction figures (Recall 0.709
/ F1 0.742 / Africa AUC 0.729) that previously appeared here; they came
from a run against the uncorrected dataset and do not match any current
artifact.**

Per-region, bootstrap-verified (see `08_bootstrap_ci.py`): the defensible
claims are (a) SFTNN beats the one-hot-region MLP on Recall and F1 in 6/7
and 7/7 regions respectively, **with identical region information
available to both models** -- i.e. the FiLM conditioning mechanism, not
the presence of region data, is what differs; (b) South Asia is a
near-complete sweep (wins or ties every metric against all three
baselines except one Precision loss to XGBoost); (c) Southeast Asia shows
robust Recall/F1 wins over XGBoost (89%/72% of bootstrap resamples) but
loses F1/ROC-AUC to TabNet. TabNet, not XGBoost, is the strongest
competitor overall: across regions it beats SFTNN on Recall 2-4-1 and on
F1 3-3-1 (loss-tie-win from SFTNN's perspective).

## Dataset corrections (applied in `01_preprocess.py`)

1. **Southeast Asia label-completeness fix.** The Failory-scraped SEA
   supplement (`data_source == "sea_failory_supplement"`, n=210) never
   captured round-level funding history: every Operating row has
   `funding_rounds_clean == 0.0` with zero variance, despite a median
   $18M disclosed funding and ages of 7-105 years. The Section 4.3.2.6
   heuristic (age >= 7 AND rounds >= 2 -> Success) therefore forced all
   210 to Failure on a missing-data artifact rather than an observed
   outcome. A scan of every (region, data_source) pair confirmed this
   pattern occurs in **0% of rows anywhere else** in the dataset. These
   rows are now excluded the same way the heuristic already excludes
   unknown-founding-year and 3-6-year-old companies. Effect: SEA success
   rate moved 26.6% -> 38.0%, SEA n moved 699 -> 489.
2. **Funding-disclosure missingness indicator.** `funding_total_usd`
   missingness is not uniform by region (Africa 27.6%, South Asia 23.5%,
   SEA 19.5%, North America 11.6%), and the unconditional `.fillna(0.0)`
   collapsed "not disclosed" into "raised nothing." Added
   `funding_total_usd_disclosed` as a binary feature (25 features total,
   up from 24) so the trunk can distinguish the two.

## New citations needed for Chapter 4/5 and the Bibliography

| Citation | Where it's used | Bibliography subsection |
|---|---|---|
| de Vries, H., Strub, F., Mary, J., Larochelle, H., Pietquin, O., & Courville, A. (2017). Modulating early visual processing by language. *Advances in Neural Information Processing Systems, 30*. | Justifies the identity-centered gamma/beta reparameterization (fix #1) | Journal Articles / Conference Proceedings (NeurIPS) |
| Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G., & McAuley, J. (2021). ReZero is all you need: Fast convergence at large depth. *Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence*, PMLR 161, 1352-1361. | Justifies zero-initializing the affine generator's final layer (fix #1) | Conference Proceedings |
| Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020). Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. *International Conference on Learning Representations*. | Group DRO (tested, not adopted -- disclose as a negative result) | Conference Proceedings |
| Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems, 29*, 3315-3323. | Per-region threshold tuning framework (tested, not primary result) | Conference Proceedings |
| Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32-35. | Youden's J threshold-selection criterion | Journal Articles |

## A few judgment calls carried over from the original handoff

1. **`category_list` is multi-valued.** The first listed category is
   used as the primary industry vertical for target encoding. One-line
   change in `primary_industry()` in `01_preprocess.py` if you want a
   different rule.
2. **Funding-instrument breakdown columns** (Series A-H, seed, debt,
   grant, etc.) are included as "funding history" per Section 4.4.1's
   phrasing, extending beyond Section 4.5.2's literal minimal example.
   Remove `FUNDING_HISTORY_COLS` in `01_preprocess.py` to match the
   methodology text more literally.
3. **Batch size and n_regions-dependent one-hot width** are
   implementation details not specified in the methodology text;
   defaults (batch size 256) can be changed directly in the scripts.

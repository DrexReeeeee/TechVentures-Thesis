# TechVenture-SFTNN: Modeling Pipeline

This project implements Chapter 4 (Sections 4.3.2 through 4.10.2) of your
methodology end to end, against your actual file
`modeling_ready_dataset_from_formulas.csv`. Every script is commented with
the exact section number it implements.

All seven scripts were smoke-tested against your real 38,090-row dataset
before being handed to you -- the pipeline runs cleanly start to finish.
What you're getting is the full six-stage development model from Section
4.6; you just need to let it run for real (more epochs, more Optuna
trials) instead of the 1-2 epoch smoke test used to validate the code.

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
  Table 4.7 exactly as six regions instead.
- Engineers `capital_velocity` from `founded_at` / `last_funding_at`.
- log1p + Robust-Scales `funding_total_usd` and all funding-instrument
  columns (Series A-H, seed, debt, grant, etc. -- these extend Section
  4.3.2's literal wording, which names `funding_total_usd` as the running
  example; drop `FUNDING_HISTORY_COLS` in `01_preprocess.py` if you want
  to match the paper's minimal feature set exactly).
- Target-encodes each startup's primary industry (first entry in
  `category_list`), fit on the training partition only.
- Splits 70/15/15, stratified on `derived_target`.
- **Enforces the label-leakage firewall**: `status`, `company_age_years`,
  `funding_rounds`/`funding_rounds_clean`, and the raw date columns are
  never exposed to any model, per Section 4.5.2.

Output: `artifacts/{train,val,test}.csv`, `artifacts/preprocessor.joblib`,
`artifacts/feature_manifest.json`.

**Check `feature_manifest.json` before continuing** -- confirm the region
list and feature count look right to you.

## 2. Stage 2 -- Region-blind MLP baseline (Section 4.7)

```bash
python 02_train_mlp.py --trials 20 --max_epochs 60
```

This runs the full ~20-trial Optuna search described in Section 4.7. On a
laptop CPU this will take a while (each trial trains up to 60 epochs with
early stopping) -- budget real time for this, or move to Colab's GPU
runtime as described in Section 4.1.

## 3. Stage 3 -- XGBoost and TabNet baselines (Section 4.7)

```bash
python 03_train_xgboost.py --trials 20
python 04_train_tabnet.py --trials 20 --max_epochs 60
```

XGBoost will be by far the fastest of the four.

## 4. Stage 4 -- Proposed SFTNN (Section 4.5.2, 4.7)

```bash
python 05_train_sftnn.py --trials 30 --max_epochs 60
```

30 trials (vs. 20 for the baselines), per Section 4.7's stated budget for
the added embedding + affine-generator branch.

## 5. Stage 5 -- Comparative evaluation (Section 4.9, 4.10)

```bash
python 06_evaluate.py
```

Produces:
- `results/table_4_6_aggregate.csv` -- fills in Table 4.6.
- `results/table_4_7_per_region.csv` -- fills in Table 4.7, for all four
  models.
- Confusion matrices printed to console (Table 4.5 structure).

This is where you actually find out whether SFTNN beats the region-blind
MLP baseline and is competitive with XGBoost/TabNet -- look specifically
at Recall and F1 on the minority regions (Africa, Latin America, South
Asia, Southeast Asia), per Section 4.10.2's stated decision criterion,
not just the aggregate row.

## 6. Stage 6 -- Interpretability (Section 4.10.2)

```bash
python 07_interpretability.py
```

Produces:
- `results/gamma_beta_heatmap.png` -- learned gamma_r/beta_r per region.
- `results/ablation_results.csv` -- the counterfactual mean-embedding
  ablation check.
- `results/region_embedding_pca.png` -- the optional PCA region-embedding
  cluster plot (Guo & Berkhahn, 2016 convention).

Take the gamma/beta heatmap and ablation table to your adviser and
(where accessible) someone with direct exposure to Southeast Asian
startup ecosystems, per Section 4.10.2's qualitative-review step -- that
part genuinely can't be automated.

## Files

| File | Methodology section |
|---|---|
| `01_preprocess.py` | 4.3.2 |
| `models.py` | 4.5.2 (Equations 1-4) |
| `data_utils.py` | shared utility, no section |
| `02_train_mlp.py` | 4.7, "Standard MLP" |
| `03_train_xgboost.py` | 4.7, "XGBoost" |
| `04_train_tabnet.py` | 4.7, "TabNet" |
| `05_train_sftnn.py` | 4.5.2 + 4.7, "Proposed: SFTNN" |
| `06_evaluate.py` | 4.9, 4.10.1 |
| `07_interpretability.py` | 4.10.2 |

## A few judgment calls I made for you

1. **Extra regions in your data.** Your `region_group` column has an
   "East Asia" value beyond the six in Table 4.7, plus "Unmapped/Missing"
   rows. "Unmapped/Missing" is always dropped (region unknown, can't
   condition SFTNN on it). East Asia is **kept as a 7th region by
   default** -- pass `--drop_extra_regions` to `01_preprocess.py` if you
   want to match Table 4.7's original six regions exactly instead. If you
   keep East Asia, remember to add a 7th row to Table 4.7 (and any
   per-region discussion in Chapter 5) when you write up results.
2. **`category_list` is multi-valued** (e.g. `"Application
   Platforms|Real Time|Social Network Media"`). I use the first listed
   category as the primary industry vertical for target encoding. If you
   want a different rule (most specific category, all categories
   multi-hot, etc.), that's a one-line change in `primary_industry()` in
   `01_preprocess.py`.
3. **Funding-instrument breakdown columns.** I included all of them
   (Series A-H, seed, debt, grant, etc.) as part of "funding history" per
   Section 4.4.1's phrasing. Section 4.5.2 literally lists x as just
   "log-scaled funding totals, capital velocity, encoded industry
   vertical bins" -- if you want to match that more literally, remove
   `FUNDING_HISTORY_COLS` from the feature list in `01_preprocess.py`.
4. **Batch size, threshold, and n_regions-dependent one-hot width** are
   implementation details not specified in the methodology text; I used
   reasonable defaults (batch size 256, decision threshold 0.5) that you
   can change directly in the scripts.

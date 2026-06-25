# Dataset Sources — TechVenture-SFM Thesis

This README documents where the datasets used in the thesis project came from, organized by region and folder. Each entry is labeled with how confident the sourcing is:

- ✅ Verified — confirmed by Claude via direct web search/fetch during this project
- 📤 User-provided — uploaded directly as a file; the original source URL was supplied by the user or could not be independently re-confirmed by Claude
- ⚠️ Inferred — Claude matched the file’s filename/schema/row-count to a likely public source, but this was not directly downloaded or confirmed by Claude itself

Always double-check ⚠️ entries yourself before citing them in your methodology — they are Claude’s best inference, not a confirmed citation.

---

## Main Base Dataset (North America, Europe, East Asia + global baseline)

**File:** `FINAL DATASET/big_startup_secsees_dataset.csv` → folded into `FINAL DATASET/final_master_dataset.csv` as `data_source = main_dataset`

⚠️ Inferred source:
`https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase`

Matched on exact filename (`big_startup_secsees_dataset.csv`), exact column count (14), and exact column names. Original data sourced from Crunchbase, covering company foundings, fundings, and outcomes through ~2017.

**Enrichment file:** `FINAL DATASET/investments_VC.csv` (round-type funding breakdown: seed, venture, round_A–H, debt, etc. — joined onto the main dataset by `permalink`)

⚠️ Inferred source:
`https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase`

Matched on schema (39 columns including `permalink`, `round_A`–`round_H`, `seed`, `venture`) and the same Crunchbase lineage as the main dataset. Approximately 98.5% of its companies overlap with the main dataset; 730 were net-new.

---

## South Asia

**File:** `FINAL DATASET/india_startups_cleaned.csv` (cleaned/consolidated version derived from the year-based funding scrape folders)

✅ Verified — GitHub mirror (no login wall):
`https://github.com/DeepakKumarGS/Indian-Startup-Funding-/blob/gh-pages/startup_funding.csv`

⚠️ Likely original Kaggle source(s):
`https://www.kaggle.com/datasets/sudalairajkumar/indian-startup-funding` (most likely candidate — could not be directly confirmed by Claude due to Kaggle’s anti-bot wall)

---

## Africa

**File:** `AFRICA/FINAL AFRICA/africa_supplement_2022_2025_clean.csv` (cleaned and consolidated supplement used for African startup coverage)

✅ Verified — Disrupt Africa’s annual “African Tech Startups Funding Report” series
(row counts in the user’s file match Disrupt Africa’s own published figures exactly: 633 startups in 2022, 406 in 2023):

- 2022 report: `https://old.disruptafrica.com/wp-content/uploads/2023/02/The-African-Tech-Startups-Funding-Report-2022.pdf`
- 2023 report: `https://disruptafrica.com/wp-content/uploads/2024/01/The-African-Tech-Startups-Funding-Report-2023.pdf`
- 2025 report: `https://disruptafrica.com/wp-content/uploads/2026/02/The-African-Tech-Startups-Funding-Report-2025.pdf`
- Report index / “Full Startup List” downloads: `https://disruptafrica.com/research/` and `https://disruptafrica.com/funding-report/`

📤 Note: no 2024 report/data exists in this lineage — confirmed gap year, not a download error.

**Other relevant files in the folder:**
- `AFRICA/AFRICA_2022.xlsx` — earlier single-year Africa dataset snapshot
- `AFRICA/Funded African tech startups 2022.xlsx` — raw 2022 funding file
- `AFRICA/Funded African Tech Startups 2023 (by country and by sector).xlsx` — raw 2023 funding file
- `AFRICA/Funded_african_2024/` — folder for 2024 materials
- `AFRICA/Funded_African_tech_startups_2025_(by_country-sector)/` — folder for 2025 materials
- `AFRICA/merge_and_clean_africa_data.py` — cleaning/merge script used for the African dataset preparation

---

## Latin America

**Files:**
- `LATIN AMERICA/brazil_startups.csv`
- `LATIN AMERICA/mexico_startups.csv`
- `LATIN AMERICA/LatinAm_brazil_columbia.xlsx`
- `LATIN AMERICA/yc_latam_companies.csv`

📤 User-provided / scraper-derived sources:

- GrowthList Mexico: `https://growthlist.co/mexico-startups/#recently-funded-startups-in-mexico`
- GrowthList Brazil: `https://growthlist.co/brazil-startups/#recently-funded-startups-in-brazil`
- YC directory: `https://www.ycombinator.com/companies/?regions=Latin%20America`
- Individual YC company profile pattern: `https://www.ycombinator.com/companies/<slug>`

These files were used for regional startup coverage and for status-label enrichment where relevant. The YC list was not treated as a standalone training-row source for the core model pipeline.

---

## Southeast Asia

No supplementary dataset was found in the local workspace. Coverage comes entirely from the main base dataset’s existing Southeast Asia rows (Singapore, Indonesia, Malaysia, Thailand, Vietnam, Philippines, Cambodia).

This remains the thinnest region in the final dataset — flagged in the regional summary files for k-fold cross-validation / wide-interval reporting rather than a single train/test split.

Multiple searches were run for Southeast Asia-specific sources during this project (Tech in Asia, e27, KrASIA, DealStreetAsia archives) — none yielded a free, bulk-downloadable, row-level dataset. Paid platforms (Crunchbase Pro, Tracxn, DealStreetAsia) were the only sources with meaningful SEA coverage found.

---

## Datasets Evaluated and Rejected (for reference / avoiding re-checking)

| Dataset | Reason rejected |
| --- | --- |
| `EXTRA DATASETS/companies.csv` | 2013 vintage, 85.8% missing funding data, worse regional skew than the main dataset |
| `EXTRA DATASETS/startup_success_dataset.csv` / `startup_valuation_dataset.csv` | Synthetic/fabricated — region-country pairings randomly assigned, zero missingness across 100K rows |
| Tech Companies Global Dataset (rashmikakr) | No funding/outcome fields at all — directory data, not funding data |
| `Acquiring_Tech_Companies.csv` | Wrong unit of analysis (36 mega-acquirers, not startups); 86% USA |
| `Startups.csv` (general YC list) | 66% USA, single-digit counts elsewhere |
| Trending Startup Data by City/Country (HackerNoon TCNP) | Self-nomination voting platform — no funding/outcome data exists at the source, not scrapeable |
| AngelList scrape (`iamtodor/angel.co-companies-list-scraping`) | 96%+ San Francisco/Bay Area |

---

## Folder-by-folder dataset guide

### Root project files

- `scrape_growthlist.py` — scraper used to collect GrowthList-based startup data for Brazil and Mexico
- `scrape_yc_latam.py` — scraper used to collect YC Latin America company listings and profiles

### `AFRICA/`

Purpose: African startup funding and company coverage, with yearly supplements and cleaned merged outputs.

Files:
- `AFRICA_2022.xlsx` — earlier African startup snapshot for 2022
- `Africa_Startups_Merged_Cleaned_20260624_225559.xlsx` — cleaned merged African dataset snapshot
- `Africa_Startups_Merged_Cleaned_20260624_225953.xlsx` — cleaned merged African dataset snapshot
- `Funded African tech startups 2022.xlsx` — raw 2022 data source snapshot
- `Funded African Tech Startups 2023 (by country and by sector).xlsx` — raw 2023 data source snapshot
- `FINAL AFRICA/` — cleaned final Africa supplement folder
- `Funded_african_2024/` — 2024 materials folder
- `Funded_African_tech_startups_2025_(by_country-sector)/` — 2025 materials folder
- `merge_and_clean_africa_data.py` — cleaning/merge script

### `EXTRA DATASETS/`

Purpose: auxiliary datasets evaluated during the project, mostly for comparison or feature enrichment testing.

Files:
- `companies.csv` — general company list dataset, later rejected for poor funding coverage and regional skew
- `startup_success_dataset.csv` — rejected as synthetic / not suitable for the thesis labeling task
- `startup_valuation_dataset.csv` — rejected for the same reason

### `FINAL DATASET/`

Purpose: the main modeling-ready datasets and the final master dataset used for training and analysis.

Files:
- `big_startup_secsees_dataset.csv` — main base dataset used as the backbone of the thesis dataset
- `final_master_dataset.csv` — final merged master dataset used for analysis
- `final_master_dataset_updated.csv` — updated version of the final master dataset
- `india_startups_cleaned.csv` — cleaned India startup dataset
- `investments_VC.csv` — round-level investment enrichment file
- `regional_distribution_summary (1).xlsx` — regional distribution summary workbook
- `regional_distribution_summary_FINAL.xlsx` — final regional distribution summary workbook

### `LATIN AMERICA/`

Purpose: Latin America startup records, mostly scraped or assembled for Brazil, Mexico, and YC-linked Latin American companies.

Files:
- `brazil_startups.csv` — Brazil startup list / funding records
- `mexico_startups.csv` — Mexico startup list / funding records
- `LatinAm_brazil_columbia.xlsx` — consolidated Latin America workbook for Brazil/Colombia-style regional coverage
- `yc_latam_companies.csv` — YC Latin America company list used for enrichment/status labeling

### `StartUp_FundingScrappingData/`

Purpose: yearly startup funding data pulled from the funding scrape pipeline, stored by year and then merged into consolidated outputs.

Files and folders:
- `2015/` to `2021/` — yearly CSV folders containing raw funding records by year
- `merged_output/` — merged yearly outputs for the funding scrape collection
- `merge_csvs.py` — script that combines yearly CSVs into consolidated outputs

---

*Compiled from project conversation history. ⚠️-flagged sources should be independently re-verified before being cited in the final thesis manuscript.*

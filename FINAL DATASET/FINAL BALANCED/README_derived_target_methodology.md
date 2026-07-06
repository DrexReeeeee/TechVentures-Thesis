# README — Deriving a Success/Failure Target Column

## Problem

The raw dataset (`final_master_dataset_sea_merged.csv`, 51,276 rows) has a `status`
field with six raw values:

| status      | rows   |
|-------------|-------:|
| operating   | 34,803 |
| NaN         |  6,564 |
| acquired    |  4,421 |
| closed      |  3,849 |
| ipo         |  1,144 |
| Active      |    494 |
| Inactive    |      1 |

Only `acquired`, `ipo`, and `closed` map cleanly onto a binary success/failure
outcome. `operating` (68% of rows) is ambiguous — a 15-year-old company that's
still "operating" without an exit and a 6-month-old company are both labeled
identically, but they mean very different things for a success/failure model.

## Approach: age as a proxy for outcome

Rather than discard the 34,803 "operating" rows, we applied an age-threshold
proxy commonly used in VC/startup-outcome research: long-surviving operating
companies without an exit are treated as de facto successes (they proved
sustainability), while young operating companies with no recent funding
activity are treated as likely-defunct ("zombie") companies whose Crunchbase
profile was simply never updated.

### Step 1 — Clean the dates

- Parsed `founded_at` and `last_funding_at` with `pd.to_datetime(..., errors='coerce')`.
- Nulled out impossible future dates (typos like founding year `2914` or
  `2115`) — anything after 2026 was treated as missing rather than guessed at.
- Computed:
  - `company_age_years` = 2026 − founded year
  - `years_since_last_funding` = 2026 − last funding year

### Step 2 — Normalize status text

- `Active` → `operating`
- `Inactive` → `closed`
(case/whitespace normalized before matching)

### Step 3 — Apply labeling rules, in this order

| # | Condition | Label | `derived_target` |
|---|-----------|-------|:---:|
| 1 | status = `acquired` or `ipo` | Success (Acquired/IPO) | **1** |
| 2 | status = `closed` (incl. normalized `Inactive`) | Failure (Closed) | **0** |
| 3 | status = `operating`/`Active` **and** `company_age_years` ≥ 7 | Success (Long-Term Survivor) | **1** |
| 4 | status = `operating`/`Active`, age < 7, **and** `years_since_last_funding` > 5 | Failure (Stagnant/Zombie) | **0** |
| 5 | status = `operating`/`Active`, age < 7, funding recent or unknown | Unclear (Too Young / Insufficient Evidence) | NaN — dropped |
| 6 | status = `operating`/`Active`, `founded_at` missing | Unclear (Unknown Founding Date) | NaN — dropped |
| 7 | `status` itself is missing | Unknown (No Status Recorded) | NaN — dropped |

**Thresholds used:** 7 years (success cutoff) and 5 years since last funding
(stagnation cutoff) — the values you specified. These are adjustable; see
"Known limitations" below.

## Results

| Label | Rows |
|---|---:|
| Success (Long-Term Survivor) | 28,675 |
| Unknown (No Status Recorded) | 6,564 |
| Unclear (Unknown Founding Date) | 6,512 |
| Success (Acquired/IPO) | 5,565 |
| Failure (Closed) | 3,850 |
| Unclear (Too Young / Insufficient Evidence) | 110 |
| **Total** | **51,276** |

- **Usable rows (`derived_target` not null): 38,090**
- Class split: **1 (success) = 34,240** / **0 (failure) = 3,850** → ~9:1 imbalance
- Cross-check: rows with an unambiguous raw status (`acquired`+`ipo`+`closed`)
  = 9,414, matching the "strict filter" baseline for sanity-checking the logic.

## Files produced

- `final_master_dataset_sea_merged_with_target.csv` — full original dataset (51,276 rows) plus the new columns: `company_age_years`, `years_since_last_funding`, `status_norm`, `success_fail_status`, `derived_target`.
- `modeling_ready_dataset.csv` — same file filtered to only the 38,090 rows with a non-null `derived_target`, ready to use as a training set.

## Known limitations / things to sanity-check before modeling

1. **Class imbalance (~9:1).** The age-based rule is generous toward labeling
   things "success" (any operating company ≥7 years old qualifies) but has a
   narrow path to "failure" (must be both young-ish *and* have a confirmed
   stale funding date). Companies with no funding history at all don't get
   caught by the zombie rule — they're routed to "unclear" instead of
   "failure." Consider loosening rule #4 to also catch old operating
   companies with **no funding history whatsoever**, if you want more
   failure-labeled examples.
2. **Old founding dates (pre-1950) were kept as-is** (e.g., a company founded
   in 1863) since these appear to be real legacy companies, not typos — only
   dates in the *future* relative to 2026 were treated as data errors.
3. **Region coverage is thin outside North America/Europe** (see prior
   breakdown) — worth checking whether the age-proxy logic behaves
   differently by region before pooling everything into one model.
4. Thresholds (7-year success cutoff, 5-year stagnation cutoff) were taken
   directly from your specification — happy to re-run a sensitivity sweep
   across a few threshold combinations if you want to see how usable-row
   count and class balance shift.

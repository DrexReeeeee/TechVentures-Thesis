# Documentation: Success/Failure Target Labeling Methodology

**Dataset:** `final_master_dataset_sea_merged.csv` (51,276 rows)
**Reference year used:** 2026

---

## 1. The Problem

The raw `status` column has six values:

| status    | rows   |
|-----------|-------:|
| operating | 34,803 |
| NaN       |  6,564 |
| acquired  |  4,421 |
| closed    |  3,849 |
| ipo       |  1,144 |
| Active    |    494 |
| Inactive  |      1 |

Only `acquired`, `ipo`, and `closed` map directly onto a binary success/failure
outcome (9,414 rows combined — the "strict" baseline). The remaining 68% of the
dataset is labeled `operating`, which is ambiguous on its own: a company that's
been operating for 15 years and one that's been operating for 6 months are
indistinguishable by that field alone.

**Goal:** recover a usable, reasonably balanced success/failure label for as
much of the `operating` population as can be defensibly justified, instead of
discarding two-thirds of the dataset.

---

## 2. Method: Age as a Proxy for Outcome

This is a standard approach in VC/startup-outcome research: a company that has
survived a long time without an exit has proven basic viability, while a
company that has had ample time but shows no growth signal is treated as part
of the "living dead" — operationally alive but not a real outcome of interest.

### Step-by-step logic

1. **Direct labels first** — `acquired`/`ipo` → Success; `closed`/`Inactive` → Failure. No inference needed.
2. **Compute company age** — `2026 − year(founded_at)`. Cleaned first: dates after 2026 (typos like a founding year of "2914") are treated as missing rather than guessed at.
3. **For everything still `operating`/`Active`:**
   - Too young (< 3 years) → **dropped** (not enough time has passed to know anything)
   - Unknown founding date → **dropped** (can't compute age at all)
   - Mid-age (3–6 years) → **dropped** (ambiguous zone — old enough to show early signs, not old enough to call confidently)
   - **7+ years old** → apply a momentum tiebreaker (see below)

### The tiebreaker for companies 7+ years old (v2 — current version)

| Condition | Label | Target |
|---|---|:---:|
| Raised **2 or more funding rounds** | Success (Long-Term Survivor, Follow-on Funded) | 1 |
| Raised **1 round or none** | Failure (Stalled — No Follow-on Funding) | 0 |

**Rationale:** if a second investor was willing to write a check after seeing
how the first round played out, that's real evidence of momentum. If a company
had 7+ years of runway and never convinced anyone to fund it again, that's a
strong signal it plateaued or quietly died without anyone updating its record.

---

## 3. Why the First Attempt (v1) Was Abandoned

The original design (matching the exact logic requested) used **calendar
recency of last funding** as the tiebreaker: "operating, under 7 years old,
but no funding activity in 5+ years → Failure."

This produced almost no failures (34,240 success vs. 3,850 failure — a 9:1
skew) and it turned out to be a **data artifact, not a real signal**:
investigating `last_funding_at` showed the bulk of the dataset's funding dates
are frozen around **2013–2015** (only 187 of 47,049 dated rows are from 2021
or later) — this looks like a legacy Crunchbase snapshot merged with a thin
trickle of newer supplemental rows. Measured against a 2026 reference point,
*almost the entire dataset* looks "stale," regardless of whether a company is
actually dead. That confound made the date-based rule nearly useless as a
discriminator.

**Fix:** replaced the calendar-based tiebreaker with a **count-based** one —
number of funding rounds raised — which doesn't depend on what year it is, and
isn't distorted by when the underlying data was collected.

---

## 4. Results

| Label | Rows |
|---|---:|
| Failure (Stalled — No Follow-on Funding) | 16,002 |
| Success (Long-Term Survivor, Follow-on Funded) | 12,673 |
| Unknown (No Status Recorded) | 6,564 |
| Unclear (Unknown Founding Date) | 6,512 |
| Success (Acquired/IPO) | 5,565 |
| Failure (Closed) | 3,850 |
| Unclear (Mid-Age, Insufficient Track Record) | 97 |
| Unclear (Too Young) | 13 |
| **Total** | **51,276** |

**Usable rows (non-null target): 38,090**
**Success (1): 18,238 — Failure (0): 19,852 — ratio ≈ 0.92:1** (near-balanced,
vs. 9:1 under the v1 approach)

### Region coverage (of the 38,090 usable rows)

| Region | Rows | % |
|---|---:|---:|
| North America | 23,976 | 62.9% |
| Europe | 7,679 | 20.2% |
| Unmapped/Missing | 2,499 | 6.6% |
| East Asia | 1,556 | 4.1% |
| South Asia | 791 | 2.1% |
| Latin America | 745 | 2.0% |
| Southeast Asia | 699 | 1.8% |
| Africa | 145 | 0.4% |

---

## 5. Deliverables

| File | What it is |
|---|---|
| `dataset_with_age_horizon_formulas.xlsx` | **Primary deliverable.** Full 51,276-row dataset with the proxy logic implemented as **live Excel formulas** (not hardcoded values), plus an editable Assumptions tab. |
| `modeling_ready_dataset_from_formulas.csv` | Static snapshot of just the 38,090 labeled rows (calculated values), for tools that can't open `.xlsx`. |

### How the Excel file is structured

- **`Assumptions` sheet** — four editable input cells (blue font, per standard modeling convention):
  - `REF_YEAR` = 2026
  - `AGE_SUCCESS_THRESHOLD` = 7
  - `AGE_TOO_YOUNG` = 3
  - `ROUNDS_THRESHOLD` = 2
- **`Data` sheet** — original 42 columns untouched, plus 5 formula-driven columns:
  - `status_norm` — normalizes `Active`→`operating`, `Inactive`→`closed`
  - `company_age_years` — `=REF_YEAR − YEAR(founded_at)`
  - `funding_rounds_clean` — safely converts `funding_rounds` to numeric
  - `success_fail_status` — full label logic as a nested `IF` formula
  - `derived_target` — same logic, output as 1 / 0 / blank

**Changing any Assumptions cell (e.g., raising the age threshold to 8) recalculates every row automatically.** Nothing is precomputed and pasted in as a static number — this was a deliberate choice so the thresholds are auditable and adjustable without needing to touch Python.

---

## 6. Known Limitations

1. **Most labels are inferred, not observed.** Of the 18,238 successes, 12,673 come from the age/momentum proxy rather than a confirmed acquisition or IPO. Of the 19,852 failures, 16,002 come from the "stalled" proxy rather than a confirmed closure. This should be disclosed alongside any model built on this target.
2. **Region coverage is heavily skewed toward North America and Europe** (~83% combined). Despite the filename (`sea_merged`), Southeast Asia only contributes 699 usable rows — worth checking whether the proxy behaves consistently across regions before pooling into a single model.
3. **A handful of source rows are corrupted independent of this work.** 4 rows in the original CSV contain literal `#REF!` text baked into fields like `city` and `funding_total_usd_verified` — this predates any processing done here and was left untouched (flagged, not altered).
4. **Thresholds (7 years, 3 years, 2 rounds) are defensible defaults, not proven-optimal values.** They're fully adjustable in the Assumptions tab if you want to run a sensitivity check across a few combinations before finalizing.

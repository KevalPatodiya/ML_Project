# UK Used Cars: Data Cleaning Pipeline

**Notebook:** `data_cleaning.ipynb`
**Input:** `uk_used_cars_autotrader.csv` (raw AutoTrader UK used-car listings)
**Final output:** `used_cars_cleaned_07.csv` (362,794 rows x 22 columns, 0 missing values; the three `*_missing` indicator columns were manually deleted from this file after the notebook ran)
**Held-out output:** `test.csv` (20% of the raw data, never touched by cleaning)
**Next step:** `linear_regression.ipynb` consumes `used_cars_cleaned_07.csv` (see `README_linear_regression.md`)

---

## Table of contents

1. [Purpose](#1-purpose)
2. [Quick summary](#2-quick-summary)
3. [Requirements](#3-requirements)
4. [How to run](#4-how-to-run)
5. [Step-by-step walkthrough](#5-step-by-step-walkthrough)
6. [Row count tracker](#6-row-count-tracker)
7. [Checkpoint files](#7-checkpoint-files)
8. [Final schema](#8-final-schema)
9. [Removed indicator columns](#9-removed-indicator-columns)
10. [Design decisions and rationale](#10-design-decisions-and-rationale)
11. [Known limitations and caveats](#11-known-limitations-and-caveats)
12. [Suggested improvements](#12-suggested-improvements)

---

## 1. Purpose

This notebook turns a raw scrape of UK used-car adverts into a modelling-ready table for predicting `price_gbp` (the advertised price in pounds). It covers:

- Hold-out splitting (so a test set is set aside before any cleaning decisions are made)
- Dropping identifier, duplicate-unit, leaky and uninformative columns
- Removing unusable rows (too many missing values, brand-new cars, implausible owner counts)
- Imputing missing values with strategies matched to each column
- Detecting and repairing implausible values (CO2, doors, seats)
- Unit conversion (g/km to g/mile for CO2)
- Saving a numbered CSV after each major stage so any stage can be inspected or resumed

---

## 2. Quick summary

| Item | Value |
|---|---|
| Rows after the 80/20 split (working set) | 383,980 |
| Columns in the raw file | 36 |
| Columns after dropping | 22 |
| Rows removed during cleaning | 21,186 (about 5.5% of the working set) |
| Rows in the final file | 362,794 |
| Columns in the final file | 22 (the 3 missing-value indicator columns were manually removed) |
| Missing values in the final file | 0 |
| Target column | `price_gbp` |
| Random seed | 42 |

---

## 3. Requirements

- Python 3.10 (the notebook kernel reports 3.10.11)
- `pandas`
- `numpy`
- `scikit-learn` (only for `train_test_split`)
- `matplotlib`
- `seaborn`
- Jupyter (Notebook, Lab or VS Code)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 4. How to run

1. Place `uk_used_cars_autotrader.csv` in the same folder as the notebook.
2. Open `data_cleaning.ipynb`.
3. Run all cells **top to bottom, in order**. Later cells reload earlier checkpoint CSVs and also depend on in-memory variables, so skipping or reordering cells will break the chain.
4. Outputs are written to the working folder: `test.csv` and `used_cars_*` CSVs.

Note: the split cell (`random_state=42`) is deterministic, so re-running produces the same train/test partition.

---

## 5. Step-by-step walkthrough

### Step 1. Load and hold out a test set

- Reads `uk_used_cars_autotrader.csv`.
- `train_test_split(test_size=0.2, random_state=42)`; the 20% test portion is saved untouched to `test.csv`.
- Indices of both parts are reset. All following cleaning works only on the remaining 80% (**383,980 rows**).
- A quick `head()` and `describe()` show the raw structure. Notable observations from `describe()`:
  - `price_evaluation` is entirely empty (count = 0).
  - `price_gbp` ranges from £75 to £9,999,999 (placeholder-style maximum).
  - `power_hp` has a minimum of 0, `reg_year` a minimum of 1936, and `previous_owners` a maximum of 99.
  - `co2_gkm` reaches 1300 and `seats` reaches 18.

### Step 2. Drop columns (checkpoint `used_cars_cols_dropped_01.csv`)

Dropped columns and the reason:

| Column | Reason |
|---|---|
| `id`, `url` | Identifiers, no predictive value |
| `title` | Free text largely duplicating make/model/version |
| `price_eur` | Direct currency conversion of the target (leakage) |
| `price_evaluation` | 100% empty |
| `mileage_km` | Duplicate of `mileage_miles` in another unit |
| `power_kw` | Duplicate of `power_hp` in another unit |
| `plate` | Registration plate, redundant with `reg_year` |
| `seller_name`, `seller_zip` | High-cardinality, identifying seller details |
| `images_count`, `is_enriched` | Scrape/listing metadata, not car characteristics |
| `first_seen_at`, `last_checked_at` | Scrape timestamps |

The notebook also contains a generic check that would drop any column with one distinct value. 36 minus 14 explicit drops equals the 22 columns observed, which implies no extra constant columns were found in this run (inferred from the shapes, not printed explicitly).

Result: **22 columns**, 383,980 rows.

### Step 3. Inspect missing values

- A table of missing counts and percentages, plus a heatmap of a random 1,000-row sample (`random_state=42`).
- Largest gaps at this point: `previous_owners` (15.91%), `emission_class` (9.77%), `annual_tax_gbp` (4.60%), `seller_county` (3.46%), `reg_year` (2.71%), `body_color` (2.69%), `co2_gkm` (2.66%), `drivetrain` (2.63%).

### Step 4. Drop rows that are mostly empty

- Rows with **more than 5 missing values** are removed: **5,761 rows** -> 378,219 remain. Inspection showed these were listings with `fuel`, `transmission`, `body_type`, and many other fields all blank, so there was too little information to impute reliably.

### Step 5. Keep only used cars

- `condition` had `Used` (365,907 rows) and `New` (12,312 rows). New cars are removed because the project targets the used-car market and new-car pricing behaves differently.
- 365,907 rows remain. After this, `condition` has a single value; it is kept in the file but is dropped again at modelling time.
- Saved as **`used_cars_cleaned_02.csv`** (365,907 x 22).

### Step 6. Fill categorical gaps and a few numeric gaps (checkpoint 03)

- `emission_class`, `seller_county`, `body_color`, `drivetrain`, `seller_city`, `version`: missing values replaced with the explicit category `"Unknown"`. This keeps all rows and lets a model learn whether "unknown" itself carries signal.
- `reg_year`: 53 rows with no registration year are **dropped** (negligible loss; registration year is a core price driver, so it is not worth guessing). 365,854 rows remain.
- `doors` and `seats`: filled with the **mode within (`make`, `model`, `body_type`)**, falling back to the global mode, then cast to integers.
- Saved as **`used_cars_cleaned_03.csv`** (365,854 x 22).

### Step 7. Outlier investigation and CO2 unit conversion (checkpoint 04)

- Box plots and IQR bounds (1.5 x IQR) were examined for `previous_owners`, `co2_gkm` and `annual_tax_gbp`.
- `previous_owners`: IQR upper bound is 6, but values go up to 99, which is implausible. **Rows with 8 or more previous owners were removed (3,060 rows)**; missing values were kept for later imputation. 362,794 rows remain.
- `annual_tax_gbp` and `co2_gkm` show many "outliers" by the IQR rule, but this is because the distributions are tightly clustered (most cars pay exactly £200 tax), not because the values are errors, so they were kept.
- **CO2 unit conversion:** `co2_gmile = co2_gkm x 1.609344`, then the original `co2_gkm` is dropped. Values are rounded to the nearest integer using the nullable `Int64` type so missing values survive.
- Saved as **`used_cars_cleaned_04.csv`** (362,794 x 22); the file was written again after rounding.

### Step 8. Impute the remaining numeric gaps (checkpoint 05)

After reloading `used_cars_cleaned_04.csv`, the remaining gaps were `previous_owners` (12.21%), `annual_tax_gbp` (3.08%) and `co2_gmile` (1.16%). For each, a 0/1 **indicator column** records which rows were originally missing, and then values are filled with medians of progressively broader groups of similar cars:

| Column | Indicator created | Fill strategy |
|---|---|---|
| `annual_tax_gbp` | `annual_tax_gbp_missing` | Median by (`make`,`model`,`version`,`fuel`,`reg_year`) -> (`make`,`model`,`fuel`,`reg_year`) -> (`fuel`,`reg_year`) -> overall median |
| `co2_gmile` | `co2_gmile_missing` | Median by (`make`,`model`,`version`,`fuel`,`reg_year`) -> (`make`,`model`,`fuel`,`reg_year`) -> (`make`,`model`,`fuel`) -> (`fuel`,`body_type`) -> overall median |
| `previous_owners` | `previous_owners_missing` | Median by `reg_year` (rounded), then overall median |

Medians are calculated only from originally recorded values (previously imputed values are masked out), so imputed numbers do not feed back into later medians.

Saved as **`used_cars_cleaned_05.csv`** (362,794 x 25). Zero missing values at this point. The three indicator columns exist in this checkpoint (and in 06), but were later deleted by hand from the final file 07 (see [section 9](#9-removed-indicator-columns)).

### Step 9. Repair suspicious CO2 values (checkpoint 06)

- An all-columns outlier table and 99th-percentile table were reviewed. Rows with `co2_gmile > 500` were inspected: supercars (Lamborghini, Ferrari, Bugatti) legitimately have very high CO2, but a group of ordinary vehicles (Ford Puma hybrid, Isuzu D-Max, GWM POER300, Nissan Juke, plug-in-hybrid Range Rover Sport, etc.) showed **1,608 or 2,092 g/mile**, which are clearly data errors (probably unit or decimal mistakes).
- Those **13 rows** had `co2_gmile` set to missing and were re-imputed with the same hierarchical median approach, now with `transmission` added to the grouping keys and `fuel` as a last resort before the overall median. Suspicious values were excluded from the reference data.
- A temporary audit flag `co2_gmile_suspicious` was used and then dropped, so it does not appear in the final file.
- Saved as **`used_cars_cleaned_06.csv`** (362,794 x 25).

### Step 10. Repair impossible `doors` and `seats` (checkpoint 07, final)

- 121 rows had `doors < 2` and 12 had `seats < 2` (131 distinct rows overall). Some are genuine data errors; vans and pickups can legitimately have unusual values, but fewer than 2 doors/seats is treated as invalid here.
- Those values are set to missing and refilled with the **mode** within (`make`,`model`,`version`,`body_type`) -> (`make`,`model`,`body_type`) -> (`body_type`) -> overall mode, then cast to `Int64`.
- The notebook saves **`used_cars_cleaned_07.csv`** as 362,794 x 25. The three `*_missing` indicator columns were then **deleted manually**, so the file as it now exists is **362,794 x 22**.
- A final check confirms **0 missing values** across all columns.

---

## 6. Row count tracker

| Stage | Action | Rows removed | Rows remaining |
|---|---|---|---|
| Start (after 80/20 split) | - | - | 383,980 |
| Step 4 | More than 5 missing values per row | 5,761 | 378,219 |
| Step 5 | `condition == "New"` | 12,312 | 365,907 |
| Step 6 | Missing `reg_year` | 53 | 365,854 |
| Step 7 | `previous_owners >= 8` | 3,060 | 362,794 |
| **Total removed** | | **21,186** | **362,794** |

Steps 8 to 10 modify values but do not remove rows.

---

## 7. Checkpoint files

| File | Shape | What it contains |
|---|---|---|
| `test.csv` | about 20% of raw rows x 36 | Untouched raw hold-out set |
| `used_cars_cols_dropped_01.csv` | 383,980 x 22 | Columns dropped, no row filtering |
| `used_cars_cleaned_02.csv` | 365,907 x 22 | Sparse rows and new cars removed |
| `used_cars_cleaned_03.csv` | 365,854 x 22 | Categorical "Unknown" fill, doors/seats filled, `reg_year` NaN rows dropped |
| `used_cars_cleaned_04.csv` | 362,794 x 22 | Owner outliers removed, CO2 converted to g/mile |
| `used_cars_cleaned_05.csv` | 362,794 x 25 | Numeric imputation plus 3 indicator columns |
| `used_cars_cleaned_06.csv` | 362,794 x 25 | Suspicious CO2 values repaired |
| `used_cars_cleaned_07.csv` | 362,794 x 22 | **Final.** Doors/seats repaired, no missing values. Indicator columns manually deleted (the notebook itself writes 25 columns) |

(The print message after the first save says `used_cars_cleaned.csv`, but the file actually written is `used_cars_cols_dropped_01.csv`.)

---

## 8. Final schema

| Column | Type | Notes |
|---|---|---|
| `make`, `model` | text | Brand and model |
| `version` | text | Trim/engine description; `"Unknown"` if absent; long free-text |
| `price_gbp` | int | **Target** |
| `mileage_miles` | int | |
| `reg_year` | float/int | Registration year |
| `fuel`, `transmission`, `body_type`, `body_color` | text | Some contain `"Unlisted"` or `"Unknown"` |
| `power_hp` | int | **0 is used for unknown power in many rows** (not imputed) |
| `doors`, `seats` | Int64 | Repaired in steps 6 and 10 |
| `previous_owners` | Int64 | Capped below 8 by row removal; imputed |
| `drivetrain`, `emission_class` | text | `"Unknown"` where missing |
| `annual_tax_gbp` | float | Imputed |
| `condition` | text | All `"Used"` (constant) |
| `seller_type`, `seller_city`, `seller_county` | text | High-cardinality for city/county |
| `co2_gmile` | Int64 | Converted from g/km, imputed and repaired |

That is 22 columns in total: 21 features plus the `price_gbp` target (`condition` is included and constant).

---

## 9. Removed indicator columns

During step 8 the notebook creates three 0/1 flag columns. They exist in `used_cars_cleaned_05.csv` and `used_cars_cleaned_06.csv`, and the notebook also writes them into `used_cars_cleaned_07.csv`. They were **manually deleted from the final CSV** and are therefore not part of the delivered dataset.

| Column (removed) | What it flagged |
|---|---|
| `annual_tax_gbp_missing` | 1 if the original tax value was absent before imputation |
| `co2_gmile_missing` | 1 if the original CO2 value was absent before imputation |
| `previous_owners_missing` | 1 if the original owner count was absent before imputation |

What this means in practice:

- **Imputed values can no longer be told apart** from recorded values in the final file. If you need that, use `used_cars_cleaned_06.csv` (it still has the flags) or re-run the notebook.
- **Re-running the notebook will bring the columns back** in `used_cars_cleaned_07.csv`; delete them again if you want the 22-column version.
- **Downstream effect on `linear_regression.ipynb`:** the notebook selects numeric columns by dtype and excludes any column ending in `_missing` from polynomial expansion, so it runs correctly without them. The only change is that the feature count is three lower, and the results printed in that notebook came from a run that **included** the flags, so re-running may give slightly different metrics.
- **Historical note:** the `co2_gmile_missing` flag was created before step 9, so the 13 rows repaired in step 9 were never flagged by it.

---

## 10. Design decisions and rationale

- **Split first, clean second.** The test set is carved out before any statistic (median, mode) is computed, which prevents test information from influencing cleaning rules.
- **Drop duplicated units** (`mileage_km`, `power_kw`, `price_eur`) to avoid redundant or leaking features.
- **`"Unknown"` instead of dropping or guessing** for categoricals: preserves rows and keeps missingness as a signal.
- **Hierarchical medians** for numerics: use the most specific similar-vehicle group available, and fall back to broader groups, so a rare car still gets a sensible estimate.
- **Indicator columns** were created in step 8 to make the imputation transparent and give models a way to discount imputed values. They were later removed by hand from the final file to keep the feature set smaller (see section 9).
- **Numbered checkpoints** make every step auditable and restartable.
- **Outliers were removed only when clearly invalid** (99 owners, 1,608 g/mile CO2, fewer than 2 doors). Extreme but legitimate values, such as £9.9M supercar prices, were kept.

---

## 11. Known limitations and caveats

1. **Extreme prices remain.** `price_gbp` still reaches £9,999,999 (a Ferrari F40 listing), and about 5.6% of rows exceed the IQR upper bound of roughly £45k. These dominate squared-error metrics downstream.
2. **`power_hp = 0` is a placeholder for unknown power**, left as is. It is not imputed, and many classics and some modern cars carry this value.
3. **Medians were computed on the 80% working set, not on a training sub-split.** The later modelling notebook makes its own train/validation split from the cleaned file, so a small amount of validation information leaks into the imputation statistics. The effect is likely small, but strict pipelines would impute inside the cross-validation loop.
4. **`test.csv` is raw.** To evaluate on it, the same cleaning steps must be replayed on it. They are not packaged as a reusable function or `sklearn` pipeline.
5. **Row removals are judgement calls.** The thresholds (more than 5 missing values per row, 8 or more owners, fewer than 2 doors/seats, the two CO2 values 1608/2092) were chosen by inspection of this dataset.
6. **`seller_city` / `seller_county` are inconsistent in casing** (for example `EXMOUTH` vs `Exmouth`, `DEVON` vs `Devon`) and were not normalised; they will be treated as separate categories.
7. **`condition` is constant after filtering** and `version` is high-cardinality free text; both are handled (dropped or vectorised) in the modelling notebook.
8. **The data is a point-in-time snapshot** (listing timestamps in October 2026), so prices reflect that market moment.
9. **The final CSV no longer records which values were imputed**, because the three indicator columns were deleted. About 12.2% of `previous_owners`, 3.1% of `annual_tax_gbp` and 1.2% of `co2_gmile` values are estimates that look identical to real ones.
10. **The CSV and the notebook are out of sync.** The notebook writes 25 columns to `used_cars_cleaned_07.csv`; the file on disk has 22 because of the manual edit.

---

## 12. Suggested improvements

- Wrap all steps into a single function or `sklearn` `Pipeline`/`ColumnTransformer` so they can be reapplied to `test.csv` and to new data.
- Normalise text casing for `seller_city`, `seller_county`, and other categoricals.
- Decide on a documented policy for extreme prices (cap, winsorise, remove or model separately, for example a supercar segment).
- Treat `power_hp == 0` as missing and impute, or add an indicator.
- Add automated sanity checks (assertions on ranges and dtypes) at the end of the notebook.
- Move imputation into the cross-validation loop to eliminate the leakage noted above.

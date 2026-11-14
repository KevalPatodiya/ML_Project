# Used Cars EDA and Cleaning (v07 to v08)

[EDA_and_More_cleaning.ipynb](EDA_and_More_cleaning.ipynb) profiles the UK used-car listings in `used_cars_cleaned_07.csv`, fixes the problems it finds, and writes a model-ready `used_cars_cleaned_08.csv`. It starts with 362,794 rows and 22 columns and ends with **361,590 rows (99.67%) and 33 columns**. The cleaned file feeds the Decision Tree experiments in `decision_tree.ipynb`.

## Files

| File | Role |
|---|---|
| `EDA_and_More_cleaning.ipynb` | The notebook: EDA, cleaning, validation checks, export |
| `used_cars_cleaned_07.csv` | Input (`RAW_PATH`) |
| `used_cars_cleaned_08.csv` | Output (`OUT_PATH`): cleaned data plus flag and derived columns |
| `cleaning_log_08.csv` | Output (`LOG_PATH`): one row per cleaning step with rows removed and values changed |

## Cleaning thresholds

All rules are set in one config cell at the top of the notebook.

| Setting | Value | Meaning |
|---|---:|---|
| `REF_YEAR` | 2026 | "Today" for vehicle age |
| `MIN_PRICE` | £500 | Below this is treated as scrap or a data-entry error and the row is dropped |
| `PRICE_PLACEHOLDER` | £9,999,999 | Dropped as a placeholder price |
| `MAX_MILEAGE` | 400,000 | Above this is a typo and is re-imputed |
| `MAX_REAL_HP` / `HYPERCAR_PRICE` | 1,000 hp / £250,000 | More than 1,000 hp is treated as an error unless the price is above £250,000 |
| `MAX_SEATS` | 9 | More seats than this is invalid and re-imputed |
| `PEER_RATIO` / `PEER_MIN_N` | 5 / 20 | Price more than 5× or less than 1/5 of the same make, model and registration-year median is an outlier, judged only when the group has at least 20 cars |
| `SENTINELS` | unknown, unlisted, none, n/a, na, nan, null, empty | Strings treated as "missing" |

## What the EDA found in the raw data

- **Duplicates and constants:** 885 exact duplicate rows, and `condition` is constant ("Used").
- **Missing values hidden as text:** 49,487 sentinel cells. The largest are `emission_class` (27,285, 7.52%), `seller_county` (7,917), `body_color` (5,288) and `drivetrain` (4,410).
- **Zero placeholders:** `previous_owners` 15,252 (4.20%), `co2_gmile` 21,951 (6.05%), `power_hp` 5,972 (1.65%), `mileage_miles` 1,826 (0.50%). Every zero-CO₂ row is Electric, Hydrogen or an unlisted fuel, so these zeros are real for EVs.
- **Inconsistent text:** unique `seller_city` labels fall from 2,808 to 1,809 and `seller_county` from 518 to 367 after case and whitespace normalisation.
- **Extreme values:**
  - Price: 1 listing at £9,999,999 (a Ferrari F40 placeholder), 90 below £500, 620 above £250,000.
  - Other fields: 23 with mileage above 400,000, 27 with more than 1,000 hp, 146 with more than 9 seats, 1,104 with more than 5 doors, and 1,048 registered before 2000.
- **Shape:** price skew is 92.53, while log10(price) is close to symmetric, which supports a log target. The strongest raw price relationships are registration year, mileage and power.

## Cleaning pipeline

| Step | Rows removed | Values changed | What it does |
|---|---:|---:|---|
| `text_normalisation` | 0 | 1,150 | Maps case and whitespace variants to the most frequent spelling. The count is the reduction in unique labels. Cities and counties get title case that keeps apostrophes intact. |
| `sentinel_to_nan` | 0 | 49,487 | Turns sentinel strings into missing values. A missing `version` becomes "Unknown". |
| `drop_constant_columns` | 0 | 0 | Drops `condition`. |
| `drop_duplicates` | 886 | 0 | Exact duplicates after normalisation. |
| `drop_invalid_rows` | 91 | 0 | 1 placeholder price and 90 prices below £500. No future or pre-1900 years were found. |
| `categorical_imputation` | 0 | 35,663 | Infers values from `version` text and peer groups (details below). |
| `mileage_fix` | 0 | 58 | Zero miles on a car at least 2 years old, or above 400,000, becomes missing, then is filled with a group median. |
| `power_fix` | 0 | 5,964 | Zero hp, or above 1,000 hp on a non-hypercar, is filled with a group median. |
| `co2_fix` | 0 | 31 | Forces 31 electric or hydrogen rows to 0 g/mile. No combustion cars needed CO₂ imputation. |
| `previous_owners_fix` | 0 | 9,426 | Zero owners on a car at least 2 years old is treated as unrecorded and filled with a registration-year median. |
| `seats_doors_fix` | 0 | 1,244 | 146 invalid seat counts and 1,098 invalid door counts are filled from the same make, model and version where possible, else the global mode. |
| `drop_price_peer_outliers` | 226 | 0 | Removes prices more than 5× or less than 1/5 of the peer median, for example a 2010 Clio at £10,950 against a £1,800 peer median. |
| `drop_duplicates_post_imputation` | 1 | 0 | Imputation made one pair of rows identical. |
| `derived_columns` | 0 | 0 | Adds the engineered columns listed below. |

Total rows removed: **1,204** (886 + 91 + 226 + 1).

**Categorical imputation details**
- **Fuel:** a fuel label found in the version text, else Electric when CO₂ is 0.
- **Transmission:** automatic or manual keywords in the version text, else the most common value for the same make, model and version.
- **Body type and drivetrain:** drivetrain is first inferred from 4WD, AWD, quattro, xDrive and similar words. Both columns then use the same make, model and version mode, falling back to make and model.
- **Emission class:** "Euro N" in the version text, else the most common class for the same registration year and fuel.
- **County:** the county seen for the same city.
- **Leftovers:** anything still unknown becomes the explicit category "Unknown".

| Column | Missing before | "Unknown" after |
|---|---:|---:|
| `version` | 0 | 463 |
| `fuel` | 179 | 74 |
| `transmission` | 216 | 105 |
| `body_type` | 992 | 174 |
| `body_color` | 5,029 | 5,029 |
| `drivetrain` | 4,393 | 555 |
| `emission_class` | 27,043 | 429 |
| `seller_city` | 3,180 | 3,180 |
| `seller_county` | 7,900 | 3,260 |

`body_color` and `seller_city` cannot be inferred from other fields, so they stay "Unknown".

## Output columns

The 21 original columns are kept in their original order (`condition` is dropped). Twelve columns are added:

| Column | Description |
|---|---|
| `vehicle_age` | `REF_YEAR` minus `reg_year`, floored at 0 |
| `log_price` | `log1p(price_gbp)` |
| `miles_per_year` | Mileage divided by age, with age floored at 1 |
| `engine_size_l` | Litres parsed from the `version` text and limited to 0.6–8.5, with electric and hydrogen cars set to 0 and the rest filled from peer medians |
| `is_performance_trim` | 1 if the version text contains AMG, M Sport, GTI, S line, ST-Line, R-Line, Type R, Cupra, Abarth, VXR or VRS |
| `is_high_value` | 1 if price is above £250,000. These real luxury cars are flagged, not dropped (594 remain). |
| `is_classic` | 1 if registered before 2000 |
| `emission_class_imputed`, `mileage_imputed`, `power_hp_imputed`, `co2_imputed`, `previous_owners_imputed` | 1 where the value was filled in, so models can tell real values from imputed ones |

## Results

| Metric | Raw | Cleaned |
|---|---:|---:|
| Rows | 362,794 | 361,590 |
| Columns | 22 | 33 |
| Maximum price | £9,999,999 | £5,000,000 |
| Price skew | 92.53 | 53.26 |
| Zero `power_hp` | 5,972 | 0 |
| Zero `previous_owners` | 15,252 | 5,736 |
| Seats above 9 | 146 | 0 |
| Duplicate rows | 885 | 0 |
| "Unknown" or "Unlisted" cells | 49,950 | 13,261 |

`log_price` skew after cleaning is −0.281. Remaining zero mileage (1,721 rows) and zero previous owners are mostly cars under 2 years old, which the rules deliberately leave alone.

Spearman correlation with `log_price` on the cleaned data:

| Feature | Correlation |
|---|---:|
| `vehicle_age` | −0.75 |
| `mileage_miles` | −0.68 |
| `power_hp` | +0.64 |
| `previous_owners` | −0.54 |
| `engine_size_l` | +0.24 |
| `co2_gmile` | +0.20 |
| `annual_tax_gbp` | +0.19 |
| `is_performance_trim` | +0.17 |

## Validation checks

The notebook asserts 12 checks and all pass: no missing values, no duplicates, no constant text columns, price within £500 up to the placeholder, `reg_year` not after 2026, power above 0, non-EV CO₂ above 0, EV CO₂ equal to 0, seats 2–9, doors 2–5, mileage at most 400,000, and no leftover "Unlisted" labels. "No missing values" means no blank cells. The 13,261 "Unknown" cells remain as a category.

## Caveats for modelling

- **Imputation uses the whole dataset.** Group medians and modes were computed before any train/validation split, so validation rows can influence imputed values. For strict evaluation, impute inside a pipeline fitted on training rows only.
- **The outlier filter uses the target.** The 226 rows removed by the peer-price rule were chosen using price and whole-dataset medians, so reported errors apply to a price-filtered population.
- **Heavily imputed fields.** `previous_owners` had 9,426 values filled with registration-year medians, which probably contributes to its −0.54 correlation with price. Use the `*_imputed` flags if this matters.
- **Redundant columns.** `price_gbp` and `log_price` are both in the file, so use one as the target and drop the other. The notebook also warns that `version` and `seller_city` are high-cardinality text.
- **Fixed reference year.** `vehicle_age` and the "at least 2 years old" rules assume 2026 and need updating for later data.

## Reproduce

1. Put `EDA_and_More_cleaning.ipynb` and `used_cars_cleaned_07.csv` in the same folder.
2. Use Jupyter with NumPy, pandas, matplotlib and seaborn.
3. Run all cells in order. The final cell writes `used_cars_cleaned_08.csv` and `cleaning_log_08.csv` and asserts the saved file has no missing values.

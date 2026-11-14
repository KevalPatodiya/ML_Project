# Used Car Price Prediction: Ensemble Methods

Predicting UK used-car listing prices (`price_gbp`) with tree-based ensemble models, compared against a log-price Ridge regression baseline. The work is in `random_forest.ipynb`.

## Data

- **File:** `used_cars_cleaned_08.csv` (not included; place it next to the notebook)
- **Target:** `price_gbp`, modelled on the log scale (`log1p` / `expm1` via `TransformedTargetRegressor`)
- **Split:** 80/20 `GroupShuffleSplit` (seed 42), grouping rows with identical model inputs so duplicates never straddle train and validation
  - Training: 289,272 listings
  - Validation: 72,318 listings

### Features

| Type | Columns |
|---|---|
| Numeric | `mileage_miles`, `reg_year`, `power_hp`, `previous_owners`, `annual_tax_gbp`, `co2_gmile` |
| Counts | `doors`, `seats` |
| Categorical | `make`, `model`, `version`, `fuel`, `transmission`, `body_type`, `body_color`, `drivetrain`, `seller_type`, `seller_city`, `seller_county` |
| Engineered | `log_mileage` (`log1p` of mileage), `make_model` (make + model combined) |

`emission_class` is deliberately excluded.

### Preprocessing

- Numeric: median imputation
- Counts: most-frequent imputation
- Categorical: one-hot encoding (unknown categories ignored)
- Ridge only: `StandardScaler(with_mean=False)` after preprocessing

## Evaluation Metrics

All models are scored on the same validation listings:

- **MAE (£)** and **RMSE (£)**
- **R²**
- **Within 10 %**: share of predictions within 10 % of the true price
- **Band accuracy**: share of predictions landing in the correct price band (< £10,000, £10,000-£25,000, > £25,000)

Hyperparameters were tuned on a held-out split *inside* the training set, so the validation set stays untouched by tuning.

## Experiments and Results

| Model | MAE (£) | RMSE (£) | R² | Within 10 % | Band acc. |
|---|---|---|---|---|---|
| Ridge (log-price, baseline) | 2,239 | 72,044 | -3.12 | 71.9 % | 94.0 % |
| **Random Forest** (40 trees, depth 24, leaf 3) | **1,850** | 24,305 | 0.531 | 68.1 % | 93.5 % |
| Random Forest, tuned (leaf 2) | 1,802 | 23,166 | 0.574 | 68.6 % | 93.5 % |
| Bagging (tuned, 20 trees) | 2,433 | 26,516 | 0.442 | 59.3 % | 91.8 % |
| Pasting (tuned, 20 trees) | 2,406 | 26,614 | 0.438 | 59.4 % | 91.8 % |
| Extra Trees | 2,001 | 24,935 | 0.506 | 65.1 % | 92.8 % |
| Extra Trees, tuned | 2,232 | 26,227 | 0.454 | 61.5 % | 92.2 % |
| AdaBoost, original | 4,242 | 21,525 | 0.632 | 32.4 % | 84.2 % |
| AdaBoost, tuned (depth 14) | 2,630 | **19,414** | **0.701** | 52.1 % | 90.2 % |
| Gradient Boosting, original | 3,094 | 26,371 | 0.448 | 46.6 % | 89.6 % |
| Gradient Boosting, tuned (depth 7) | 2,700 | 25,525 | 0.483 | 52.4 % | 91.0 % |

### Notebook walkthrough

1. **Ridge vs Random Forest:** baseline comparison; the forest has much lower MAE and RMSE.
2. **Random Forest tuning:** a small grid over `min_samples_leaf` and `max_features`; `min_samples_leaf=2, max_features=0.7` wins.
3. **Bagging vs Pasting:** decision-tree ensembles with and without bootstrap sampling, then tuned over `max_depth` and `min_samples_leaf`.
4. **Extra Trees:** original and tuned versions.
5. **AdaBoost:** original and tuned (deeper base trees helped a lot).
6. **Gradient Boosting:** original and tuned (deeper trees helped).

## Key Takeaways

- **Random Forest has the best MAE** (about £1,800-£1,850) and the best "within 10 %" score among the tree ensembles.
- **Ridge's RMSE is huge (R² = -3.1)** because exponentiating log-scale predictions amplifies a few extreme errors, even though its MAE and "within 10 %" scores are reasonable.
- **AdaBoost has the best RMSE and R²** (19,414 and 0.701) but a worse MAE, so it handles large errors on expensive cars better while being less accurate on typical listings.
- **Bagging/Pasting and Extra Trees did not beat the forest.** Pasting and Bagging were close to each other, and tuning Extra Trees did not help on validation.
- **Boosting needed deeper trees.** Both boosters improved markedly when depth increased from the shallow defaults.
- Which model is "best" depends on the metric: MAE favours the forest, RMSE and R² favour AdaBoost.

## Caveats

- Several tuning runs use subsamples (e.g., 60,000 training rows for AdaBoost and Gradient Boosting; 15,000-40,000 rows per tree for Bagging and Extra Trees) and few trees (8-60) to keep runtime manageable on 2 CPU cores. Full-data, larger ensembles would likely score better.
- Tuning grids are small and not exhaustive.
- Results come from a single train/validation split and a single seed.
- The Bagging notebook cell uses a hard-coded forest MAE (`FOREST_MAE = 1816.211`) for its "gain vs forest" column, which differs slightly from the forest MAE reported elsewhere (1,850).
- The full Random Forest fit takes about 22 minutes with 2 workers.

## Requirements

- Python 3.10
- `numpy`, `pandas`, `scikit-learn`, `ipython` / Jupyter

## Running

1. Place `used_cars_cleaned_08.csv` in the same folder as the notebook.
2. Open `random_forest.ipynb` and run the cells **in order**. Later cells reuse objects from earlier ones (`preprocessor`, `forest`, `X_train`, `X_val`, `extra_trees`, `adaboost`, and others).
3. Budget substantial time: the full Random Forest fit and the Extra Trees and Bagging refits take several minutes each.

All randomness is seeded (`SEED = 42`) for reproducibility.

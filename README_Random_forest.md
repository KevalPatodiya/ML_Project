# Used Car Price Prediction: Ensemble Methods

Predicting UK used-car listing prices (`price_gbp`) with tree-based ensemble models, compared against a log-price Ridge regression baseline. The work is in `random_forest.ipynb`.

## Historical notebook data

- **File:** `used_cars_cleaned_08.csv` (not included; place it next to the notebook)
- **Target:** `price_gbp`, modelled on the log scale (`log1p` / `expm1` via `TransformedTargetRegressor`)
- **Split:** 80/20 `GroupShuffleSplit` (seed 42), grouping rows with identical model inputs so exact matches of the selected inputs stay in one split
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

Hyperparameters were selected using an internal training split. The outer validation results were subsequently inspected across experiments.

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

## Historical observations

- **Random Forest has the best MAE** (about £1,800-£1,850) and the best "within 10 %" score among the tree ensembles.
- **Ridge's RMSE is huge (R² = -3.1)** with a much larger RMSE than MAE. Exponentiating log-scale predictions can amplify extreme errors, while its MAE and "within 10 %" scores are reasonable.
- **AdaBoost has the best RMSE and R²** (19,414 and 0.701) but a worse MAE, while its larger MAE indicates worse average absolute error. Price-segment performance requires separate evaluation.
- **Bagging/Pasting and Extra Trees did not beat the forest.** Pasting and Bagging were close to each other, and tuning Extra Trees did not help on validation.
- **Boosting needed deeper trees.** Both boosters improved markedly when depth increased from the shallow defaults.
- Which model is "best" depends on the metric: MAE favours the forest, RMSE and R² favour AdaBoost.

## Historical limitations

- Several tuning runs use subsamples (e.g., 60,000 training rows for AdaBoost and Gradient Boosting; 15,000-40,000 rows per tree for Bagging and Extra Trees) and few trees (8-60) to keep runtime manageable on 2 CPU cores. Performance with larger fitting budgets was unverified in these historical experiments.
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

## Later notebook and chat experiments

Updated 18 November 2026. Cell numbers are one-based positions in the supplied 13-cell notebook. Cells 1–10 contain the historical validation experiments above. Cells 11–12 add the Random Forest/Lasso blend and model export. The later coarse-to-fine forest search and updated blend were run separately during this chat.

### Exact historical notebook observations

The saved notebook outputs provide these more precise scores. They use the 72,318-row validation split of used_cars_cleaned_08.csv.

| Model | MAE (£) | RMSE (£) | R² | Within 10% (%) | Band accuracy (%) |
|---|---:|---:|---:|---:|---:|
| Refit log-price Ridge | 2,239.498 | 72,044.113 | -3.122 | 71.869 | 93.999 |
| Original Random Forest | 1,850.264 | 24,304.879 | 0.531 | 68.072 | 93.451 |
| Tuned Random Forest | 1,802.159 | 23,166.157 | 0.574 | 68.589 | 93.519 |
| Original Bagging | 2,652.760 | 26,743.986 | 0.432 | 55.010 | 90.915 |
| Original Pasting | 2,637.344 | 26,802.553 | 0.430 | 55.190 | 91.037 |
| Tuned Bagging | 2,433.219 | 26,515.785 | 0.442 | 59.284 | 91.797 |
| Tuned Pasting | 2,406.249 | 26,614.072 | 0.438 | 59.350 | 91.840 |
| Original Extra Trees | 2,001.191 | 24,934.983 | 0.506 | 65.123 | 92.767 |
| Tuned Extra Trees | 2,232.131 | 26,227.384 | 0.454 | 61.496 | 92.150 |
| Original AdaBoost | 4,242.299 | 21,524.890 | 0.632 | 32.372 | 84.227 |
| Tuned AdaBoost | 2,630.020 | 19,413.520 | 0.701 | 52.062 | 90.242 |
| Original Gradient Boosting | 3,093.858 | 26,370.788 | 0.448 | 46.633 | 89.588 |
| Tuned Gradient Boosting | 2,699.983 | 25,525.213 | 0.483 | 52.405 | 91.002 |

The forest tuning in cell 2 selected minimum leaf size 2 and max_features=0.7. Its three internal tuning MAEs were £2,383.31 for leaf 3/features 0.7, £2,325.89 for leaf 2/features 0.7, and £2,451.97 for leaf 3/features 0.4. The final selected forest retained 40 trees, depth 24 and max_samples=0.6.

Tuned Bagging and Pasting selected depth 24 and leaf size 2. The Extra Trees search selected depth 24, leaf size 2 and max_features=0.7, but its refitted validation scores worsened relative to the original Extra Trees run. AdaBoost selected depth 14 and Gradient Boosting selected depth 7. These searches were limited and used different fitting budgets.

MAPE is absent from these historical output tables. It cannot be recovered from MAE, RMSE or Within 10% without the corresponding predictions. The later blend experiments report MAPE directly.

Earlier messages in this chat contain different historical run scores, such as £1,816.211 MAE for Random Forest. The tables here use the supplied notebook's saved outputs. Its scores and the older chat run should not be mixed into one leaderboard without identifying the run.

### Original Random Forest + tuned log-price Lasso + Gradient Boosting

Cell 11 uses train_clean.csv with 352,458 training rows and test_clean.csv with 88,120 test rows. These differ from the historical cleaned_08 validation experiment.

The loader excludes price_gbp, log_price, price_eur, is_high_value, price_evaluation and emission_class from predictors. It adds log_mileage and make_model. Numeric columns receive median imputation. Categories receive constant imputation and one-hot encoding. The forest groups categories occurring fewer than 20 times. Lasso uses sparse scaling and the previously tuned alpha=0.00003, fitted with a sparse solver and duality-gap convergence check.

The original blend uses 85% Lasso and 15% forest. Both bases learn log-price and convert predictions to GBP before blending. The forest has 12 trees, depth 24, minimum leaf size 2, max_features=0.7, bootstrap sampling and 20,000 draws per tree.

Five grouped folds generate an out-of-fold blend prediction for every training row. The correction target is actual price minus that prediction in GBP. Gradient Boosting uses Huber loss, 40 estimators, learning rate 0.05, depth 3 and minimum leaf size 30. Final predictions add the predicted correction to the fitted base blend.

| Model | Split | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Lasso | Training, fitted | 1,187.520 | 5,421.394 | 7.627 | 0.9727 | 77.682 |
| Old forest | Training, fitted | 2,308.866 | 22,587.712 | 12.564 | 0.5258 | 60.436 |
| Old forest + Lasso | Training, fitted | 1,211.327 | 6,463.532 | 7.630 | 0.9612 | 78.218 |
| Old forest + Lasso + correction | Training, fitted | 1,186.471 | 6,451.489 | 7.541 | 0.9613 | 78.880 |
| Lasso | Test | 1,513.960 | 10,185.208 | 9.753 | 0.9000 | 73.056 |
| Old forest | Test | 2,406.350 | 21,778.966 | 13.163 | 0.5428 | 59.066 |
| Old forest + Lasso | Test | 1,474.315 | 10,682.457 | 9.457 | 0.8900 | 74.222 |
| Old forest + Lasso + correction | Test | 1,447.464 | 10,667.604 | 9.346 | 0.8903 | 75.010 |

The uncorrected blend's pooled out-of-fold training MAE was £1,557.839, with 73.975% within 10%. Its fitted-training scores describe in-sample performance.

### Coarse-to-fine tuning of the standalone Random Forest

The chat then tuned the forest alone using a grouped split inside train_clean.csv, with 281,971 fitting rows and 70,487 validation rows. Selection prioritised Within 10%, with MAE as the tie-breaker. The test file was evaluated after selecting the configuration. All candidates used log-price targets, the same forest preprocessing, seed 42 and two workers.

The reference for this search was the 12-tree forest from the blend experiment. It differs from the 40-tree forest in cells 1–2.

| Stage | Trees | Max depth | Min leaf | Max features | Bootstrap draws per tree | Tuning MAE (£) | Within 10% (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old reference | 12 | 24 | 2 | 0.7 | 20,000 | 2,643.493 | 58.347 |
| Coarse | 12 | 36 | 2 | 0.7 | 20,000 | 2,644.770 | 58.489 |
| Coarse | 12 | 36 | 1 | 0.5 | 20,000 | 2,614.715 | 59.183 |
| Coarse winner | 12 | 32 | 2 | 0.7 | 40,000 | 2,362.245 | 62.587 |
| Coarse | 12 | 32 | 3 | 0.5 | 40,000 | 2,450.674 | 61.223 |
| Fine | 12 | 36 | 2 | 0.7 | 40,000 | 2,386.699 | 62.542 |
| Fine | 12 | 32 | 2 | 0.8 | 40,000 | 2,365.130 | 62.465 |
| Fine | 12 | 32 | 1 | 0.7 | 40,000 | 2,325.915 | 63.173 |
| Fine | 12 | 32 | 2 | 0.7 | 50,000 | 2,298.673 | 63.654 |
| Fine, more trees | 24 | 32 | 2 | 0.7 | 50,000 | 2,244.868 | 64.925 |
| **Selected** | **36** | **32** | **2** | **0.7** | **50,000** | **2,224.801** | **65.323** |

Both the old and selected standalone models were refitted on all 352,458 training rows.

| Split | Forest | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Training | Old tuned | 2,308.866 | 22,587.712 | 12.564 | 0.5258 | 60.436 |
| Training | New tuned | 1,791.995 | 21,478.022 | 9.921 | 0.5712 | 70.035 |
| Test | Old tuned | 2,406.350 | 21,778.966 | 13.163 | 0.5428 | 59.066 |
| Test | New tuned | 1,994.323 | 20,613.538 | 11.237 | 0.5904 | 66.122 |

The new forest lowered test MAE by £412.03 and improved Within 10% by 7.056 percentage points, putting 6,218 more test listings within that margin. The new training/test proximity rates differ by 3.913 percentage points. This search improved the forest substantially within the tested budget. It does not establish a global optimum.

### Newly tuned Random Forest + Lasso + Gradient Boosting

The next experiment used the selected 36-tree forest with Lasso alpha=0.00003 and retained the 85% Lasso / 15% forest weights. It regenerated all five-fold residual targets and refitted the Huber correction.

| Split | Ensemble | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Training, fitted | Old forest + Lasso + correction | 1,186.471 | 6,451.489 | 7.541 | 0.9613 | 78.880 |
| Training, fitted | New forest + Lasso + correction | 1,150.136 | 6,329.610 | 7.342 | 0.9628 | 79.665 |
| Test | Old forest + Lasso + correction | 1,447.464 | 10,667.604 | 9.346 | 0.8903 | 75.010 |
| Test | New forest + Lasso, before correction | 1,454.834 | 10,590.006 | 9.348 | 0.8919 | 74.629 |
| Test | New forest + Lasso + correction | 1,427.423 | 10,576.601 | 9.236 | 0.8922 | 75.394 |

The uncorrected new blend achieved £1,534.640 pooled out-of-fold MAE and 74.360% within 10%. The old blend's corresponding figures were £1,557.839 and 73.975%.

The new corrected ensemble lowered test MAE by £20.04 relative to the old corrected ensemble and gained 338 listings within 10%. The correction itself lowered the new blend's MAE by £27.41 and increased Within 10% from 74.629% to 75.394%.

Each forest fold took approximately eight minutes in the observed run. Total runtime includes five folds, a full-data base refit and the correction fit. These timings are machine- and configuration-dependent.

### Comparison with the best observed Decision Tree blend

These ensembles use the same 88,120 test rows. The tree blend uses 75% Lasso and 25% tree, while the forest blend uses 85% Lasso and 15% forest.

| Corrected ensemble | Test MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|
| Depth-33 Decision Tree + Lasso | 1,414.637 | 10,903.144 | 9.193 | 0.8854 | 75.555 |
| Newly tuned Random Forest + Lasso | 1,427.423 | 10,576.601 | 9.236 | 0.8922 | 75.394 |

The forest blend has lower RMSE and higher R². The tree blend has lower MAE and higher Within 10%. The forest blend is the best observed Random Forest ensemble from these chat experiments.

### Extra Trees coarse-to-fine search status

A separate coarse-to-fine Extra Trees experiment was started using train_clean.csv, then the user requested a pause. Only the following completed trial scores were reported before that pause.

| Trial | Trees | Depth | Min leaf | Max features | Draws per tree | Tuning MAE (£) | Within 10% (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier tuned reference | 12 | 30 | 2 | 0.7 | 20,000 | 2,624.738 | 58.831 |
| Coarse | 12 | 36 | 2 | 0.7 | 20,000 | 2,644.118 | 58.778 |
| Coarse | 12 | 36 | 1 | 0.5 | 20,000 | 2,634.552 | 58.922 |

These are partial tuning observations. A final new-versus-old Extra Trees test comparison was not delivered. The reference here differs from the 20-tree Extra Trees model in cells 5–6.

## Validation and interpretation of the later runs

Within 10% measures the share satisfying abs(actual minus predicted) <= 0.10 times actual. MAPE averages absolute percentage errors. Band accuracy measures agreement with broad price categories and can remain high despite sizeable price errors.

Grouped tuning and out-of-fold splits keep exact prepared input matches together. They do not identify every duplicate vehicle or relisting. The later train/test audit found 156 test listings whose predictor values matched a training row.

| Standalone forest | Test rows without an exact training-input match | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|---:|
| Old tuned | 87,964 | 2,398.887 | 21,674.645 | 13.173 | 0.5433 | 59.029 |
| New tuned | 87,964 | 1,986.891 | 20,499.341 | 11.246 | 0.5915 | 66.090 |

The new forest's improvement remained after excluding those exact matches. The later corrected forest ensemble table uses all 88,120 test rows.

Preprocessing is fitted within the relevant training folds in the later runs. The complete provenance of cleaning in train_clean.csv and test_clean.csv was not re-audited. Grouping at model-training time cannot undo any earlier cleaning leakage.

The test file has been inspected across many experiments, so it now informs model selection. A fresh untouched holdout is required for an independent final generalisation estimate. Historical cleaned_08 validation scores and later cleaned train/test scores should be assessed within their respective experiment groups.

High fitted-training scores do not establish generalisation. The new corrected ensemble scored 79.665% within 10% on fitted training data and 75.394% on test data. Out-of-fold scores provide a separate view of base-blend performance.

## Saved model and reproducing the later experiments

Cell 12 saves randomforest+lasso+gradientboosting.joblib from cell 11. It contains tuned_lasso and tuned_forest tuples, the correction pipeline, blend weights, feature_columns and excluded_columns. The saved forest has 12 trees, depth 24 and 20,000 bootstrap draws per tree. The newer 36-tree experiment does not update this export automatically.

Cells 11–12 require train_clean.csv and test_clean.csv, SciPy, NumPy, pandas, scikit-learn and joblib. The historical cells additionally use used_cars_cleaned_08.csv. Update PROJECT_DIR to the data folder, since it still points to the former Sample Project directory even though this notebook now resides in the ML_Project repository.

Run cell 11 for the original Lasso/forest/correction experiment and cell 12 to save its bundle. To reproduce the new ensemble, use 36 estimators, depth 32, minimum leaf size 2, max_features=0.7 and 50,000 bootstrap draws per tree in every forest fit. Regenerate the five-fold residual targets before fitting the correction. Keep the 85% Lasso / 15% forest weights for the comparison documented here.

No original MAPE, classification precision/recall/F1 or unreported test result has been inferred from the available aggregate metrics.

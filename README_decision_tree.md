# Used Cars Price Estimation and Deal Finder

This project estimates the advertised asking price of UK used cars in pounds. [decision_tree.ipynb](decision_tree.ipynb) starts with a Decision Tree, compares tuning and text encoding, rebuilds validation to address data leakage, selects a log-price Ridge reference, and later blends a tree with log-price linear models and Gradient Boosting residual correction. The price estimate can help screen listings for potential deals. It does not establish sale price, vehicle condition, or whether a listing is a genuine bargain.

## Decision Tree experiments

The original notebook used `used_cars_cleaned_08.csv` and an 80/20 random training–validation split (72,318 validation listings). Numeric inputs were imputed, categorical inputs were one-hot encoded, and the tree was evaluated on exact asking prices and on three price bands: under £10,000, £10,000 to under £25,000, and £25,000 or more.

| Experiment | Depth | Minimum leaf size | CV MAE | Validation MAE | Within 10% | Band accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Basic tree | 15 | 10 | £2,885.74 | £2,795.99 | 49.003% | 89.326% |
| Initial tuning | 20 | 5 | £2,467.08 | £2,262.02 | 57.301% | 91.246% |
| Coarse-to-fine, one-hot version | 25 | 2 | £2,292.98 | £2,144.36 | 61.054% | 91.902% |
| Same tuned tree, CountVectorizer version | 25 | 2 | — | £2,092.77 | 61.537% | 91.970% |

- **Initial search.** A six-combination grid compared depths 10, 15 and 20 with leaf sizes 5 and 10 using three-fold CV. A separate sequential search proposed five hyperparameters and required at least a £5 CV MAE improvement to continue. Depth 20 improved CV MAE by £317.59 and leaf size 5 by a further £101.07. Increasing `min_samples_split` to 30 worsened CV MAE by £99.27, so the search stopped. The proposed `max_leaf_nodes` and `max_features` changes were skipped.
- **Coarse-to-fine search.** Depth 30 with leaf size 5 lowered CV MAE to £2,348.13. Depth 25 with leaf size 2 lowered it further to £2,292.98. The third coarse trial (unrestricted depth, leaf size 10) worsened CV MAE by £178.32, so the search stopped before the fine trials (depth 24/leaf 4, 26/3 and 28/3) were evaluated.
- **Version encoding.** CountVectorizer used unigrams and bigrams from the trim description. On this split it lowered validation MAE by £51.59 and RMSE by £852.15, raised R² from 0.651 to 0.684, and gained 0.483 percentage points within 10%. The rebuilt grouped-validation pipeline below still one-hot encodes `version`, and a later grouped text-tree comparison is documented below.
- **Fit diagnosis.** The tuned one-hot tree had £1,102.73 training MAE and £2,144.36 validation MAE. Its within-10% rate fell from 76.271% to 61.054%, and R² fell from 0.965 to 0.651. Validation RMSE (£17,624.09) was about 2.8 times training RMSE (£6,185.33), while median absolute error rose from £550.00 to £952.88. The gap indicates overfitting. RMSE rose proportionally more than the median error, which suggests a small number of large errors drive it.

## Validation rebuild and leakage audit

An earlier audit of the original cleaned CSV (`used_cars_cleaned_07.csv`) found that it already contained group medians and modes computed before the notebook made its training–validation split. That allowed validation and CV rows to influence imputation. The same audit found 268 original validation rows whose saved values, including price, matched a training row. Matching values alone cannot identify the same vehicle. Numeric Spearman correlations with price were strongest for registration year (+0.753), mileage (−0.677), and horsepower (+0.623).

The revised notebook starts from `uk_used_cars_autotrader.csv`, recreates the original 20% holdout with seed 42, then splits development data by groups of identical prepared predictors (73,695 grouped-validation listings). It applies fixed eligibility rules for used listings with a known registration year and fewer than eight previous owners when recorded. It normalises missing categories, masks invalid door, seat and horsepower values, converts CO₂ to grams per mile, and masks the two flagged emissions values. The pipeline learns numeric medians, door and seat modes, and categorical vocabularies from training rows. The predictor allowlist excludes both the asking price and its euro equivalent.

The grouped split prevents exact prepared predictor matches across the development training and validation sets. The original holdout was split by row, so exact overlaps with development and relisted vehicles remain possible. The revised results use different validation rows and preprocessing from the original tree experiments. The grouped-validation tree cell hard-codes depth 28 and minimum leaf size 3, which differs from the depth 25 and leaf size 2 tree selected in the coarse-to-fine search above.

| Model on grouped validation | MAE | RMSE | R² | Within 10% |
|---|---:|---:|---:|---:|
| Tuned Decision Tree (depth 28, leaf 3) | £2,362.49 | £20,600.67 | 0.651 | 59.182% |
| Log-price Ridge baseline | £1,789.08 | £17,412.04 | 0.751 | 71.822% |

Ridge uses L2 regularisation with `alpha=0.1`. It learns on `log1p(price_gbp)` and converts its predictions back to pounds with `expm1`. The log transform compresses the range of asking prices during fitting.

## Feature experiments and model selection

All feature experiments reused the same grouped split. MAE in pounds was the primary selection metric.

| Experiment | Validation MAE | Decision |
|---|---:|---|
| Log-price Ridge baseline | £1,789.08 | Reference |
| Add log mileage alongside raw mileage | £1,784.91 | Keep |
| Add age × log mileage to the log-mileage model | £1,795.98 | Omit |
| Add combined make and model to the log-mileage model | £1,768.18 | Keep |
| Add battery capacity from version text to the make–model model | £1,768.41 | Omit |
| Remove emission class from the make–model model | **£1,764.78** | Selected |

Battery capacity was extracted for 89.727% of electric and 99.032% of plug-in hybrid validation listings (10.390% of all listings), yet it barely changed MAE for either group. In the ablation tests, removing `version` increased MAE by £931.02, and removing `seller_city` increased it by £20.98. Removing annual tax improved MAE by £0.23 while worsening RMSE and the within-10% rate. Removing emission class improved MAE by £3.40 while increasing RMSE by £43.30. The saved model retains annual tax, seller location, and one-hot version encoding.

The selected model achieved **£1,764.78 MAE**, **£17,201.32 RMSE**, **0.757 R²**, **71.864% within 10%**, and **94.133% price-band accuracy** on grouped validation. The small emission-class gain was observed on one validation split.

## Saved model and holdout sample

The notebook writes `used_car_log_price_ridge.joblib` with the fitted model under the `model` key and the scikit-learn version as metadata. The saved pipeline includes fitted imputation, one-hot encoding, and scaling. Fixed listing preparation and the two engineered columns are applied before prediction.

The notebook loaded the raw 36-column `test.csv`, applied the same eligibility rules, and sampled 1,000 of the 92,098 eligible listings using seed 42. The raw holdout contained 95,995 listings.

| Metric | Holdout sample |
|---|---:|
| MAE | £1,520.06 |
| RMSE | £5,478.03 |
| R² | 0.877 |
| Within 10% of asking price | 70.90% |
| Price-band accuracy | 93.60% |
| Macro precision / recall / F1 | 0.929 / 0.933 / 0.931 |
| Mean signed residual | −£208.43 |
| Mean signed percentage residual | −3.98% |

| Price band | Precision | Recall | F1 | Listings |
|---|---:|---:|---:|---:|
| Under £10,000 | 0.967 | 0.965 | 0.966 | 339 |
| £10,000 to under £25,000 | 0.936 | 0.928 | 0.932 | 470 |
| £25,000+ | 0.883 | 0.906 | 0.894 | 191 |

The notebook displays the first 15 actual and predicted prices with their signed residuals. Residual equals actual minus predicted, so the negative mean indicates overprediction on this sample. The percentage averages each residual divided by its positive asking price. Band accuracy measures agreement with broad price ranges. The within-10% rate measures proximity to the actual price. These are sample scores. A full evaluation of that raw holdout workflow was still pending at that stage, and RMSE is particularly sensitive to the mix of very expensive listings.

## Reproduce the notebook

1. Put `decision_tree.ipynb`, `uk_used_cars_autotrader.csv`, and `test.csv` in the same working folder. Add `used_cars_cleaned_08.csv` to rerun the historical tree cells.
2. Use a Jupyter environment with NumPy, pandas, scikit-learn, and joblib. The verified export used scikit-learn 1.5.2 and joblib 1.4.2.
3. Run the notebook from the cell that loads `uk_used_cars_autotrader.csv` through the feature experiments and model export to reproduce the selected model.
4. Run the test sample cells to reload the export, call `prepare_listings()`, add `log_mileage` and `make_model`, and compute metrics. To evaluate all eligible holdout listings, use the full prepared test set in place of the sample.

## Later notebook experiments and observations from this chat

Updated 18 November 2026. Cell numbers are one-based positions in the supplied 27-cell notebook. The depth-33 tuning, PCA experiment and latest CountVectorizer blend were run separately during this chat. Their results extend the experiments saved in the notebook.

### Standalone text trees and residual correction

Cell 20 compares three trees on 294,618 training and 73,659 grouped-validation listings prepared from the raw CSV.

| Model | Training MAE (£) | Validation MAE (£) | Validation RMSE (£) | Validation MAPE (%) | Validation R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|---:|
| One-hot reference, depth 28 | 1,476.262 | 2,415.275 | 40,692.904 | 15.038 | 0.430 | 59.507 |
| CountVectorizer tree, GBP target | 1,592.804 | 2,514.203 | 40,997.009 | 16.272 | 0.422 | 55.768 |
| CountVectorizer tree, log-price target | 1,055.293 | 2,356.649 | 41,805.202 | 14.083 | 0.399 | 61.471 |

The log-price text tree improved MAE and percentage proximity. The one-hot reference had lower RMSE and higher R². Fit times were 125.699, 124.159 and 247.901 seconds respectively.

Cell 21 retrains the coarse-to-fine CountVectorizer tree on an 80/20 row split of used_cars_cleaned_08.csv. Its target is price in GBP. It uses depth 25, minimum leaf size 2, unigram/bigram counts, min_df=3, a 15,000-feature vocabulary limit and a token pattern that retains dots and hyphens within tokens.

| Split | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|
| Training | 1,082.410 | 6,130.494 | 8.032 | 0.966 | 76.808 |
| Validation | 2,092.772 | 16,771.939 | 12.944 | 0.684 | 61.537 |

The training–validation gap indicates overfitting. These scores use different rows and preparation from cell 20 and the later train_clean.csv experiments.

Cell 22 fits Gradient Boosting to residuals from three grouped out-of-fold tree predictions. It uses 40 estimators, depth 3, minimum leaf size 30 and learning rate 0.05.

| Model | Validation MAE (£) | RMSE (£) | MAPE (%) | Within 10% (%) |
|---|---:|---:|---:|---:|
| Decision Tree | 2,092.772 | 16,771.939 | 12.944 | 61.537 |
| Tree + residual correction | 2,118.720 | 16,958.532 | 13.204 | 61.512 |

This correction worsened every reported metric. Cell 23 generated residuals using five grouped folds.

| Fold | Listings scored | MAE (£) | RMSE (£) | MAPE (%) |
|---|---:|---:|---:|---:|
| 1 | 57,855 | 2,177.096 | 17,152.943 | 13.503 |
| 2 | 57,855 | 2,181.289 | 21,764.447 | 13.630 |
| 3 | 57,854 | 2,201.279 | 20,122.388 | 13.307 |
| 4 | 57,854 | 2,069.871 | 11,956.977 | 13.557 |
| 5 | 57,854 | 2,059.841 | 10,516.040 | 13.533 |

Pooled out-of-fold MAE was £2,137.88 across 289,272 training listings. Grouping within these folds does not resolve any earlier cleaning leakage or duplicate overlap in the outer row split.

### Tree blends with log-price Ridge and Lasso

Cells 24–26 use train_clean.csv with 352,458 rows and test_clean.csv with 88,120 rows. They exclude price_gbp, log_price, price_eur, is_high_value, price_evaluation and emission_class from predictors, and add log_mileage and make_model.

Numeric inputs receive median imputation. Categorical inputs receive constant imputation and one-hot encoding. The tree groups categories occurring fewer than 20 times. The linear models use sparse scaling. Both base models learn log-price and convert predictions back to GBP before blending.

The weights are 75% linear model and 25% tree. The tree uses depth 30 and minimum leaf size 5. Lasso uses the previously tuned alpha=0.00003 and a sparse solver with a duality-gap convergence check. Gradient Boosting learns actual price minus the out-of-fold blend prediction in GBP, using five grouped folds. Its settings are Huber loss, 40 estimators, learning rate 0.05, depth 3 and minimum leaf size 30. Final predictions add the predicted residual to the blend.

These test scores come from saved outputs in cells 24 and 25.

| Model | Test MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|
| Log-price Ridge | 1,564.051 | 10,841.214 | 10.035 | 0.887 | 72.988 |
| Log-price Lasso | 1,513.960 | 10,185.208 | 9.753 | 0.900 | 73.056 |
| Depth-30 Decision Tree | 2,054.673 | 18,718.546 | 12.628 | 0.662 | 63.323 |
| Tree + Ridge | 1,453.418 | 10,828.786 | 9.481 | 0.887 | 74.956 |
| Tree + Ridge + Gradient Boosting | 1,433.475 | 10,825.170 | 9.365 | 0.887 | 75.398 |
| Tree + Lasso | 1,438.263 | 10,908.310 | 9.305 | 0.885 | 74.899 |
| Tree + Lasso + Gradient Boosting | 1,415.896 | 10,903.117 | 9.193 | 0.885 | 75.508 |

The corrected Lasso blend had the lowest MAE and highest Within 10% among these notebook experiments. Standalone Lasso had the lowest RMSE and highest R² in this table.

### Further coarse-to-fine tuning of the standalone tree

This chat tested the depth-30 tree using a grouped tuning split of 281,971 fitting and 70,487 validation rows from train_clean.csv. Selection prioritised Within 10%, with MAE as tie-breaker. The test file was evaluated after selection.

| Stage | Max depth | Minimum leaf size | Tuning MAE (£) | Tuning Within 10% (%) |
|---|---:|---:|---:|---:|
| Old tuned reference | 30 | 5 | 2,327.703 | 62.051 |
| Coarse | 24 | 3 | 2,345.529 | 61.496 |
| Coarse | 32 | 2 | 2,470.016 | 61.613 |
| Coarse | 38 | 3 | 2,328.090 | 61.738 |
| Coarse | 38 | 7 | 2,337.399 | 61.708 |
| Fine | 27 | 5 | 2,324.290 | 62.004 |
| **Fine, selected** | **33** | **5** | **2,322.383** | **62.091** |
| Fine | 30 | 4 | 2,307.775 | 61.979 |
| Fine | 30 | 6 | 2,323.012 | 61.949 |

Depth 30 with leaf size 4 had the lowest tuning MAE. Depth 33 with leaf size 5 won the requested percentage-proximity criterion. Both old and selected trees were refitted on all training rows.

| Split | Tree | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Training | Old depth 30 | 1,356.155 | 18,872.489 | 7.620 | 0.6689 | 77.328 |
| Training | New depth 33 | 1,342.785 | 18,870.890 | 7.548 | 0.6690 | 77.660 |
| Test | Old depth 30 | 2,054.673 | 18,718.546 | 12.628 | 0.6623 | 63.323 |
| Test | New depth 33 | 2,048.124 | 18,699.146 | 12.611 | 0.6630 | 63.350 |

The new tree lowered test MAE by £6.55 and gained 24 predictions within 10%. The small gain and remaining training–test gap limit the improvement claim.

### Updated Lasso blend with the depth-33 tree

The updated experiment retained Lasso alpha=0.00003 and the 75% Lasso / 25% tree weights. It regenerated the five-fold residual targets and refitted the Huber correction.

| Split | Blend with Gradient Boosting | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Training, fitted | Old depth-30 tree | 1,071.616 | 6,997.250 | 6.708 | 0.95449 | 82.006 |
| Training, fitted | New depth-33 tree | 1,068.788 | 6,996.786 | 6.698 | 0.95450 | 82.083 |
| Test | Old depth-30 tree | 1,415.896 | 10,903.117 | 9.193 | 0.88542 | 75.508 |
| Test | New depth-33 tree | 1,414.637 | 10,903.144 | 9.193 | 0.88542 | 75.555 |

The updated blend reduced test MAE by £1.26 and put 41 additional listings within 10%. RMSE increased by about £0.03. Before correction, the depth-33 blend scored £1,437.035 MAE and 74.925% within 10%.

### PCA experiment

The earlier depth-30 Lasso/tree/Gradient Boosting blend was rebuilt with centered PCA on standardized numeric inputs. Categorical inputs retained one-hot encoding. PCA was fitted within each training fold. A 95% variance threshold retained 21 of the 28 numeric components in the final fit, explaining 96.103% of numeric variance.

| Split | PCA blend with Gradient Boosting | MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---|---:|---:|---:|---:|---:|
| Training, fitted | Numeric PCA | 1,227.770 | 7,176.770 | 7.607 | 0.9521 | 77.770 |
| Test | Numeric PCA | 1,685.981 | 11,736.691 | 10.727 | 0.8672 | 69.453 |

Relative to the depth-30 reference, test MAE increased by £270.09 and Within 10% fell by 6.055 percentage points. The reference placed 5,336 more test listings within 10%. PCA was omitted from the selected approach.

### CountVectorizer in the latest blend

The latest chat experiment replaced one-hot version encoding with CountVectorizer in the depth-33 tree, Lasso and correction preprocessing. Other categorical inputs retained one-hot encoding. It used single words and word pairs, min_df=5, the default token pattern and a 15,000-feature vocabulary limit. Vocabularies were fitted within each training fold.

This experiment reused the previously tuned tree and Lasso settings and fixed blend weights. Those settings were not retuned for the new text representation.

| CountVectorizer model | Test MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|
| Log-price Lasso | 1,803.754 | 13,677.043 | 10.705 | 0.8197 | 66.482 |
| Depth-33 Decision Tree | 2,022.272 | 18,977.815 | 12.466 | 0.6529 | 64.253 |
| Tree + Lasso | 1,609.537 | 11,750.267 | 9.900 | 0.8669 | 70.630 |
| Tree + Lasso + Gradient Boosting | 1,569.638 | 11,737.168 | 9.794 | 0.8672 | 71.922 |

The final CountVectorizer ensemble scored £1,316.556 fitted-training MAE and 75.801% within 10%. Relative to the depth-33 one-hot ensemble, test MAE increased by £155.00 and Within 10% fell by 3.633 percentage points. Text counts improved standalone tree MAE and percentage proximity, while Lasso worsened enough to lower blend performance. The one-hot ensemble remains selected on these observations.

### Related Random Forest comparison from this chat

A separate experiment combined the newly tuned 36-tree Random Forest, depth 32, leaf size 2, max_features=0.7 and 50,000 bootstrap draws per tree with tuned Lasso and a five-fold Huber correction. Its weights were 85% Lasso and 15% forest.

| Corrected ensemble | Test MAE (£) | RMSE (£) | MAPE (%) | R² | Within 10% (%) |
|---|---:|---:|---:|---:|---:|
| Depth-33 Decision Tree + Lasso | 1,414.637 | 10,903.144 | 9.193 | 0.8854 | 75.555 |
| Newly tuned Random Forest + Lasso | 1,427.423 | 10,576.601 | 9.236 | 0.8922 | 75.394 |

The tree blend leads on MAE and Within 10%. The forest blend has lower RMSE and higher R². These separately weighted ensembles were evaluated on the same test rows.

## Metric interpretation and validation limits

Within 10% counts predictions satisfying abs(actual minus predicted) <= 0.10 times actual. MAPE averages absolute percentage errors. MAE reports average absolute error in pounds. RMSE gives larger errors more weight. Final fitted-training scores measure in-sample fit. Out-of-fold predictions come from models that excluded the scored training rows.

Grouping identical prepared predictors keeps exact matches together within tuning and out-of-fold splits. The later audit found 156 test listings whose predictors matched a training row. Matching inputs do not verify vehicle identity or detect every relisting.

| Model | Test rows without an exact training-input match | MAE (£) | Within 10% (%) |
|---|---:|---:|---:|
| Depth-30 standalone tree | 87,964 | 2,049.141 | 63.280 |
| Depth-33 standalone tree | 87,964 | 2,042.724 | 63.305 |
| Depth-33 Lasso/tree/Gradient Boosting | 87,964 | 1,406.622 | 75.531 |
| Latest CountVectorizer ensemble | 87,964 | 1,561.415 | 71.895 |

The later pipelines fit imputation, scaling and vocabularies inside training folds. The complete provenance of cleaning in train_clean.csv and test_clean.csv was not re-audited in these runs. Earlier cleaning may already have used information across a later split.

The same test file has been inspected across many experiments, so these results now inform model selection. A fresh untouched holdout is needed for an independent final generalisation estimate. Results using different datasets, eligibility rules or splits should be assessed within their respective experiment groups.

## Current selection, export and reproduction

The depth-33 one-hot Decision Tree + log-price Lasso + Gradient Boosting is the best observed Decision Tree ensemble in the recent 88,120-row evaluation by MAE and Within 10%. Its improvement over the depth-30 blend is small.

Cell 26 saves decisiontree+lasso+gradientboosting.joblib using the depth-30 tree from cell 25. The bundle contains the Lasso preprocessing/weights/intercept tuple, fitted tree, booster, blend weights, expected feature columns and excluded-column list. The depth-33 chat results do not update this export automatically. The earlier used_car_log_price_ridge.joblib belongs to the Ridge-only workflow.

For cells 24–26, install SciPy alongside the dependencies listed earlier and supply train_clean.csv and test_clean.csv. Update PROJECT_DIR to the data folder. Several cells still point to the former Sample Project directory, while the supplied notebook is now in the ML_Project repository.

Run cell 24 for the Ridge blend or cell 25 for the Lasso blend, then cell 26 to export the Lasso blend. To reproduce the depth-33 experiment, change the base tree depth and regenerate all five-fold residual targets before fitting the correction. The PCA and latest CountVectorizer blend experiments were run outside the supplied notebook.

The historical sections retain earlier results. Their 1,000-row raw holdout sample and the later 88,120-row cleaned test evaluation use different preparation.

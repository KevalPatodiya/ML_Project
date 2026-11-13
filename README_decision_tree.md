# Used Cars Price Estimation and Deal Finder

This project estimates the advertised asking price of UK used cars in pounds. [decision_tree.ipynb](decision_tree.ipynb) starts with a Decision Tree, compares tuning and text encoding, rebuilds validation to address data leakage, and selects a log-price Ridge model. The price estimate can help screen listings for potential deals. It does not establish sale price, vehicle condition, or whether a listing is a genuine bargain.

## Decision Tree experiments

The original notebook used `used_cars_cleaned_07.csv` and an 80/20 random training–validation split. Numeric inputs were imputed, categorical inputs were one-hot encoded, and the tree was evaluated on exact asking prices and on three price bands: under £10,000, £10,000 to under £25,000, and £25,000 or more.

| Experiment | Depth | Minimum leaf size | CV MAE | Validation MAE | Within 10% | Band accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Basic tree | 15 | 10 | £2,992.36 | £3,000.68 | 47.608% | 88.826% |
| Initial tuning | 20 | 5 | £2,559.38 | £2,485.55 | 56.349% | 90.612% |
| Coarse-to-fine, one-hot version | 28 | 3 | £2,365.59 | £2,267.53 | 61.681% | 91.797% |
| Same tuned tree, CountVectorizer version | 28 | 3 | — | £2,278.31 | 62.114% | 91.899% |

- **Initial search.** A six-combination grid compared depths 10, 15 and 20 with leaf sizes 5 and 10 using three-fold CV. A separate sequential search proposed five hyperparameters and required at least a £5 CV MAE improvement to continue. Increasing `min_samples_split` to 30 worsened CV MAE by £68.91, so the search stopped. The proposed `max_leaf_nodes` and `max_features` changes were skipped.
- **Coarse-to-fine search.** Depths 26 and 28 lowered CV MAE to £2,432.54 and £2,426.29. Reducing the leaf size to 3 brought it to £2,365.59.
- **Version encoding.** CountVectorizer used unigrams and bigrams from the trim description. It gained 0.433 percentage points within 10% but increased MAE by £10.78. One-hot encoding remained the tree baseline.
- **Fit diagnosis.** The tuned one-hot tree had £1,218.20 training MAE and £2,267.53 validation MAE. Its within-10% rate fell from 76.893% to 61.681%. The gap indicates overfitting. Validation RMSE reached £39,046.99, and the validation set included a £9,999,999 listing above the tree's £5 million training maximum.

## Validation rebuild and leakage audit

The original cleaned CSV already contained group medians and modes computed before the notebook made its training–validation split. That allowed validation and CV rows to influence imputation. The audit also found 268 original validation rows whose saved values, including price, matched a training row. Matching values alone cannot identify the same vehicle. Numeric Spearman correlations with price were strongest for registration year (+0.753), mileage (−0.677), and horsepower (+0.623).

The revised notebook starts from `uk_used_cars_autotrader.csv`, recreates the original 20% holdout with seed 42, then splits development data by groups of identical prepared predictors. It applies fixed eligibility rules for used listings with a known registration year and fewer than eight previous owners when recorded. It normalises missing categories, masks invalid door, seat and horsepower values, converts CO₂ to grams per mile, and masks the two flagged emissions values. The pipeline learns numeric medians, door and seat modes, and categorical vocabularies from training rows. The predictor allowlist excludes both the asking price and its euro equivalent.

The grouped split prevents exact prepared predictor matches across the development training and validation sets. The original holdout was split by row, so exact overlaps with development and relisted vehicles remain possible. The revised results use different validation rows and preprocessing from the original tree experiments.

| Model on grouped validation | MAE | RMSE | R² | Within 10% |
|---|---:|---:|---:|---:|
| Tuned Decision Tree | £2,362.49 | £20,600.67 | 0.651 | 59.182% |
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

Battery capacity barely changed MAE for either electric or plug-in hybrid listings. In the ablation tests, removing `version` increased MAE by £931.02, and removing `seller_city` increased it by £20.98. Removing annual tax improved MAE by £0.23 while worsening RMSE and the within-10% rate. Removing emission class improved MAE by £3.40 while increasing RMSE by £43.30. The saved model retains annual tax, seller location, and one-hot version encoding.

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

The notebook displays the first 15 actual and predicted prices with their signed residuals. Residual equals actual minus predicted, so the negative mean indicates overprediction on this sample. The percentage averages each residual divided by its positive asking price. Band accuracy measures agreement with broad price ranges. The within-10% rate measures proximity to the actual price. These are sample scores. A full holdout score is still pending, and RMSE is particularly sensitive to the mix of very expensive listings.

## Reproduce the notebook

1. Put `decision_tree.ipynb`, `uk_used_cars_autotrader.csv`, and `test.csv` in the same working folder. Add `used_cars_cleaned_07.csv` to rerun the historical tree cells.
2. Use a Jupyter environment with NumPy, pandas, scikit-learn, and joblib. The verified export used scikit-learn 1.5.2 and joblib 1.4.2.
3. Run the notebook from the cell that loads `uk_used_cars_autotrader.csv` through the feature experiments and model export to reproduce the selected model.
4. Run the test sample cells to reload the export, call `prepare_listings()`, add `log_mileage` and `make_model`, and compute metrics. To evaluate all eligible holdout listings, use the full prepared test set in place of the sample.

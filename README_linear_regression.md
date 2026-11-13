# UK Used Cars: Linear Regression Experiments

**Notebook:** `linear_regression.ipynb`
**Input:** `used_cars_cleaned_07.csv` (produced by `data_cleaning.ipynb`, see `README_data_cleaning.md`)
**Target:** `price_gbp` (advertised used-car price in pounds)
**Task:** Supervised regression, comparing several linear-model families and feature strategies

---

## Table of contents

1. [Purpose](#1-purpose)
2. [Results at a glance](#2-results-at-a-glance)
3. [Requirements](#3-requirements)
4. [How to run](#4-how-to-run)
5. [Data and shared setup](#5-data-and-shared-setup)
6. [Experiment-by-experiment walkthrough](#6-experiment-by-experiment-walkthrough)
7. [Model comparison](#7-model-comparison)
8. [Key findings](#8-key-findings)
9. [Metrics explained](#9-metrics-explained)
10. [Cell dependencies](#10-cell-dependencies)
11. [Limitations and caveats](#11-limitations-and-caveats)
12. [Suggested next steps](#12-suggested-next-steps)

---

## 1. Purpose

This notebook asks: **how well can linear models predict UK used-car prices, and which variations help?** It tests:

1. Plain ordinary least squares (OLS) linear regression
2. Ridge regression tuned with cross-validation
3. Polynomial (degree 2) features
4. A joint search over polynomial degree and regularisation strength
5. Stochastic gradient descent (SGD) regression
6. Ridge on a log-transformed target
7. Bag-of-words text features from the `version` column
8. Dimensionality reduction (TruncatedSVD) before linear regression
9. Error analysis of the worst predictions

Everything uses the same train/validation split, so the results are directly comparable.

---

## 2. Results at a glance

All numbers are on the **validation set** (about 20% of the cleaned data) in pounds.

| # | Model | Val MAE (£) | Val RMSE (£) | Val R² |
|---|---|---|---|---|
| 1 | Linear regression (OLS) | 2,382.69 | 30,518.42 | **0.591** |
| 2 | Ridge (alpha = 0.1) | 2,361.28 | 32,369.49 | 0.540 |
| 3 | Polynomial deg 2 + Ridge (alpha = 100, untuned) | 3,873.20 | 40,692.88 | 0.273 |
| 4 | Tuned poly/Ridge (best: degree 1, alpha = 0.1) | 2,361.07 | 32,369.42 | 0.540 |
| 5 | SGDRegressor (tuned) | 4,004.54 | 40,274.38 | 0.288 |
| 6 | Ridge on log1p(price) target (alpha = 0.1) | **1,722.45** | 37,244.65 | 0.391 |
| 7 | Ridge + CountVectorizer on `version` | 2,618.29 | 34,017.90 | 0.492 |
| 8 | TruncatedSVD (100 comps) + Linear regression | 6,905.43 | 43,595.43 | 0.166 |

**Best by MAE:** log-target Ridge (£1,722).
**Best by RMSE and R²:** plain OLS linear regression (£30,518; R² 0.591).
Which model is "best" depends on the metric; see [Key findings](#8-key-findings).

---

## 3. Requirements

- Python 3.10 (the notebook kernel reports 3.10.11)
- `numpy`, `pandas`
- `scikit-learn` (`LinearRegression`, `Ridge`, `SGDRegressor`, `GridSearchCV`, `KFold`, `ColumnTransformer`, `Pipeline`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`, `PolynomialFeatures`, `CountVectorizer`, `TruncatedSVD`, `TransformedTargetRegressor`)
- Jupyter (Notebook, Lab or VS Code)

```bash
pip install numpy pandas scikit-learn jupyter
```

---

## 4. How to run

1. Run `data_cleaning.ipynb` first (or otherwise obtain `used_cars_cleaned_07.csv`) and place the CSV beside this notebook.
2. Open `linear_regression.ipynb` and run the cells **in order, top to bottom**. Later cells reuse objects created earlier (see [Cell dependencies](#10-cell-dependencies)).
3. The grid searches use `n_jobs=2`, and the SVD search uses `n_jobs=1`, to keep memory use manageable. With about 290k training rows and one-hot encoded high-cardinality columns, the heavier cells (polynomial grid, SGD, SVD) can take a while.

---

## 5. Data and shared setup

### Features and target

```python
X = df.drop(columns=["price_gbp", "condition"], errors="ignore")
y = df["price_gbp"]
```

- `condition` is dropped because it has a single value (`Used`) after cleaning.
- 23 feature columns remain: numeric measurements, categorical descriptors, high-cardinality text-like fields (`model`, `version`, `seller_city`, `seller_county`), and the three `*_missing` indicators.

### Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Roughly 290,000 training rows and 72,500 validation rows (80/20 of 362,794). The **same split** is reused by every experiment so comparisons are fair.

### Column types

- **Numeric columns** = all numeric dtypes (this includes the three `*_missing` indicator columns).
- **Categorical columns** = `object`, `string`, `category` dtypes.

### Base preprocessing (`preprocessor`)

| Branch | Steps |
|---|---|
| Numeric | `SimpleImputer(median)` -> `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")` |

All learned preprocessing lives inside `Pipeline` objects, so it is fitted **only on training data** (and only on the training fold during cross-validation).

### Cross-validation

`KFold(n_splits=3, shuffle=True, random_state=42)` with `scoring="neg_mean_absolute_error"`. Models are tuned on the **training set only**; the validation set is then used to report performance.

---

## 6. Experiment-by-experiment walkthrough

### Experiment 1: OLS linear regression (baseline)

`Pipeline(preprocessor -> LinearRegression)`

| Split | MAE (£) | RMSE (£) | R² |
|---|---|---|---|
| Training | 1,623.39 | 8,036.43 | 0.942 |
| Validation | 2,382.69 | 30,518.42 | 0.591 |

- A large gap between training R² (0.942) and validation R² (0.591) signals **overfitting**, driven largely by the very wide one-hot encoding of high-cardinality columns.
- **1,915 validation predictions were negative**, which is impossible for a price. Linear models do not constrain the output range.
- A sample of ten predictions vs. actuals is shown with absolute errors.

### Experiment 2: Ridge regression (L2 regularisation)

`Ridge(solver="lsqr")`, grid `alpha in [0.1, 1, 10, 100, 1000]`, 3-fold CV on MAE (15 fits).

- Best: **alpha = 0.1**, CV MAE £2,370.80.
- Validation: MAE £2,361.28, RMSE £32,369.49, R² 0.5401.
- Slightly better MAE than OLS but **worse RMSE and R²**. The weakest regularisation won, meaning stronger shrinkage hurt.

### Experiment 3: Polynomial features (degree 2) + Ridge

- Polynomial expansion is applied only to the numeric measurement columns; the `*_missing` indicators are passed through, categoricals one-hot encoded.
- Steps for the numeric branch: median impute -> `PolynomialFeatures(degree=2, include_bias=False)` -> `StandardScaler`.
- Fixed `Ridge(alpha=100)`.

| Split | MAE (£) | RMSE (£) | R² |
|---|---|---|---|
| Training | 3,736.62 | 22,454.47 | 0.547 |
| Validation | 3,873.20 | 40,692.88 | 0.273 |

This untuned run underfits badly (training R² is only 0.547), mainly because alpha = 100 is heavy for this setup.

### Experiment 4: Joint grid over polynomial degree and alpha

Grid: `degree in [1, 2]` x `alpha in [0.1, 1, 10, 100, 1000]` (10 candidates, 30 fits).

Top CV results (MAE, £):

| Rank | Degree | Alpha | CV MAE |
|---|---|---|---|
| 1 | 1 | 0.1 | 2,372.05 |
| 2 | 2 | 1 | 2,431.80 |
| 3 | 1 | 1 | 2,556.61 |
| 4 | 2 | 0.1 | 3,034.02 |
| 5 | 2 | 10 | 3,126.22 |
| ... | ... | ... | ... |
| 10 | 1 | 1000 | 6,088.76 |

- **Degree 1 (no polynomial terms) won.** Quadratic terms did not help.
- Best model: training MAE £1,819.04 / RMSE £8,307.62 / R² 0.938; validation MAE £2,361.07 / RMSE £32,369.42 / R² 0.540. This is essentially identical to the tuned Ridge of Experiment 2.
- Two-way grid shows degree 2 at alpha = 1 is competitive, which suggests that with the right regularisation, polynomial terms are nearly neutral, never clearly better.

### Experiment 5: SGDRegressor (gradient-descent linear model)

Pipeline reuses the preprocessor from the best polynomial model (degree 1 was selected, so it is effectively linear). `SGDRegressor(loss="squared_error", penalty="l2", max_iter=3000, tol=1e-4, average=True)`.

Grid: `alpha in [1e-4, 1e-3, 1e-2]`, `learning_rate="adaptive"`, `eta0 in [1e-4, 1e-3]` (6 candidates, 18 fits).

- Best: alpha = 0.0001, eta0 = 0.001, adaptive.
- Validation: MAE £4,004.54, RMSE £40,274.38, R² 0.288.
- Much worse than closed-form Ridge. SGD likely has not fully converged on this large, sparse, wide design matrix at these hyperparameters; a finer search (more iterations, other learning rates) could narrow the gap.

### Experiment 6: Log-transformed target

`TransformedTargetRegressor(regressor=Pipeline(preprocessor -> Ridge), func=np.log1p, inverse_func=np.expm1)`, grid `alpha in [0.01, 0.1, 1, 10, 100]` (15 fits).

- Best: alpha = 0.1.
- Validation: **MAE £1,722.45** (best of all experiments), RMSE £37,244.65, R² 0.3911.
- Predictions are made in log space and automatically converted back to pounds, which also guarantees **non-negative predictions**.
- MAE improves by about £640 compared with Ridge, yet RMSE and R² worsen. This is the classic log-target trade-off: log training equalises *relative* errors, giving much better typical-car accuracy, but when converting back, errors on very expensive cars are magnified in pounds.

### Experiment 7: Text features from `version` with `CountVectorizer`

- `version` is removed from the one-hot columns and instead vectorised as bag-of-words: lowercase, `strip_accents="unicode"`, token pattern `(?u)\b\w[\w.-]*\b`, `min_df=3`, `max_features=15000`.
- Grid: `ngram_range in [(1,1), (1,2)]` x `alpha in [0.01, 0.1, 1, 10]` (8 candidates, 24 fits).
- Best: ngram_range = (1, 2), alpha = 1.
- Validation: MAE £2,618.29, RMSE £34,017.90, R² 0.492.
- This was **worse than treating `version` as a plain one-hot category** (Experiment 2: MAE £2,361). One-hot matches exact trim strings, which the scraped data repeats often; bag-of-words splits them into tokens that lose some exact-trim information.

### Experiment 8: Error analysis (worst 20 predictions)

Using the tuned Ridge, the 20 largest absolute errors on validation were listed. They are overwhelmingly **supercars and ultra-rare cars**:

| Car | Actual price | Approx. error |
|---|---|---|
| Ferrari F40 | £9,999,999 | £7.8M |
| Ferrari 275 GTB/4 | £3,349,990 | £3.0M |
| Rimac Nevera | £1,500,000 | £1.4M |
| Ferrari 599 GTO | £1,299,995 | £1.0M |
| Alfa Romeo Giulietta Sprint Zagato | £395,995 | £434k |
| Rolls-Royce Ghost, Lamborghini Aventador, Mercedes G63, SLS, Huracan, etc. | £130k to £900k | £150k to £320k |

Observations:
- Several of these have `power_hp = 0` (an unknown-value placeholder), so the model lacks a key signal.
- A handful of listings dominate squared-error metrics. This explains the gap between a modest MAE (about £2.4k) and a very large RMSE (about £30k+).
- Some errors are also data-driven (for example, the £9,999,999 F40 listing may be a "price on application" placeholder).

### Experiment 9: TruncatedSVD dimensionality reduction + OLS

`Pipeline(text_preprocessor -> TruncatedSVD -> LinearRegression)`, grid `n_components in [50, 100, 200]` (9 fits).

- Best: 100 components, retaining **80.1%** of feature variance.
- Validation: MAE £6,905.43, RMSE £43,595.43, R² 0.1657 (the worst model).
- Compressing the wide sparse design matrix into 100 dense components discards too much price-relevant information, especially fine-grained model and trim identity.

---

## 7. Model comparison

| Model | Strength | Weakness |
|---|---|---|
| OLS | Best RMSE and R²; simple | Overfits (train R² 0.94 vs val 0.59); negative price predictions |
| Ridge (alpha 0.1) | Slightly better MAE than OLS | Slightly worse RMSE/R²; negligible regularisation benefit |
| Polynomial + Ridge | Captures nonlinearity in principle | No gain once tuned; untuned version underfits badly |
| SGD | Scales to huge data | Poor fit here (needs more tuning/convergence) |
| Log-target Ridge | **Best MAE**; no negative predictions | Worse RMSE/R² on expensive cars |
| Ridge + `version` BoW | Handles unseen trim wording | Worse than exact one-hot trim |
| SVD + OLS | Compact features | Heavy information loss; worst results |

---

## 8. Key findings

1. **Regularisation barely helps.** The best alpha was always the smallest tested (0.1), and Ridge matched OLS on MAE while losing on RMSE. The overfitting appears to come from the structure of the data (high-cardinality identifiers), not from coefficient magnitude.
2. **Polynomial terms did not help** once tuned: degree 1 won the grid search.
3. **Metric choice changes the winner.** The log-target model is best for typical errors (MAE £1,722), whereas plain OLS is best for squared error (RMSE £30.5k, R² 0.591).
4. **Heavy-tailed prices drive RMSE and R².** A small number of supercars and classic cars with prices from £130k to nearly £10M account for most of the squared error.
5. **Linear models can predict negative prices** (1,915 cases for OLS). The log-target approach avoids this by construction.
6. **Compression and text-splitting both hurt.** SVD and bag-of-words on `version` were each worse than the simple one-hot baseline.
7. **Overfitting persists for the unconstrained model:** training R² 0.94 versus validation R² 0.59, which suggests richer nonlinear models or better handling of rare categories could do better.

---

## 9. Metrics explained

- **MAE (Mean Absolute Error):** average absolute pounds error. Robust to outliers; reads as "typical miss".
- **RMSE (Root Mean Squared Error):** square root of the average squared error. Penalises large misses heavily, so it is dominated by expensive-car errors.
- **R² (coefficient of determination):** share of price variance explained (1.0 is perfect, 0 is equivalent to predicting the mean, negative is worse than the mean).

All three are reported because they disagree here, and that disagreement is informative.

---

## 10. Cell dependencies

The notebook is sequential. Key hand-offs:

| Cell (experiment) | Needs from earlier cells |
|---|---|
| 2 (Ridge) | `preprocessor`, `X_train`, `X_val`, `y_train`, `y_val` from cell 1 |
| 3 (Polynomial) | `numeric_cols`, `categorical_cols`, `categorical_pipeline` from cell 1 |
| 4 (Poly grid) | `poly_model` (cell 3), `cv` (cell 2) |
| 5 (SGD) | `best_poly_model` (cell 4), `cv` |
| 6 (Log target) | `preprocessor`, `cv` |
| 7 (Text) | `X_train`, `X_val`, `cv`; **redefines `numeric_cols`** |
| 8 (Error analysis) | `best_ridge` (cell 2) |
| 9 (SVD) | `text_preprocessor`, `X_train_text`, `X_val_text` (cell 7) |

If a kernel is restarted, run from the top rather than jumping into the middle.

---

## 11. Limitations and caveats

1. **No separate test evaluation.** Hyperparameters are tuned with training-only cross-validation, but the validation set is then used to compare **all** models, so reported validation scores are slightly optimistic. The raw `test.csv` from the cleaning notebook has not been evaluated; it would first need the same cleaning steps applied.
2. **Outliers were deliberately kept.** Prices up to £9,999,999 remain in both train and validation sets, affecting fitted coefficients and RMSE/R².
3. **Possible minor leakage from cleaning.** Imputation medians in the cleaning notebook were computed before this split, so validation rows influenced some imputed values.
4. **`power_hp = 0` is a placeholder for unknown power** and is treated as a real number here.
5. **High-cardinality one-hot encoding** (`model`, `version`, `seller_city`, `seller_county`) creates a very wide sparse matrix, increasing overfitting risk and runtime.
6. **Tuning breadth is modest:** 3-fold CV, a coarse alpha grid, and small SGD/SVD grids. A finer search might change rankings slightly, especially for SGD.
7. **Linear models only.** Interactions between make, age, mileage, and power are inherently nonlinear in price; this notebook is a baseline, not a ceiling.
8. **Variable name reuse:** `numeric_cols` is redefined in the text-feature cell, and generic names such as `predictions` are overwritten between experiments. Re-running individual cells out of order can give confusing results.
9. **Prices are a single-time snapshot** (October 2026 listings) of advertised prices, not final sale prices.

---

## 12. Suggested next steps

- **Nonlinear models:** gradient boosting (such as `HistGradientBoostingRegressor`), random forests, or LightGBM/XGBoost. These handle interactions, rare categories, and heavy tails more naturally.
- **Target-aware handling of extremes:** cap or winsorise price, train separate models for mainstream and supercar segments, or use a robust loss (Huber or quantile loss).
- **Feature engineering:** car age (listing year minus `reg_year`), mileage per year, power-to-weight proxies, make/model frequency encoding or target encoding in place of raw one-hot.
- **Treat `power_hp == 0` as missing** and impute or add an indicator.
- **Report percentage errors** (MAPE or median absolute percentage error) so that results are comparable between cheap and expensive cars.
- **Clip log-model predictions** and check bias after `expm1` back-transformation.
- **Final evaluation on `test.csv`** after replaying the cleaning pipeline, and only for the one chosen model.
- **Increase CV folds** (5 or 10) and widen alpha ranges below 0.1 since the boundary value kept winning.

# UK Used Cars: Linear Regression Experiments

**Notebook:** `linear_regression.ipynb`
**Input:** `used_cars_cleaned_08.csv` (produced by `data_cleaning.ipynb`, see `README_data_cleaning.md`)
**Target:** `price_gbp` (advertised used-car price in pounds)
**Task:** Supervised regression, comparing several linear-model families and feature strategies
**Saved artefact:** `linear_reg.joblib` (the baseline OLS pipeline, see [Experiment 10](#experiment-10-saving-the-baseline-model))

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
8. Error analysis of the worst predictions
9. Dimensionality reduction (TruncatedSVD) before linear regression
10. Saving the baseline model with `joblib`

Everything uses the same train/validation split, so the results are directly comparable.

---

## 2. Results at a glance

All numbers are on the **validation set** (72,324 listings, about 20% of the cleaned data) in pounds.

| # | Model | Val MAE (£) | Val RMSE (£) | Val R² |
|---|---|---|---|---|
| 1 | Linear regression (OLS) | 2,180.20 | 13,021.90 | 0.791 |
| 2 | Ridge (alpha = 0.1) | 2,235.42 | 13,071.57 | 0.789 |
| 3 | Polynomial deg 2 + Ridge (alpha = 100, untuned) | 3,448.12 | 15,361.76 | 0.708 |
| 4 | Tuned poly/Ridge (best: degree 2, alpha = 1) | 2,375.48 | 12,461.34 | 0.808 |
| 5 | SGDRegressor (tuned, polynomial features) | 3,114.28 | 14,794.66 | 0.730 |
| 6 | Ridge on log1p(price) target (alpha = 0.1) | **1,610.46** | **12,257.21** | **0.814** |
| 7 | Ridge + CountVectorizer on `version` | 2,491.25 | 13,184.33 | 0.785 |
| 8 | TruncatedSVD (100 comps) + Linear regression | 6,707.07 | 20,602.26 | 0.476 |

**Best on all three metrics:** the log-target Ridge (MAE £1,610; RMSE £12,257; R² 0.814).
Among models trained on the raw price, the tuned degree-2 Ridge has the best RMSE and R², while plain OLS has the best MAE.

---

## 3. Requirements

- Python 3.10 (the notebook kernel reports 3.10.11)
- `numpy`, `pandas`
- `scikit-learn` (`LinearRegression`, `Ridge`, `SGDRegressor`, `GridSearchCV`, `KFold`, `GroupKFold`, `GroupShuffleSplit`, `ColumnTransformer`, `Pipeline`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`, `PolynomialFeatures`, `CountVectorizer`, `TruncatedSVD`, `TransformedTargetRegressor`)
- `joblib` (installed with scikit-learn; used to save the model)
- Jupyter (Notebook, Lab or VS Code)

```bash
pip install numpy pandas scikit-learn joblib jupyter
```

---

## 4. How to run

1. Run `data_cleaning.ipynb` first (or otherwise obtain `used_cars_cleaned_08.csv`) and place the CSV beside this notebook.
2. Open `linear_regression.ipynb` and run the cells **in order, top to bottom**. Later cells reuse objects created earlier (see [Cell dependencies](#10-cell-dependencies)).
3. The grid searches use `n_jobs=2`, and the SVD search uses `n_jobs=1`, to keep memory use manageable. With about 289k training rows and one-hot encoded high-cardinality columns, the heavier cells (polynomial grid, SGD, SVD) can take a while.
4. The last code cell writes `linear_reg.joblib` into the current working directory.

---

## 5. Data and shared setup

### Features and target

The notebook now selects features **explicitly** instead of dropping columns:

```python
feature_cols = [
    "make", "model", "version", "mileage_miles", "reg_year",
    "fuel", "transmission", "power_hp", "body_type", "body_color",
    "doors", "seats", "previous_owners", "drivetrain", "emission_class",
    "annual_tax_gbp", "seller_type", "seller_city", "seller_county",
    "co2_gmile",
]
X = df.loc[:, feature_cols].copy()
y = df["price_gbp"].copy()
```

- 20 feature columns: numeric measurements, categorical descriptors, and high-cardinality text-like fields (`model`, `version`, `seller_city`, `seller_county`).
- Only information available **before** the price is estimated is used.
- There are no `*_missing` indicator columns in this feature set, so the "indicator" branch of the polynomial preprocessor is empty.

### Split (grouped, to prevent duplicate leakage)

```python
groups = pd.util.hash_pandas_object(X, index=False)
GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

Rows with **identical predictor values** are hashed into the same group and kept on the same side of the split, even if their prices differ. An `assert` confirms that no validation group appears in training.

| Set | Listings |
|---|---|
| Training | 289,266 |
| Validation | 72,324 |

The **same split** is reused by every experiment so comparisons are fair.

### Column types

- **Numeric columns** = all numeric dtypes in `X_train`.
- **Categorical columns** = `object`, `string`, `category` dtypes.

### Base preprocessing (`preprocessor`)

| Branch | Steps |
|---|---|
| Numeric | `SimpleImputer(median)` -> `StandardScaler` |
| Categorical | `SimpleImputer(most_frequent)` -> `OneHotEncoder(handle_unknown="ignore")` |

All learned preprocessing lives inside `Pipeline` objects, so it is fitted **only on training data** (and only on the training fold during cross-validation).

### Cross-validation

Cell 1 precomputes grouped folds (`GroupKFold(n_splits=3)`) and assigns them to `cv`. However, **cell 2 immediately overwrites `cv`** with `KFold(n_splits=3, shuffle=True, random_state=42)`, and every later search uses that `KFold` (cell 4 redefines it identically). So the grouped folds are never actually used for tuning; see [Limitations](#11-limitations-and-caveats).

All searches score with `neg_mean_absolute_error`. Models are tuned on the **training set only**; the validation set is then used to report performance.

---

## 6. Experiment-by-experiment walkthrough

### Experiment 1: OLS linear regression (baseline)

`Pipeline(preprocessor -> LinearRegression)`

| Split | MAE (£) | RMSE (£) | R² |
|---|---|---|---|
| Training | 1,605.73 | 7,824.66 | 0.945 |
| Validation | 2,180.20 | 13,021.90 | 0.791 |

- A gap between training R² (0.945) and validation R² (0.791) signals **overfitting**, driven largely by the very wide one-hot encoding of high-cardinality columns.
- **1,728 validation predictions were negative**, which is impossible for a price. Linear models do not constrain the output range (for example, a £1,499 car was predicted at about -£3,320).
- A sample of ten predictions vs. actuals is shown with absolute errors.

### Experiment 2: Ridge regression (L2 regularisation)

`Ridge(solver="lsqr")`, grid `alpha in [0.1, 1, 10, 100, 1000]`, 3-fold CV on MAE (15 fits).

- Best: **alpha = 0.1**, CV MAE £2,312.37.
- Validation: MAE £2,235.42, RMSE £13,071.57, R² 0.7889.
- This is **slightly worse than plain OLS** on all three validation metrics. The weakest regularisation won, meaning stronger shrinkage hurt.

### Experiment 3: Polynomial features (degree 2) + Ridge

- Polynomial expansion is applied to the numeric columns; categoricals are one-hot encoded. (The `*_missing` passthrough branch exists in the code but is empty for this feature set.)
- Steps for the numeric branch: median impute -> `PolynomialFeatures(degree=2, include_bias=False)` -> `StandardScaler`.
- Fixed `Ridge(alpha=100)`.

| Split | MAE (£) | RMSE (£) | R² |
|---|---|---|---|
| Training | 3,455.49 | 22,311.72 | 0.554 |
| Validation | 3,448.12 | 15,361.76 | 0.708 |

This untuned run underfits (training R² is only 0.554), mainly because alpha = 100 is heavy for this setup.

### Experiment 4: Joint grid over polynomial degree and alpha

Grid: `degree in [1, 2]` x `alpha in [0.1, 1, 10, 100, 1000]` (10 candidates, 30 fits).

Full CV results (MAE, £):

| Rank | Degree | Alpha | CV MAE |
|---|---|---|---|
| 1 | 2 | 1 | 2,253.29 |
| 2 | 1 | 0.1 | 2,310.70 |
| 3 | 1 | 1 | 2,462.60 |
| 4 | 2 | 10 | 2,857.17 |
| 5 | 2 | 0.1 | 3,077.16 |
| 6 | 1 | 10 | 3,288.40 |
| 7 | 2 | 100 | 3,687.59 |
| 8 | 2 | 1000 | 4,489.84 |
| 9 | 1 | 100 | 4,609.63 |
| 10 | 1 | 1000 | 5,984.97 |

- **Degree 2 with alpha = 1 won** this time, but only narrowly over degree 1 (CV MAE £2,253 vs £2,311).
- Best model: training MAE £2,131.25 / RMSE £9,827.77 / R² 0.914; validation MAE £2,375.48 / RMSE £12,461.34 / R² 0.808.
- Mixed verdict on validation: this model has better RMSE and R² than the plain Ridge (Experiment 2), but a **worse MAE** (£2,375 vs £2,235). Quadratic terms help on expensive cars but not on typical ones.

### Experiment 5: SGDRegressor (gradient-descent linear model)

The pipeline reuses the preprocessor from the best polynomial model. Because degree 2 was selected, this model **does use polynomial features**. `SGDRegressor(loss="squared_error", penalty="l2", max_iter=3000, tol=1e-4, average=True)`.

Grid: `alpha in [1e-4, 1e-3, 1e-2]`, `learning_rate="adaptive"`, `eta0 in [1e-4, 1e-3]` (6 candidates, 18 fits).

- Best: alpha = 0.0001, eta0 = 0.001, adaptive.
- Validation: MAE £3,114.28, RMSE £14,794.66, R² 0.7296.
- Clearly worse than closed-form Ridge. SGD likely has not fully converged on this large, wide design matrix at these hyperparameters; a finer search (more iterations, other learning rates) could narrow the gap. The best alpha and eta0 sit at the grid edge for alpha (smallest value), so smaller values are worth trying.

### Experiment 6: Log-transformed target

`TransformedTargetRegressor(regressor=Pipeline(preprocessor -> Ridge), func=np.log1p, inverse_func=np.expm1)`, grid `alpha in [0.01, 0.1, 1, 10, 100]` (15 fits).

- Best: alpha = 0.1.
- Validation: **MAE £1,610.46, RMSE £12,257.21, R² 0.8144**, the best of all experiments on every metric.
- Predictions are made in log space and automatically converted back to pounds, which also guarantees **non-negative predictions**.
- MAE improves by about £625 compared with plain Ridge, and RMSE and R² improve as well. Training on the log scale equalises *relative* errors, which suits prices that span several orders of magnitude.

### Experiment 7: Text features from `version` with `CountVectorizer`

- `version` is filled with an empty string when missing, removed from the one-hot columns, and instead vectorised as bag-of-words: lowercase, `strip_accents="unicode"`, token pattern `(?u)\b\w[\w.-]*\b`, `min_df=3`, `max_features=15000`.
- This cell **redefines `numeric_cols`** and builds `X_train_text` / `X_val_text` (copies of the split).
- Grid: `ngram_range in [(1,1), (1,2)]` x `alpha in [0.01, 0.1, 1, 10]` (8 candidates, 24 fits).
- Best: ngram_range = (1, 2), alpha = 1.
- Validation: MAE £2,491.25, RMSE £13,184.33, R² 0.7853.
- This was **worse than treating `version` as a plain one-hot category** (Experiment 2: MAE £2,235, RMSE £13,072). One-hot matches exact trim strings, which the scraped data repeats often; bag-of-words splits them into tokens that lose some exact-trim information.

### Experiment 8: Error analysis (worst 20 predictions)

Using the tuned Ridge (`best_ridge`, Experiment 2), the 20 largest absolute errors on validation were listed. They are overwhelmingly **supercars, ultra-rare cars and large luxury V12s**:

| Car | Actual price | Approx. error |
|---|---|---|
| Aston Martin Valour (2024) | £1,785,000 | £1.61M |
| Porsche 911 993 GT2 | £1,600,000 | £1.49M |
| Aston Martin Valour (2025) | £1,499,995 | £1.34M |
| Rimac Nevera | £1,500,000 | £1.31M |
| Aston Martin Valhalla | £1,099,995 | £889k |
| Rolls-Royce Phantom Drophead (2017) | £849,000 | £551k |
| Ferrari 599 GTB Fiorano (private seller) | £159,000 | £359k |
| Ferrari 488 Pista Spider | £999,995 | £338k |
| Rolls-Royce Ghost, Wraith Black Badge, Mercedes SLS, G63 (x2) | £396k to £500k | £250k to £319k |
| Lamborghini Aventador SVJ, Huracan STO, Murcielago | £600k to £1M | £223k to £243k |
| Ferrari 599 (40,400 miles), Rolls-Royce Phantom (x2), McLaren GTS | £60k to £164k | £201k to £223k |

Observations:
- The errors are now **two-sided**. Brand-new or ultra-rare models (Valour, Valhalla, Nevera, 993 GT2) are heavily **under-predicted**, while several older, much cheaper cars (the £79,900 Ferrari 599, the £59,950 and £67,500 Rolls-Royce Phantoms, the £159,000 599) are **over-predicted** by about £200k to £360k, because the linear model extrapolates the high make/power effect.
- Two rows have `version = "Unknown"` (Aston Martin Valhalla, McLaren GTS), so the model lacks trim information for them.
- The worst single error is about £1.6M, much smaller than in earlier runs on the `_07` data, where listings priced near £10M dominated; those listings no longer appear in the top 20.
- A handful of listings still dominate squared-error metrics, which explains the gap between a modest MAE (about £2.2k) and a much larger RMSE (about £13k).

### Experiment 9: TruncatedSVD dimensionality reduction + OLS

`Pipeline(text_preprocessor -> TruncatedSVD -> LinearRegression)`, grid `n_components in [50, 100, 200]` (9 fits).

- Best: 100 components, retaining **80.5%** of feature variance.
- Validation: MAE £6,707.07, RMSE £20,602.26, R² 0.4757 (the worst model).
- Compressing the wide sparse design matrix into 100 dense components discards too much price-relevant information, especially fine-grained model and trim identity. The best value is in the middle of the grid, so the search was not boundary-limited.

### Experiment 10: Saving the baseline model

```python
check_is_fitted(linear_model)
joblib.dump(linear_model, Path("linear_reg.joblib"), compress=3)
```

- Saves the **fitted OLS pipeline from Experiment 1** (`linear_model`), preprocessing included, so it can be reloaded and used with `joblib.load(...).predict(new_df)` on a DataFrame with the same 20 feature columns.
- Note that this is the **baseline**, not the best-performing model. The log-target Ridge (`log_search.best_estimator_`) would be the better one to save, since it is the best on every metric and cannot output negative prices.

---

## 7. Model comparison

| Model | Strength | Weakness |
|---|---|---|
| OLS | Best MAE among raw-price models; simple | Overfits (train R² 0.945 vs val 0.791); 1,728 negative predictions |
| Ridge (alpha 0.1) | Tuned and stable | Slightly worse than OLS on validation; negligible regularisation benefit |
| Polynomial + Ridge (tuned) | Best RMSE and R² among raw-price models | Worse MAE than plain Ridge; untuned version underfits |
| SGD | Scales to huge data | Poor fit here (needs more tuning/convergence) |
| Log-target Ridge | **Best on MAE, RMSE and R²**; no negative predictions | Back-transformation can bias predictions; still large misses on supercars |
| Ridge + `version` BoW | Handles unseen trim wording | Worse than exact one-hot trim |
| SVD + OLS | Compact features | Heavy information loss; worst results |

---

## 8. Key findings

1. **The log-target Ridge is the clear winner**, best on MAE (£1,610), RMSE (£12,257) and R² (0.814). This is a change from earlier runs on the `_07` data, where the log target won on MAE but lost on RMSE and R².
2. **Regularisation barely helps.** The best alpha was the smallest tested (0.1) for the Ridge and log-Ridge searches, and plain OLS beat tuned Ridge on validation. The overfitting appears to come from the structure of the data (high-cardinality identifiers), not from coefficient magnitude.
3. **Polynomial terms help only partially.** Degree 2 won the CV search and improved RMSE and R², but its validation MAE was worse than plain Ridge, so the gain is concentrated on expensive cars.
4. **Heavy-tailed prices still drive RMSE and R².** A small number of supercars and ultra-rare cars account for most of the squared error, though the worst misses are smaller than before (about £1.6M vs about £7.8M).
5. **Linear models can predict negative prices** (1,728 cases for OLS on validation). The log-target approach avoids this by construction.
6. **Compression and text-splitting both hurt.** SVD and bag-of-words on `version` were each worse than the simple one-hot baseline.
7. **Overfitting persists for the unconstrained model:** training R² 0.945 versus validation R² 0.791, which suggests richer nonlinear models or better handling of rare categories could do better.
8. **The grouped split matters.** Keeping identical predictor rows together removes duplicate leakage, so the validation scores here are more trustworthy than a purely random split.

---

## 9. Metrics explained

- **MAE (Mean Absolute Error):** average absolute pounds error. Robust to outliers; reads as "typical miss".
- **RMSE (Root Mean Squared Error):** square root of the average squared error. Penalises large misses heavily, so it is dominated by expensive-car errors.
- **R² (coefficient of determination):** share of price variance explained (1.0 is perfect, 0 is equivalent to predicting the mean, negative is worse than the mean).

All three are reported because they can disagree, and that disagreement is informative.

---

## 10. Cell dependencies

The notebook is sequential. Key hand-offs:

| Cell (experiment) | Needs from earlier cells |
|---|---|
| 2 (Ridge) | `preprocessor`, `X_train`, `X_val`, `y_train`, `y_val` from cell 1; overwrites `cv` |
| 3 (Polynomial) | `numeric_cols`, `categorical_cols`, `categorical_pipeline` from cell 1 |
| 4 (Poly grid) | `poly_model` (cell 3); redefines `cv` |
| 5 (SGD) | `best_poly_model` (cell 4), `cv` |
| 6 (Log target) | `preprocessor`, `cv` |
| 7 (Text) | `X_train`, `X_val`, `cv`; **redefines `numeric_cols`** |
| 8 (Error analysis) | `best_ridge` (cell 2), `X_val`, `y_val` |
| 9 (SVD) | `text_preprocessor`, `X_train_text`, `X_val_text` (cell 7), `cv` |
| 10 (Save model) | `linear_model` (cell 1) |

If a kernel is restarted, run from the top rather than jumping into the middle.

---

## 11. Limitations and caveats

1. **The grouped CV folds are unused.** `grouped_cv` is built in cell 1 but `cv` is overwritten by a shuffled `KFold` in cell 2, so hyperparameter tuning can still see near-duplicate listings across folds. CV scores are therefore somewhat optimistic, and the chosen alphas may lean towards too little regularisation. Using `cv = grouped_cv` throughout (and not redefining `cv`) would fix this.
2. **No separate test evaluation.** The validation set is used to compare **all** models, so reported validation scores are slightly optimistic. The raw `test.csv` from the cleaning notebook has not been evaluated; it would first need the same cleaning steps applied.
3. **Outliers were deliberately kept.** Very high prices (over £1M) remain in both train and validation sets, affecting fitted coefficients and RMSE/R².
4. **Possible minor leakage from cleaning.** If imputation medians in the cleaning notebook were computed before this split, validation rows influenced some imputed values.
5. **Sentinel values such as `power_hp = 0`** (if present in the data) would be treated as real numbers here.
6. **High-cardinality one-hot encoding** (`model`, `version`, `seller_city`, `seller_county`) creates a very wide sparse matrix, increasing overfitting risk and runtime.
7. **Tuning breadth is modest:** 3-fold CV, a coarse alpha grid, and small SGD/SVD grids. Several winners sit at the edge of their grid (alpha = 0.1 for Ridge and log-Ridge, alpha = 0.0001 for SGD), so a wider search is warranted.
8. **Linear models only.** Interactions between make, age, mileage, and power are inherently nonlinear in price; this notebook is a baseline, not a ceiling.
9. **Variable name reuse:** `numeric_cols` and `cv` are redefined mid-notebook, and generic names such as `predictions` are overwritten between experiments. Re-running individual cells out of order can give confusing results.
10. **The saved `linear_reg.joblib` is the OLS baseline**, not the best model, and the notebook prints an absolute Windows save path that will differ on other machines.
11. **Prices are a single-time snapshot** (October 2026 listings) of advertised prices, not final sale prices.

---

## 12. Suggested next steps

- **Fix the CV setup:** use the precomputed grouped folds for all searches, and extend CV to 5 or 10 folds.
- **Widen the grids** below the current boundary values (alpha under 0.1 for Ridge and log-Ridge; smaller alpha and eta0 for SGD).
- **Nonlinear models:** gradient boosting (such as `HistGradientBoostingRegressor`), random forests, or LightGBM/XGBoost. These handle interactions, rare categories, and heavy tails more naturally, and can be combined with the log target.
- **Target-aware handling of extremes:** cap or winsorise price, train separate models for mainstream and supercar segments, or use a robust loss (Huber or quantile loss).
- **Feature engineering:** car age (listing year minus `reg_year`), mileage per year, make/model frequency encoding or target encoding in place of raw one-hot, and a flag for `version == "Unknown"`.
- **Report percentage errors** (MAPE or median absolute percentage error) so that results are comparable between cheap and expensive cars.
- **Clip log-model predictions** and check bias after `expm1` back-transformation.
- **Save the best model** (the log-target Ridge) with `joblib` instead of, or alongside, the OLS baseline.
- **Final evaluation on `test.csv`** after replaying the cleaning pipeline, and only for the one chosen model.

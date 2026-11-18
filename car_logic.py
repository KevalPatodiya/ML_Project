"""Pure-Python logic for the used-car price app (no Streamlit imports here).

Keeping this separate from app.py means it can be tested on its own, and the
feature engineering stays identical to the notebooks:
  * EDA and More Cleaning.ipynb  -> vehicle_age, miles_per_year, engine_size_l, flags
  * decision_tree / ensembling notebooks -> log_mileage, make_model, prediction blend
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

REF_YEAR = 2026  # same fixed reference year used when the data was cleaned

# The 40 columns every sub-model in ensemble_used_car_price.joblib expects.
FEATURE_COLUMNS = [
    "make", "model", "version", "mileage_miles", "reg_year", "fuel", "transmission",
    "power_hp", "body_type", "body_color", "doors", "seats", "previous_owners",
    "drivetrain", "annual_tax_gbp", "seller_type", "seller_city", "seller_county",
    "co2_gmile", "seller_county_imputed", "reg_year_imputed", "fuel_imputed",
    "transmission_imputed", "body_type_imputed", "drivetrain_imputed",
    "emission_class_imputed", "mileage_imputed", "power_hp_imputed", "co2_imputed",
    "annual_tax_imputed", "previous_owners_imputed", "seats_imputed", "doors_imputed",
    "vehicle_age", "miles_per_year", "engine_size_l", "is_performance_trim",
    "is_classic", "log_mileage", "make_model",
]

IMPUTED_FLAGS = [c for c in FEATURE_COLUMNS if c.endswith("_imputed")]

PERFORMANCE_RE = re.compile(
    r"\b(?:amg|m sport|m performance|gti|gtd|s line|st-line|r-line|type r|cupra|abarth|vxr|vrs)\b"
)
ENGINE_RE = re.compile(r"(?<![\d.])(\d\.\d)(?=\D|$)")
ELECTRIC_FUELS = {"Electric", "Hydrogen"}

# The ensemble sorts every estimate into one of three price bands.
DEFAULT_CUTOFFS = np.array([10_000, 25_000])
DEFAULT_BAND_NAMES = ["Under £10,000", "£10,000 to under £25,000", "£25,000+"]

MODEL_LABELS = {
    "decision_tree": "Decision Tree + Lasso + Boosting",
    "random_forest": "Random Forest + Lasso + Boosting",
    "extra_trees": "Extra Trees + Ridge + Boosting",
}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def parse_engine_size(version: str, fuel: str, fallback: float = 1.5) -> float:
    """Litres from the version text ('2.0 TFSI ...'); 0 for electric/hydrogen."""
    if fuel in ELECTRIC_FUELS:
        return 0.0
    match = ENGINE_RE.search(str(version))
    if match:
        litres = float(match.group(1))
        if 0.6 <= litres <= 8.5:
            return litres
    return float(fallback)


def is_performance_trim(version: str) -> int:
    return int(bool(PERFORMANCE_RE.search(str(version).lower())))


def build_features(rows: pd.DataFrame | dict | Iterable[dict]) -> pd.DataFrame:
    """Turn raw car details into the exact 40-column frame the models were trained on.

    Required keys: make, model, mileage_miles, reg_year. Everything else is expected
    to be present too (the app fills sensible defaults before calling this).
    """
    if isinstance(rows, dict):
        df = pd.DataFrame([rows])
    elif isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(list(rows))

    df["mileage_miles"] = pd.to_numeric(df["mileage_miles"], errors="coerce").fillna(0).clip(lower=0)
    df["reg_year"] = pd.to_numeric(df["reg_year"], errors="coerce").fillna(REF_YEAR).astype(int)

    df["vehicle_age"] = (REF_YEAR - df["reg_year"]).clip(lower=0)
    df["miles_per_year"] = (df["mileage_miles"] / df["vehicle_age"].clip(lower=1)).round(1)
    df["log_mileage"] = np.log1p(df["mileage_miles"])
    df["make_model"] = (
        df["make"].fillna("Unknown").astype(str) + " | " + df["model"].fillna("Unknown").astype(str)
    )
    df["is_classic"] = (df["reg_year"] < 2000).astype(int)

    if "is_performance_trim" not in df.columns:
        df["is_performance_trim"] = df["version"].map(is_performance_trim)
    if "engine_size_l" not in df.columns:
        df["engine_size_l"] = [parse_engine_size(v, f) for v, f in zip(df["version"], df["fuel"])]

    # The user supplies real values, so none of them are "imputed".
    for flag in IMPUTED_FLAGS:
        if flag not in df.columns:
            df[flag] = 0

    int_cols = [
        "mileage_miles", "reg_year", "power_hp", "doors", "seats", "previous_owners",
        "co2_gmile", "vehicle_age", "is_performance_trim", "is_classic", *IMPUTED_FLAGS,
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype("int64")
    for col in ["annual_tax_gbp", "engine_size_l", "miles_per_year", "log_mileage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns when building features: {missing}")
    return df[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# Prediction (mirrors cell 28 / 29 of ensembling.ipynb)
# --------------------------------------------------------------------------- #
def _predict_lasso(fitted, features):
    preprocessor, weights, intercept = fitted
    result = preprocessor.transform(features) @ weights + intercept
    return np.asarray(np.expm1(result)).reshape(-1)


def _predict_log_tree_tuple(fitted, features):
    preprocessor, model = fitted
    return np.asarray(np.expm1(model.predict(preprocessor.transform(features)))).reshape(-1)


def _quiet(estimator) -> None:
    """Switch off per-tree progress printing if the estimator has it."""
    try:
        estimator.verbose = 0
    except Exception:
        pass


def predict_one_model(model_bundle: dict, features: pd.DataFrame, name: str) -> np.ndarray:
    """Price estimate in pounds from one of the three saved sub-models."""
    if name == "decision_tree":
        lasso = _predict_lasso(model_bundle["lasso"], features)
        tree = np.asarray(model_bundle["tree"].predict(features)).reshape(-1)
        estimate = model_bundle["lasso_weight"] * lasso + model_bundle["tree_weight"] * tree
    elif name == "random_forest":
        lasso = _predict_lasso(model_bundle["tuned_lasso"], features)
        forest = _predict_log_tree_tuple(model_bundle["tuned_forest"], features)
        estimate = model_bundle["lasso_weight"] * lasso + model_bundle["forest_weight"] * forest
    else:  # extra_trees
        ridge = np.expm1(model_bundle["ridge"].predict(features))
        extra = _predict_log_tree_tuple(model_bundle["extra_trees"], features)
        estimate = model_bundle["ridge_weight"] * np.asarray(ridge).reshape(-1) \
            + model_bundle["extra_trees_weight"] * extra

    # Every sub-model adds a Gradient Boosting correction (in pounds).
    estimate = np.asarray(estimate).reshape(-1) + np.asarray(model_bundle["booster"].predict(features)).reshape(-1)
    return np.maximum(estimate, 0.0)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def band_probabilities(prices: np.ndarray, cutoffs, log_spread: float) -> np.ndarray:
    """Soft band probabilities per model, shape (n_rows, n_models, n_bands)."""
    cutoffs = np.asarray(cutoffs, dtype=float)
    log_prices = np.log(np.maximum(prices, 1.0))
    cdf = _normal_cdf((np.log(cutoffs) - log_prices[:, :, None]) / log_spread)
    return np.stack([cdf[:, :, 0], cdf[:, :, 1] - cdf[:, :, 0], 1.0 - cdf[:, :, 1]], axis=-1)


def predict_all(bundle: dict, features: pd.DataFrame) -> dict:
    """Run all three sub-models and combine them the way the notebook scored best.

    Final price  = mean of the three estimates
    Final band   = soft vote (average of each model's band probabilities)
    """
    names = list(bundle["models"].keys())
    columns = []
    for name in names:
        model_bundle = bundle["models"][name]
        expected = model_bundle.get("feature_columns", FEATURE_COLUMNS)
        columns.append(predict_one_model(model_bundle, features[expected], name))
    prices = np.column_stack(columns)

    cutoffs = np.asarray(bundle.get("cutoffs", DEFAULT_CUTOFFS))
    spread = float(bundle.get("log_spread", 0.10))
    per_model_probs = band_probabilities(prices, cutoffs, spread)
    soft = per_model_probs.mean(axis=1)

    return {
        "names": names,
        "per_model": prices,
        "mean_price": prices.mean(axis=1),
        "median_price": np.median(prices, axis=1),
        "band_probs": soft,
        "band_index": soft.argmax(axis=1),
        "band_names": list(bundle.get("band_names", DEFAULT_BAND_NAMES)),
    }


def estimate_prices(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    """Just the final mean price for each row (used by the what-if sweeps)."""
    return predict_all(bundle, features)["mean_price"]


# --------------------------------------------------------------------------- #
# Reference-data helpers (pre-fill specs, batch defaults)
# --------------------------------------------------------------------------- #
def model_defaults(ref: dict, make: str, model: str, version: str | None = None) -> dict:
    """Typical specs for a make/model (and version when known)."""
    glob = ref["global"]
    entry = ref["makes"].get(make, {}).get(model)
    base = {
        "hp": 148, "doors": 5, "seats": 5, "co2": 150, "tax": 200, "eng": 1.5,
        "fuel": ref["options"]["fuel"][0], "trans": ref["options"]["transmission"][0],
        "body": ref["options"]["body_type"][0], "drive": ref["options"]["drivetrain"][0],
        "colour": glob["colour"], "owners": glob["owners"],
    }
    spec_keys = ("hp", "doors", "seats", "co2", "tax", "eng", "fuel", "trans", "body", "drive")
    if entry:
        base.update({k: entry[k] for k in spec_keys + ("colour", "owners")})
        if version and version in entry["versions"]:
            base.update({k: entry["versions"][version][k] for k in spec_keys})
    return base


def fill_batch_defaults(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Complete an uploaded table: missing optional columns get typical values per make/model."""
    out = df.copy()
    glob = ref["global"]
    needed = ["make", "model", "mileage_miles", "reg_year"]
    absent = [c for c in needed if c not in out.columns]
    if absent:
        raise ValueError(f"Your file needs these columns: {absent}")

    mapping = {  # output column -> (reference key, fallback)
        "power_hp": "hp", "doors": "doors", "seats": "seats", "co2_gmile": "co2",
        "annual_tax_gbp": "tax", "fuel": "fuel", "transmission": "trans",
        "body_type": "body", "drivetrain": "drive", "body_color": "colour",
        "previous_owners": "owners",
    }
    cache: dict[tuple, dict] = {}
    for col in ("version", "seller_type", "seller_county", "seller_city"):
        if col not in out.columns:
            out[col] = {"version": "", "seller_type": glob["seller_type"],
                        "seller_county": glob["county"], "seller_city": "Unknown"}[col]
    for col in list(mapping) + ["engine_size_l"]:
        if col not in out.columns:
            out[col] = None
        out[col] = out[col].astype(object)  # allows mixing numbers and text while filling

    for idx, row in out.iterrows():
        key = (row["make"], row["model"], row["version"] if isinstance(row["version"], str) else "")
        if key not in cache:
            cache[key] = model_defaults(ref, key[0], key[1], key[2] or None)
        d = cache[key]
        for col, ref_key in mapping.items():
            if pd.isna(out.at[idx, col]) or out.at[idx, col] == "":
                out.at[idx, col] = d[ref_key]
        if pd.isna(out.at[idx, "engine_size_l"]):
            fuel = out.at[idx, "fuel"]
            out.at[idx, "engine_size_l"] = 0.0 if fuel in ELECTRIC_FUELS else d["eng"]
    for col in ("seller_type", "seller_county", "seller_city", "version"):
        out[col] = out[col].fillna("").astype(str)
    out["seller_type"] = out["seller_type"].replace("", glob["seller_type"])
    out["seller_county"] = out["seller_county"].replace("", glob["county"])
    out["seller_city"] = out["seller_city"].replace("", "Unknown")
    return out


def price_verdict(asking: float, estimate: float, tolerance: float = 0.10) -> tuple[str, float]:
    """Compare an asking price with the estimate. Returns (label, % difference)."""
    diff = (asking - estimate) / estimate if estimate > 0 else 0.0
    if diff < -tolerance:
        return "Below estimate", diff * 100
    if diff > tolerance:
        return "Above estimate", diff * 100
    return "In line with estimate", diff * 100

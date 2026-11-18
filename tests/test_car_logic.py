"""Sanity tests for car_logic.py.

Run from the app folder:   python -m pytest tests -q

Set CAR_TRAIN_CSV=path/to/train_clean.csv to also check that the app's feature
engineering reproduces the engineered columns stored in your training data.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import car_logic as cl  # noqa: E402

REF = json.loads((ROOT / "reference_data.json").read_text(encoding="utf-8"))


class _Const:
    """Stand-in for a fitted model: returns a fixed value per row."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class _Ident:
    def transform(self, X):
        return np.zeros((len(X), 2))


def fake_bundle(price_a=8000.0, price_b=9000.0, price_c=12000.0):
    """A bundle with the same keys as ensemble_used_car_price.joblib."""
    log = lambda p: np.log1p(p)  # noqa: E731
    return {
        "models": {
            "decision_tree": {
                "lasso": (_Ident(), np.zeros(2), log(price_a)),
                "tree": _Const(price_a), "booster": _Const(0.0),
                "lasso_weight": 0.75, "tree_weight": 0.25,
                "feature_columns": cl.FEATURE_COLUMNS,
            },
            "random_forest": {
                "tuned_lasso": (_Ident(), np.zeros(2), log(price_b)),
                "tuned_forest": (_Ident(), _Const(log(price_b))), "booster": _Const(0.0),
                "lasso_weight": 0.85, "forest_weight": 0.15,
                "feature_columns": cl.FEATURE_COLUMNS,
            },
            "extra_trees": {
                "ridge": _Const(log(price_c)),
                "extra_trees": (_Ident(), _Const(log(price_c))), "booster": _Const(0.0),
                "ridge_weight": 0.8, "extra_trees_weight": 0.2,
                "feature_columns": cl.FEATURE_COLUMNS,
            },
        },
        "cutoffs": np.array([10_000, 25_000]),
        "band_names": np.array(cl.DEFAULT_BAND_NAMES),
        "log_spread": 0.10,
    }


def sample_car(**overrides):
    car = dict(
        make="Kia", model="Picanto", version="1.2 GT-Line S AMT Euro 6 (s/s) 5dr",
        mileage_miles=9250, reg_year=2025, fuel="Petrol", transmission="Automatic",
        power_hp=77, body_type="Hatchback", body_color="White", doors=5, seats=5,
        previous_owners=2, drivetrain="Front Wheel Drive", annual_tax_gbp=200.0,
        seller_type="Trade", seller_city="Paignton", seller_county="Devon", co2_gmile=206,
    )
    car.update(overrides)
    return car


def test_feature_frame_has_exact_columns_and_order():
    feats = cl.build_features(sample_car())
    assert list(feats.columns) == cl.FEATURE_COLUMNS
    assert len(feats) == 1
    assert feats.loc[0, "vehicle_age"] == 1
    assert feats.loc[0, "make_model"] == "Kia | Picanto"
    assert feats.loc[0, "is_performance_trim"] == 0  # "GT-Line" is not in the trim list
    assert feats.loc[0, "engine_size_l"] == 1.2
    assert (feats[cl.IMPUTED_FLAGS] == 0).all().all()


def test_engine_parsing_and_flags():
    assert cl.parse_engine_size("2.0 TFSI quattro", "Petrol") == 2.0
    assert cl.parse_engine_size("anything", "Electric") == 0.0
    assert cl.parse_engine_size("no number here", "Petrol", fallback=1.6) == 1.6
    assert cl.is_performance_trim("2.0 TSI GTI 5dr") == 1
    assert cl.is_performance_trim("1.0 Zetec 5dr") == 0
    feats = cl.build_features(sample_car(reg_year=1995, mileage_miles=120000))
    assert feats.loc[0, "is_classic"] == 1
    assert feats.loc[0, "miles_per_year"] == round(120000 / 31, 1)


def test_future_or_zero_age_cars_do_not_divide_by_zero():
    feats = cl.build_features(sample_car(reg_year=2026, mileage_miles=10))
    assert feats.loc[0, "vehicle_age"] == 0
    assert feats.loc[0, "miles_per_year"] == 10.0


def test_predict_all_blends_and_bands():
    out = cl.predict_all(fake_bundle(), cl.build_features(sample_car()))
    assert out["per_model"].shape == (1, 3)
    # each fake sub-model returns (almost exactly) its target price
    np.testing.assert_allclose(out["per_model"][0], [8000, 9000, 12000], rtol=1e-6)
    assert out["mean_price"][0] == pytest.approx(np.mean([8000, 9000, 12000]))
    assert out["band_probs"].shape == (1, 3)
    assert out["band_probs"].sum() == pytest.approx(1.0)
    # mean 9667 sits just under the £10,000 line
    assert out["band_index"][0] == 0


def test_batch_prediction_vectorised():
    cars = [sample_car(mileage_miles=m) for m in (1000, 50000, 90000)]
    out = cl.predict_all(fake_bundle(), cl.build_features(cars))
    assert out["mean_price"].shape == (3,)


def test_price_verdict():
    assert cl.price_verdict(8000, 10000)[0] == "Below estimate"
    assert cl.price_verdict(10500, 10000)[0] == "In line with estimate"
    assert cl.price_verdict(12000, 10000)[0] == "Above estimate"


def test_reference_data_is_usable():
    make = REF["make_order"][0]
    model = next(iter(REF["makes"][make]))
    d = cl.model_defaults(REF, make, model)
    assert d["fuel"] in REF["options"]["fuel"] or d["fuel"] == "Unknown"
    version = next(iter(REF["makes"][make][model]["versions"]))
    d2 = cl.model_defaults(REF, make, model, version)
    assert d2["hp"] > 0


def test_batch_defaults_fill_missing_columns():
    make = REF["make_order"][0]
    model = next(iter(REF["makes"][make]))
    df = pd.DataFrame({"make": [make], "model": [model], "mileage_miles": [40000], "reg_year": [2019]})
    filled = cl.fill_batch_defaults(df, REF)
    feats = cl.build_features(filled)
    assert list(feats.columns) == cl.FEATURE_COLUMNS
    assert not feats.isna().any().any()


@pytest.mark.skipif(not os.environ.get("CAR_TRAIN_CSV"), reason="set CAR_TRAIN_CSV to compare with training data")
def test_features_match_training_csv():
    """Rebuild engineered columns from raw ones and compare with what the notebooks stored."""
    df = pd.read_csv(os.environ["CAR_TRAIN_CSV"]).sample(5000, random_state=0)
    raw = df.drop(columns=["vehicle_age", "miles_per_year", "is_performance_trim", "is_classic"])
    mine = cl.build_features(raw.assign(engine_size_l=df["engine_size_l"].to_numpy()))
    for col in ["vehicle_age", "miles_per_year", "is_performance_trim", "is_classic"]:
        np.testing.assert_allclose(mine[col].to_numpy(), df[col].to_numpy(), rtol=1e-6, err_msg=col)
    np.testing.assert_allclose(mine["log_mileage"], np.log1p(df["mileage_miles"]), rtol=1e-9)
    # engine size from the version text should agree for the large majority of non-EVs
    parsed = np.array([cl.parse_engine_size(v, f, fallback=np.nan) for v, f in zip(df["version"], df["fuel"])])
    comparable = ~np.isnan(parsed)
    assert (np.abs(parsed[comparable] - df["engine_size_l"].to_numpy()[comparable]) < 1e-9).mean() > 0.98

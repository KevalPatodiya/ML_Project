"""Rebuild the small lookup files the Streamlit app needs.

Run this once (or whenever the training data changes):

    python build_reference_data.py path/to/train_clean.csv

It writes two files next to this script:
  * reference_data.json    - dropdown options and typical specs per make/model/version,
                             so the app can pre-fill the form without the 84 MB CSV
  * market_sample.csv.gz   - a light sample of real listings for the "Market view" tab
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train_clean.csv")
OUT = Path(__file__).parent

MAX_VERSIONS_PER_MODEL = 30
MIN_OPTION_ROWS = 20
MARKET_ROWS_PER_MODEL = 120

NUM_COLS = ["power_hp", "doors", "seats", "co2_gmile", "annual_tax_gbp", "engine_size_l"]
CAT_COLS = ["fuel", "transmission", "body_type", "drivetrain"]


def mode_by(df, keys, col):
    """Most common value of `col` within each group of `keys` (fast, no lambdas)."""
    counts = df.groupby(keys + [col], observed=True).size().reset_index(name="_n")
    counts = counts.sort_values("_n").drop_duplicates(keys, keep="last")
    return counts.set_index(keys)[col]


def spec_table(df, keys):
    """One row per group with median numbers, modal categories and a row count."""
    out = df.groupby(keys, observed=True)[NUM_COLS].median()
    for col in CAT_COLS:
        out[col] = mode_by(df, keys, col)
    out["n"] = df.groupby(keys, observed=True).size()
    return out


def pack(row):
    return {
        "n": int(row["n"]),
        "hp": int(round(row["power_hp"])),
        "doors": int(round(row["doors"])),
        "seats": int(round(row["seats"])),
        "co2": int(round(row["co2_gmile"])),
        "tax": int(round(row["annual_tax_gbp"])),
        "eng": round(float(row["engine_size_l"]), 1),
        "fuel": row["fuel"],
        "trans": row["transmission"],
        "body": row["body_type"],
        "drive": row["drivetrain"],
    }


def main():
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC}")

    # ---- dropdown options -------------------------------------------------
    def options(col, drop=("Unknown",)):
        vc = df[col].value_counts()
        return [v for v, n in vc.items() if n >= MIN_OPTION_ROWS and v not in drop]

    opts = {
        "fuel": options("fuel"),
        "transmission": options("transmission"),
        "body_type": options("body_type"),
        "body_color": options("body_color"),
        "drivetrain": options("drivetrain"),
        "seller_type": options("seller_type"),
    }

    # ---- location lookup: county -> cities --------------------------------
    county_counts = df["seller_county"].value_counts()
    counties = [c for c, n in county_counts.items() if n >= MIN_OPTION_ROWS and c != "Unknown"]
    cities = {}
    for county in counties:
        sub = df.loc[df["seller_county"] == county, "seller_city"].value_counts()
        cities[county] = [c for c, n in sub.head(60).items() if n >= 5 and c != "Unknown"]

    # ---- make -> model -> version defaults --------------------------------
    model_specs = spec_table(df, ["make", "model"])
    version_specs = spec_table(df, ["make", "model", "version"])
    colour_mode = mode_by(df, ["make", "model"], "body_color")
    owners_median = df.groupby(["make", "model"])["previous_owners"].median()

    makes = {}
    for make, make_rows in model_specs.groupby(level=0):
        models = {}
        for (_, model), row in make_rows.sort_values("n", ascending=False).iterrows():
            entry = pack(row)
            entry["colour"] = colour_mode.get((make, model), opts["body_color"][0])
            entry["owners"] = int(round(owners_median.get((make, model), 2)))
            vsub = version_specs.loc[(make, model)].sort_values("n", ascending=False)
            keep = vsub[vsub["n"] >= 2].head(MAX_VERSIONS_PER_MODEL)
            if keep.empty:
                keep = vsub.head(1)
            entry["versions"] = {ver: pack(r) for ver, r in keep.iterrows()}
            models[model] = entry
        makes[make] = models
    make_order = list(df["make"].value_counts().index)

    # ---- global fallbacks and slider ranges -------------------------------
    glob = {
        "owners": int(df["previous_owners"].median()),
        "colour": df["body_color"].mode().iat[0],
        "seller_type": df["seller_type"].mode().iat[0],
        "county": county_counts.index[0],
        "mileage_median": int(df["mileage_miles"].median()),
        "year_min": int(df["reg_year"].min()),
        "year_max": int(df["reg_year"].max()),
        "price_median": int(df["price_gbp"].median()),
    }

    ref = {
        "options": opts,
        "counties": counties,
        "cities": cities,
        "make_order": make_order,
        "makes": makes,
        "global": glob,
    }
    (OUT / "reference_data.json").write_text(json.dumps(ref, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print("Wrote reference_data.json:", (OUT / "reference_data.json").stat().st_size // 1024, "KB")

    # ---- market sample -----------------------------------------------------
    keep_cols = ["make", "model", "reg_year", "mileage_miles", "fuel", "transmission", "price_gbp"]
    market = df.loc[df["price_gbp"] <= 1_000_000, keep_cols]
    # shuffle, then keep at most N rows per make/model (works the same on every pandas version)
    market = (
        market.sample(frac=1, random_state=42)
        .groupby(["make", "model"], sort=False)
        .head(MARKET_ROWS_PER_MODEL)
        .reset_index(drop=True)
    )
    market.to_csv(OUT / "market_sample.csv.gz", index=False, compression="gzip")
    print(f"Wrote market_sample.csv.gz: {len(market):,} rows,",
          (OUT / "market_sample.csv.gz").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

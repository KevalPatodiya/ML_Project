from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "ensemble_used_car_price.joblib"

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")
st.title("Used Car Price Estimator")
st.write("Upload a CSV containing one car listing to estimate its price.")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

def predict_lasso(fitted, features):
    preprocessor, weights, intercept = fitted
    result = preprocessor.transform(features) @ weights + intercept
    return np.asarray(np.expm1(result)).reshape(-1)

def predict_log_tree_tuple(fitted, features):
    preprocessor, model = fitted
    return np.asarray(
        np.expm1(model.predict(preprocessor.transform(features)))
    ).reshape(-1)

def predict_one_model(bundle, features, model_name):
    if model_name == "decision_tree":
        lasso = predict_lasso(bundle["lasso"], features)
        tree = bundle["tree"].predict(features)
        estimate = (
            bundle["lasso_weight"] * lasso
            + bundle["tree_weight"] * tree
        )

    elif model_name == "random_forest":
        lasso = predict_lasso(bundle["tuned_lasso"], features)
        forest = predict_log_tree_tuple(bundle["tuned_forest"], features)
        estimate = (
            bundle["lasso_weight"] * lasso
            + bundle["forest_weight"] * forest
        )

    else:  # extra_trees
        ridge = np.expm1(bundle["ridge"].predict(features))
        extra_trees = predict_log_tree_tuple(bundle["extra_trees"], features)
        estimate = (
            bundle["ridge_weight"] * ridge
            + bundle["extra_trees_weight"] * extra_trees
        )

    # The notebook adds a Gradient Boosting correction to each model.
    estimate = estimate + bundle["booster"].predict(features)
    return np.maximum(np.asarray(estimate).reshape(-1), 0)

uploaded = st.file_uploader("Choose a one-car CSV file", type="csv")

if uploaded is not None:
    bundle = load_bundle()
    data = pd.read_csv(uploaded)

    if len(data) != 1:
        st.error("For now, upload a CSV with exactly one car listing.")
        st.stop()

    # Remove prices and other target-derived columns if they are present.
    features = data.drop(
        columns=[
            "price_gbp",
            "log_price",
            "price_eur",
            "is_high_value",
            "price_evaluation",
        ],
        errors="ignore",
    ).copy()

    required_base = ["make", "model", "mileage_miles"]
    missing_base = [col for col in required_base if col not in features.columns]

    if missing_base:
        st.error(f"Your CSV is missing required columns: {missing_base}")
        st.stop()

    # These are the two extra features created in the notebook.
    features["log_mileage"] = np.log1p(
        pd.to_numeric(features["mileage_miles"], errors="coerce")
    )
    features["make_model"] = (
        features["make"].fillna("Unknown").astype(str)
        + " | "
        + features["model"].fillna("Unknown").astype(str)
    )

    predictions = {}
    for name, model_bundle in bundle["models"].items():
        expected = model_bundle["feature_columns"]
        missing = [col for col in expected if col not in features.columns]

        if missing:
            st.error(
                f"CSV is missing columns required by {name}: {missing}"
            )
            st.stop()

        predictions[name] = predict_one_model(
            model_bundle,
            features[expected],
            name,
        )[0]

    average = float(np.mean(list(predictions.values())))
    band_number = int(np.digitize([average], bundle["cutoffs"])[0])
    price_band = bundle["band_names"][band_number]

    st.subheader("Estimated prices")
    st.dataframe(
        pd.DataFrame({
            "Model": list(predictions.keys()),
            "Estimated price (£)": [
                round(value, 2) for value in predictions.values()
            ],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.metric("Average estimate", f"£{average:,.2f}")
    st.write(f"**Estimated price band:** {price_band}")
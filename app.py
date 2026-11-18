"""UK Used Car Price Estimator - Streamlit app for ensemble_used_car_price.joblib.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import car_logic as cl

APP_DIR = Path(__file__).parent
MODEL_CANDIDATES = [APP_DIR / "ensemble_used_car_price.joblib", APP_DIR.parent / "ensemble_used_car_price.joblib"]
EXPECTED_SKLEARN = "1.5.2"   # the version the model file was trained and saved with
OTHER = "Other (type my own)"
NOT_SPECIFIED = "(not specified)"
BATCH_ROW_LIMIT = 10_000

ACCENT = "#2F6FED"
GOOD = "#1F9D6B"
WARN = "#E08A00"
BAD = "#D64545"
MUTED = "#8A94A6"

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1250px;}
      div[data-testid="stMetric"] {background: rgba(47,111,237,0.06); border: 1px solid rgba(47,111,237,0.18);
                                   padding: 14px 18px; border-radius: 14px;}
      div[data-testid="stMetricValue"] {font-size: 2rem;}
      .band-pill {display:inline-block; padding:4px 12px; border-radius:999px; font-weight:600;
                  background:rgba(47,111,237,0.12); color:#2F6FED;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the ensemble model (about 55 MB)...")
def load_bundle():
    path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("ensemble_used_car_price.joblib was not found next to app.py")
    bundle = joblib.load(path)
    for model_bundle in bundle["models"].values():       # stop trees printing progress
        for value in model_bundle.values():
            parts = value if isinstance(value, tuple) else (value,)
            for part in parts:
                if hasattr(part, "verbose"):
                    try:
                        part.verbose = 0
                    except Exception:
                        pass
    return bundle


@st.cache_data(show_spinner=False)
def load_reference() -> dict:
    return json.loads((APP_DIR / "reference_data.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_market() -> pd.DataFrame:
    return pd.read_csv(APP_DIR / "market_sample.csv.gz")


REF = load_reference()
OPT = REF["options"]
GLOB = REF["global"]


# --------------------------------------------------------------------------- #
# Form state: choosing a make/model/version pre-fills the typical specs
# --------------------------------------------------------------------------- #
def choices(options: list, current) -> list:
    """Make sure the currently selected value is always a valid option."""
    options = list(options)
    if current not in options and current is not None:
        options.append(current)
    return options


def _clamp(value, low, high):
    return max(low, min(high, value))


def apply_specs() -> None:
    ss = st.session_state
    version = ss.version if ss.version != OTHER else None
    d = cl.model_defaults(REF, ss.make, ss.model, version)
    ss.power_hp = int(_clamp(d["hp"], 20, 1000))
    ss.doors = int(_clamp(d["doors"], 2, 5))
    ss.seats = int(_clamp(d["seats"], 2, 9))
    ss.co2 = int(_clamp(d["co2"], 0, 600))
    ss.tax = int(_clamp(d["tax"], 0, 800))
    ss.engine = float(_clamp(d["eng"], 0.0, 8.5))
    ss.fuel, ss.transmission = d["fuel"], d["trans"]
    ss.body_type, ss.drivetrain = d["body"], d["drive"]


def on_version_change() -> None:
    apply_specs()


def on_model_change() -> None:
    ss = st.session_state
    versions = list(REF["makes"][ss.make][ss.model]["versions"])
    ss.version = versions[0]
    d = cl.model_defaults(REF, ss.make, ss.model)
    ss.body_color, ss.previous_owners = d["colour"], int(_clamp(d["owners"], 0, 7))
    apply_specs()


def on_make_change() -> None:
    ss = st.session_state
    ss.model = next(iter(REF["makes"][ss.make]))
    on_model_change()


def on_custom_version_change() -> None:
    ss = st.session_state
    ss.engine = float(cl.parse_engine_size(ss.version_custom, ss.fuel, fallback=ss.engine))


def on_county_change() -> None:
    st.session_state.city = NOT_SPECIFIED


def init_state() -> None:
    ss = st.session_state
    if ss.get("_ready"):
        return
    ss.make = REF["make_order"][0]
    ss.model = next(iter(REF["makes"][ss.make]))
    ss.version = next(iter(REF["makes"][ss.make][ss.model]["versions"]))
    ss.version_custom = ""
    ss.reg_year, ss.mileage = 2020, int(round(_clamp(GLOB["mileage_median"], 0, 250_000) / 500) * 500)
    ss.seller_type, ss.county, ss.city = GLOB["seller_type"], GLOB["county"], NOT_SPECIFIED
    d = cl.model_defaults(REF, ss.make, ss.model)
    ss.body_color, ss.previous_owners = d["colour"], int(_clamp(d["owners"], 0, 7))
    apply_specs()
    ss.asking = 0
    ss._ready = True


def sidebar_form() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("## 🚗 Describe the car")
        st.selectbox("Make", REF["make_order"], key="make", on_change=on_make_change)
        st.selectbox("Model", list(REF["makes"][ss.make]), key="model", on_change=on_model_change)
        versions = list(REF["makes"][ss.make][ss.model]["versions"]) + [OTHER]
        st.selectbox("Version / trim", versions, key="version", on_change=on_version_change)
        if ss.version == OTHER:
            st.text_input("Type the version", key="version_custom", on_change=on_custom_version_change,
                          placeholder="e.g. 2.0 TDI SE 5dr")

        c1, c2 = st.columns(2)
        c1.number_input("Registration year", min_value=1937, max_value=cl.REF_YEAR, step=1, key="reg_year")
        c2.number_input("Previous owners", min_value=0, max_value=7, step=1, key="previous_owners")
        st.slider("Mileage (miles)", 0, 250_000, step=500, key="mileage")

        with st.expander("Specs (pre-filled, editable)", expanded=False):
            a, b = st.columns(2)
            a.selectbox("Fuel", choices(OPT["fuel"], ss.fuel), key="fuel")
            b.selectbox("Gearbox", choices(OPT["transmission"], ss.transmission), key="transmission")
            a.selectbox("Body type", choices(OPT["body_type"], ss.body_type), key="body_type")
            b.selectbox("Drivetrain", choices(OPT["drivetrain"], ss.drivetrain), key="drivetrain")
            st.selectbox("Colour", choices(OPT["body_color"], ss.body_color), key="body_color")
            a.number_input("Power (hp)", 20, 1000, step=5, key="power_hp")
            b.number_input("Engine (litres)", 0.0, 8.5, step=0.1, format="%.1f", key="engine")
            a.number_input("Doors", 2, 5, key="doors")
            b.number_input("Seats", 2, 9, key="seats")
            a.number_input("CO₂ (g/mile)", 0, 600, step=5, key="co2")
            b.number_input("Annual tax (£)", 0, 800, step=5, key="tax")
            st.button("↩ Reset specs to typical values", on_click=apply_specs, use_container_width=True)

        with st.expander("Seller & location", expanded=False):
            st.radio("Seller", choices(OPT["seller_type"], ss.seller_type), key="seller_type", horizontal=True)
            st.selectbox("County", choices(REF["counties"], ss.county), key="county", on_change=on_county_change)
            cities = [NOT_SPECIFIED] + REF["cities"].get(ss.county, [])
            st.selectbox("City / town", choices(cities, ss.city), key="city")

        st.caption("Estimates update instantly as you change anything here.")


def current_car() -> dict:
    ss = st.session_state
    version = ss.version_custom.strip() if ss.version == OTHER else ss.version
    return dict(
        make=ss.make, model=ss.model, version=version, mileage_miles=int(ss.mileage),
        reg_year=int(ss.reg_year), fuel=ss.fuel, transmission=ss.transmission,
        power_hp=int(ss.power_hp), body_type=ss.body_type, body_color=ss.body_color,
        doors=int(ss.doors), seats=int(ss.seats), previous_owners=int(ss.previous_owners),
        drivetrain=ss.drivetrain, annual_tax_gbp=float(ss.tax), seller_type=ss.seller_type,
        seller_city="Unknown" if ss.city == NOT_SPECIFIED else ss.city, seller_county=ss.county,
        co2_gmile=int(ss.co2), engine_size_l=float(ss.engine),
    )


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        title=dict(text=title, x=0.0, font=dict(size=16)) if title else None,
        font=dict(size=13), hoverlabel=dict(font_size=13),
    )
    return fig


def gbp(value: float) -> str:
    return f"£{value:,.0f}"


def models_chart(res: dict) -> go.Figure:
    labels = [cl.MODEL_LABELS.get(n, n) for n in res["names"]]
    prices = res["per_model"][0]
    fig = go.Figure(go.Bar(
        x=prices, y=labels, orientation="h", marker_color=ACCENT, opacity=0.85,
        text=[gbp(p) for p in prices], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:£,.0f}<extra></extra>",
    ))
    fig.add_vline(x=float(res["mean_price"][0]), line_dash="dash", line_color=GOOD,
                  annotation_text="Final estimate", annotation_position="top")
    fig.update_xaxes(tickprefix="£", tickformat=",", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return style(fig, 260, "What each of the three models says")


def band_chart(res: dict) -> go.Figure:
    probs = res["band_probs"][0] * 100
    colors = [ACCENT if i == int(res["band_index"][0]) else "#C5CFE3" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=res["band_names"], y=probs, marker_color=colors,
        text=[f"{p:.0f}%" for p in probs], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 115], ticksuffix="%")
    return style(fig, 260, "How sure are the models about the price band?")


def deal_chart(asking: float, estimate: float) -> go.Figure:
    lo, hi = estimate * 0.9, estimate * 1.1
    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=0.2, y1=0.8, fillcolor="rgba(31,157,107,0.25)", line_width=0)
    fig.add_trace(go.Scatter(x=[estimate], y=[0.5], mode="markers", name="Estimate",
                             marker=dict(size=14, color=GOOD, symbol="diamond"),
                             hovertemplate="Estimate %{x:£,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[asking], y=[0.5], mode="markers", name="Asking price",
                             marker=dict(size=16, color=BAD, symbol="line-ns-open", line=dict(width=4, color=BAD)),
                             hovertemplate="Asking %{x:£,.0f}<extra></extra>"))
    span = max(abs(asking - estimate), estimate * 0.2)
    fig.update_xaxes(range=[min(asking, estimate) - span * 0.6, max(asking, estimate) + span * 0.6],
                     tickprefix="£", tickformat=",")
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(legend=dict(orientation="h", y=-0.35))
    return style(fig, 170)


# --------------------------------------------------------------------------- #
# What-if sweeps (cached: the same question is not recomputed on every rerun)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=48)
def run_sweep(_bundle, car_json: str, variable: str, scale_mileage: bool) -> pd.DataFrame:
    car = json.loads(car_json)
    base_age = max(cl.REF_YEAR - car["reg_year"], 1)
    miles_per_year = car["mileage_miles"] / base_age
    rows, labels = [], []

    if variable == "Mileage":
        top = int(max(120_000, car["mileage_miles"] * 1.5))
        values = list(np.linspace(0, top, 31).astype(int))
        for v in values:
            rows.append({**car, "mileage_miles": v}); labels.append(v)
    elif variable == "Registration year":
        values = list(range(max(1995, cl.REF_YEAR - 30), cl.REF_YEAR + 1))
        for v in values:
            miles = int(miles_per_year * max(cl.REF_YEAR - v, 1)) if scale_mileage else car["mileage_miles"]
            rows.append({**car, "reg_year": v, "mileage_miles": miles}); labels.append(v)
    elif variable == "Previous owners":
        values = list(range(0, 8))
        for v in values:
            rows.append({**car, "previous_owners": v}); labels.append(v)
    elif variable == "Fuel type":
        values = [f for f in OPT["fuel"]]
        for v in values:
            eng = 0.0 if v in cl.ELECTRIC_FUELS else (car["engine_size_l"] or 1.5)
            rows.append({**car, "fuel": v, "engine_size_l": eng}); labels.append(v)
    elif variable == "Gearbox":
        values = OPT["transmission"]
        for v in values:
            rows.append({**car, "transmission": v}); labels.append(v)
    else:  # Seller type
        values = OPT["seller_type"]
        for v in values:
            rows.append({**car, "seller_type": v}); labels.append(v)

    prices = cl.estimate_prices(_bundle, cl.build_features(rows))
    return pd.DataFrame({"value": labels, "price": prices})


def sweep_chart(df: pd.DataFrame, variable: str, car: dict, current_price: float) -> go.Figure:
    current = {"Mileage": car["mileage_miles"], "Registration year": car["reg_year"],
               "Previous owners": car["previous_owners"], "Fuel type": car["fuel"],
               "Gearbox": car["transmission"], "Seller type": car["seller_type"]}[variable]
    fig = go.Figure()
    if variable in ("Mileage", "Registration year"):
        fig.add_trace(go.Scatter(x=df["value"], y=df["price"], mode="lines+markers", line=dict(color=ACCENT, width=3),
                                 marker=dict(size=6), name="Estimate",
                                 hovertemplate=f"{variable}: %{{x:,}}<br>Estimate: %{{y:£,.0f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[current], y=[current_price], mode="markers", name="Your car",
                                 marker=dict(size=16, color=BAD, symbol="star"),
                                 hovertemplate="Your car<br>%{y:£,.0f}<extra></extra>"))
        fig.update_xaxes(title=variable + (" (miles)" if variable == "Mileage" else ""))
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.1))
    else:
        colors = [BAD if str(v) == str(current) else ACCENT for v in df["value"]]
        fig.add_trace(go.Bar(x=df["value"].astype(str), y=df["price"], marker_color=colors,
                             text=[gbp(p) for p in df["price"]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>%{y:£,.0f}<extra></extra>"))
        fig.update_xaxes(title=variable)
    fig.update_yaxes(title="Estimated price", tickprefix="£", tickformat=",", rangemode="tozero")
    return style(fig, 420, f"Estimated price as {variable.lower()} changes")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_estimate(res: dict, car: dict) -> None:
    price = float(res["mean_price"][0])
    band = res["band_names"][int(res["band_index"][0])]
    band_prob = float(res["band_probs"][0, int(res["band_index"][0])]) * 100

    title = f"{car['make']} {car['model']}"
    st.markdown(f"### {title}")
    st.caption(f"{car['version']}  ·  {car['reg_year']}  ·  {car['mileage_miles']:,} miles  ·  "
               f"{car['fuel']}  ·  {car['transmission']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated asking price", gbp(price))
    m2.metric("Likely range (±10%)", f"{gbp(price * 0.9)} – {gbp(price * 1.1)}")
    m3.metric("Price band", band, f"{band_prob:.0f}% sure", delta_color="off")
    st.caption("In testing on 88,120 unseen listings, about 3 in 4 estimates landed within 10% of the real asking price. "
               "Treat the range as a guide, not a guarantee.")

    left, right = st.columns(2)
    left.plotly_chart(models_chart(res), use_container_width=True)
    right.plotly_chart(band_chart(res), use_container_width=True)

    st.markdown("#### Is this a good deal?")
    st.number_input("Enter the seller's asking price (£) to compare", min_value=0, max_value=5_000_000,
                    step=250, key="asking")
    asking = float(st.session_state.asking)
    if asking > 0:
        label, diff = cl.price_verdict(asking, price)
        text = f"**{label}**: the asking price is {abs(diff):.1f}% {'below' if diff < 0 else 'above'} the model's estimate " \
               f"({gbp(asking - price) if diff >= 0 else '-' + gbp(price - asking)})."
        if label == "Below estimate":
            st.success(text + " Could be a bargain, or a sign to check condition, history and mileage carefully.")
        elif label == "Above estimate":
            st.warning(text + " There may be room to negotiate, or the car may have extras the model cannot see.")
        else:
            st.info(text + " That is a normal price for this kind of car.")
        st.plotly_chart(deal_chart(asking, price), use_container_width=True)


def tab_whatif(bundle: dict, car: dict, price: float) -> None:
    st.markdown("### What would change the price?")
    st.write("Pick one thing to vary. Everything else about the car stays as you described it in the sidebar.")
    c1, c2 = st.columns([2, 3])
    variable = c1.selectbox("Vary", ["Mileage", "Registration year", "Previous owners", "Fuel type", "Gearbox", "Seller type"],
                            key="sweep_var")
    scale = True
    if variable == "Registration year":
        scale = c2.toggle("Keep yearly mileage the same (older car = more miles)", value=True, key="sweep_scale")
    with st.spinner("Re-estimating..."):
        df = run_sweep(bundle, json.dumps(car, sort_keys=True), variable, scale)
    st.plotly_chart(sweep_chart(df, variable, car, price), use_container_width=True)
    if variable == "Fuel type":
        st.caption("Other specs (power, CO₂, tax, body) are kept the same, so unusual combinations may be unrealistic.")
    with st.expander("See the numbers"):
        show = df.rename(columns={"value": variable, "price": "Estimated price (£)"})
        show["Estimated price (£)"] = show["Estimated price (£)"].round(0)
        st.dataframe(show, hide_index=True, use_container_width=True)


def tab_market(car: dict, price: float) -> None:
    st.markdown("### Where does this car sit in the market?")
    market = load_market()
    comps = market[(market["make"] == car["make"]) & (market["model"] == car["model"])]
    scope = f"{car['make']} {car['model']}"
    if len(comps) < 30:
        comps = market[market["make"] == car["make"]]
        scope = f"{car['make']} (all models)"
    if comps.empty:
        st.info("No comparison listings are available for this car.")
        return

    pct = float((comps["price_gbp"] < price).mean() * 100)
    st.write(f"Compared with **{len(comps):,}** sampled listings for **{scope}**, your estimate is higher than "
             f"**{pct:.0f}%** of them.")

    cap = float(max(comps["price_gbp"].quantile(0.99), price * 1.2))
    shown = comps[comps["price_gbp"] <= cap]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=shown["mileage_miles"], y=shown["price_gbp"], mode="markers", name="Listings",
        marker=dict(size=6, opacity=0.55, color=shown["reg_year"], colorscale="Viridis",
                    colorbar=dict(title="Reg. year"), showscale=True),
        customdata=shown[["reg_year", "fuel", "transmission"]].to_numpy(dtype=object),
        hovertemplate="£%{y:,.0f}<br>%{x:,} miles<br>Reg. %{customdata[0]} · %{customdata[1]} · %{customdata[2]}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[car["mileage_miles"]], y=[price], mode="markers", name="Your car (estimate)",
                             marker=dict(size=18, color=BAD, symbol="star", line=dict(width=1, color="white"))))
    fig.update_xaxes(title="Mileage", tickformat=",")
    fig.update_yaxes(title="Asking price", tickprefix="£", tickformat=",")
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    st.plotly_chart(style(fig, 460, "Price against mileage (colour = registration year)"), use_container_width=True)

    hist = go.Figure(go.Histogram(x=shown["price_gbp"], nbinsx=40, marker_color="#9DB7F5"))
    hist.add_vline(x=price, line_color=BAD, line_width=3, annotation_text="Your estimate")
    hist.update_xaxes(title="Asking price", tickprefix="£", tickformat=",")
    hist.update_yaxes(title="Listings")
    st.plotly_chart(style(hist, 300, "How asking prices are spread"), use_container_width=True)
    st.caption("Comparison listings are a random sample from the project's training data, not live adverts.")


def template_csv() -> bytes:
    example = pd.DataFrame([
        dict(make="Ford", model="Focus", version="1.0 EcoBoost Zetec 5dr", mileage_miles=45000, reg_year=2019,
             fuel="Petrol", transmission="Manual", seller_type="Trade", seller_county="Lancashire"),
        dict(make="BMW", model="3 Series", version="2.0 320i M Sport Auto Euro 6 (s/s) 4dr", mileage_miles=60000,
             reg_year=2018, fuel="Petrol", transmission="Automatic", seller_type="Private", seller_county="Essex"),
    ])
    return example.to_csv(index=False).encode("utf-8")


def tab_batch(bundle: dict) -> None:
    st.markdown("### Price many cars at once")
    st.write("Upload a CSV with at least **make, model, mileage_miles, reg_year**. Anything else you leave out "
             "(fuel, gearbox, power, tax...) is filled with typical values for that make and model.")
    st.download_button("⬇ Download a template CSV", template_csv(), "car_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload your CSV", type="csv", key="batch_file")
    if uploaded is None:
        return
    try:
        raw = pd.read_csv(uploaded)
        if len(raw) > BATCH_ROW_LIMIT:
            st.warning(f"Only the first {BATCH_ROW_LIMIT:,} rows were used.")
            raw = raw.head(BATCH_ROW_LIMIT)
        raw = raw.drop(columns=["log_price", "is_high_value", "price_eur"], errors="ignore")
        with st.spinner(f"Estimating {len(raw):,} cars..."):
            filled = cl.fill_batch_defaults(raw, REF)
            out = cl.predict_all(bundle, cl.build_features(filled))
    except Exception as exc:  # show a readable message instead of a stack trace
        st.error(f"Could not process that file: {exc}")
        return

    result = raw.copy()
    for i, name in enumerate(out["names"]):
        result[f"{name}_estimate_gbp"] = out["per_model"][:, i].round(0)
    result["estimated_price_gbp"] = out["mean_price"].round(0)
    result["price_band"] = [out["band_names"][i] for i in out["band_index"]]
    result["band_confidence_pct"] = (out["band_probs"].max(axis=1) * 100).round(1)
    actual_col = next((c for c in ("price_gbp", "asking_price_gbp", "asking_price") if c in raw.columns), None)
    if actual_col:
        actual = pd.to_numeric(raw[actual_col], errors="coerce")
        result["difference_pct"] = ((actual - result["estimated_price_gbp"]) / result["estimated_price_gbp"] * 100).round(1)

    st.success(f"Estimated {len(result):,} cars.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Cars priced", f"{len(result):,}")
    k2.metric("Median estimate", gbp(float(result["estimated_price_gbp"].median())))
    if actual_col:
        within = float((result["difference_pct"].abs() <= 10).mean() * 100)
        k3.metric(f"Within 10% of {actual_col}", f"{within:.0f}%")
    else:
        k3.metric("Highest estimate", gbp(float(result["estimated_price_gbp"].max())))

    st.dataframe(result, hide_index=True, use_container_width=True)
    hist = go.Figure(go.Histogram(x=result["estimated_price_gbp"].clip(upper=result["estimated_price_gbp"].quantile(0.99)),
                                  nbinsx=40, marker_color=ACCENT))
    hist.update_xaxes(title="Estimated price", tickprefix="£", tickformat=",")
    hist.update_yaxes(title="Cars")
    st.plotly_chart(style(hist, 300, "Distribution of estimates"), use_container_width=True)
    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    st.download_button("⬇ Download results (CSV)", buffer.getvalue().encode("utf-8"), "car_price_estimates.csv", "text/csv")


def tab_about(bundle: dict) -> None:
    st.markdown("### How the estimate is made")
    st.write(
        "The app loads `ensemble_used_car_price.joblib`, which holds **three separately trained blends**. Each blend mixes a "
        "linear model (Lasso or Ridge) with a tree model and then adds a Gradient Boosting correction in pounds. "
        "The final price is the **average of the three**, and the price band comes from a **soft vote** over how likely each "
        "model thinks every band is."
    )
    rows = []
    for name, mb in bundle["models"].items():
        weights = {k: v for k, v in mb.items() if k.endswith("_weight")}
        rows.append({"Model": cl.MODEL_LABELS.get(name, name),
                     "Blend weights": ", ".join(f"{k.replace('_weight', '')} {v:.0%}" for k, v in weights.items())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### Test results (88,120 unseen listings, from `ensembling.ipynb`)")
    st.dataframe(pd.DataFrame([
        ["Decision Tree + Lasso + GB", 75.51, 1416, 0.885, 94.84],
        ["Random Forest + Lasso + GB", 75.01, 1447, 0.890, 94.75],
        ["Extra Trees + Ridge + GB", 75.07, 1476, 0.883, 94.73],
        ["Average of the three (used here)", 75.59, 1414, 0.890, 94.86],
    ], columns=["Model", "Within 10% (%)", "MAE (£)", "R²", "Band accuracy (%)"]),
        hide_index=True, use_container_width=True)

    st.markdown("#### Good to know")
    st.markdown(
        "- The model predicts **asking prices on UK adverts**, not what a car finally sells for.\n"
        "- It does not see condition, service history, options or accident damage.\n"
        "- Ages are counted from **2026**, the reference year used when the data was cleaned.\n"
        "- Very rare, classic or six-figure cars are estimated less reliably than everyday cars."
    )
    import sklearn
    st.caption(f"scikit-learn in this app: {sklearn.__version__} (the model was saved with {EXPECTED_SKLEARN}).")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🚗 UK Used Car Price Estimator")
    st.write("Describe a car in the sidebar and an ensemble of three machine-learning models estimates its asking price.")

    import sklearn
    if sklearn.__version__ != EXPECTED_SKLEARN:
        st.warning(f"This app has scikit-learn {sklearn.__version__} installed, but the model was saved with "
                   f"{EXPECTED_SKLEARN}. If loading fails or predictions look odd, run "
                   f"`pip install -r requirements.txt` in a fresh environment.")
    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(f"The model could not be loaded: {exc}")
        st.info("Check that `ensemble_used_car_price.joblib` sits next to `app.py` and that the packages in "
                "`requirements.txt` (especially scikit-learn 1.5.2 and numpy 1.26) are installed.")
        st.stop()

    init_state()
    sidebar_form()
    car = current_car()

    try:
        res = cl.predict_all(bundle, cl.build_features(car))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()
    price = float(res["mean_price"][0])

    tabs = st.tabs(["💷 Estimate", "🎛 What-if", "📊 Market view", "📁 Batch", "ℹ️ About"])
    with tabs[0]:
        tab_estimate(res, car)
    with tabs[1]:
        tab_whatif(bundle, car, price)
    with tabs[2]:
        tab_market(car, price)
    with tabs[3]:
        tab_batch(bundle)
    with tabs[4]:
        tab_about(bundle)


main()

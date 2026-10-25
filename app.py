
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Responsible Car Price Analytics",
    page_icon="🚗",
    layout="wide",
)


# =====================================================
# COMPACT STYLING
# =====================================================
st.markdown(
    """
    <style>
        header, footer {visibility: hidden;}

        .block-container {
            padding: 0.4rem 0.8rem !important;
            max-width: 100% !important;
        }

        h1 {
            font-size: 1.15rem !important;
            margin: 0.1rem 0 !important;
        }

        h2 {font-size: 0.95rem !important;}
        h3 {font-size: 0.82rem !important;}

        p, label, li {
            font-size: 0.72rem !important;
            line-height: 1.15 !important;
        }

        div[data-testid="stMetric"] {
            padding: 0.2rem 0.35rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 0.95rem !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.3rem !important;
        }

        section[data-testid="stSidebar"] {
            min-width: 190px;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.4rem !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# FILE PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = OUTPUT_DIR / "car_prices_cleaned.csv"
MODEL_PATH = OUTPUT_DIR / "random_forest_model.pkl"
PREPROCESSOR_PATH = OUTPUT_DIR / "preprocessor.pkl"


# =====================================================
# MODEL FEATURES
# =====================================================
FEATURES = [
    "symboling",
    "fueltype",
    "aspiration",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginetype",
    "cylindernumber",
    "enginesize",
    "fuelsystem",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "brand",
    "car_model",
    "power_to_weight",
    "avg_mpg",
]


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required = ["price"] + FEATURES
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    if data.empty:
        raise ValueError("The cleaned dataset is empty.")

    return data


# =====================================================
# LOAD MODEL AND PREPROCESSOR
# =====================================================
@st.cache_resource
def load_model_files():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


try:
    df_raw = load_data()
    model, preprocessor = load_model_files()

except Exception as exc:
    st.error(
        "Could not load the dataset, model or preprocessor. "
        "Check that all required files are in the outputs folder."
    )
    st.caption(f"Technical details: {exc}")
    st.stop()


# =====================================================
# NAVIGATION
# =====================================================
st.sidebar.title("🚗 Car Price Analytics")

page = st.sidebar.radio(
    "Select section",
    [
        "Project Overview",
        "Price Predictor",
        "EDA Insights",
        "Model Performance & Ethics",
    ],
)


# =====================================================
# SIDEBAR FILTERS
# =====================================================
df = df_raw.copy()

if page in ["Project Overview", "EDA Insights"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset filters")

    for column, label in [
        ("brand", "Brand"),
        ("fueltype", "Fuel type"),
        ("drivewheel", "Drive wheel"),
    ]:
        options = sorted(
            df_raw[column].dropna().unique().tolist(),
            key=str,
        )

        selected = st.sidebar.selectbox(
            label,
            ["All"] + options,
            key=f"filter_{column}",
        )

        if selected != "All":
            df = df[df[column] == selected]


# =====================================================
# SECTION 1: PROJECT OVERVIEW
# =====================================================
if page == "Project Overview":

    st.title("🚗 Car Price Analysis — Overview")

    if df.empty:
        st.warning("No cars match the selected filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Cars", f"{len(df):,}")
    c2.metric("Average price", f"${df['price'].mean():,.0f}")
    c3.metric("Median price", f"${df['price'].median():,.0f}")
    c4.metric("Maximum price", f"${df['price'].max():,.0f}")

    left, right = st.columns(2, gap="small")

    with left:
        brand_prices = (
            df.groupby("brand", as_index=False)["price"]
            .mean()
            .sort_values("price", ascending=False)
            .head(8)
        )

        fig = px.bar(
            brand_prices,
            x="brand",
            y="price",
            title="Average Price by Brand ($)",
            labels={
                "brand": "Brand",
                "price": "Average price ($)",
            },
            template="plotly_white",
        )

        fig.update_layout(
            height=225,
            margin=dict(l=25, r=8, t=35, b=25),
            font=dict(size=9),
            title_font_size=12,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        fig = px.histogram(
            df,
            x="price",
            nbins=25,
            title="Car Price Distribution ($)",
            labels={"price": "Price ($)"},
            template="plotly_white",
        )

        fig.update_layout(
            height=225,
            margin=dict(l=25, r=8, t=35, b=25),
            font=dict(size=9),
            title_font_size=12,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.subheader("Key findings")

    st.markdown(
        "- Car prices vary across brands and vehicle characteristics.\n"
        "- Engine size, horsepower and curb weight are useful variables "
        "to investigate when studying price differences.\n"
        "- These findings describe this dataset and may not represent "
        "the current used-car market."
    )


# =====================================================
# SECTION 2: PRICE PREDICTOR
# =====================================================
elif page == "Price Predictor":

    st.title("💡 Car Price Prediction")

    st.caption(
        "Educational estimates only. Predictions are not guaranteed "
        "market valuations."
    )

    # Use typical values from the cleaned dataset.
    defaults = {}

    for column in FEATURES:
        series = df_raw[column]

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            defaults[column] = series.median()
        else:
            modes = series.dropna().mode()
            defaults[column] = (
                modes.iloc[0] if not modes.empty else None
            )

    primary_features = [
        ("horsepower", "Horsepower"),
        ("enginesize", "Engine size"),
        ("curbweight", "Curb weight (lbs)"),
        ("avg_mpg", "Average MPG"),
        ("cylindernumber", "Cylinders"),
        ("drivewheel", "Drive wheel"),
    ]

    inputs = defaults.copy()
    left, right = st.columns(2, gap="small")

    for index, (column, label) in enumerate(primary_features):
        container = left if index % 2 == 0 else right
        series = df_raw[column]

        with container:
            if (
                pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series)
            ):
                low = float(series.min())
                high = float(series.max())
                default = float(defaults[column])

                if low == high:
                    inputs[column] = low
                    st.caption(f"{label}: {low:g}")
                else:
                    inputs[column] = st.slider(
                        label,
                        min_value=low,
                        max_value=high,
                        value=min(max(default, low), high),
                        key=f"main_{column}",
                    )

            else:
                options = sorted(
                    series.dropna().unique().tolist(),
                    key=str,
                )

                if options:
                    default = defaults[column]
                    default_index = (
                        options.index(default)
                        if default in options
                        else 0
                    )

                    inputs[column] = st.selectbox(
                        label,
                        options=options,
                        index=default_index,
                        key=f"main_{column}",
                    )

    with st.expander("Additional vehicle characteristics"):

        st.caption(
            "The remaining model features are included. "
            "Defaults come from the cleaned dataset."
        )

        primary_names = {item[0] for item in primary_features}
        remaining = [
            column for column in FEATURES
            if column not in primary_names
        ]

        extra_left, extra_right = st.columns(2, gap="small")

        for index, column in enumerate(remaining):
            container = extra_left if index % 2 == 0 else extra_right
            series = df_raw[column]
            label = column.replace("_", " ").title()

            with container:
                if (
                    pd.api.types.is_numeric_dtype(series)
                    and not pd.api.types.is_bool_dtype(series)
                ):
                    low = float(series.min())
                    high = float(series.max())
                    default = float(defaults[column])

                    if low == high:
                        inputs[column] = low
                        st.caption(f"{label}: {low:g}")
                    else:
                        inputs[column] = st.slider(
                            label,
                            min_value=low,
                            max_value=high,
                            value=min(max(default, low), high),
                            key=f"extra_{column}",
                        )

                else:
                    options = sorted(
                        series.dropna().unique().tolist(),
                        key=str,
                    )

                    if options:
                        default = defaults[column]
                        default_index = (
                            options.index(default)
                            if default in options
                            else 0
                        )

                        inputs[column] = st.selectbox(
                            label,
                            options=options,
                            index=default_index,
                            key=f"extra_{column}",
                        )

    if st.button(
        "Calculate Estimated Price",
        type="primary",
        use_container_width=True,
    ):

        input_data = pd.DataFrame(
            [[inputs[column] for column in FEATURES]],
            columns=FEATURES,
        )

        try:
            encoded_data = preprocessor.transform(input_data)
            prediction = float(model.predict(encoded_data)[0])

            if not np.isfinite(prediction):
                st.error(
                    "The model returned an invalid value. "
                    "No price estimate is available."
                )
            else:
                st.success(
                    f"Estimated Car Price: **${prediction:,.2f}**"
                )

        except Exception as exc:
            st.error(
                "Prediction failed. No price estimate was generated."
            )
            st.caption(
                "Check the training features, data types and fitted "
                "preprocessor. No fallback formula is used."
            )
            st.caption(f"Technical details: {exc}")


# =====================================================
# SECTION 3: COMPACT EDA INSIGHTS
# =====================================================
elif page == "EDA Insights":

    st.title("📊 Exploratory Data Analysis")

    if df.empty:
        st.warning("No cars match the selected filters.")
        st.stop()

    left, right = st.columns(2, gap="small")

    with left:
        fig = px.scatter(
            df,
            x="enginesize",
            y="price",
            color="drivewheel",
            title="Engine Size vs Price",
            labels={
                "enginesize": "Engine size",
                "price": "Price ($)",
                "drivewheel": "Drive wheel",
            },
            template="plotly_white",
        )

        fig.update_layout(
            height=195,
            margin=dict(l=30, r=5, t=30, b=25),
            font=dict(size=8),
            title_font_size=11,
            legend=dict(title_font_size=8, font_size=8),
        )

        fig.update_traces(marker=dict(size=5))

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        fig = px.scatter(
            df,
            x="horsepower",
            y="price",
            color="fueltype",
            title="Horsepower vs Price",
            labels={
                "horsepower": "Horsepower",
                "price": "Price ($)",
                "fueltype": "Fuel type",
            },
            template="plotly_white",
        )

        fig.update_layout(
            height=195,
            margin=dict(l=30, r=5, t=30, b=25),
            font=dict(size=8),
            title_font_size=11,
            legend=dict(title_font_size=8, font_size=8),
        )

        fig.update_traces(marker=dict(size=5))

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    left, right = st.columns(2, gap="small")

    with left:
        fig = px.box(
            df,
            x="carbody",
            y="price",
            color="carbody",
            title="Price by Body Style",
            labels={
                "carbody": "Body style",
                "price": "Price ($)",
            },
            template="plotly_white",
        )

        fig.update_layout(
            height=195,
            margin=dict(l=30, r=5, t=30, b=30),
            font=dict(size=8),
            title_font_size=11,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        fig = px.scatter(
            df,
            x="curbweight",
            y="price",
            color="cylindernumber",
            title="Curb Weight vs Price",
            labels={
                "curbweight": "Curb weight (lbs)",
                "price": "Price ($)",
                "cylindernumber": "Cylinders",
            },
            template="plotly_white",
        )

        fig.update_layout(
            height=195,
            margin=dict(l=30, r=5, t=30, b=25),
            font=dict(size=8),
            title_font_size=11,
            legend=dict(title_font_size=8, font_size=8),
        )

        fig.update_traces(marker=dict(size=5))

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )


# =====================================================
# SECTION 4: MODEL PERFORMANCE & ETHICS
# =====================================================
elif page == "Model Performance & Ethics":

    st.title("📈 Model Performance & Ethical Use")

    st.info(
        "The values below are previously reported results. Verify them "
        "against the final executed machine-learning notebook before submission."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Reported Test R²", "0.956")
    c2.metric("Reported Test MAE", "$1,355")
    c3.metric("Reported Test RMSE", "$1,863")

    left, right = st.columns(2, gap="small")

    with left:
        st.subheader("Model reliability")

        st.markdown(
            "- Accuracy can vary across individual vehicles.\n"
            "- Test performance does not guarantee accuracy for future "
            "market conditions or every vehicle group.\n"
            "- Use prediction-error analysis to investigate groups where "
            "errors are higher.\n"
            "- Predictions are estimates, not guaranteed valuations."
        )

    with right:
        st.subheader("Ethics and limitations")

        st.markdown(
            "- **Fairness:** Compare errors across vehicle groups before "
            "drawing conclusions about performance differences.\n"
            "- **Data provenance:** Document the dataset source and "
            "applicable reuse conditions.\n"
            "- **Transparency:** Explain the model's features and limitations.\n"
            "- **Privacy:** Make only claims supported by dataset documentation.\n"
            "- **Target leakage:** Confirm that target-derived features "
            "were excluded from model training."
        )

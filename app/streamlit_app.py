from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model" / "cycling_traffic_model.joblib"
IMAGE_DIR = ROOT / "pics"


st.set_page_config(
    page_title="Cycling traffic in Paris",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --blue:#087ca7; --navy:#073b4c; --mint:#e8f7f5; }
    .stApp { background: linear-gradient(180deg, #f7fbfc 0%, #ffffff 35%); }
    h1, h2, h3 { color: var(--navy); }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] * { color: white; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #d9e7eb; border-radius: 12px;
        padding: 14px; box-shadow: 0 4px 18px rgba(7,59,76,.06);
    }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #073b4c, #0b8fac);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0 0 .35rem 0; }
    .hero p { font-size: 1.08rem; margin: 0; opacity: .94; }
    .note { background:#e8f7f5; border-left:4px solid #0b8fac; padding:1rem; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    return json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_counters() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "counters.parquet")


@st.cache_data(show_spinner=False)
def load_patterns() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "traffic_patterns.parquet")


@st.cache_data(show_spinner=False)
def load_sites() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "site_totals.parquet")


@st.cache_data(show_spinner=False)
def load_daily_history() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "daily_history.parquet")


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    month = frame["month"].astype("float32")
    day_value = frame["day"].astype("float32")
    hour = frame["hour"].astype("float32")
    weekday = frame["weekday_index"].astype("float32")
    return pd.DataFrame(
        {
            "counter_code": frame["counter_code"].astype("float32"),
            "latitude": frame["latitude"].astype("float32"),
            "longitude": frame["longitude"].astype("float32"),
            "year": frame["year"].astype("float32"),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "day_sin": np.sin(2 * np.pi * day_value / 31),
            "day_cos": np.cos(2 * np.pi * day_value / 31),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "weekend": (weekday >= 5).astype("float32"),
        }
    )


summary = load_summary()
pages = [
    "Project overview",
    "Data exploration",
    "Traffic patterns",
    "Data preparation",
    "Model evaluation",
    "Predictions",
]

st.sidebar.title("Cycling traffic")
page = st.sidebar.radio("Explore the project", pages)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Source:** [Paris Open Data](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/)"
)
st.sidebar.caption("Data from March 2022 to May 2024")


if page == "Project overview":
    st.markdown(
        """
        <div class="hero">
          <h1>Cycling traffic in Paris</h1>
          <p>Explore when and where bicycle traffic is highest, then estimate hourly traffic for 70 counting sites.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Why this project?")
        st.write(
            "Paris uses permanent bicycle counters to understand how cycling evolves. "
            "This dashboard turns almost one million hourly observations into practical "
            "views of demand by location, month, weekday and hour."
        )
        st.write(
            "The prediction page estimates the expected number of bicycles for a selected "
            "counter and time using a compact machine-learning model."
        )
        st.markdown(
            '<div class="note">The app uses pre-aggregated files, so the original 1.33 GB CSV never needs to be loaded by Streamlit.</div>',
            unsafe_allow_html=True,
        )
    with right:
        image_path = IMAGE_DIR / "bike.jpg"
        if image_path.exists() and image_path.stat().st_size > 1000:
            st.image(str(image_path), caption="Cycling in Paris", width="stretch")

    st.subheader("Dataset at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hourly observations", f"{summary['rows']:,}")
    c2.metric("Counting sites", summary["counters"])
    c3.metric("Average per hour", f"{summary['mean_hourly_count']:.0f}")
    c4.metric("Median per hour", f"{summary['median_hourly_count']:.0f}")


elif page == "Data exploration":
    st.title("Data exploration")
    st.write(
        "The target is the hourly bicycle count. Its distribution is strongly right-skewed: "
        "most observations are moderate, with occasional very high peaks."
    )
    stats = pd.DataFrame(
        {
            "Statistic": ["Minimum", "First quartile", "Median", "Mean", "Third quartile", "Maximum"],
            "Hourly bicycles": [
                summary["min_hourly_count"],
                summary["q1_hourly_count"],
                summary["median_hourly_count"],
                round(summary["mean_hourly_count"], 1),
                summary["q3_hourly_count"],
                summary["max_hourly_count"],
            ],
        }
    )
    st.dataframe(stats, hide_index=True, width="stretch")
    fig = px.bar(
        stats,
        x="Statistic",
        y="Hourly bicycles",
        title="Distribution landmarks",
        color="Hourly bicycles",
        color_continuous_scale="Teal",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig)
    st.info(
        "The maximum is an exceptional peak and should not be interpreted as a typical hour. "
        "Median and quartiles better describe normal traffic."
    )


elif page == "Traffic patterns":
    st.title("Traffic patterns")
    patterns = load_patterns()
    years = sorted(patterns["year"].unique())
    selected_year = st.select_slider("Year", options=years, value=2023 if 2023 in years else years[-1])
    current = patterns.loc[patterns["year"] == selected_year].copy()
    current["day_type"] = np.where(current["weekday_index"] >= 5, "Weekend", "Weekday")

    monthly = current.groupby("month", as_index=False)["total_count"].sum()
    fig_month = px.bar(
        monthly,
        x="month",
        y="total_count",
        labels={"month": "Month", "total_count": "Total bicycles"},
        title=f"Recorded bicycle traffic by month — {selected_year}",
        color="total_count",
        color_continuous_scale="Teal",
    )
    fig_month.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_month)

    by_hour = (
        current.groupby(["day_type", "hour"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "mean_hourly_count": np.average(
                        group["mean_hourly_count"], weights=group["observations"]
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    fig_hour = px.line(
        by_hour,
        x="hour",
        y="mean_hourly_count",
        color="day_type",
        markers=True,
        labels={"hour": "Hour", "mean_hourly_count": "Average bicycles", "day_type": "Day type"},
        title="Average hourly profile",
    )
    st.plotly_chart(fig_hour)

    sites = load_sites()
    top = sites.head(10).sort_values("total_count")
    fig_sites = px.bar(
        top,
        x="total_count",
        y="counter_name",
        orientation="h",
        labels={"total_count": "Total bicycles", "counter_name": "Counting site"},
        title="Top 10 counting sites",
        color="total_count",
        color_continuous_scale="Teal",
    )
    fig_sites.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_sites)

    map_sites = pd.concat([sites.head(10), sites.tail(10)]).copy()
    map_sites["group"] = ["Top 10"] * 10 + ["Lowest 10"] * 10
    fig_map = px.scatter_map(
        map_sites,
        lat="latitude",
        lon="longitude",
        color="group",
        size="mean_count",
        hover_name="counter_name",
        zoom=10,
        height=540,
        title="Locations of the busiest and quietest sites",
    )
    fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0})
    st.plotly_chart(fig_map)


elif page == "Data preparation":
    st.title("Data preparation")
    st.write(
        "The original dataset contains more than 965,000 rows and a 1.33 GB source CSV. "
        "Loading it for every visitor would exceed the practical limits of a free hosted app."
    )
    steps = pd.DataFrame(
        {
            "Step": [
                "Select useful columns",
                "Validate dates and coordinates",
                "Aggregate historical views",
                "Encode cyclical time",
                "Train outside Streamlit",
                "Export compact artifacts",
            ],
            "Purpose": [
                "Remove URLs, image metadata and unused identifiers",
                "Keep only valid observations",
                "Avoid recomputing charts from one million rows",
                "Represent repeating hourly, weekly and monthly patterns",
                "Prevent expensive training on the web server",
                "Load only a few hundred kilobytes at runtime",
            ],
        }
    )
    st.dataframe(steps, hide_index=True, width="stretch")
    st.success(
        "The deployed data tables and model together are below 0.5 MB, excluding the cover image."
    )


elif page == "Model evaluation":
    st.title("Model evaluation")
    bundle = load_model_bundle()
    metrics = bundle["metrics"]
    st.write(
        "A compact histogram gradient boosting regressor predicts expected hourly traffic. "
        "It was trained on 2022–2023 and evaluated on the later 2024 period."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("R² on 2024", f"{metrics['r2']:.2f}")
    c2.metric("Mean absolute error", f"{metrics['mae']:.1f} bicycles")
    c3.metric("Model size", f"{MODEL_PATH.stat().st_size / 1024:.0f} KB")

    comparison = pd.DataFrame(
        {
            "Method": ["Compact ML model", "Naive historical mean"],
            "MAE": [metrics["mae"], metrics["baseline_mae"]],
            "RMSE": [metrics["rmse"], metrics["baseline_rmse"]],
        }
    ).melt(id_vars="Method", var_name="Metric", value_name="Error")
    fig = px.bar(
        comparison,
        x="Metric",
        y="Error",
        color="Method",
        barmode="group",
        title="Prediction error on unseen 2024 data — lower is better",
    )
    st.plotly_chart(fig)
    st.caption(
        "A temporal holdout is stricter than a random split because the model must predict a later period."
    )


elif page == "Predictions":
    st.title("Predict hourly bicycle traffic")
    counters = load_counters().copy()
    selected_name = st.selectbox("Counting site", counters["counter_name"].tolist())
    selected = counters.loc[counters["counter_name"] == selected_name].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Date", value=date.today())
    with col2:
        selected_time = st.time_input("Hour", value=time(8, 0), step=3600)

    bundle = load_model_bundle()
    if st.button("Estimate traffic", type="primary"):
        input_row = pd.DataFrame(
            {
                "counter_code": [selected["counter_code"]],
                "latitude": [selected["latitude"]],
                "longitude": [selected["longitude"]],
                "year": [selected_date.year],
                "month": [selected_date.month],
                "day": [selected_date.day],
                "hour": [selected_time.hour],
                "weekday_index": [selected_date.weekday()],
            }
        )
        prediction = max(float(bundle["model"].predict(feature_frame(input_row))[0]), 0)
        st.success(
            f"Expected traffic at {selected_name}: approximately **{prediction:.0f} bicycles** "
            f"between {selected_time.hour:02d}:00 and {(selected_time.hour + 1) % 24:02d}:00."
        )
        st.caption(
            f"Typical model error on the 2024 test period: about {bundle['metrics']['mae']:.0f} bicycles."
        )

    history = load_daily_history()
    site_history = history.loc[history["counter_code"] == selected["counter_code"]].copy()
    fig_history = px.line(
        site_history,
        x="date",
        y="daily_count",
        labels={"date": "Date", "daily_count": "Daily bicycles"},
        title=f"Historical daily traffic — {selected_name}",
    )
    st.plotly_chart(fig_history)

    location = pd.DataFrame(
        {
            "counter_name": [selected_name],
            "latitude": [selected["latitude"]],
            "longitude": [selected["longitude"]],
        }
    )
    fig_location = px.scatter_map(
        location,
        lat="latitude",
        lon="longitude",
        hover_name="counter_name",
        zoom=13,
        height=380,
    )
    fig_location.update_layout(map_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_location)

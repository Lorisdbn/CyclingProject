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

INK = "#172033"
MUTED = "#5F6C7B"
NAVY = "#0B1F33"
TEAL = "#0F766E"
BLUE = "#2563EB"
GRID = "#E6EBF1"
MONTHS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


st.set_page_config(
    page_title="Paris Cycling Analytics | Data Portfolio",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #5F6C7B;
        --navy: #0B1F33;
        --teal: #0F766E;
        --blue: #2563EB;
        --surface: #FFFFFF;
        --canvas: #F6F8FB;
        --line: #E3E9F0;
    }
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1320px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }
    header[data-testid="stHeader"] {
        background: rgba(246, 248, 251, .92);
        border-bottom: 1px solid rgba(227, 233, 240, .9);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--ink);
    }
    .stApp h1 { letter-spacing: -.035em; font-weight: 760; }
    .stApp h2 { letter-spacing: -.025em; font-weight: 720; }
    .stApp h3 { letter-spacing: -.015em; font-weight: 680; }
    [data-testid="stMarkdownContainer"] p { line-height: 1.65; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] a { color: #6EE7D8 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .34rem .5rem;
        border-radius: 8px;
        transition: background .15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,255,255,.07);
    }

    div[data-testid="stMetric"] {
        min-height: 112px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 26px rgba(11, 31, 51, .045);
    }
    [data-testid="stMetricLabel"] { font-size: .82rem; font-weight: 650; }
    [data-testid="stMetricValue"] { font-weight: 760; letter-spacing: -.03em; }
    [data-testid="stMetricDelta"] { font-weight: 650; }
    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border-color: #CDD6E1 !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--teal);
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        padding: .62rem 1.15rem;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"] p { color: #FFFFFF !important; }
    .stButton > button[kind="primary"]:hover { background: #0B5F59; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .eyebrow {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin: 0 0 .45rem 0;
    }
    .page-title {
        color: var(--ink);
        font-size: clamp(2rem, 3.6vw, 3.25rem);
        line-height: 1.06;
        letter-spacing: -.045em;
        margin: 0;
    }
    .page-deck {
        color: var(--muted);
        font-size: 1.06rem;
        line-height: 1.6;
        max-width: 850px;
        margin: .75rem 0 1.65rem 0;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.35rem;
        border-radius: 20px;
        color: white;
        background: var(--navy);
        box-shadow: 0 18px 45px rgba(11,31,51,.14);
        margin-bottom: 1.25rem;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 340px;
        height: 340px;
        right: -100px;
        top: -160px;
        border: 58px solid rgba(45, 212, 191, .16);
        border-radius: 50%;
    }
    .hero .kicker { color: #6EE7D8; font-size: .78rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    .hero h1 { color: white !important; font-size: clamp(2.25rem, 4.5vw, 4rem); line-height: 1.02; margin: .55rem 0 .8rem; max-width: 820px; }
    .hero p { color: #DCE7F1 !important; font-size: 1.12rem; line-height: 1.6; margin: 0; max-width: 780px; }
    .tag-row { display: flex; gap: .45rem; flex-wrap: wrap; margin-top: 1.3rem; }
    .tag { padding: .32rem .6rem; border-radius: 999px; background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.15); color: #F8FAFC; font-size: .76rem; font-weight: 650; }

    .insight {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        padding: 1rem 1.05rem;
        border-radius: 10px;
        color: var(--ink);
        line-height: 1.55;
        margin: .7rem 0 1rem;
    }
    .insight strong { color: var(--navy); }
    .skill-grid, .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: .4rem 0 1.3rem;
    }
    .skill-card, .pipeline-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 1rem;
        box-shadow: 0 7px 22px rgba(11,31,51,.035);
    }
    .skill-card b, .pipeline-card b { display: block; color: var(--ink); margin-bottom: .28rem; }
    .skill-card span, .pipeline-card span { color: var(--muted); font-size: .88rem; line-height: 1.5; }
    .step-number { color: var(--teal); font-size: .73rem; font-weight: 800; letter-spacing: .09em; margin-bottom: .45rem; }
    .result-card {
        background: var(--navy);
        color: white;
        border-radius: 15px;
        padding: 1.25rem 1.35rem;
        margin: .8rem 0 1.1rem;
    }
    .result-card small { color: #9FB3C8; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; }
    .result-card .value { color: white; font-size: 2.15rem; font-weight: 780; letter-spacing: -.04em; margin: .25rem 0; }
    .result-card p { color: #DCE7F1 !important; margin: 0; }
    .sidebar-brand { color: white; font-size: 1.42rem; font-weight: 780; letter-spacing: -.035em; margin-bottom: .15rem; }
    .sidebar-sub { color: #9FB3C8; font-size: .8rem; line-height: 1.45; margin-bottom: 1rem; }

    @media (max-width: 800px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.4rem; }
        .hero { padding: 1.55rem 1.35rem; border-radius: 15px; }
        .skill-grid, .pipeline-grid { grid-template-columns: 1fr; }
        .page-deck { font-size: .98rem; }
    }
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


def page_header(kicker: str, title: str, deck: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{kicker}</div><h1 class="page-title">{title}</h1>'
        f'<p class="page-deck">{deck}</p>',
        unsafe_allow_html=True,
    )


def chart_title(title: str, subtitle: str) -> str:
    return f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED};font-weight:400'>{subtitle}</span>"


def style_chart(fig, height: int = 430, legend: bool = False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, Segoe UI, Arial", "color": INK, "size": 13},
        title={"font": {"color": INK, "size": 18}, "x": 0, "xanchor": "left"},
        margin={"l": 28, "r": 24, "t": 105, "b": 35},
        hoverlabel={"bgcolor": NAVY, "font_color": "white", "bordercolor": NAVY},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.14, "x": 1, "xanchor": "right", "title_text": ""},
        showlegend=legend,
    )
    fig.update_xaxes(showline=True, linecolor=GRID, gridcolor=GRID, zeroline=False, title_standoff=14)
    fig.update_yaxes(showline=False, gridcolor=GRID, zeroline=False, title_standoff=14)
    return fig


def show_chart(fig, height: int = 430, legend: bool = False) -> None:
    st.plotly_chart(
        style_chart(fig, height=height, legend=legend),
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False, "responsive": True},
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

st.sidebar.markdown(
    '<div class="sidebar-brand">Paris Cycling Analytics</div>'
    '<div class="sidebar-sub">End-to-end data portfolio project</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio("Explore the project", pages)
st.sidebar.markdown("---")
st.sidebar.markdown("**Stack**")
st.sidebar.caption("Python · pandas · Plotly · scikit-learn · Streamlit")
st.sidebar.markdown(
    "**Source:** [Paris Open Data](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/)"
)
st.sidebar.caption("Coverage: 27 Mar 2022 – 19 May 2024")
st.sidebar.markdown(
    "[View source code ↗](https://github.com/Lorisdbn/cycling-traffic-paris-streamlit-py)"
)


if page == "Project overview":
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">Data science portfolio · Paris mobility</div>
          <h1>From 1.33 GB of raw data to a fast predictive app.</h1>
          <p>An end-to-end analysis of hourly bicycle traffic across 70 Paris counters — from data engineering and exploratory analysis to temporal validation and cloud deployment.</p>
          <div class="tag-row">
            <span class="tag">965k observations</span><span class="tag">Feature engineering</span>
            <span class="tag">Gradient boosting</span><span class="tag">Temporal holdout</span>
            <span class="tag">Cloud optimisation</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hourly observations", f"{summary['rows'] / 1_000:.0f}k", help="Clean hourly counter records")
    k2.metric("Counting sites", summary["counters"], help="Permanent counters across Paris")
    k3.metric("Median hourly traffic", f"{summary['median_hourly_count']:.0f}", help="Bicycles per counter-hour")
    k4.metric("Model R²", f"{summary['model_metrics_2024']['r2']:.2f}", help="Measured on unseen 2024 data")

    st.markdown("### The analytical challenge")
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.write(
            "Paris publishes rich hourly bicycle-counter data, but the raw source is too large "
            "for a responsive free-tier web application. The project therefore separates heavy "
            "offline preparation from lightweight online exploration and prediction."
        )
        st.markdown(
            '<div class="insight"><strong>Design decision.</strong> Pre-aggregated analytical tables and a compact trained model reduce the deployed assets to under 0.5 MB, while preserving the views needed for analysis.</div>',
            unsafe_allow_html=True,
        )
    with right:
        image_path = IMAGE_DIR / "bike.jpg"
        if image_path.exists() and image_path.stat().st_size > 1000:
            st.image(str(image_path), caption="Paris mobility context", width="stretch")

    st.markdown("### Skills demonstrated")
    st.markdown(
        """
        <div class="skill-grid">
          <div class="skill-card"><b>Data engineering</b><span>Schema selection, validation, aggregation and memory-aware Parquet artifacts.</span></div>
          <div class="skill-card"><b>Exploratory analysis</b><span>Distribution, seasonality, commuting profiles, rankings and spatial patterns.</span></div>
          <div class="skill-card"><b>Machine learning</b><span>Cyclical feature engineering, gradient boosting and a strict temporal holdout.</span></div>
          <div class="skill-card"><b>Model evaluation</b><span>MAE, RMSE and R² compared against a transparent historical-mean baseline.</span></div>
          <div class="skill-card"><b>Data storytelling</b><span>Summary-first dashboard design with concise interpretation beside every visual.</span></div>
          <div class="skill-card"><b>Production thinking</b><span>Caching, bounded assets, dependency control, deployment and automated monitoring.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Data exploration":
    page_header(
        "01 · Understand the target",
        "Hourly bicycle traffic is highly skewed.",
        "A typical counter-hour records dozens of bicycles, while rare extreme peaks reach several thousand. Robust summaries are therefore more informative than the mean alone.",
    )

    q1, median, mean, q3 = (
        summary["q1_hourly_count"],
        summary["median_hourly_count"],
        summary["mean_hourly_count"],
        summary["q3_hourly_count"],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("First quartile", f"{q1:.0f}", help="25% of observations are at or below this value")
    c2.metric("Median", f"{median:.0f}", help="The central observed counter-hour")
    c3.metric("Mean", f"{mean:.0f}", delta=f"+{mean - median:.0f} vs median", delta_color="off")
    c4.metric("Third quartile", f"{q3:.0f}", help="75% of observations are at or below this value")

    typical = pd.DataFrame(
        {
            "Statistic": ["First quartile", "Median", "Mean", "Third quartile"],
            "Hourly bicycles": [q1, median, mean, q3],
        }
    )
    fig = px.bar(
        typical,
        x="Statistic",
        y="Hourly bicycles",
        text="Hourly bicycles",
        labels={"Hourly bicycles": "Bicycles per counter-hour"},
        title=chart_title(
            "Typical distribution landmarks",
            "All 965,173 hourly observations · extreme maximum shown separately",
        ),
    )
    fig.update_traces(marker_color=["#B9DEDA", TEAL, BLUE, "#89B8D8"], texttemplate="%{text:.0f}", textposition="outside")
    fig.update_yaxes(range=[0, q3 * 1.28])
    show_chart(fig, height=410)

    st.markdown(
        f'<div class="insight"><strong>Outlier context.</strong> The maximum is {summary["max_hourly_count"]:,.0f} bicycles — {summary["max_hourly_count"] / median:.0f}× the median. This confirms a long right tail and motivates robust metrics plus a tree-based model.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View the complete descriptive statistics"):
        stats = pd.DataFrame(
            {
                "Statistic": ["Minimum", "First quartile", "Median", "Mean", "Third quartile", "Maximum"],
                "Hourly bicycles": [summary["min_hourly_count"], q1, median, round(mean, 1), q3, summary["max_hourly_count"]],
            }
        )
        st.dataframe(stats, hide_index=True, width="stretch")


elif page == "Traffic patterns":
    page_header(
        "02 · Explore demand",
        "Seasonality, commuting hours and place shape demand.",
        "Use the year control to inspect aggregate volume, then compare weekday and weekend profiles and the spatial concentration of traffic.",
    )
    patterns = load_patterns()
    years = sorted(patterns["year"].unique())
    selected_year = st.select_slider("Analysis year", options=years, value=2023 if 2023 in years else years[-1])
    current = patterns.loc[patterns["year"] == selected_year].copy()
    current["day_type"] = np.where(current["weekday_index"] >= 5, "Weekend", "Weekday")

    monthly = current.groupby("month", as_index=False)["total_count"].sum()
    monthly["month_name"] = monthly["month"].map(MONTHS)
    peak_month_row = monthly.loc[monthly["total_count"].idxmax()]

    by_hour = (
        current.assign(weighted=lambda x: x["mean_hourly_count"] * x["observations"])
        .groupby(["day_type", "hour"], as_index=False)
        .agg(weighted=("weighted", "sum"), observations=("observations", "sum"))
    )
    by_hour["mean_hourly_count"] = by_hour["weighted"] / by_hour["observations"]
    weekday_rows = by_hour.loc[by_hour["day_type"] == "Weekday"]
    weekday_peak = weekday_rows.loc[weekday_rows["mean_hourly_count"].idxmax()]
    day_averages = by_hour.groupby("day_type").apply(
        lambda x: np.average(x["mean_hourly_count"], weights=x["observations"]), include_groups=False
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recorded volume", f"{monthly['total_count'].sum() / 1_000_000:.1f}M", help=f"Sum of observations recorded in {selected_year}")
    m2.metric("Peak month", str(peak_month_row["month_name"]), help=f"{peak_month_row['total_count'] / 1_000_000:.1f}M recorded bicycles")
    m3.metric("Weekday peak", f"{int(weekday_peak['hour']):02d}:00", help=f"{weekday_peak['mean_hourly_count']:.0f} average bicycles")
    m4.metric("Weekday / weekend", f"{day_averages['Weekday'] / day_averages['Weekend']:.1f}×", help="Average hourly weekday traffic divided by weekend traffic")

    fig_month = px.bar(
        monthly,
        x="month_name",
        y="total_count",
        category_orders={"month_name": [MONTHS[m] for m in monthly["month"]]},
        labels={"month_name": "Month", "total_count": "Recorded bicycles"},
        title=chart_title("Monthly recorded traffic", f"{selected_year} · totals reflect available observations in the source"),
    )
    fig_month.update_traces(marker_color=TEAL, hovertemplate="%{x}: %{y:,.0f}<extra></extra>")
    show_chart(fig_month)

    fig_hour = px.line(
        by_hour,
        x="hour",
        y="mean_hourly_count",
        color="day_type",
        markers=True,
        color_discrete_map={"Weekday": TEAL, "Weekend": BLUE},
        labels={"hour": "Hour of day", "mean_hourly_count": "Average bicycles", "day_type": "Day type"},
        title=chart_title("Average hourly profile", f"{selected_year} · weighted across all available counter observations"),
    )
    fig_hour.update_traces(line={"width": 3}, marker={"size": 6})
    fig_hour.update_xaxes(dtick=2)
    show_chart(fig_hour, legend=True)
    st.markdown(
        '<div class="insight"><strong>Demand signature.</strong> Weekdays show two sharp commuting peaks, while weekend traffic follows a flatter leisure-oriented profile.</div>',
        unsafe_allow_html=True,
    )

    sites = load_sites()
    top = sites.head(8).sort_values("total_count")
    fig_sites = px.bar(
        top,
        x="total_count",
        y="counter_name",
        orientation="h",
        text="total_count",
        labels={"total_count": "Recorded bicycles", "counter_name": ""},
        title=chart_title("Highest-volume counting sites", "Top 8 sites across the full source period · 27 Mar 2022 – 19 May 2024"),
    )
    fig_sites.update_traces(marker_color=TEAL, texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
    fig_sites.update_layout(margin={"l": 20, "r": 70, "t": 92, "b": 35})
    show_chart(fig_sites, height=470)

    map_sites = pd.concat([sites.head(8), sites.tail(8)]).copy()
    map_sites["Volume group"] = ["Highest 8"] * 8 + ["Lowest 8"] * 8
    fig_map = px.scatter_map(
        map_sites,
        lat="latitude",
        lon="longitude",
        color="Volume group",
        color_discrete_map={"Highest 8": TEAL, "Lowest 8": BLUE},
        size="mean_count",
        hover_name="counter_name",
        hover_data={"mean_count": ":.1f", "latitude": False, "longitude": False},
        zoom=10,
        height=520,
        title=chart_title("Spatial contrast in recorded traffic", "Eight highest- and eight lowest-volume counters across the full source period"),
    )
    fig_map.update_layout(
        map_style="carto-positron",
        paper_bgcolor="#FFFFFF",
        font={"family": "Inter, Segoe UI, Arial", "color": INK},
        title={"font": {"color": INK, "size": 18}, "x": 0, "xanchor": "left"},
        margin={"r": 0, "t": 92, "l": 0, "b": 0},
        legend={"orientation": "h", "y": 1.01},
    )
    st.plotly_chart(fig_map, use_container_width=True, theme=None, config={"displayModeBar": False})


elif page == "Data preparation":
    page_header(
        "03 · Engineer for production",
        "Heavy processing happens once — not on every page load.",
        "The raw source is cleaned, validated, aggregated and transformed offline. Streamlit receives only the compact artifacts required for exploration and inference.",
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw CSV", "1.33 GB")
    d2.metric("Raw observations", "965k")
    d3.metric("Deployed artifacts", "< 0.5 MB")
    d4.metric("Size reduction", "> 99.9%")

    st.markdown("### Reproducible pipeline")
    st.markdown(
        """
        <div class="pipeline-grid">
          <div class="pipeline-card"><div class="step-number">STEP 01</div><b>Select</b><span>Keep analytical fields; remove URLs, image metadata and unused identifiers.</span></div>
          <div class="pipeline-card"><div class="step-number">STEP 02</div><b>Validate</b><span>Parse dates, check coordinates and retain valid counter observations.</span></div>
          <div class="pipeline-card"><div class="step-number">STEP 03</div><b>Aggregate</b><span>Build daily, seasonal and site-level analytical tables in Parquet.</span></div>
          <div class="pipeline-card"><div class="step-number">STEP 04</div><b>Engineer</b><span>Encode hour, weekday, day and month as cyclical sine/cosine features.</span></div>
          <div class="pipeline-card"><div class="step-number">STEP 05</div><b>Train</b><span>Fit and validate the model offline using an out-of-time test period.</span></div>
          <div class="pipeline-card"><div class="step-number">STEP 06</div><b>Deploy</b><span>Load cached tables and a compact model for fast, repeatable inference.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="insight"><strong>Production trade-off.</strong> The app preserves analytical depth without loading the 1.33 GB source or retraining the model on the web server.</div>',
        unsafe_allow_html=True,
    )


elif page == "Model evaluation":
    page_header(
        "04 · Validate honestly",
        "The model is tested on the future, not a random sample.",
        "A compact histogram gradient boosting regressor is trained on 2022–2023 and evaluated on unseen 2024 observations. This temporal split better reflects real deployment conditions.",
    )
    bundle = load_model_bundle()
    metrics = bundle["metrics"]
    mae_gain = 1 - metrics["mae"] / metrics["baseline_mae"]
    rmse_gain = 1 - metrics["rmse"] / metrics["baseline_rmse"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² on unseen 2024", f"{metrics['r2']:.2f}", help="Share of test-set variance explained")
    c2.metric("MAE · bicycles / hour", f"{metrics['mae']:.1f}", help="Typical absolute prediction error")
    c3.metric("MAE improvement", f"−{mae_gain:.0%}", help="Relative error reduction versus historical-mean baseline")
    c4.metric("Model size", f"{MODEL_PATH.stat().st_size / 1024:.0f} KB", help="Small enough for fast cloud loading")

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
        text="Error",
        color_discrete_map={"Compact ML model": TEAL, "Naive historical mean": "#AAB6C3"},
        labels={"Error": "Error (bicycles per hour)"},
        title=chart_title("Prediction error by method", "Unseen 2024 observations · lower is better"),
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_yaxes(range=[0, comparison["Error"].max() * 1.23])
    show_chart(fig, height=450, legend=True)

    st.markdown(
        f'<div class="insight"><strong>Benchmark result.</strong> The compact model reduces MAE by {mae_gain:.0%} and RMSE by {rmse_gain:.0%} versus the historical-mean baseline, while remaining small enough for a free cloud deployment.</div>',
        unsafe_allow_html=True,
    )
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("#### Evaluation design")
        st.write(f"Training observations: **{metrics['train_rows']:,}**  ")
        st.write(f"Temporal test observations: **{metrics['test_rows']:,}**")
    with e2:
        st.markdown("#### Model inputs")
        st.write("Counter identity and coordinates, year, cyclical month/day/hour/weekday features, and a weekend indicator.")


elif page == "Predictions":
    page_header(
        "05 · Operationalise the model",
        "Estimate hourly traffic for any counter and date.",
        "Select a counting site and time. The same feature pipeline used during training is applied before the compact model produces an estimate.",
    )
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
        st.session_state["last_prediction"] = {
            "value": prediction,
            "site": selected_name,
            "date": selected_date,
            "hour": selected_time.hour,
        }

    prediction_result = st.session_state.get("last_prediction")
    if prediction_result and prediction_result["site"] == selected_name:
        hour = prediction_result["hour"]
        st.markdown(
            f'<div class="result-card"><small>Expected hourly traffic</small><div class="value">{prediction_result["value"]:.0f} bicycles</div><p>{selected_name} · {prediction_result["date"]:%d %b %Y} · {hour:02d}:00–{(hour + 1) % 24:02d}:00</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Typical absolute error on the 2024 test period: approximately {bundle['metrics']['mae']:.0f} bicycles per hour.")

    history = load_daily_history()
    site_history = history.loc[history["counter_code"] == selected["counter_code"]].copy()
    fig_history = px.line(
        site_history,
        x="date",
        y="daily_count",
        labels={"date": "Date", "daily_count": "Daily bicycles"},
        title=chart_title("Historical daily traffic", f"{selected_name} · recorded daily totals"),
    )
    fig_history.update_traces(line={"color": TEAL, "width": 2.2}, hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} bicycles<extra></extra>")
    show_chart(fig_history, height=430)

    location = pd.DataFrame(
        {"counter_name": [selected_name], "latitude": [selected["latitude"]], "longitude": [selected["longitude"]]}
    )
    fig_location = px.scatter_map(
        location,
        lat="latitude",
        lon="longitude",
        hover_name="counter_name",
        zoom=13,
        height=390,
        title=chart_title("Counter location", selected_name),
    )
    fig_location.update_traces(marker={"color": TEAL, "size": 18})
    fig_location.update_layout(
        map_style="carto-positron",
        paper_bgcolor="#FFFFFF",
        font={"family": "Inter, Segoe UI, Arial", "color": INK},
        title={"font": {"color": INK, "size": 18}, "x": 0, "xanchor": "left"},
        margin={"r": 0, "t": 90, "l": 0, "b": 0},
    )
    st.plotly_chart(fig_location, use_container_width=True, theme=None, config={"displayModeBar": False})

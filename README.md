# Paris Cycling Analytics

An end-to-end data science portfolio project that turns **965,173 hourly bicycle-counter observations** into an interactive analytical dashboard and a lightweight prediction service.

**[Open the live Streamlit application](https://cycling-traffic-paris.streamlit.app/)**

## Business question

When and where is bicycle demand highest in Paris, and can hourly traffic be estimated for a selected counter and future time?

The application moves from descriptive analysis to operational prediction:

- Explore the skewed target distribution and robust summary statistics.
- Compare monthly seasonality and weekday/weekend hourly profiles.
- Rank high-volume counting sites and inspect spatial patterns.
- Explain the production data pipeline and feature engineering.
- Evaluate the model on an unseen future period.
- Generate an hourly prediction for any of 70 counting sites.

## Results

| Metric | Result |
| --- | ---: |
| Temporal test period | 2024 |
| Test observations | 325,672 |
| R² | 0.67 |
| Mean absolute error | 29.6 bicycles/hour |
| MAE reduction vs historical-mean baseline | 57% |
| Model artifact size | 335 KB |

The model is a compact histogram gradient boosting regressor trained on 2022–2023 data. A temporal holdout is used instead of a random split so evaluation better reflects a real deployment scenario.

## Production architecture

The original CSV is **1.33 GB**, which is not practical to load for every visitor on a free hosted app. Heavy work is performed offline:

1. Select and validate analytical fields.
2. Build compact daily, seasonal and site-level Parquet tables.
3. Engineer cyclical time features for hour, weekday, day and month.
4. Train and evaluate the model outside Streamlit.
5. Deploy less than 0.5 MB of analytical tables and model artifacts.

This separation keeps the application responsive while preserving the most useful analytical views.

## Technology

Python · pandas · NumPy · Plotly · scikit-learn · PyArrow · Streamlit · GitHub Actions

## Data source

[Paris Open Data — bicycle counter records](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/)

Data coverage: 27 March 2022 to 19 May 2024.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

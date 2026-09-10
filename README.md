# Cycling traffic in Paris

An interactive Streamlit dashboard for exploring Paris bicycle-counter data and estimating hourly traffic at 70 counting sites.

## Live application

Deploy `app/streamlit_app.py` on Streamlit Community Cloud.

## Cloud-ready architecture

The original CSV is 1.33 GB and is intentionally not loaded by the hosted app. The repository contains compact, precomputed Parquet tables and a 340 KB machine-learning model instead. This keeps startup fast and memory usage low.

The model is a histogram gradient boosting regressor trained on 2022–2023 data and evaluated on the later 2024 period. Its temporal holdout score is R² 0.67 with a mean absolute error of about 30 bicycles per hour.

## Data source

[Paris Open Data — bicycle counter records](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/)

The original analysis was completed during a data-analysis bootcamp in 2024.

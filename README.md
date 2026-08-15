# Urban Mobility Demand Forecasting Platform

An industry oriented machine learning project that forecasts hourly taxi pickup demand for each New York City taxi zone. It uses official NYC Taxi and Limousine Commission trip records and evaluates models with chronological splits that reproduce real deployment conditions.

## Business objective

Predict the number of taxi pickups in every zone for future hours so that an operations team can identify demand hotspots, allocate supply, and understand peak period risk.

## First milestone

1. Download official monthly Yellow Taxi Parquet files.
2. Aggregate individual trips into zone and hour demand.
3. Build calendar, lag, and rolling demand features without temporal leakage.
4. Compare same hour last week against a global LightGBM model.
5. Evaluate overall demand, peak demand, and multiple forecast periods.

## Verified first benchmark

The verified three month run processes 9.55 million trips into 572,208 zone hour observations. LightGBM improves test RMSE at one and six hour horizons, while the weekly baseline is slightly stronger at twenty four hours. An eighty percent probabilistic interval achieves 81.09% empirical coverage. These results are documented in [`RESULTS.md`](RESULTS.md) and visualized in the Streamlit dashboard.

## Data

The official TLC trip records contain pickup and dropoff timestamps, taxi zone identifiers, passenger counts, distances, fares, and payment information. Raw files are excluded from Git because each month can contain millions of trips.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download --year 2024 --months 1 2 3
python -m src.prepare
python -m src.train
streamlit run app.py
```

Start the prediction API after training:

```bash
uvicorn src.api:app --reload
```

Interactive API documentation is then available at `http://127.0.0.1:8000/docs`.

Run feature drift monitoring against the latest seven days:

```bash
python -m src.monitor
```

## Leakage controls

All lag and rolling features are shifted before aggregation. Validation and test periods are later than the training period. The test period is never used for feature selection or model tuning.

## Planned product layer

The current product layer includes multi horizon point forecasts, one hour probabilistic forecasts, a FastAPI inference service, PSI feature drift monitoring, and an operations dashboard. Future milestones add weather, holidays, geographic zone maps, experiment tracking, and automated scheduled retraining.

## API contract

The `POST /forecast` endpoint validates zone, calendar, lag, and rolling demand features. It supports one, six, and twenty four hour point forecasts. The one hour response also returns P10, P50, and P90 predictions. The `GET /health` endpoint reports model availability for every supported horizon.

## Monitoring

Training stores a reference distribution for five historical demand features. The monitoring job compares recent data with that reference using Population Stability Index. Values below 0.10 are healthy, values from 0.10 to 0.25 produce a warning, and values above 0.25 produce an alert.

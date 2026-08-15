# Urban Mobility Demand Forecasting Platform

An end to end machine learning system that predicts hourly taxi pickup demand for every New York City taxi zone. It transforms official trip records into operational forecasts, uncertainty estimates, an inference API, a monitoring report, and an interactive dashboard.

<p><strong><a href="https://urban&#45;mobility&#45;demand&#45;forecasting&#45;xhzfr7pczc7u9jkmvgphri.streamlit.app/">Open the live dashboard</a></strong></p>

## Why this project exists

Urban mobility platforms must position limited supply before demand appears. This project answers a practical question: how many pickups should an operations team expect in each zone over the next one, six, and twenty four hours?

The output can support driver positioning, fleet allocation, hotspot detection, and peak period planning.

## What the system does

1. Downloads official monthly Yellow Taxi records from the NYC Taxi and Limousine Commission.
2. Validates timestamps and zone identifiers, then creates a complete hourly grid that preserves zero demand periods.
3. Builds calendar, lag, and rolling features using only information available at prediction time.
4. Trains global LightGBM models for one, six, and twenty four hour horizons.
5. Compares every model with the demand observed at the same hour one week earlier.
6. Produces P10, P50, and P90 forecasts for short term capacity planning.
7. Serves predictions through FastAPI and presents model performance through Streamlit.
8. Detects feature distribution changes with Population Stability Index monitoring.

## System architecture

```text
Official TLC records
        ↓
Validation and hourly aggregation
        ↓
Leakage safe feature pipeline
        ↓
Temporal validation and model training
        ↓
Point forecasts and probability intervals
        ↓
FastAPI service   Streamlit dashboard   PSI monitoring
```

## Verified results

The experiment processes 9.55 million trips from January through March 2024 into 572,208 zone and hour observations. The final fourteen days are kept as an untouched test period.

<table>
<tr><th>Horizon</th><th>Weekly baseline RMSE</th><th>LightGBM RMSE</th><th>Relative change</th><th>Selected approach</th></tr>
<tr><td>1 hour</td><td>12.82</td><td>11.17</td><td>12.85% better</td><td>LightGBM</td></tr>
<tr><td>6 hours</td><td>12.82</td><td>12.35</td><td>3.66% better</td><td>LightGBM</td></tr>
<tr><td>24 hours</td><td>12.82</td><td>12.94</td><td>0.99% worse</td><td>Weekly baseline</td></tr>
</table>

The eighty percent probability interval reaches 81.09% empirical coverage. The result also shows why model choice should depend on forecast horizon: recent demand signals help short term forecasts, while weekly seasonality remains more reliable at twenty four hours.

Detailed metrics and interpretation are available in [`RESULTS.md`](RESULTS.md).

## Product surfaces

The Streamlit application presents headline metrics, horizon comparisons, calibrated intervals, and zone level error analysis.

The FastAPI service exposes a health endpoint and a forecast endpoint. One hour responses include point and probability forecasts. Six and twenty four hour responses provide point forecasts.

The monitoring job compares recent feature distributions with the training reference profile. PSI below 0.10 is healthy, values from 0.10 through 0.25 trigger a warning, and values above 0.25 trigger an alert.

## Run locally

Create and activate a Python 3.11 environment, then run:

```text
make install
make test
make dashboard
```

The repository contains the trained model artifacts and evaluation samples required by the dashboard and API. To reproduce the complete experiment from official source data, run:

```text
make pipeline
```

Start the prediction service with:

```text
make api
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

Run the latest seven day drift check with:

```text
make monitor
```

## Run with Docker

```text
make container
```

The included container starts the Streamlit dashboard on port 8501.

## Reliability choices

1. Rolling statistics are shifted before aggregation, preventing the current target from entering its own features.
2. Validation and test windows occur strictly after training data.
3. The test period is not used for feature selection or tuning.
4. Every complex model is evaluated against a strong seasonal baseline.
5. Automated tests cover zero demand aggregation, leakage controls, temporal ordering, horizon alignment, drift alerts, input validation, and inference.
6. Continuous integration runs the complete test suite for every change to the main branch.

## Repository guide

<table>
<tr><th>Path</th><th>Responsibility</th></tr>
<tr><td><code>src/download.py</code></td><td>Official data acquisition</td></tr>
<tr><td><code>src/prepare.py</code></td><td>Validation and hourly aggregation</td></tr>
<tr><td><code>src/features.py</code></td><td>Leakage safe feature generation</td></tr>
<tr><td><code>src/train.py</code></td><td>Training, evaluation, and artifact creation</td></tr>
<tr><td><code>src/api.py</code></td><td>Online inference service</td></tr>
<tr><td><code>src/monitor.py</code></td><td>PSI drift monitoring</td></tr>
<tr><td><code>app.py</code></td><td>Interactive operations dashboard</td></tr>
<tr><td><code>tests</code></td><td>Pipeline and API regression tests</td></tr>
<tr><td><code>artifacts</code></td><td>Models, metrics, samples, and monitoring output</td></tr>
</table>

## Data source

The project uses public Yellow Taxi trip records published by the NYC Taxi and Limousine Commission. Raw files remain outside version control because each monthly file contains millions of trips. Source details and retrieval instructions are recorded in [`data/SOURCES.md`](data/SOURCES.md).

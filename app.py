from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Urban Mobility Demand Forecasting", layout="wide")
st.title("Urban Mobility Demand Forecasting")
st.caption("Hourly taxi pickup forecasts across NYC taxi zones using official TLC trip records")

metrics_path = Path("artifacts/metrics.csv")
predictions_path = Path("artifacts/prediction_sample.csv")
probability_path = Path("artifacts/probabilistic_metrics.json")
interval_path = Path("artifacts/interval_sample.csv")
if not metrics_path.exists() or not predictions_path.exists():
    st.error("Run the preparation and training pipeline before opening the dashboard.")
    st.stop()

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path, parse_dates=["target_timestamp"])
probability = json.loads(probability_path.read_text()) if probability_path.exists() else None
test_metrics = metrics[metrics.split.eq("test")].copy()
one_hour = test_metrics[test_metrics.horizon_hours.eq(1)].set_index("model")
improvement = 1 - one_hour.loc["lightgbm", "rmse"] / one_hour.loc["same_hour_last_week", "rmse"]

one, two, three, four = st.columns(4)
one.metric("Raw trips", "9.55M", help="Official trip records across January to March 2024")
two.metric("Zone hour rows", "572,208")
three.metric("1 hour RMSE", f"{one_hour.loc['lightgbm', 'rmse']:.2f}")
four.metric("RMSE improvement", f"{improvement:.1%}", help="Compared with the same hour last week")

st.subheader("Performance by forecast horizon")
metric_choice = st.selectbox("Metric", ["rmse", "mae", "wmape", "peak_rmse"], index=0)
figure = px.bar(
    test_metrics, x="horizon_hours", y=metric_choice, color="model", barmode="group",
    labels={"horizon_hours": "Forecast horizon in hours", metric_choice: metric_choice.upper(), "model": "Model"},
)
st.plotly_chart(figure, width="stretch")
st.info("LightGBM leads at one and six hours. At twenty four hours, the weekly seasonal baseline is slightly stronger. Model selection should therefore be specific to each forecast horizon.")

if probability and interval_path.exists():
    st.subheader("Probabilistic forecast")
    coverage_one, coverage_two = st.columns(2)
    coverage_one.metric("Nominal interval", f"{probability['nominal_coverage']:.0%}")
    coverage_two.metric("Empirical coverage", f"{probability['empirical_coverage']:.1%}")
    interval = pd.read_csv(interval_path, parse_dates=["target_timestamp"])
    zone = st.selectbox("High demand zone", sorted(interval.zone_id.unique()))
    zone_interval = interval[interval.zone_id.eq(zone)].sort_values("target_timestamp")
    interval_figure = px.line(zone_interval, x="target_timestamp", y=["target", "p10", "p50", "p90"],
                              labels={"value": "Hourly pickups", "target_timestamp": "Target time", "variable": "Series"})
    st.plotly_chart(interval_figure, width="stretch")
    st.caption("P10 and P90 form an expected eighty percent prediction interval. Empirical coverage measures calibration on the untouched test period.")

st.subheader("Zone error explorer")
horizon = st.selectbox("Forecast horizon", sorted(predictions.horizon_hours.unique()))
selected = predictions[predictions.horizon_hours.eq(horizon)].copy()
selected["absolute_error"] = (selected.target - selected.prediction).abs()
selected["baseline_absolute_error"] = (selected.target - selected.baseline).abs()
zone_summary = selected.groupby("zone_id", as_index=False).agg(
    observations=("target", "size"),
    average_demand=("target", "mean"),
    model_mae=("absolute_error", "mean"),
    baseline_mae=("baseline_absolute_error", "mean"),
)
zone_summary["mae_improvement"] = zone_summary.baseline_mae - zone_summary.model_mae
left, right = st.columns([2, 1])
left.plotly_chart(
    px.scatter(zone_summary, x="average_demand", y="model_mae", color="mae_improvement",
               hover_data=["zone_id", "observations"], color_continuous_scale="RdYlGn",
               labels={"average_demand": "Average hourly pickups", "model_mae": "LightGBM MAE"}),
    width="stretch",
)
right.dataframe(zone_summary.sort_values("mae_improvement").head(15), width="stretch", hide_index=True)

st.subheader("Evaluation protocol")
st.markdown(
    "Features are calculated at the forecast origin. Targets are shifted one, six, or twenty four hours into the future. "
    "The last fourteen days form the test period, while the preceding fourteen days form validation. "
    "The baseline uses demand from the same target hour one week earlier."
)

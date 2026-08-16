from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.api import ForecastRequest, predict_request
from src.history import historical_features


st.set_page_config(page_title="Urban Mobility Demand Forecasting", layout="wide")
st.title("Urban Mobility Demand Forecasting")
st.caption("Hourly taxi pickup forecasts across NYC taxi zones using official TLC trip records")

metrics_path = Path("artifacts/metrics.csv")
predictions_path = Path("artifacts/prediction_sample.csv")
probability_path = Path("artifacts/probabilistic_metrics.json")
interval_path = Path("artifacts/interval_sample.csv")
metadata_path = Path("artifacts/metadata.json")
backtest_path = Path("artifacts/backtest_metrics.csv")
backtest_summary_path = Path("artifacts/backtest_summary.json")
history_path = Path("data/processed/hourly_zone_demand.parquet")
if not metrics_path.exists() or not predictions_path.exists():
    st.error("Run the preparation and training pipeline before opening the dashboard.")
    st.stop()

metrics = pd.read_csv(metrics_path)
predictions = pd.read_csv(predictions_path, parse_dates=["target_timestamp"])
probability = json.loads(probability_path.read_text()) if probability_path.exists() else None
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
backtest_summary = json.loads(backtest_summary_path.read_text()) if backtest_summary_path.exists() else {}
hourly_history = pd.read_parquet(history_path) if history_path.exists() else None
test_metrics = metrics[metrics.split.eq("test")].copy()
one_hour = test_metrics[test_metrics.horizon_hours.eq(1)].set_index("model")
improvement = 1 - one_hour.loc["lightgbm", "rmse"] / one_hour.loc["same_hour_last_week", "rmse"]
dataset = metadata.get("dataset", {})
trip_count = dataset.get("total_trips", 41_169_300)
zone_hour_rows = dataset.get("zone_hour_rows", 2_310_192)
backtest_improvement = backtest_summary.get("1", {}).get("improvement_mean", improvement)

one, two, three, four = st.columns(4)
one.metric("Raw trips", f"{trip_count / 1_000_000:.2f}M", help="Official trip records across all twelve months of 2024")
two.metric("Zone hour rows", f"{zone_hour_rows:,}")
three.metric("1 hour RMSE", f"{one_hour.loc['lightgbm', 'rmse']:.2f}")
four.metric("Backtest RMSE gain", f"{backtest_improvement:.1%}", help="Mean improvement across three future test windows")

st.subheader("Interactive demand forecast")
st.write(
    "Configure an operating scenario and run the trained model directly. "
    "Demand features are loaded automatically from the deployed TLC history snapshot."
)

history_mode = st.radio(
    "Demand input mode",
    ["Automatic from TLC history", "Custom scenario"],
    horizontal=True,
    help="Automatic mode reproduces a historical forecast. Custom mode supports what if analysis.",
)

with st.form("forecast_simulator"):
    setup_one, setup_two, setup_three = st.columns(3)
    horizon_input = setup_one.selectbox("Forecast horizon", [1, 6, 24], format_func=lambda value: f"Next {value} hour" if value == 1 else f"Next {value} hours")
    zone_input = setup_two.number_input("NYC taxi zone", min_value=1, max_value=265, value=161, step=1)
    forecast_origin = datetime.combine(
        setup_three.date_input("Forecast origin date", value=datetime(2024, 12, 29).date()),
        setup_three.time_input("Forecast origin time", value=datetime.strptime("18:00", "%H:%M").time()),
    )

    if history_mode == "Custom scenario":
        st.caption("Recent observed pickups in the selected zone")
        history_one, history_two, history_three, history_four, history_five = st.columns(5)
        lag_one = history_one.number_input("Previous hour", min_value=0.0, value=120.0, step=1.0)
        lag_twenty_four = history_two.number_input("Same hour yesterday", min_value=0.0, value=110.0, step=1.0)
        lag_week = history_three.number_input("Same hour last week", min_value=0.0, value=105.0, step=1.0)
        mean_day = history_four.number_input("Average over 24 hours", min_value=0.0, value=100.0, step=1.0)
        mean_week = history_five.number_input("Average over 7 days", min_value=0.0, value=95.0, step=1.0)
    else:
        st.caption("The five historical demand features will be calculated when the forecast runs.")
    submitted = st.form_submit_button("Generate forecast", type="primary", width="stretch")

if submitted:
    if history_mode == "Automatic from TLC history":
        if hourly_history is None:
            st.error("The deployed history snapshot is unavailable. Select Custom scenario to continue.")
            st.stop()
        try:
            features = historical_features(hourly_history, int(zone_input), forecast_origin)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        lag_one = features["lag_1"]
        lag_twenty_four = features["lag_24"]
        lag_week = features["lag_168"]
        mean_day = features["rolling_mean_24"]
        mean_week = features["rolling_mean_168"]

    target_time = forecast_origin + timedelta(hours=horizon_input)
    request = ForecastRequest(
        horizon_hours=horizon_input,
        zone_id=int(zone_input),
        target_hour=target_time.hour,
        target_day_of_week=target_time.weekday(),
        target_is_weekend=int(target_time.weekday() >= 5),
        target_month=target_time.month,
        lag_1=lag_one,
        lag_24=lag_twenty_four,
        lag_168=lag_week,
        rolling_mean_24=mean_day,
        rolling_mean_168=mean_week,
    )
    forecast = predict_request(request)
    expected_change = forecast.prediction / max(mean_day, 1.0) - 1

    result_one, result_two, result_three = st.columns(3)
    result_one.metric("Expected pickups", f"{forecast.prediction:.0f}")
    result_two.metric("Target time", target_time.strftime("%a %d %b, %H:%M"))
    result_three.metric("Versus recent average", f"{expected_change:+.1%}")

    if history_mode == "Automatic from TLC history":
        with st.expander("Historical features used", expanded=True):
            feature_one, feature_two, feature_three, feature_four, feature_five = st.columns(5)
            feature_one.metric("Previous hour", f"{lag_one:.0f}")
            feature_two.metric("Same hour yesterday", f"{lag_twenty_four:.0f}")
            feature_three.metric("Same hour last week", f"{lag_week:.0f}")
            feature_four.metric("Average over 24 hours", f"{mean_day:.1f}")
            feature_five.metric("Average over 7 days", f"{mean_week:.1f}")

    if forecast.p10 is not None:
        interval_frame = pd.DataFrame({
            "estimate": ["P10 conservative", "P50 central", "P90 high demand"],
            "pickups": [forecast.p10, forecast.p50, forecast.p90],
        })
        st.plotly_chart(
            px.bar(
                interval_frame,
                x="estimate",
                y="pickups",
                color="estimate",
                text_auto=".0f",
                labels={"estimate": "Planning scenario", "pickups": "Expected hourly pickups"},
            ),
            width="stretch",
        )
        st.info(
            f"For risk aware allocation, plan around {forecast.p50:.0f} pickups under the central scenario "
            f"and reserve capacity up to {forecast.p90:.0f} pickups under the high demand scenario."
        )
    elif horizon_input == 24:
        st.info(
            "The twenty four hour model improved RMSE in all three rolling test windows, with more variation than shorter horizons. "
            "Use the result as a planning forecast and retain the weekly baseline as a production fallback."
        )
    else:
        st.info("This horizon provides a point forecast. Probability intervals are currently calibrated for the one hour horizon.")

st.subheader("Performance by forecast horizon")
metric_choice = st.selectbox("Metric", ["rmse", "mae", "wmape", "peak_rmse"], index=0)
figure = px.bar(
    test_metrics, x="horizon_hours", y=metric_choice, color="model", barmode="group",
    labels={"horizon_hours": "Forecast horizon in hours", metric_choice: metric_choice.upper(), "model": "Model"},
)
st.plotly_chart(figure, width="stretch")
st.info("With a full year of training data, LightGBM leads the weekly seasonal baseline at every horizon on the final test period.")

if backtest_path.exists():
    st.subheader("Rolling backtest stability")
    backtest = pd.read_csv(backtest_path, parse_dates=["origin"])
    model_backtest = backtest[backtest.model.eq("lightgbm")].copy()
    stability_one, stability_two, stability_three = st.columns(3)
    for column, horizon_value in zip((stability_one, stability_two, stability_three), (1, 6, 24)):
        summary = backtest_summary[str(horizon_value)]
        column.metric(
            f"{horizon_value} hour mean gain",
            f"{summary['improvement_mean']:.1%}",
            help=f"LightGBM beat the weekly baseline in {summary['improved_folds']} of {summary['folds']} folds",
        )
    backtest_figure = px.line(
        model_backtest,
        x="origin",
        y="rmse_improvement",
        color="horizon_hours",
        markers=True,
        labels={"origin": "Future test window start", "rmse_improvement": "RMSE improvement", "horizon_hours": "Horizon"},
    )
    backtest_figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(backtest_figure, width="stretch")
    st.caption("Each point trains on earlier observations, validates on the following fourteen days, and evaluates on a later twenty eight day window.")

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

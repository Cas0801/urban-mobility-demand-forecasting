from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_forecast_simulator_runs_default_scenario():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.subheader[0].value == "Interactive demand forecast"

    app.button[0].click().run()

    assert not app.exception
    result_metrics = {metric.label: metric.value for metric in app.metric}
    assert float(result_metrics["Expected pickups"]) >= 0
    assert "Target time" in result_metrics
    assert "Versus recent average" in result_metrics

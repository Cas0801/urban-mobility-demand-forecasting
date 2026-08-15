# Multi horizon benchmark

## Dataset

The verified experiment uses official NYC Yellow Taxi records from January through March 2024.

1. Aggregated zone hour rows: 572,208
2. Forecast horizons: one, six, and twenty four hours
3. Validation period: fourteen days
4. Final test period: fourteen days
5. Primary model: one global LightGBM model per forecast horizon
6. Business baseline: demand from the same target hour one week earlier

Monthly timestamp boundaries are enforced before aggregation. This removes invalid records outside the month declared by each source file.

## Final test results

<table>
  <tr><th>Horizon</th><th>Model</th><th>RMSE</th><th>MAE</th><th>W MAPE</th><th>Peak RMSE</th></tr>
  <tr><td>1 hour</td><td>Weekly baseline</td><td>12.816</td><td>3.836</td><td>21.26%</td><td>37.744</td></tr>
  <tr><td>1 hour</td><td>LightGBM</td><td><strong>11.170</strong></td><td><strong>3.485</strong></td><td><strong>19.32%</strong></td><td><strong>32.853</strong></td></tr>
  <tr><td>6 hours</td><td>Weekly baseline</td><td>12.816</td><td>3.836</td><td>21.26%</td><td>37.744</td></tr>
  <tr><td>6 hours</td><td>LightGBM</td><td><strong>12.347</strong></td><td><strong>3.716</strong></td><td><strong>20.59%</strong></td><td><strong>36.602</strong></td></tr>
  <tr><td>24 hours</td><td><strong>Weekly baseline</strong></td><td><strong>12.816</strong></td><td><strong>3.836</strong></td><td><strong>21.26%</strong></td><td><strong>37.744</strong></td></tr>
  <tr><td>24 hours</td><td>LightGBM</td><td>12.943</td><td>3.852</td><td>21.35%</td><td>38.130</td></tr>
</table>

## Interpretation

At one hour, LightGBM reduces RMSE by 12.85% and peak RMSE by 12.96% relative to the weekly baseline. At six hours, the RMSE gain falls to 3.66%. At twenty four hours, LightGBM is 0.99% worse than the weekly baseline.

This is a useful production finding. One model should not automatically serve every horizon. Short horizon operations benefit from recent demand signals, while longer horizons depend more strongly on stable weekly seasonality. The next iteration will add weather and calendar data, then test whether those known future variables improve the twenty four hour forecast.

## Probabilistic forecast

Three additional LightGBM models estimate the tenth, fiftieth, and ninetieth demand percentiles for the one hour horizon. The nominal eighty percent interval achieves 81.09% empirical coverage on the untouched test period, with an average width of 12.18 pickups.

This means the first probabilistic model is close to its target coverage. The dashboard exposes zone level intervals for the twenty busiest zones so that an operations user can choose a capacity level based on risk rather than relying on a single point estimate.

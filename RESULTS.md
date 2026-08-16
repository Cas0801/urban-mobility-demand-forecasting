# Multi horizon benchmark

## Dataset

The verified experiment uses official NYC Yellow Taxi records across all twelve months of 2024.

1. Raw trip records: 41,169,720
2. Valid trips after timestamp and zone checks: 41,169,300
3. Aggregated zone hour rows: 2,310,192
4. Covered taxi zones: 263
5. Forecast horizons: one, six, and twenty four hours
6. Primary model: one global LightGBM model per forecast horizon
7. Business baseline: demand from the same target hour one week earlier

Monthly timestamp boundaries are enforced before aggregation. The pipeline reads and aggregates one month at a time, so the complete raw dataset never needs to reside in memory at once.

## Rolling backtest

Three expanding window folds evaluate separate future periods beginning on May 28, August 31, and December 4. Each fold trains only on earlier observations, validates on the next fourteen days, and evaluates on a later twenty eight day period.

<table>
  <tr><th>Horizon</th><th>Mean LightGBM RMSE</th><th>RMSE standard deviation</th><th>Mean gain over baseline</th><th>Improved folds</th></tr>
  <tr><td>1 hour</td><td>12.393</td><td>1.277</td><td>31.17%</td><td>3 of 3</td></tr>
  <tr><td>6 hours</td><td>13.159</td><td>1.497</td><td>27.14%</td><td>3 of 3</td></tr>
  <tr><td>24 hours</td><td>14.243</td><td>2.104</td><td>21.65%</td><td>3 of 3</td></tr>
</table>

## Final test results

The final model uses data before December 4 for training, December 4 through December 17 for validation, and December 18 through December 31 as the untouched test period.

<table>
  <tr><th>Horizon</th><th>Model</th><th>RMSE</th><th>MAE</th><th>W MAPE</th><th>Peak RMSE</th></tr>
  <tr><td>1 hour</td><td>Weekly baseline</td><td>26.591</td><td>7.411</td><td>47.15%</td><td>81.740</td></tr>
  <tr><td>1 hour</td><td>LightGBM</td><td><strong>12.151</strong></td><td><strong>3.778</strong></td><td><strong>24.03%</strong></td><td><strong>37.610</strong></td></tr>
  <tr><td>6 hours</td><td>Weekly baseline</td><td>26.591</td><td>7.411</td><td>47.15%</td><td>81.740</td></tr>
  <tr><td>6 hours</td><td>LightGBM</td><td><strong>15.174</strong></td><td><strong>4.498</strong></td><td><strong>28.62%</strong></td><td><strong>46.310</strong></td></tr>
  <tr><td>24 hours</td><td>Weekly baseline</td><td>26.591</td><td>7.411</td><td>47.15%</td><td>81.740</td></tr>
  <tr><td>24 hours</td><td>LightGBM</td><td><strong>17.351</strong></td><td><strong>5.023</strong></td><td><strong>31.95%</strong></td><td><strong>52.709</strong></td></tr>
</table>

## Interpretation

The final year end period contains stronger demand changes than the same period one week earlier, which makes the seasonal baseline unusually weak. LightGBM reduces final test RMSE by 54.30% at one hour, 42.93% at six hours, and 34.75% at twenty four hours.

The rolling backtest is the more conservative headline result. It confirms that the improvement is not isolated to the year end test: all three horizons outperform the baseline in every evaluated future window. Gains decline as the horizon increases, which is consistent with recent demand becoming less informative farther into the future.

## Probabilistic forecast

Three additional LightGBM models estimate the tenth, fiftieth, and ninetieth demand percentiles for the one hour horizon. The nominal eighty percent interval achieves 84.87% empirical coverage on the final test period, with an average width of 11.93 pickups.

The interval is slightly conservative because empirical coverage is higher than its target. The dashboard exposes probability scenarios so an operations user can allocate capacity according to risk rather than relying on a single point estimate.

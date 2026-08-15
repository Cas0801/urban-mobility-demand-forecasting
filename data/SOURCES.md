# Data sources

## NYC Taxi and Limousine Commission

Dataset: Yellow Taxi Trip Records

Official page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Monthly file pattern: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`

Relevant input fields:

1. `tpep_pickup_datetime`
2. `PULocationID`
3. `passenger_count`
4. `trip_distance`

The TLC notes that trip records are submitted by authorized technology providers and may contain accuracy or completeness issues. This project applies timestamp, zone, and range validation before aggregation.

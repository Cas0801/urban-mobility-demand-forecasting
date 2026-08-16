# Data sources

## NYC Taxi and Limousine Commission

Dataset: Yellow Taxi Trip Records

Official page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Monthly file pattern: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`

Experiment coverage: January through December 2024

Raw records downloaded: 41,169,720

Relevant input fields:

1. `tpep_pickup_datetime`
2. `PULocationID`

The TLC notes that trip records are submitted by authorized technology providers and may contain accuracy or completeness issues. This project applies timestamp, zone, and range validation before aggregation.

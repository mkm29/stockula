-- TimescaleDB Compression and Retention Policies
-- This file contains compression and retention policy setup for optimal storage

-- Enable compression on price_history
ALTER TABLE price_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,interval_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy (compress data older than 30 days)
SELECT add_compression_policy('price_history', INTERVAL '30 days');

-- Add retention policy (keep data for 5 years)
SELECT add_retention_policy('price_history', INTERVAL '5 years');

-- Add compression policy for ETL runs (compress after 7 days)
ALTER TABLE etl_runs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pipeline_name',
    timescaledb.compress_orderby = 'start_time DESC'
);

SELECT add_compression_policy('etl_runs', INTERVAL '7 days');

-- Add retention policy for ETL runs (keep for 1 year)
SELECT add_retention_policy('etl_runs', INTERVAL '1 year');

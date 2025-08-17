-- Data Retention and Compression Policies for TimescaleDB
-- Optimized for financial data lifecycle management

-- Price history data retention and compression
-- Keep detailed data for 2 years, then compress
-- Drop data older than 10 years
SELECT add_retention_policy(
    'price_history',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'price_history',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- Dividends data retention (keep longer for historical analysis)
-- Compress after 1 year, retain for 20 years
SELECT add_retention_policy(
    'dividends',
    INTERVAL '20 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'dividends',
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Splits data retention (keep for historical accuracy)
-- Compress after 1 year, retain for 20 years
SELECT add_retention_policy(
    'splits',
    INTERVAL '20 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'splits',
    INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Options data retention (shorter retention due to expiration)
-- Compress after 3 months, retain for 3 years
SELECT add_retention_policy(
    'options_calls',
    INTERVAL '3 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'options_calls',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'options_puts',
    INTERVAL '3 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'options_puts',
    INTERVAL '3 months',
    if_not_exists => TRUE
);

-- Market summary data (compress after 6 months, keep for 10 years)
SELECT add_retention_policy(
    'market_summary_daily',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'market_summary_daily',
    INTERVAL '6 months',
    if_not_exists => TRUE
);

-- Sector performance data (compress after 6 months, keep for 10 years)
SELECT add_retention_policy(
    'sector_performance',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'sector_performance',
    INTERVAL '6 months',
    if_not_exists => TRUE
);

-- Create continuous aggregates for common queries
-- Daily OHLCV aggregation from intraday data
CREATE MATERIALIZED VIEW IF NOT EXISTS price_history_daily
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', time) AS day,
    first(open_price, time) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, time) AS close_price,
    sum(volume) AS volume,
    count(*) AS num_trades,
    avg(close_price) AS avg_price,
    stddev(close_price) AS price_stddev
FROM price_history
WHERE interval != '1d' -- Exclude already daily data
GROUP BY symbol, day;

-- Weekly price aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS price_history_weekly
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 week', time) AS week,
    first(open_price, time) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, time) AS close_price,
    sum(volume) AS volume,
    count(*) AS num_trades,
    avg(close_price) AS avg_price,
    stddev(close_price) AS price_stddev
FROM price_history
GROUP BY symbol, week;

-- Monthly price aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS price_history_monthly
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 month', time) AS month,
    first(open_price, time) AS open_price,
    max(high_price) AS high_price,
    min(low_price) AS low_price,
    last(close_price, time) AS close_price,
    sum(volume) AS volume,
    count(*) AS num_trades,
    avg(close_price) AS avg_price,
    stddev(close_price) AS price_stddev
FROM price_history
GROUP BY symbol, month;

-- Market volume analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS market_volume_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    sum(volume) AS total_volume,
    sum(trading_value) AS total_value,
    count(DISTINCT symbol) AS active_symbols,
    avg(close_price) AS avg_price,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY close_price) AS median_price
FROM price_history
WHERE volume > 0
GROUP BY hour;

-- Options volume analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS options_volume_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    expiration_date,
    sum(volume) AS total_call_volume,
    sum(open_interest) AS total_call_oi,
    avg(implied_volatility) AS avg_iv,
    count(*) AS num_strikes
FROM options_calls
WHERE volume > 0
GROUP BY day, symbol, expiration_date;

-- Add refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy(
    'price_history_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'price_history_weekly',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'price_history_monthly',
    start_offset => INTERVAL '2 months',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'market_volume_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy(
    'options_volume_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '2 hours',
    if_not_exists => TRUE
);

-- Add compression policies for continuous aggregates
SELECT add_compression_policy(
    'price_history_daily',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'price_history_weekly',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy(
    'price_history_monthly',
    INTERVAL '180 days',
    if_not_exists => TRUE
);

-- Create indexes on continuous aggregates
CREATE INDEX IF NOT EXISTS idx_price_history_daily_symbol_day
ON price_history_daily(symbol, day DESC);

CREATE INDEX IF NOT EXISTS idx_price_history_weekly_symbol_week
ON price_history_weekly(symbol, week DESC);

CREATE INDEX IF NOT EXISTS idx_price_history_monthly_symbol_month
ON price_history_monthly(symbol, month DESC);

CREATE INDEX IF NOT EXISTS idx_market_volume_hourly_hour
ON market_volume_hourly(hour DESC);

CREATE INDEX IF NOT EXISTS idx_options_volume_daily_symbol_day
ON options_volume_daily(symbol, day DESC);

-- TimescaleDB Continuous Aggregates
-- This file contains continuous aggregate views for common analytical queries

-- Daily price aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_prices
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket('1 day', timestamp) AS day,
       first(open_price, timestamp) AS open_price,
       max(high_price) AS high_price,
       min(low_price) AS low_price,
       last(close_price, timestamp) AS close_price,
       sum(volume) AS volume,
       count(*) AS trade_count
FROM price_history
WHERE interval_type = '1d'
GROUP BY symbol, day;

-- Weekly price aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_prices
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket('1 week', timestamp) AS week,
       first(open_price, timestamp) AS open_price,
       max(high_price) AS high_price,
       min(low_price) AS low_price,
       last(close_price, timestamp) AS close_price,
       sum(volume) AS volume
FROM price_history
WHERE interval_type = '1d'
GROUP BY symbol, week;

-- Monthly price aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_prices
WITH (timescaledb.continuous) AS
SELECT symbol,
       time_bucket('1 month', timestamp) AS month,
       first(open_price, timestamp) AS open_price,
       max(high_price) AS high_price,
       min(low_price) AS low_price,
       last(close_price, timestamp) AS close_price,
       sum(volume) AS volume
FROM price_history
WHERE interval_type = '1d'
GROUP BY symbol, month;

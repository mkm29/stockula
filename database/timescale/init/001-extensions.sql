-- TimescaleDB Extensions and Setup
-- This script initializes TimescaleDB with required extensions for financial time series

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgai extension for vector operations and ML features
CREATE EXTENSION IF NOT EXISTS ai CASCADE;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create custom functions for financial calculations
CREATE OR REPLACE FUNCTION calculate_returns(
    current_price NUMERIC,
    previous_price NUMERIC
) RETURNS NUMERIC AS $$
BEGIN
    IF previous_price IS NULL OR previous_price = 0 THEN
        RETURN NULL;
    END IF;
    RETURN ((current_price - previous_price) / previous_price) * 100;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate volatility over a period
CREATE OR REPLACE FUNCTION calculate_volatility(
    symbol TEXT,
    start_date DATE,
    end_date DATE
) RETURNS NUMERIC AS $$
DECLARE
    volatility NUMERIC;
BEGIN
    SELECT STDDEV(
        LN(close_price / LAG(close_price) OVER (ORDER BY date))
    ) * SQRT(252) * 100
    INTO volatility
    FROM price_history
    WHERE symbol = $1
    AND date BETWEEN start_date AND end_date
    AND close_price IS NOT NULL
    AND LAG(close_price) OVER (ORDER BY date) IS NOT NULL;

    RETURN volatility;
END;
$$ LANGUAGE plpgsql;

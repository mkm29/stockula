-- TimescaleDB Index Creation Scripts
-- This file contains optimized indexes for time-series queries

-- Price history indexes
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_time
ON price_history (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_price_history_symbol_interval
ON price_history (symbol, interval_type, timestamp DESC);

-- Recent data index (last year)
CREATE INDEX IF NOT EXISTS idx_price_history_recent
ON price_history (timestamp DESC)
WHERE timestamp >= NOW() - INTERVAL '1 year';

-- Dividends indexes
CREATE INDEX IF NOT EXISTS idx_dividends_symbol_date
ON dividends (symbol, date DESC);

-- Splits indexes
CREATE INDEX IF NOT EXISTS idx_splits_symbol_date
ON splits (symbol, date DESC);

-- Options indexes
CREATE INDEX IF NOT EXISTS idx_options_calls_symbol_exp
ON options_calls (symbol, expiration_date);

CREATE INDEX IF NOT EXISTS idx_options_puts_symbol_exp
ON options_puts (symbol, expiration_date);

-- Stock info JSONB index
CREATE INDEX IF NOT EXISTS idx_stock_info_gin
ON stock_info USING GIN (info_json);

-- ETL monitoring indexes
CREATE INDEX IF NOT EXISTS idx_etl_runs_pipeline_time
ON etl_runs (pipeline_name, start_time DESC);

CREATE INDEX IF NOT EXISTS idx_etl_runs_status
ON etl_runs (status, start_time DESC);

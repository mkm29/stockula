-- Comprehensive Indexing Strategy for TimescaleDB Financial Data
-- Optimized for common query patterns in financial analysis

-- Stocks table indexes
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
CREATE INDEX IF NOT EXISTS idx_stocks_industry ON stocks(industry);
CREATE INDEX IF NOT EXISTS idx_stocks_exchange ON stocks(exchange);
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap ON stocks(market_cap);
CREATE INDEX IF NOT EXISTS idx_stocks_name_trgm ON stocks USING gin(name gin_trgm_ops);

-- Price history indexes (time-series optimized)
-- Primary time-based queries
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_time ON price_history(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_time_symbol ON price_history(time DESC, symbol);
CREATE INDEX IF NOT EXISTS idx_price_history_date_symbol ON price_history(date DESC, symbol);

-- Volume and price range queries
CREATE INDEX IF NOT EXISTS idx_price_history_volume ON price_history(volume DESC) WHERE volume > 0;
CREATE INDEX IF NOT EXISTS idx_price_history_close_price ON price_history(close_price);
CREATE INDEX IF NOT EXISTS idx_price_history_high_low ON price_history(high_price, low_price);

-- Technical analysis indexes
CREATE INDEX IF NOT EXISTS idx_price_history_returns ON price_history(daily_return) WHERE daily_return IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_price_history_volatility ON price_history(volatility) WHERE volatility IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_price_history_trading_value ON price_history(trading_value DESC);

-- Interval-specific queries
CREATE INDEX IF NOT EXISTS idx_price_history_interval_symbol_time ON price_history(interval, symbol, time DESC);

-- Composite index for common range queries
CREATE INDEX IF NOT EXISTS idx_price_history_symbol_date_interval ON price_history(symbol, date DESC, interval);

-- Dividends indexes
CREATE INDEX IF NOT EXISTS idx_dividends_symbol_time ON dividends(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_dividends_amount ON dividends(amount DESC);
CREATE INDEX IF NOT EXISTS idx_dividends_type ON dividends(dividend_type);

-- Splits indexes
CREATE INDEX IF NOT EXISTS idx_splits_symbol_time ON splits(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_splits_ratio ON splits(ratio);

-- Options calls indexes
CREATE INDEX IF NOT EXISTS idx_options_calls_symbol_exp ON options_calls(symbol, expiration_date DESC);
CREATE INDEX IF NOT EXISTS idx_options_calls_strike ON options_calls(strike);
CREATE INDEX IF NOT EXISTS idx_options_calls_volume ON options_calls(volume DESC) WHERE volume > 0;
CREATE INDEX IF NOT EXISTS idx_options_calls_oi ON options_calls(open_interest DESC) WHERE open_interest > 0;
CREATE INDEX IF NOT EXISTS idx_options_calls_iv ON options_calls(implied_volatility) WHERE implied_volatility IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_options_calls_itm ON options_calls(in_the_money);
CREATE INDEX IF NOT EXISTS idx_options_calls_greeks ON options_calls(delta, gamma, theta) WHERE delta IS NOT NULL;

-- Options puts indexes (mirror of calls)
CREATE INDEX IF NOT EXISTS idx_options_puts_symbol_exp ON options_puts(symbol, expiration_date DESC);
CREATE INDEX IF NOT EXISTS idx_options_puts_strike ON options_puts(strike);
CREATE INDEX IF NOT EXISTS idx_options_puts_volume ON options_puts(volume DESC) WHERE volume > 0;
CREATE INDEX IF NOT EXISTS idx_options_puts_oi ON options_puts(open_interest DESC) WHERE open_interest > 0;
CREATE INDEX IF NOT EXISTS idx_options_puts_iv ON options_puts(implied_volatility) WHERE implied_volatility IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_options_puts_itm ON options_puts(in_the_money);
CREATE INDEX IF NOT EXISTS idx_options_puts_greeks ON options_puts(delta, gamma, theta) WHERE delta IS NOT NULL;

-- Stock info JSONB indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_stock_info_json_gin ON stock_info USING gin(info_json);
-- Specific JSONB field indexes for common queries
CREATE INDEX IF NOT EXISTS idx_stock_info_sector ON stock_info USING gin((info_json->>'sector'));
CREATE INDEX IF NOT EXISTS idx_stock_info_industry ON stock_info USING gin((info_json->>'industry'));
CREATE INDEX IF NOT EXISTS idx_stock_info_market_cap ON stock_info((info_json->>'marketCap')::bigint);

-- Strategy and preset indexes
CREATE INDEX IF NOT EXISTS idx_strategies_category ON strategies(category);
CREATE INDEX IF NOT EXISTS idx_strategies_active ON strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_strategy_presets_strategy_id ON strategy_presets(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_presets_default ON strategy_presets(is_default) WHERE is_default = TRUE;

-- AutoTS model indexes
CREATE INDEX IF NOT EXISTS idx_autots_models_categories ON autots_models USING gin(categories);
CREATE INDEX IF NOT EXISTS idx_autots_models_gpu ON autots_models(is_gpu_enabled) WHERE is_gpu_enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_autots_models_slow ON autots_models(is_slow);

-- AutoTS preset indexes
CREATE INDEX IF NOT EXISTS idx_autots_presets_models ON autots_presets USING gin(models);
CREATE INDEX IF NOT EXISTS idx_autots_presets_use_case ON autots_presets(use_case);

-- Market summary indexes
CREATE INDEX IF NOT EXISTS idx_market_summary_time ON market_summary_daily(time DESC);
CREATE INDEX IF NOT EXISTS idx_market_summary_volume ON market_summary_daily(total_volume DESC);
CREATE INDEX IF NOT EXISTS idx_market_summary_volatility ON market_summary_daily(volatility);

-- Sector performance indexes
CREATE INDEX IF NOT EXISTS idx_sector_performance_time_sector ON sector_performance(time DESC, sector);
CREATE INDEX IF NOT EXISTS idx_sector_performance_sector_time ON sector_performance(sector, time DESC);
CREATE INDEX IF NOT EXISTS idx_sector_performance_return ON sector_performance(avg_return DESC);

-- Unique constraints to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_price_history_unique
ON price_history(symbol, date, interval);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dividends_unique
ON dividends(symbol, date);

CREATE UNIQUE INDEX IF NOT EXISTS idx_splits_unique
ON splits(symbol, date);

CREATE UNIQUE INDEX IF NOT EXISTS idx_options_calls_unique
ON options_calls(symbol, expiration_date, strike, contract_symbol);

CREATE UNIQUE INDEX IF NOT EXISTS idx_options_puts_unique
ON options_puts(symbol, expiration_date, strike, contract_symbol);

-- Partial indexes for data quality and performance
CREATE INDEX IF NOT EXISTS idx_price_history_valid_prices
ON price_history(symbol, time DESC)
WHERE close_price IS NOT NULL AND close_price > 0;

CREATE INDEX IF NOT EXISTS idx_price_history_high_volume
ON price_history(symbol, time DESC, volume)
WHERE volume > 1000000;

CREATE INDEX IF NOT EXISTS idx_options_liquid_calls
ON options_calls(symbol, expiration_date, strike)
WHERE volume > 10 OR open_interest > 100;

CREATE INDEX IF NOT EXISTS idx_options_liquid_puts
ON options_puts(symbol, expiration_date, strike)
WHERE volume > 10 OR open_interest > 100;

-- Function-based indexes for common calculations
CREATE INDEX IF NOT EXISTS idx_price_history_log_returns
ON price_history(symbol, time DESC)
WHERE close_price IS NOT NULL AND close_price > 0;

-- Create materialized view indexes for aggregated data
CREATE INDEX IF NOT EXISTS idx_price_history_monthly
ON price_history(symbol, date_trunc('month', time), interval);

CREATE INDEX IF NOT EXISTS idx_price_history_weekly
ON price_history(symbol, date_trunc('week', time), interval);

-- PRODUCTION CRITICAL INDEXES - Create these IMMEDIATELY after migration
-- These indexes are essential for production query performance

-- Multi-column covering indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_covering_main
ON price_history(symbol, time DESC)
INCLUDE (close_price, volume, high_price, low_price, open_price);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_covering_analytics
ON price_history(symbol, date DESC, interval)
INCLUDE (close_price, adj_close_price, volume, daily_return, volatility);

-- Critical BTREE indexes for range scans
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_time_range
ON price_history(time) WHERE time >= '2020-01-01';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_symbol_time_range
ON price_history(symbol, time) WHERE time >= '2020-01-01';

-- BRIN indexes for time-series data (space efficient)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_time_brin
ON price_history USING brin(time) WITH (pages_per_range = 128);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_date_brin
ON price_history USING brin(date) WITH (pages_per_range = 128);

-- Hash indexes for exact symbol lookups (fastest for equality)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_symbol_hash
ON price_history USING hash(symbol);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_symbol_hash
ON stocks USING hash(symbol);

-- Specialized indexes for financial calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_price_volume
ON price_history(close_price, volume DESC)
WHERE close_price > 0 AND volume > 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_ohlc
ON price_history(symbol, time DESC, open_price, high_price, low_price, close_price)
WHERE open_price IS NOT NULL AND close_price IS NOT NULL;

-- Options-specific production indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_calls_production
ON options_calls(symbol, expiration_date, strike, implied_volatility)
WHERE implied_volatility IS NOT NULL AND volume > 0;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_puts_production
ON options_puts(symbol, expiration_date, strike, implied_volatility)
WHERE implied_volatility IS NOT NULL AND volume > 0;

-- Performance monitoring indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_updated_at
ON price_history(updated_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stocks_updated_at
ON stocks(updated_at DESC);

-- Index for data freshness checks
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_price_history_freshness
ON price_history(symbol, time DESC)
WHERE time >= CURRENT_DATE - INTERVAL '7 days';

-- Analyze tables after index creation for optimal query plans
ANALYZE stocks;
ANALYZE price_history;
ANALYZE dividends;
ANALYZE splits;
ANALYZE options_calls;
ANALYZE options_puts;
ANALYZE stock_info;

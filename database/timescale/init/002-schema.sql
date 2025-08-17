-- TimescaleDB Optimized Schema for Financial Time Series
-- Based on existing SQLModel schema with TimescaleDB optimizations

-- Create sequences for IDs
CREATE SEQUENCE IF NOT EXISTS price_history_id_seq;
CREATE SEQUENCE IF NOT EXISTS dividends_id_seq;
CREATE SEQUENCE IF NOT EXISTS splits_id_seq;
CREATE SEQUENCE IF NOT EXISTS options_calls_id_seq;
CREATE SEQUENCE IF NOT EXISTS options_puts_id_seq;
CREATE SEQUENCE IF NOT EXISTS strategies_id_seq;
CREATE SEQUENCE IF NOT EXISTS strategy_presets_id_seq;
CREATE SEQUENCE IF NOT EXISTS autots_models_id_seq;
CREATE SEQUENCE IF NOT EXISTS autots_presets_id_seq;

-- Stocks table (reference data - not time-series)
CREATE TABLE IF NOT EXISTS stocks (
    symbol VARCHAR(20) PRIMARY KEY,
    name TEXT,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    exchange VARCHAR(50),
    currency VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Price history hypertable (main time-series table)
CREATE TABLE IF NOT EXISTS price_history (
    id BIGINT DEFAULT nextval('price_history_id_seq') NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    open_price NUMERIC(15,4),
    high_price NUMERIC(15,4),
    low_price NUMERIC(15,4),
    close_price NUMERIC(15,4),
    adj_close_price NUMERIC(15,4),
    volume BIGINT,
    interval VARCHAR(10) DEFAULT '1d',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Additional calculated fields for performance
    daily_return NUMERIC(10,6),
    volatility NUMERIC(10,6),
    trading_value NUMERIC(20,2) GENERATED ALWAYS AS (close_price * volume) STORED,

    CONSTRAINT pk_price_history PRIMARY KEY (symbol, date, interval, id),
    CONSTRAINT fk_price_history_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable(
    'price_history',
    'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Dividends hypertable
CREATE TABLE IF NOT EXISTS dividends (
    id BIGINT DEFAULT nextval('dividends_id_seq') NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    amount NUMERIC(10,4) NOT NULL,
    dividend_type VARCHAR(20) DEFAULT 'regular',
    currency VARCHAR(10) DEFAULT 'USD',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_dividends PRIMARY KEY (symbol, date, id),
    CONSTRAINT fk_dividends_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

SELECT create_hypertable(
    'dividends',
    'time',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Stock splits hypertable
CREATE TABLE IF NOT EXISTS splits (
    id BIGINT DEFAULT nextval('splits_id_seq') NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ratio NUMERIC(10,4) NOT NULL,
    split_type VARCHAR(20) DEFAULT 'stock_split',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_splits PRIMARY KEY (symbol, date, id),
    CONSTRAINT fk_splits_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

SELECT create_hypertable(
    'splits',
    'time',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Options calls hypertable
CREATE TABLE IF NOT EXISTS options_calls (
    id BIGINT DEFAULT nextval('options_calls_id_seq') NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    expiration_date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    strike NUMERIC(15,4) NOT NULL,
    last_price NUMERIC(10,4),
    bid NUMERIC(10,4),
    ask NUMERIC(10,4),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility NUMERIC(8,6),
    delta NUMERIC(8,6),
    gamma NUMERIC(8,6),
    theta NUMERIC(8,6),
    vega NUMERIC(8,6),
    rho NUMERIC(8,6),
    in_the_money BOOLEAN,
    contract_symbol VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_options_calls PRIMARY KEY (symbol, expiration_date, strike, contract_symbol, id),
    CONSTRAINT fk_options_calls_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

SELECT create_hypertable(
    'options_calls',
    'time',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

-- Options puts hypertable
CREATE TABLE IF NOT EXISTS options_puts (
    id BIGINT DEFAULT nextval('options_puts_id_seq') NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    expiration_date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    strike NUMERIC(15,4) NOT NULL,
    last_price NUMERIC(10,4),
    bid NUMERIC(10,4),
    ask NUMERIC(10,4),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility NUMERIC(8,6),
    delta NUMERIC(8,6),
    gamma NUMERIC(8,6),
    theta NUMERIC(8,6),
    vega NUMERIC(8,6),
    rho NUMERIC(8,6),
    in_the_money BOOLEAN,
    contract_symbol VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_options_puts PRIMARY KEY (symbol, expiration_date, strike, contract_symbol, id),
    CONSTRAINT fk_options_puts_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

SELECT create_hypertable(
    'options_puts',
    'time',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE
);

-- Stock info table (raw yfinance data as JSONB for better performance)
CREATE TABLE IF NOT EXISTS stock_info (
    symbol VARCHAR(20) PRIMARY KEY,
    info_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_stock_info_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol) ON DELETE CASCADE
);

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id BIGINT DEFAULT nextval('strategies_id_seq') PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    class_name VARCHAR(200) NOT NULL,
    module_path VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Strategy presets table
CREATE TABLE IF NOT EXISTS strategy_presets (
    id BIGINT DEFAULT nextval('strategy_presets_id_seq') PRIMARY KEY,
    strategy_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
    parameters_json JSONB NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_strategy_presets_strategy FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
    CONSTRAINT unique_strategy_preset UNIQUE (strategy_id, name)
);

-- AutoTS models table
CREATE TABLE IF NOT EXISTS autots_models (
    id BIGINT DEFAULT nextval('autots_models_id_seq') PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    categories JSONB DEFAULT '[]'::jsonb,
    is_slow BOOLEAN DEFAULT FALSE,
    is_gpu_enabled BOOLEAN DEFAULT FALSE,
    requires_regressor BOOLEAN DEFAULT FALSE,
    min_data_points INTEGER DEFAULT 10,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- AutoTS presets table
CREATE TABLE IF NOT EXISTS autots_presets (
    id BIGINT DEFAULT nextval('autots_presets_id_seq') PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    models JSONB NOT NULL,
    description TEXT,
    use_case VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Market data aggregation tables for analytics
CREATE TABLE IF NOT EXISTS market_summary_daily (
    date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_volume BIGINT,
    total_value NUMERIC(20,2),
    avg_price NUMERIC(15,4),
    volatility NUMERIC(10,6),
    num_stocks INTEGER,
    market_cap NUMERIC(20,2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_market_summary_daily PRIMARY KEY (date)
);

SELECT create_hypertable(
    'market_summary_daily',
    'time',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

-- Sector performance tracking
CREATE TABLE IF NOT EXISTS sector_performance (
    date DATE NOT NULL,
    time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sector VARCHAR(100) NOT NULL,
    avg_return NUMERIC(10,6),
    total_volume BIGINT,
    total_market_cap NUMERIC(20,2),
    num_stocks INTEGER,
    volatility NUMERIC(10,6),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT pk_sector_performance PRIMARY KEY (date, sector)
);

SELECT create_hypertable(
    'sector_performance',
    'time',
    chunk_time_interval => INTERVAL '1 year',
    if_not_exists => TRUE
);

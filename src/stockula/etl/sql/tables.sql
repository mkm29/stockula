-- TimescaleDB Table Creation Scripts
-- This file contains all table creation statements for the Stockula financial data platform

-- Stocks metadata table
CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    industry TEXT,
    market_cap BIGINT,
    exchange TEXT,
    currency TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Price history hypertable (time-series data)
CREATE TABLE IF NOT EXISTS price_history (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval_type TEXT NOT NULL DEFAULT '1d',
    open_price NUMERIC(12,4),
    high_price NUMERIC(12,4),
    low_price NUMERIC(12,4),
    close_price NUMERIC(12,4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT price_history_pkey PRIMARY KEY (symbol, timestamp, interval_type),
    CONSTRAINT fk_price_history_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

-- Dividends table
CREATE TABLE IF NOT EXISTS dividends (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    amount NUMERIC(10,4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_dividends_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol),
    CONSTRAINT unique_dividend UNIQUE (symbol, date)
);

-- Stock splits table
CREATE TABLE IF NOT EXISTS splits (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    ratio NUMERIC(8,4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_splits_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol),
    CONSTRAINT unique_split UNIQUE (symbol, date)
);

-- Options calls table
CREATE TABLE IF NOT EXISTS options_calls (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    expiration_date DATE NOT NULL,
    strike NUMERIC(10,4) NOT NULL,
    last_price NUMERIC(10,4),
    bid NUMERIC(10,4),
    ask NUMERIC(10,4),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility NUMERIC(8,6),
    in_the_money BOOLEAN,
    contract_symbol TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_options_calls_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol),
    CONSTRAINT unique_option_call UNIQUE (symbol, expiration_date, strike, contract_symbol)
);

-- Options puts table
CREATE TABLE IF NOT EXISTS options_puts (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    expiration_date DATE NOT NULL,
    strike NUMERIC(10,4) NOT NULL,
    last_price NUMERIC(10,4),
    bid NUMERIC(10,4),
    ask NUMERIC(10,4),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility NUMERIC(8,6),
    in_the_money BOOLEAN,
    contract_symbol TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_options_puts_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol),
    CONSTRAINT unique_option_put UNIQUE (symbol, expiration_date, strike, contract_symbol)
);

-- Stock info table for raw JSON data
CREATE TABLE IF NOT EXISTS stock_info (
    symbol TEXT PRIMARY KEY,
    info_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT fk_stock_info_symbol FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

-- ETL monitoring table
CREATE TABLE IF NOT EXISTS etl_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    records_processed INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

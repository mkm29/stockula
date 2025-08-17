-- Initialize Stockula database with TimescaleDB extensions
-- This script runs automatically during container initialization

-- Create the stockula database if it doesn't exist
SELECT 'CREATE DATABASE stockula'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'stockula');

-- Connect to the stockula database
\c stockula

-- Create TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Create roles for different access levels
DO $$
BEGIN
    -- ETL worker role (read/write access)
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'stockula_etl') THEN
        CREATE ROLE stockula_etl WITH LOGIN PASSWORD 'etl_password_change_me';
    END IF;

    -- Read-only analyst role
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'stockula_analyst') THEN
        CREATE ROLE stockula_analyst WITH LOGIN PASSWORD 'analyst_password_change_me';
    END IF;

    -- Application role (limited read/write)
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'stockula_app') THEN
        CREATE ROLE stockula_app WITH LOGIN PASSWORD 'app_password_change_me';
    END IF;
END
$$;

-- Grant appropriate permissions
GRANT CONNECT ON DATABASE stockula TO stockula_etl, stockula_analyst, stockula_app;
GRANT USAGE ON SCHEMA public TO stockula_etl, stockula_analyst, stockula_app;

-- ETL role gets full access
GRANT CREATE ON SCHEMA public TO stockula_etl;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO stockula_etl;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO stockula_etl;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO stockula_etl;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO stockula_etl;

-- Analyst role gets read-only access
GRANT SELECT ON ALL TABLES IN SCHEMA public TO stockula_analyst;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO stockula_analyst;

-- App role gets limited read/write
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO stockula_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO stockula_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO stockula_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO stockula_app;

-- Create custom functions for data management
CREATE OR REPLACE FUNCTION cleanup_old_data(table_name TEXT, retention_period INTERVAL)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    EXECUTE format(
        'DELETE FROM %I WHERE timestamp < NOW() - %L',
        table_name, retention_period
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get table statistics
CREATE OR REPLACE FUNCTION get_table_stats(table_name TEXT)
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT,
    last_vacuum TIMESTAMP,
    last_analyze TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.schemaname || '.' || t.tablename as table_name,
        t.n_tup_ins + t.n_tup_upd - t.n_tup_del as row_count,
        pg_size_pretty(pg_relation_size(c.oid)) as table_size,
        pg_size_pretty(pg_indexes_size(c.oid)) as index_size,
        pg_size_pretty(pg_total_relation_size(c.oid)) as total_size,
        t.last_vacuum,
        t.last_analyze
    FROM pg_stat_user_tables t
    JOIN pg_class c ON c.relname = t.relname
    WHERE t.schemaname || '.' || t.tablename = $1 OR t.tablename = $1;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze data quality
CREATE OR REPLACE FUNCTION analyze_data_quality(table_name TEXT, column_name TEXT)
RETURNS TABLE(
    null_count BIGINT,
    null_percentage NUMERIC,
    distinct_count BIGINT,
    min_value TEXT,
    max_value TEXT
) AS $$
DECLARE
    total_rows BIGINT;
BEGIN
    -- Get total row count
    EXECUTE format('SELECT COUNT(*) FROM %I', table_name) INTO total_rows;

    -- Analyze the specified column
    RETURN QUERY
    EXECUTE format(
        'SELECT
            COUNT(*) FILTER (WHERE %I IS NULL) as null_count,
            ROUND(COUNT(*) FILTER (WHERE %I IS NULL) * 100.0 / %s, 2) as null_percentage,
            COUNT(DISTINCT %I) as distinct_count,
            MIN(%I::TEXT) as min_value,
            MAX(%I::TEXT) as max_value
        FROM %I',
        column_name, column_name, total_rows, column_name, column_name, column_name, table_name
    );
END;
$$ LANGUAGE plpgsql;

-- Create a view for monitoring hypertable health
CREATE OR REPLACE VIEW hypertable_health AS
SELECT
    ht.hypertable_name,
    ht.num_dimensions,
    ht.num_chunks,
    ht.compression_state,
    ht.compressed_heap_size,
    ht.uncompressed_heap_size,
    CASE
        WHEN ht.uncompressed_heap_size > 0
        THEN ROUND((1 - ht.compressed_heap_size::NUMERIC / ht.uncompressed_heap_size) * 100, 2)
        ELSE 0
    END as compression_ratio_percent
FROM timescaledb_information.hypertables ht;

-- Create materialized view for quick metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_metrics AS
SELECT
    'stockula' as database_name,
    CURRENT_DATE as metric_date,
    COUNT(*) as total_tables,
    SUM(pg_total_relation_size(schemaname||'.'||tablename)) as total_size_bytes
FROM pg_tables
WHERE schemaname = 'public'
WITH NO DATA;

-- Create index on the materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_metrics_date
ON daily_metrics (metric_date);

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW daily_metrics;

-- Log successful initialization
INSERT INTO pg_stat_statements_info VALUES ('Stockula database initialized successfully');

COMMENT ON DATABASE stockula IS 'Stockula financial data warehouse with TimescaleDB';
COMMENT ON FUNCTION cleanup_old_data IS 'Clean up old data based on retention period';
COMMENT ON FUNCTION get_table_stats IS 'Get comprehensive table statistics';
COMMENT ON FUNCTION analyze_data_quality IS 'Analyze data quality for a specific column';
COMMENT ON VIEW hypertable_health IS 'Monitor TimescaleDB hypertable health and compression';

-- Enable row-level security (can be used for multi-tenant scenarios)
-- ALTER TABLE future_tables ENABLE ROW LEVEL SECURITY;

COMMIT;

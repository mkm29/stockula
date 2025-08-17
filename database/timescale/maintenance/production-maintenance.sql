-- Production Maintenance Schedule for TimescaleDB
-- Critical automated maintenance tasks for immediate production deployment

-- Daily maintenance tasks
DO $$
DECLARE
    table_name TEXT;
    query_text TEXT;
BEGIN
    -- Update table statistics for query optimizer
    RAISE NOTICE 'Starting daily maintenance at %', now();

    -- Analyze critical tables for fresh statistics
    FOR table_name IN
        SELECT schemaname||'.'||tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename IN ('price_history', 'stocks', 'options_calls', 'options_puts', 'dividends')
    LOOP
        query_text := 'ANALYZE ' || table_name;
        EXECUTE query_text;
        RAISE NOTICE 'Analyzed table: %', table_name;
    END LOOP;

    RAISE NOTICE 'Daily maintenance completed at %', now();
END $$;

-- Aggressive VACUUM settings for time-series data
-- Run this during low-traffic hours (2-4 AM)
CREATE OR REPLACE FUNCTION production_vacuum_maintenance()
RETURNS VOID AS $$
DECLARE
    table_rec RECORD;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
BEGIN
    start_time := clock_timestamp();
    RAISE NOTICE 'Starting production vacuum maintenance at %', start_time;

    -- Vacuum high-write tables more aggressively
    FOR table_rec IN
        SELECT schemaname, tablename,
               CASE
                   WHEN tablename = 'price_history' THEN 'VACUUM (ANALYZE, VERBOSE, BUFFER_USAGE_LIMIT ''256MB'')'
                   WHEN tablename IN ('options_calls', 'options_puts') THEN 'VACUUM (ANALYZE, VERBOSE, BUFFER_USAGE_LIMIT ''128MB'')'
                   ELSE 'VACUUM (ANALYZE, VERBOSE)'
               END as vacuum_cmd
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename IN ('price_history', 'options_calls', 'options_puts', 'dividends', 'splits')
    LOOP
        EXECUTE table_rec.vacuum_cmd || ' ' || quote_ident(table_rec.schemaname) || '.' || quote_ident(table_rec.tablename);
        RAISE NOTICE 'Vacuumed table: %.%', table_rec.schemaname, table_rec.tablename;
    END LOOP;

    -- Update TimescaleDB statistics
    SELECT timescaledb_experimental.refresh_continuous_aggregate_stats();

    end_time := clock_timestamp();
    RAISE NOTICE 'Production vacuum maintenance completed in % seconds', EXTRACT(EPOCH FROM (end_time - start_time));
END;
$$ LANGUAGE plpgsql;

-- Reindex maintenance for production
CREATE OR REPLACE FUNCTION production_reindex_maintenance()
RETURNS VOID AS $$
DECLARE
    index_rec RECORD;
    start_time TIMESTAMP;
BEGIN
    start_time := clock_timestamp();
    RAISE NOTICE 'Starting production reindex maintenance at %', start_time;

    -- Reindex bloated indexes
    FOR index_rec IN
        SELECT schemaname, tablename, indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND tablename IN ('price_history', 'stocks')
        AND indexname LIKE 'idx_%'
    LOOP
        EXECUTE 'REINDEX INDEX CONCURRENTLY ' || quote_ident(index_rec.schemaname) || '.' || quote_ident(index_rec.indexname);
        RAISE NOTICE 'Reindexed: %', index_rec.indexname;
    END LOOP;

    RAISE NOTICE 'Production reindex maintenance completed in % seconds',
                 EXTRACT(EPOCH FROM (clock_timestamp() - start_time));
END;
$$ LANGUAGE plpgsql;

-- Connection and lock monitoring
CREATE OR REPLACE FUNCTION check_production_health()
RETURNS TABLE(
    metric TEXT,
    value NUMERIC,
    threshold NUMERIC,
    status TEXT
) AS $$
BEGIN
    -- Check active connections
    RETURN QUERY
    SELECT
        'active_connections'::TEXT,
        count(*)::NUMERIC,
        400::NUMERIC,
        CASE WHEN count(*) > 400 THEN 'CRITICAL'
             WHEN count(*) > 300 THEN 'WARNING'
             ELSE 'OK' END::TEXT
    FROM pg_stat_activity
    WHERE state = 'active';

    -- Check idle in transaction
    RETURN QUERY
    SELECT
        'idle_in_transaction'::TEXT,
        count(*)::NUMERIC,
        50::NUMERIC,
        CASE WHEN count(*) > 50 THEN 'CRITICAL'
             WHEN count(*) > 25 THEN 'WARNING'
             ELSE 'OK' END::TEXT
    FROM pg_stat_activity
    WHERE state = 'idle in transaction'
    AND now() - state_change > interval '5 minutes';

    -- Check locks
    RETURN QUERY
    SELECT
        'waiting_locks'::TEXT,
        count(*)::NUMERIC,
        10::NUMERIC,
        CASE WHEN count(*) > 10 THEN 'CRITICAL'
             WHEN count(*) > 5 THEN 'WARNING'
             ELSE 'OK' END::TEXT
    FROM pg_locks
    WHERE NOT granted;

    -- Check replication lag (if applicable)
    RETURN QUERY
    SELECT
        'wal_lag_bytes'::TEXT,
        COALESCE(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn), 0)::NUMERIC,
        104857600::NUMERIC, -- 100MB
        CASE WHEN COALESCE(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn), 0) > 104857600 THEN 'CRITICAL'
             WHEN COALESCE(pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn), 0) > 52428800 THEN 'WARNING'
             ELSE 'OK' END::TEXT
    FROM pg_stat_replication
    LIMIT 1;

    -- Database size monitoring
    RETURN QUERY
    SELECT
        'database_size_gb'::TEXT,
        (pg_database_size(current_database()) / 1024 / 1024 / 1024)::NUMERIC,
        1000::NUMERIC, -- 1TB
        CASE WHEN pg_database_size(current_database()) / 1024 / 1024 / 1024 > 1000 THEN 'WARNING'
             ELSE 'OK' END::TEXT;

    -- Table bloat check for critical tables
    RETURN QUERY
    SELECT
        'price_history_bloat_ratio'::TEXT,
        COALESCE((pg_stat_get_live_tuples(c.oid)::NUMERIC / NULLIF(pg_stat_get_tuples_inserted(c.oid) + pg_stat_get_tuples_updated(c.oid), 0)), 1)::NUMERIC,
        0.8::NUMERIC,
        CASE WHEN COALESCE((pg_stat_get_live_tuples(c.oid)::NUMERIC / NULLIF(pg_stat_get_tuples_inserted(c.oid) + pg_stat_get_tuples_updated(c.oid), 0)), 1) < 0.8 THEN 'WARNING'
             ELSE 'OK' END::TEXT
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relname = 'price_history';

END;
$$ LANGUAGE plpgsql;

-- Performance monitoring queries
CREATE OR REPLACE FUNCTION get_slow_queries(minutes_back INTEGER DEFAULT 60)
RETURNS TABLE(
    query_hash TEXT,
    calls BIGINT,
    total_time NUMERIC,
    avg_time NUMERIC,
    query TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        substring(pss.query, 1, 8) as query_hash,
        pss.calls,
        round(pss.total_exec_time::numeric, 2) as total_time,
        round(pss.mean_exec_time::numeric, 2) as avg_time,
        pss.query
    FROM pg_stat_statements pss
    WHERE pss.mean_exec_time > 1000 -- queries taking more than 1 second
    ORDER BY pss.mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Data freshness monitoring
CREATE OR REPLACE FUNCTION check_data_freshness()
RETURNS TABLE(
    table_name TEXT,
    latest_timestamp TIMESTAMPTZ,
    age_hours NUMERIC,
    status TEXT
) AS $$
BEGIN
    -- Price history freshness
    RETURN QUERY
    SELECT
        'price_history'::TEXT,
        max(time)::TIMESTAMPTZ,
        EXTRACT(EPOCH FROM (now() - max(time))) / 3600::NUMERIC,
        CASE
            WHEN max(time) < now() - interval '2 days' THEN 'STALE'
            WHEN max(time) < now() - interval '1 day' THEN 'WARNING'
            ELSE 'FRESH'
        END::TEXT
    FROM price_history;

    -- Options data freshness
    RETURN QUERY
    SELECT
        'options_calls'::TEXT,
        max(time)::TIMESTAMPTZ,
        EXTRACT(EPOCH FROM (now() - max(time))) / 3600::NUMERIC,
        CASE
            WHEN max(time) < now() - interval '1 day' THEN 'STALE'
            WHEN max(time) < now() - interval '12 hours' THEN 'WARNING'
            ELSE 'FRESH'
        END::TEXT
    FROM options_calls;

END;
$$ LANGUAGE plpgsql;

-- Automated cleanup of old data
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 2555) -- ~7 years
RETURNS VOID AS $$
DECLARE
    cutoff_date DATE;
    deleted_rows BIGINT;
BEGIN
    cutoff_date := CURRENT_DATE - retention_days;

    RAISE NOTICE 'Starting cleanup of data older than %', cutoff_date;

    -- Clean old options data (expires quickly)
    DELETE FROM options_calls WHERE expiration_date < cutoff_date;
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    RAISE NOTICE 'Deleted % expired options_calls records', deleted_rows;

    DELETE FROM options_puts WHERE expiration_date < cutoff_date;
    GET DIAGNOSTICS deleted_rows = ROW_COUNT;
    RAISE NOTICE 'Deleted % expired options_puts records', deleted_rows;

    -- Archive old price data if needed (keep 7 years by default)
    -- Note: Be very careful with this in production!
    IF retention_days < 2555 THEN
        RAISE NOTICE 'Archiving price_history data older than % days is disabled for safety', retention_days;
        -- Uncomment only if you want to delete old price data
        -- DELETE FROM price_history WHERE date < cutoff_date;
        -- GET DIAGNOSTICS deleted_rows = ROW_COUNT;
        -- RAISE NOTICE 'Archived % old price_history records', deleted_rows;
    END IF;

    RAISE NOTICE 'Cleanup completed';
END;
$$ LANGUAGE plpgsql;

-- Index usage monitoring
CREATE OR REPLACE FUNCTION monitor_index_usage()
RETURNS TABLE(
    table_name TEXT,
    index_name TEXT,
    scans BIGINT,
    tuples_read BIGINT,
    tuples_fetched BIGINT,
    efficiency NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname||'.'||relname as table_name,
        indexrelname as index_name,
        idx_scan as scans,
        idx_tup_read as tuples_read,
        idx_tup_fetch as tuples_fetched,
        CASE WHEN idx_tup_read > 0
             THEN round((idx_tup_fetch::NUMERIC / idx_tup_read) * 100, 2)
             ELSE 0
        END as efficiency
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    ORDER BY idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

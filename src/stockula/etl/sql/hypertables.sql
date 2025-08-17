-- TimescaleDB Hypertable Creation Scripts
-- This file contains hypertable creation statements for time-series optimization

-- Create hypertable for price_history if not already created
SELECT CASE
    WHEN NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'price_history'
    )
    THEN create_hypertable('price_history', 'timestamp', chunk_time_interval => INTERVAL '1 week')
    ELSE NULL
END;

-- Create hypertable for etl_runs if not already created
SELECT CASE
    WHEN NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'etl_runs'
    )
    THEN create_hypertable('etl_runs', 'start_time', chunk_time_interval => INTERVAL '1 month')
    ELSE NULL
END;

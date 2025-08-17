# TimescaleDB Setup for Stockula Trading Platform

This directory contains a complete TimescaleDB setup optimized for financial time series data, providing
high-performance storage, advanced analytics, and comprehensive monitoring for the Stockula trading platform.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Stockula App  │    │  PgBouncer   │    │   TimescaleDB   │
│                 │───▶│ (Connection  │───▶│   (Primary)     │
│                 │    │  Pooling)    │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────┐                        ┌─────────────────┐
│     Redis       │                        │   Monitoring    │
│   (Caching)     │                        │   (Prometheus/  │
│                 │                        │    Grafana)     │
└─────────────────┘                        └─────────────────┘
```

## Features

### 🚀 Performance Optimizations

- **Hypertables**: Automatic partitioning by time for optimal query performance
- **Compression**: Advanced compression algorithms reducing storage by 90%+
- **Continuous Aggregates**: Pre-computed OHLCV data at multiple time intervals
- **Intelligent Indexing**: Optimized indexes for financial query patterns
- **Connection Pooling**: PgBouncer for efficient connection management

### 📊 Financial Data Models

- **Price History**: OHLCV data with support for multiple intervals
- **Options Chain**: Complete options data with Greeks calculation
- **Corporate Actions**: Dividends, splits, and other corporate events
- **Market Analytics**: Sector performance and market-wide metrics
- **Strategy Backtesting**: Integrated strategy and performance tracking

### 🔍 Monitoring & Alerting

- **Real-time Metrics**: Comprehensive monitoring via Prometheus
- **Custom Dashboards**: Pre-built Grafana dashboards for financial data
- **Intelligent Alerts**: Early warning system for data quality and performance
- **Health Checks**: Automated monitoring of database and application health

### 🔒 Operational Excellence

- **Automated Backups**: Full and incremental backups with retention policies
- **Point-in-Time Recovery**: Complete disaster recovery capabilities
- **Maintenance Automation**: Scheduled maintenance tasks and optimization
- **Security**: Secure configuration with authentication and access controls

## Quick Start

### 1. Environment Setup

Create a `.env` file with your configuration:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_grafana_password

# Performance Tuning (adjust based on your hardware)
TS_TUNE_MEMORY=4GB
TS_TUNE_NUM_CPUS=4
TS_TUNE_MAX_CONNS=200

# Backup Configuration
RETENTION_DAYS=30
BACKUP_DIR=/backups
```

### 2. Start the Stack

```bash
# Start all services
docker-compose -f docker-compose.timescale.yml up -d

# Start specific services
docker-compose -f docker-compose.timescale.yml up -d timescaledb pgbouncer redis

# Include monitoring
docker-compose -f docker-compose.timescale.yml up -d timescaledb pgbouncer redis grafana prometheus
```

### 3. Verify Installation

```bash
# Check database connectivity
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "SELECT version();"

# Verify TimescaleDB extension
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "SELECT extversion FROM pg_extension WHERE extname='timescaledb';"

# Check hypertables
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "SELECT * FROM timescaledb_information.hypertables;"
```

### 4. Access Monitoring

- **Grafana**: http://localhost:3000 (admin/your_grafana_password)
- **Prometheus**: http://localhost:9090
- **Database**: localhost:5432 (via TimescaleDB) or localhost:6432 (via PgBouncer)

## Database Schema

### Core Tables

#### Stocks (Reference Data)

```sql
-- Basic stock metadata
stocks (
    symbol VARCHAR(20) PRIMARY KEY,
    name TEXT,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    exchange VARCHAR(50),
    currency VARCHAR(10),
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
```

#### Price History (Hypertable)

```sql
-- Main time-series table for OHLCV data
price_history (
    id BIGINT,
    symbol VARCHAR(20),
    date DATE,
    time TIMESTAMPTZ,  -- Partition key
    open_price NUMERIC(15,4),
    high_price NUMERIC(15,4),
    low_price NUMERIC(15,4),
    close_price NUMERIC(15,4),
    adj_close_price NUMERIC(15,4),
    volume BIGINT,
    interval VARCHAR(10),
    daily_return NUMERIC(10,6),
    volatility NUMERIC(10,6),
    trading_value NUMERIC(20,2) GENERATED ALWAYS AS (close_price * volume) STORED
)
```

#### Options Data (Hypertables)

```sql
-- Options calls and puts with Greeks
options_calls/options_puts (
    id BIGINT,
    symbol VARCHAR(20),
    expiration_date DATE,
    time TIMESTAMPTZ,  -- Partition key
    strike NUMERIC(15,4),
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
    contract_symbol VARCHAR(50)
)
```

### Continuous Aggregates

Pre-computed aggregations for performance:

- `price_history_daily` - Daily OHLCV from intraday data
- `price_history_weekly` - Weekly aggregations
- `price_history_monthly` - Monthly aggregations
- `market_volume_hourly` - Market-wide volume analysis
- `options_volume_daily` - Options activity summaries

## Performance Tuning

### Memory Configuration

For a system with 8GB RAM:

```sql
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 64MB
maintenance_work_mem = 512MB
```

### TimescaleDB Settings

```sql
-- Parallel processing
timescaledb.max_background_workers = 16
max_worker_processes = 32
max_parallel_workers = 8

-- Compression settings
timescaledb.compress_segmentby = 'symbol'
timescaledb.compress_orderby = 'time DESC'
```

### Index Strategy

Key indexes for optimal performance:

```sql
-- Time-based queries (most common)
CREATE INDEX ON price_history (symbol, time DESC);
CREATE INDEX ON price_history (time DESC, symbol);

-- Volume analysis
CREATE INDEX ON price_history (volume DESC) WHERE volume > 0;

-- Technical analysis
CREATE INDEX ON price_history (daily_return) WHERE daily_return IS NOT NULL;
CREATE INDEX ON price_history (volatility) WHERE volatility IS NOT NULL;
```

## Backup and Recovery

### Automated Backups

```bash
# Run full backup (Sundays)
docker-compose -f docker-compose.timescale.yml run --rm backup

# Force full backup
FORCE_FULL_BACKUP=true docker-compose -f docker-compose.timescale.yml run --rm backup

# Run incremental backup
docker-compose -f docker-compose.timescale.yml run --rm backup
```

### Restore Operations

```bash
# Restore from full backup
./database/timescale/backup/restore.sh /backups/full/stockula_backup_20240816_120000.dump

# Restore with database recreation
./database/timescale/backup/restore.sh --drop-existing /backups/full/stockula_backup_20240816_120000.dump

# Verify backup integrity
./database/timescale/backup/restore.sh --verify-only /backups/full/stockula_backup_20240816_120000.dump
```

## Maintenance Operations

### Regular Maintenance

```bash
# Run all maintenance tasks
./database/timescale/backup/maintenance.sh

# Specific maintenance tasks
./database/timescale/backup/maintenance.sh vacuum
./database/timescale/backup/maintenance.sh statistics
./database/timescale/backup/maintenance.sh reindex
./database/timescale/backup/maintenance.sh timescaledb

# Health checks
./database/timescale/backup/maintenance.sh health
```

### Compression Management

```sql
-- Check compression status
SELECT
    hypertable_name,
    compression_enabled,
    compressed_chunks,
    uncompressed_chunks
FROM timescaledb_information.hypertables;

-- Manual compression
SELECT compress_chunk(i)
FROM show_chunks('price_history', older_than => INTERVAL '1 month') i;

-- Decompress if needed
SELECT decompress_chunk(i)
FROM show_chunks('price_history', newer_than => INTERVAL '1 week') i;
```

## Monitoring and Alerting

### Key Metrics

1. **Database Performance**

   - Connection count and utilization
   - Query response times
   - Cache hit ratios
   - Lock contention

1. **TimescaleDB Specific**

   - Compression ratios
   - Chunk sizes and counts
   - Background job status
   - Continuous aggregate freshness

1. **Data Quality**

   - Missing price data
   - Stale data detection
   - Data ingestion rates
   - Anomaly detection

1. **System Resources**

   - CPU and memory usage
   - Disk space and I/O
   - Network connectivity
   - Backup success rates

### Alert Thresholds

Critical alerts:

- Database connectivity failures
- Connection pool exhaustion (>95%)
- Disk space critical (\<5%)
- Backup failures

Warning alerts:

- High connection usage (>80%)
- Long-running queries (>1 hour)
- Low compression ratios (\<50%)
- Stale data (>1 hour old)

## Security Configuration

### Database Security

```sql
-- Create read-only user for reporting
CREATE USER stockula_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE stockula TO stockula_readonly;
GRANT USAGE ON SCHEMA public TO stockula_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO stockula_readonly;

-- Create backup user
CREATE USER stockula_backup WITH PASSWORD 'backup_password';
GRANT CONNECT ON DATABASE stockula TO stockula_backup;
ALTER USER stockula_backup WITH REPLICATION;
```

### Network Security

- TimescaleDB only accessible via PgBouncer
- Internal network isolation
- SSL/TLS encryption for external connections
- Firewall rules for specific ports

## Troubleshooting

### Common Issues

1. **Connection Pool Exhaustion**

   ```bash
   # Check active connections
   SELECT count(*) FROM pg_stat_activity;

   # Increase pool size in pgbouncer.ini
   default_pool_size = 100
   ```

1. **Slow Queries**

   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

1. **Compression Issues**

   ```sql
   -- Check compression status
   SELECT * FROM timescaledb_information.compression_settings;

   -- Recompress chunks
   SELECT compress_chunk(i, if_not_compressed => true)
   FROM show_chunks('price_history') i;
   ```

1. **Memory Issues**

   ```bash
   # Check memory usage
   SELECT name, setting, unit FROM pg_settings WHERE name LIKE '%mem%';

   # Adjust work_mem for large queries
   SET work_mem = '256MB';
   ```

### Log Analysis

```bash
# View PostgreSQL logs
docker-compose -f docker-compose.timescale.yml logs timescaledb

# Monitor real-time logs
docker-compose -f docker-compose.timescale.yml logs -f timescaledb

# Check specific errors
docker-compose -f docker-compose.timescale.yml logs timescaledb | grep ERROR
```

## Integration with Stockula

### Connection Configuration

Update your Stockula configuration:

```python
# Database URL for TimescaleDB
DATABASE_URL = "postgresql://stockula:password@localhost:6432/stockula"

# Or via PgBouncer for connection pooling
DATABASE_URL = "postgresql://stockula:password@localhost:6432/stockula"

# Redis for caching
REDIS_URL = "redis://localhost:6379/0"
```

### SQLModel Integration

The schema is fully compatible with your existing SQLModel models. No changes required to your application code.

### Performance Recommendations

1. **Batch Inserts**: Use batch operations for bulk data insertion
1. **Prepared Statements**: Leverage prepared statements for repeated queries
1. **Connection Pooling**: Always use PgBouncer for production
1. **Caching**: Implement Redis caching for frequently accessed data
1. **Compression**: Let TimescaleDB handle compression automatically

## Support and Maintenance

### Scheduled Tasks

Recommended maintenance schedule:

- **Daily**: Health checks, backup verification
- **Weekly**: Statistics update, index maintenance
- **Monthly**: Full vacuum, compression review
- **Quarterly**: Performance analysis, capacity planning

### Capacity Planning

Monitor these metrics for capacity planning:

- Database growth rate (GB/month)
- Query performance trends
- Connection usage patterns
- Compression effectiveness
- I/O utilization

### Upgrade Path

For TimescaleDB upgrades:

1. Test in staging environment
1. Create full backup
1. Perform rolling upgrade
1. Verify functionality
1. Monitor performance

## Resources

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## License

This TimescaleDB setup is part of the Stockula project and follows the same licensing terms.

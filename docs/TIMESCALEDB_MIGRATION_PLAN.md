# TimescaleDB Migration Plan for Stockula - IMMEDIATE DEPLOYMENT

## Executive Summary

This document outlines the **immediate migration plan** to deploy Stockula with TimescaleDB within **24-48 hours**. The
migration leverages existing automation scripts, Docker-based deployment, and parallel execution to minimize downtime
while delivering 10-100x performance improvements and reduced API dependency.

## 🎯 Migration Objectives

1. **Reduce Yahoo Finance API Calls**: Store all historical data locally in TimescaleDB
1. **Improve Performance**: Leverage TimescaleDB's time series optimizations for 10-100x query speedup
1. **Enable Advanced Analytics**: Use continuous aggregates, AI/ML capabilities via pgai
1. **Maintain Architecture**: Preserve existing SQLModel ORM and dependency injection patterns
1. **Ensure Data Quality**: Implement comprehensive validation and monitoring

## 📊 Current State Analysis

### Existing Architecture

- **Database**: SQLite with SQLModel ORM
- **Data Source**: Yahoo Finance API via yfinance library
- **Caching**: Simple staleness checks in SQLite
- **Tables**: Stocks, PriceHistory, Dividends, Splits, Options, StockInfo
- **Pattern**: Repository pattern with dependency injection

### Pain Points

- Frequent API rate limiting from Yahoo Finance
- Slow historical data queries in SQLite
- Limited analytical capabilities
- No real-time data processing
- Storage inefficiency for time series data

## 🏗️ Target Architecture

### Technology Stack

- **Database**: TimescaleDB (PostgreSQL 16 with TimescaleDB extension)
- **Extensions**: pgai (AI/ML), vector (embeddings), plpython3u
- **Connection Pooling**: PgBouncer
- **Orchestration**: Apache Airflow for ETL
- **Monitoring**: Prometheus + Grafana
- **Caching**: Redis for session management
- **Deployment**: Docker Compose with optional Kubernetes

### Database Design

#### Hypertables (Time Series Optimized)

```sql
-- Price history with 1-month chunks
CREATE TABLE price_history (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DECIMAL(20,6),
    high DECIMAL(20,6),
    low DECIMAL(20,6),
    close DECIMAL(20,6),
    adjusted_close DECIMAL(20,6),
    volume BIGINT,
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('price_history', 'time', chunk_time_interval => INTERVAL '1 month');

-- Dividends with 1-year chunks
CREATE TABLE dividends (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    amount DECIMAL(20,6),
    PRIMARY KEY (symbol, time)
);
SELECT create_hypertable('dividends', 'time', chunk_time_interval => INTERVAL '1 year');
```

#### Continuous Aggregates

```sql
-- Pre-computed OHLC at different intervals
CREATE MATERIALIZED VIEW ohlc_daily
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day'::interval, time) AS date,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume
FROM price_history
GROUP BY symbol, date;
```

### Performance Optimizations

- **Compression**: Automatic compression for data >90 days old (90% storage reduction)
- **Indexes**: Optimized for symbol+time queries
- **Connection Pooling**: PgBouncer with transaction-level pooling
- **Batch Operations**: COPY protocol for bulk inserts
- **Parallel Processing**: Multiple symbols processed concurrently

## ⚡ IMMEDIATE DEPLOYMENT PLAN (24-48 Hours)

### 🚀 Quick Start Guide (Hour 0-2)

**Prerequisites Check:**

```bash
# Verify Docker and dependencies
docker --version && docker-compose --version && uv --version
cd /Users/mitchellmurphy/Developer/github.com/mkm29/stockula
git checkout feature/timescaledb  # Use existing branch
```

**One-Command Deployment:**

```bash
# Deploy complete TimescaleDB stack with monitoring
./scripts/setup-timescaledb.sh --production --monitoring --auto-migrate
```

This automated script will:

- Deploy TimescaleDB + pgai + PgBouncer + Redis
- Set up Prometheus + Grafana monitoring
- Create database schema with hypertables
- Migrate existing SQLite data
- Configure continuous aggregates
- Start health checks

### 📋 Parallel Execution Phases

#### Phase A: Infrastructure (Hours 0-4) - **AUTOMATED**

Run in parallel with other phases:

```bash
# Terminal 1: Infrastructure deployment
./scripts/setup-timescaledb.sh --stack-only &
INFRA_PID=$!

# Terminal 2: Database preparation
./scripts/prepare-database.sh &
DB_PID=$!

# Terminal 3: Code preparation
./scripts/prepare-code.sh &
CODE_PID=$!

# Wait for all parallel tasks
wait $INFRA_PID $DB_PID $CODE_PID
```

**Deliverables (Pre-built):**

- ✅ `docker-compose.timescale.yml` - Complete stack definition
- ✅ `scripts/setup-timescaledb.sh` - Automated setup script
- ✅ `database/timescale/` - Configuration files
- ✅ Monitoring dashboards (Grafana)

#### Phase B: Data Migration (Hours 2-8) - **PARALLEL**

```bash
# Run multiple migration streams in parallel
./scripts/migrate-data.sh --parallel --workers=4 --symbols=AAPL,GOOGL,MSFT,TSLA &
./scripts/migrate-data.sh --parallel --workers=4 --symbols=META,AMZN,NFLX,NVDA &
./scripts/migrate-data.sh --parallel --workers=4 --symbols=SPY,QQQ,VTI,BND &

# Monitor progress
watch -n 5 "docker exec stockula-timescale psql -U stockula -d stockula -c \"SELECT symbol, COUNT(*) FROM price_history GROUP BY symbol;\""
```

**Deliverables (Auto-generated):**

- ✅ Database schema with hypertables
- ✅ Continuous aggregates for OHLC data
- ✅ Compression policies (90-day threshold)
- ✅ Data validation reports
- ✅ Performance indexes

#### Phase C: Code Integration (Hours 4-12) - **MINIMAL CHANGES**

**Key Changes Required:**

```python
# src/stockula/config/database.py - ADD ONLY
TIMESCALE_CONFIG = {
    "url": "postgresql://stockula:password@localhost:5432/stockula",
    "pool_size": 20,
    "echo": False
}

# src/stockula/container.py - SINGLE LINE CHANGE
def create_container(use_timescale: bool = True):
    # Change from: use_sqlite_repository()
    # Change to: use_timescale_repository() if use_timescale else use_sqlite_repository()
```

**Backward Compatibility Maintained:**

- SQLite fallback via environment variable
- Existing CLI commands unchanged
- No breaking API changes

#### Phase D: Validation (Hours 8-16) - **AUTOMATED**

```bash
# Comprehensive validation suite
./scripts/validate-migration.sh --comprehensive

# Performance benchmarking
./scripts/benchmark.sh --compare-sqlite-timescale

# Data integrity checks
./scripts/data-integrity.sh --full-validation
```

**Success Criteria (Automated Checks):**

- ✅ All existing tests pass (pytest)
- ✅ 10x+ query performance improvement
- ✅ \<0.1% data discrepancy vs SQLite
- ✅ 99.9% data completeness
- ✅ Sub-second response times

#### Phase E: Production Cutover (Hours 16-24) - **CONTROLLED**

```bash
# Blue-green deployment approach
export STOCKULA_DB_TYPE=timescale
export STOCKULA_FALLBACK_ENABLED=true

# Test production workload
uv run python -m stockula --ticker AAPL --mode ta --use-timescale
uv run python -m stockula --config portfolio.yaml --mode backtest --use-timescale

# Monitor for 2 hours, then disable fallback
export STOCKULA_FALLBACK_ENABLED=false
```

## 🔄 Immediate Migration Strategy

### Zero-Downtime Approach

1. **Parallel Deployment** (Hours 0-8)

   - Deploy TimescaleDB alongside SQLite
   - Automated data sync with validation
   - Real-time consistency monitoring

1. **Feature-Flag Cutover** (Hours 8-16)

   - Route read queries to TimescaleDB via environment flag
   - Maintain SQLite for writes during validation
   - Automated performance comparison

1. **Full Migration** (Hours 16-24)

   - Switch all operations to TimescaleDB
   - Keep SQLite backup for 48 hours
   - Monitor production metrics

### 🚨 Rollback Plan (< 5 Minutes)

**Immediate Rollback Command:**

```bash
# Emergency rollback to SQLite
export STOCKULA_DB_TYPE=sqlite
export STOCKULA_EMERGENCY_MODE=true

# Restart services
docker-compose restart stockula-app

# Verify rollback
uv run python -m stockula --ticker AAPL --mode ta --verify-db-type
```

**Rollback Triggers:**

- Query performance degradation >50%
- Data inconsistency >0.1%
- System error rate >1%
- Response time >2x baseline

**Safety Measures:**

- SQLite database preserved for 48 hours
- Automated health checks every 30 seconds
- Circuit breaker pattern for database connections
- Real-time alerting on critical metrics

## 📈 Expected Benefits

### Performance Improvements

| Operation              | SQLite | TimescaleDB | Improvement |
| ---------------------- | ------ | ----------- | ----------- |
| Daily OHLC Query       | 500ms  | 10ms        | 50x         |
| 1-Year Historical      | 5s     | 100ms       | 50x         |
| Bulk Insert (10k rows) | 10s    | 500ms       | 20x         |
| Storage (1M rows)      | 100MB  | 10MB        | 10x         |

### Operational Benefits

- **Reduced API Calls**: 99% reduction in Yahoo Finance requests
- **Real-time Analytics**: Sub-second queries on millions of rows
- **Advanced Features**: AI/ML capabilities, vector search
- **Cost Optimization**: 90% storage reduction via compression
- **Scalability**: Handle 100x more symbols

## 🛠️ Technical Implementation Details

### Connection Configuration

```python
# src/stockula/config/database.py
class TimescaleDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "stockula"
    user: str = "stockula"
    password: SecretStr
    pool_size: int = 20
    max_overflow: int = 40

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
```

### Repository Pattern Updates

```python
# src/stockula/data/timescale_repository.py
class TimescaleDataRepository(DataRepository):
    def __init__(self, session: Session):
        self.session = session

    async def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        # Use continuous aggregates for performance
        query = """
            SELECT * FROM ohlc_daily
            WHERE symbol = :symbol
            AND date BETWEEN :start AND :end
            ORDER BY date
        """
        return pd.read_sql(query, self.session.bind, params={
            "symbol": symbol,
            "start": start,
            "end": end
        })
```

### Dependency Injection Updates

```python
# src/stockula/container.py
def create_container(config: Config) -> Container:
    container = Container()

    if config.database.type == "timescaledb":
        from .data.timescale_repository import TimescaleDataRepository
        container.register(DataRepository, TimescaleDataRepository)
    else:
        # Fallback to SQLite
        from .data.sqlite_repository import SQLiteDataRepository
        container.register(DataRepository, SQLiteDataRepository)

    return container
```

## 📊 Day-1 Operations Runbook

### Immediate Monitoring Setup

**Health Check Dashboard:**

```bash
# Access monitoring endpoints
echo "TimescaleDB Health: http://localhost:3000/d/timescale-health"
echo "Application Metrics: http://localhost:3000/d/stockula-app"
echo "Query Performance: http://localhost:3000/d/query-performance"

# Quick health verification
curl -f http://localhost:5432 && echo "TimescaleDB: UP"
curl -f http://localhost:3000 && echo "Grafana: UP"
curl -f http://localhost:9090 && echo "Prometheus: UP"
```

### Critical Metrics to Watch (First 24 Hours)

| Metric              | Threshold       | Action                       |
| ------------------- | --------------- | ---------------------------- |
| Query Response Time | >1s             | Check continuous aggregates  |
| Data Ingestion Rate | \<1000 rows/min | Verify ETL pipeline          |
| Memory Usage        | >80%            | Scale PgBouncer pool         |
| Disk Space          | >70%            | Check compression policies   |
| Error Rate          | >0.5%           | Investigate logs immediately |

### Alert Configuration

**Immediate Alerts (Slack/PagerDuty):**

```yaml
# prometheus/rules.yml
groups:
  - name: stockula_critical
    rules:
      - alert: DatabaseDown
        expr: up{job="timescaledb"} == 0
        for: 30s

      - alert: QueryLatencyHigh
        expr: histogram_quantile(0.95, query_duration_seconds_bucket) > 1.0
        for: 2m

      - alert: DataStaleness
        expr: time() - last_price_update_timestamp > 86400
        for: 5m
```

### Operational Commands

**Daily Operations:**

```bash
# Check database health
docker exec stockula-timescale pg_isready -U stockula

# Verify data freshness
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT symbol, MAX(time) as last_update FROM price_history GROUP BY symbol ORDER BY last_update DESC LIMIT 10;"

# Monitor compression status
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT chunk_schema, chunk_name, compression_status FROM timescaledb_information.chunks ORDER BY chunk_name;"

# Check continuous aggregate refresh
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT view_name, completed_threshold FROM timescaledb_information.continuous_aggregates;"
```

### Performance Optimization

**Immediate Tuning (Applied Automatically):**

```sql
-- Connection pooling (PgBouncer)
max_client_conn = 200
default_pool_size = 20
server_round_robin = 1

-- TimescaleDB settings
shared_preload_libraries = 'timescaledb'
max_connections = 100
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 256MB
maintenance_work_mem = 1GB
```

### Backup Verification

**Automated Backup Check:**

```bash
# Verify backup integrity (runs every 6 hours)
./scripts/verify-backup.sh --check-integrity --test-restore

# Manual backup verification
docker exec stockula-timescale pg_dump -U stockula -d stockula | head -20
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### 1. **Slow Query Performance**

**Symptoms:**

- Queries taking >1 second
- High CPU usage on TimescaleDB
- Application timeouts

**Diagnosis:**

```bash
# Check active queries
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT query, state, query_start FROM pg_stat_activity WHERE state = 'active';"

# Check index usage
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT schemaname, tablename, attname, n_distinct, correlation FROM pg_stats WHERE tablename = 'price_history';"
```

**Solutions:**

```bash
# Refresh continuous aggregates
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "CALL refresh_continuous_aggregate('ohlc_daily', NULL, NULL);"

# Check compression
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT compress_chunk(chunk) FROM show_chunks('price_history', older_than => INTERVAL '90 days') AS chunk;"
```

#### 2. **Data Inconsistency**

**Symptoms:**

- Validation scripts failing
- Missing price data
- Duplicate records

**Diagnosis:**

```bash
# Data completeness check
./scripts/data-integrity.sh --symbols=AAPL,GOOGL --date-range=30d

# Compare with SQLite backup
./scripts/compare-databases.sh --sqlite-backup=./stockula_backup.db --timescale
```

**Solutions:**

```bash
# Re-run migration for specific symbols
./scripts/migrate-data.sh --force --symbols=AAPL --start-date=2024-01-01

# Fix duplicate records
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "DELETE FROM price_history WHERE ctid NOT IN (SELECT min(ctid) FROM price_history GROUP BY symbol, time);"
```

#### 3. **Connection Pool Exhaustion**

**Symptoms:**

- "Connection pool exhausted" errors
- Application hanging
- High connection count

**Diagnosis:**

```bash
# Check active connections
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# PgBouncer status
docker exec stockula-pgbouncer psql -p 6432 -U stockula pgbouncer -c "SHOW POOLS;"
```

**Solutions:**

```bash
# Increase pool size temporarily
docker exec stockula-pgbouncer psql -p 6432 -U stockula pgbouncer -c "SET default_pool_size = 30;"

# Restart PgBouncer
docker-compose restart stockula-pgbouncer
```

#### 4. **Disk Space Issues**

**Symptoms:**

- Disk usage >80%
- Insert failures
- Slow compression

**Diagnosis:**

```bash
# Check chunk sizes
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT chunk_schema, chunk_name, pg_size_pretty(total_bytes) FROM timescaledb_information.chunks ORDER BY total_bytes DESC LIMIT 10;"
```

**Solutions:**

```bash
# Force compression on old chunks
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT compress_chunk(chunk) FROM show_chunks('price_history', older_than => INTERVAL '30 days') AS chunk;"

# Drop old chunks (if data retention policy allows)
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "SELECT drop_chunks('price_history', older_than => INTERVAL '2 years');"
```

### Emergency Procedures

#### Complete System Reset (Nuclear Option)

**⚠️ WARNING: This will destroy all TimescaleDB data**

```bash
# 1. Export current data
./scripts/emergency-export.sh --output=/tmp/stockula_export.sql

# 2. Reset TimescaleDB stack
docker-compose -f docker-compose.timescale.yml down -v
docker-compose -f docker-compose.timescale.yml up -d

# 3. Wait for services to be ready (2-3 minutes)
./scripts/wait-for-services.sh

# 4. Re-run migration
./scripts/setup-timescaledb.sh --auto-migrate --force

# 5. Verify system health
./scripts/validate-migration.sh --quick
```

#### Performance Emergency

```bash
# 1. Enable emergency mode (reduces precision for speed)
export STOCKULA_EMERGENCY_MODE=true
export STOCKULA_QUERY_TIMEOUT=5

# 2. Disable continuous aggregate refreshes temporarily
docker exec stockula-timescale psql -U stockula -d stockula -c \
  "ALTER MATERIALIZED VIEW ohlc_daily SET (timescaledb.materialized_only = false);"

# 3. Monitor for 30 minutes, then re-enable
export STOCKULA_EMERGENCY_MODE=false
```

### Escalation Procedures

**Level 1 - Automated Recovery:** Scripts handle common issues **Level 2 - Manual Intervention:** Follow troubleshooting
guide **Level 3 - Emergency Rollback:** Use 5-minute rollback procedure **Level 4 - Complete Reset:** Nuclear option
with data export

### Log Analysis

**Key Log Locations:**

```bash
# Application logs
docker logs stockula-app --tail=100 -f

# TimescaleDB logs
docker logs stockula-timescale --tail=100 -f

# PgBouncer logs
docker logs stockula-pgbouncer --tail=100 -f

# Grafana logs
docker logs stockula-grafana --tail=100 -f
```

**Critical Error Patterns:**

```bash
# Search for critical errors
docker logs stockula-app 2>&1 | grep -E "(CRITICAL|ERROR|FATAL)" | tail -20

# Check for connection issues
docker logs stockula-app 2>&1 | grep -E "(connection|timeout|pool)" | tail -20

# Monitor query performance
docker logs stockula-timescale 2>&1 | grep -E "(slow|duration)" | tail -20
```

## 🔐 Security Considerations

- Encrypted connections (SSL/TLS)
- Role-based access control
- Audit logging
- Secrets management via environment variables
- Network isolation with Docker networks

## 📚 Documentation & Training

### Documentation Updates

- [ ] Update README with TimescaleDB setup
- [ ] API documentation for new queries
- [ ] Operational runbooks
- [ ] Troubleshooting guide

### Team Training

- [ ] TimescaleDB fundamentals
- [ ] Time series best practices
- [ ] Monitoring and alerting
- [ ] Incident response procedures

## ✅ Immediate Success Criteria (24-48 Hours)

### Hour 8 Validation (Infrastructure Ready)

- [ ] All Docker services running (TimescaleDB, Grafana, Prometheus)
- [ ] Database schema created with hypertables
- [ ] Monitoring dashboards accessible
- [ ] Basic connectivity tests pass

### Hour 16 Validation (Data Migration Complete)

- [ ] 100% data migration success rate
- [ ] \<0.1% data discrepancy vs SQLite source
- [ ] All price history, dividends, splits migrated
- [ ] Continuous aggregates populated

### Hour 24 Validation (Production Ready)

- [ ] **Performance**: 10x+ improvement in query speed
- [ ] **Reliability**: Zero critical alerts for 2+ hours
- [ ] **Compatibility**: All existing CLI commands work
- [ ] **Monitoring**: Real-time metrics collection active
- [ ] **Fallback**: Emergency rollback tested and functional

### Week 1 Validation (Stability Confirmed)

- [ ] 99.9% uptime achieved
- [ ] Query performance maintained under load
- [ ] Data pipeline automated via Airflow
- [ ] Backup/restore procedures verified
- [ ] Team trained on operational procedures

### Automated Validation Commands

```bash
# Run complete validation suite
./scripts/validate-deployment.sh --comprehensive

# Expected output:
# ✅ Infrastructure: PASS
# ✅ Data Migration: PASS
# ✅ Performance: PASS (15.2x improvement)
# ✅ Monitoring: PASS
# ✅ Fallback: PASS
#
# 🎉 TimescaleDB migration SUCCESSFUL
# Ready for production workloads
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
cd /Users/mitchellmurphy/Developer/github.com/mkm29/stockula

# 2. Deploy TimescaleDB stack
./scripts/setup-timescaledb.sh --monitoring --app

# 3. Run migration
uv run python -m stockula.etl.migrate --from-sqlite --to-timescale

# 4. Verify deployment
docker-compose -f docker-compose.timescale.yml ps

# 5. Access services
# Database: postgresql://stockula:password@localhost:5432/stockula
# Grafana: http://localhost:3000 (admin/admin)
# Airflow: http://localhost:8080 (admin/admin)
```

## ⏱️ 24-Hour Timeline Summary

| Hour  | Phase            | Key Activities                      | Deliverables                           |
| ----- | ---------------- | ----------------------------------- | -------------------------------------- |
| 0-4   | Infrastructure   | Deploy stack, setup monitoring      | Docker services UP, Grafana dashboards |
| 2-8   | Data Migration   | Parallel data sync, validation      | Historical data in TimescaleDB         |
| 4-12  | Code Integration | Minimal changes, testing            | Feature flags, fallback ready          |
| 8-16  | Validation       | Performance tests, integrity checks | Benchmarks, validation reports         |
| 16-24 | Production       | Cutover, monitoring, optimization   | Live system, alerts configured         |
| 24-48 | Stabilization    | Fine-tuning, documentation          | Operational readiness                  |

### Key Milestones

- **Hour 2**: Infrastructure deployed and healthy
- **Hour 8**: Data migration 50% complete
- **Hour 16**: Full data migration validated
- **Hour 20**: Production cutover complete
- **Hour 24**: System stable, fallback tested
- **Hour 48**: Team trained, documentation complete

### Parallel Track Summary

Multiple teams can work simultaneously:

- **Infrastructure Team**: Docker deployment, monitoring setup
- **Data Team**: Migration scripts, validation tools
- **Development Team**: Code changes, testing
- **Operations Team**: Runbooks, alerting, procedures

This parallel approach reduces the critical path from 6 weeks to 24-48 hours.

## 🔗 Resources

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [pgai Extension Guide](https://github.com/timescale/pgai)
- [Docker Compose Configuration](./docker-compose.timescale.yml)
- [ETL Pipeline Design](./user-guide/etl_pipeline_design.md)
- [Setup Script](./scripts/setup-timescaledb.sh)

______________________________________________________________________

This migration plan provides a clear path from SQLite to TimescaleDB while maintaining Stockula's clean architecture and
ensuring minimal disruption to existing functionality. The phased approach allows for careful validation and rollback
capabilities at each stage.

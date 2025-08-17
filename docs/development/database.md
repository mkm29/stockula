# Database Development Guide

This document provides comprehensive guidance for database development in Stockula, including architecture, performance
requirements, and best practices.

## Database Architecture Overview

### Consolidated TimescaleDB Implementation

Stockula uses a **pure TimescaleDB implementation** with a dramatically simplified architecture:

```
src/stockula/database/
├── models.py           # SQLModel definitions with TimescaleDB hypertables
├── manager.py          # Consolidated DatabaseManager (1,784 lines)
├── interfaces.py       # Single IDatabaseManager interface
└── utilities/          # Database utilities
    ├── batch_operations.py # Batch data operations
    ├── monitoring.py       # Performance monitoring
    ├── reporting.py        # Database reporting
    └── validation.py       # Data validation
```

### Key Features

- **Single Manager Class**: All connection logic, query operations, and analytics in one consolidated manager
- **Pure TimescaleDB**: No SQLite fallback, optimized time-series handling
- **Integrated Analytics**: 8 built-in methods (moving averages, RSI, Bollinger Bands, etc.)
- **Connection Management**: Built-in pooling, session handling, and configuration management
- **Hypertable Optimization**: Automatic partitioning for price data by time
- **Type Safety**: Full SQLModel integration with comprehensive type hints

### Database Models

#### Core Tables (TimescaleDB Hypertables)

```python
class PriceHistory(SQLModel, table=True):
    """Time-series price data with automatic hypertable partitioning."""
    __tablename__ = "price_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="stocks.symbol")
    timestamp: datetime = Field(index=True)
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[int] = None
    interval: str = Field(default="1d")

class Stock(SQLModel, table=True):
    """Stock metadata and information."""
    __tablename__ = "stocks"

    symbol: str = Field(primary_key=True)
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
```

#### Supporting Tables

- **Dividends**: Dividend payment records
- **Splits**: Stock split history
- **Options**: Options chain data
- **StockInfo**: Extended stock metadata

## SQLModel Integration

### SQLModel Implementation ✅

Stockula uses SQLModel for type-safe database operations, providing:

#### Before: Raw SQL Approach

```python
# Creating tables with raw SQL
conn.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        -- ...
    )
""")

# Inserting data with manual parameter binding
conn.execute(
    "INSERT OR REPLACE INTO stocks (symbol, name, sector) VALUES (?, ?, ?)",
    (symbol, name, sector)
)
```

#### After: SQLModel Approach

```python
# Model definition with full type safety
class Stock(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    name: Optional[str] = None
    sector: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Type-safe operations
with session:
    stock = Stock(symbol=symbol, name=name, sector=sector)
    session.add(stock)
    session.commit()
```

### Benefits Achieved

| Feature                  | Raw SQL         | SQLModel     |
| ------------------------ | --------------- | ------------ |
| Type Safety              | ❌ No           | ✅ Yes       |
| SQL Injection Protection | ⚠️ Manual       | ✅ Automatic |
| IDE Support              | ❌ Limited      | ✅ Full      |
| Relationships            | ❌ Manual joins | ✅ Automatic |
| Validation               | ❌ None         | ✅ Pydantic  |
| Schema Management        | ✅ Alembic      | ✅ Alembic   |

### Current Implementation

```python
from stockula.database.manager import DatabaseManager

# Uses SQLModel internally for type safety
db = DatabaseManager()

# All operations are type-safe with full IDE support
stock_info = db.get_stock_info("AAPL")  # Returns Optional[Dict[str, Any]]
price_history = db.get_price_history("AAPL", "2024-01-01", "2024-12-31")  # Returns pd.DataFrame
```

## Performance Requirements

### Query Performance Targets

#### Time-Series Data Retrieval

- **Basic price history queries**: < 100ms for 1 year of daily data
- **Aggregated queries**: < 500ms for 5 years of data with time bucketing
- **Recent price changes**: < 200ms for 100 symbols over 24 hours
- **Volume profile analysis**: < 1000ms for 1 year of data with 20 price bins

#### Data Insertion Performance

- **Single symbol batch insert**: < 50ms for 1000 records
- **Multi-symbol batch insert**: < 500ms for 10,000 records across 100 symbols
- **Async batch operations**: 2x faster than synchronous equivalents

#### Memory Usage

- **Connection pools**: Maximum 20 connections per pool
- **Query result caching**: Maximum 100MB memory footprint
- **Batch operations**: Process 10,000+ records without memory spikes > 200MB

### Scalability Contracts

#### Data Volume Handling

- **Daily data ingestion**: Support 500+ symbols daily without performance degradation
- **Historical data storage**: Efficiently handle 5+ years of minute-level data
- **Concurrent operations**: Support 10+ concurrent read/write operations

#### Database Growth Management

- **Hypertable chunking**: Automatic chunk management for optimal performance
- **Compression policies**: 70%+ compression ratio for data older than 7 days
- **Retention policies**: Automatic cleanup of data older than 5 years

### Reliability Contracts

#### Connection Management

- **Connection pooling**: Automatic connection recycling every 3600 seconds
- **Failover handling**: Graceful degradation within 5 seconds
- **Recovery time**: Resume TimescaleDB operations within 30 seconds of restoration

#### Data Consistency

- **ACID compliance**: All multi-table operations must be transactional
- **Conflict resolution**: Upsert operations for duplicate timestamp/symbol pairs
- **Data Integrity**: ACID compliance for all transactions

## TimescaleDB Optimization

### Required Indexes

```sql
-- Primary time-based index (automatic with hypertables)
CREATE INDEX ON price_history (symbol, timestamp DESC);

-- Analytics support indexes
CREATE INDEX ON price_history (symbol, interval, timestamp);
CREATE INDEX ON price_history (timestamp, symbol) WHERE close_price IS NOT NULL;
```

### Optimized Query Patterns

```sql
-- Efficient time-range queries
SELECT * FROM price_history
WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
ORDER BY timestamp;

-- Optimized aggregations with time_bucket
SELECT time_bucket('1 day', timestamp) as bucket,
       first(open_price, timestamp),
       max(high_price),
       min(low_price),
       last(close_price, timestamp)
FROM price_history
WHERE symbol = $1 AND timestamp >= $2
GROUP BY bucket ORDER BY bucket;
```

### Connection Pool Configuration

```python
TimescaleDBConfig(
    pool_size=10,           # Base connection pool size
    max_overflow=20,        # Maximum additional connections
    pool_timeout=30,        # Connection acquisition timeout
    pool_recycle=3600,      # Connection recycling interval
    pool_pre_ping=True,     # Verify connections before use
)
```

## Analytics Methods

### Built-in Analytics (8 Methods)

The consolidated DatabaseManager includes 8 analytics methods with full TimescaleDB optimization:

#### 1. Moving Averages

```python
def get_moving_averages(self, symbol: str, periods: list[int] = None,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Calculate moving averages using TimescaleDB window functions."""
    # Uses optimized SQL with PARTITION BY and window functions
```

#### 2. Bollinger Bands

```python
def get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0,
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Calculate Bollinger Bands with standard deviation calculations."""
```

#### 3. RSI (Relative Strength Index)

```python
def get_rsi(self, symbol: str, period: int = 14,
           start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Calculate RSI using complex CTE-based SQL with LAG functions."""
```

#### 4. Price Momentum

```python
def get_price_momentum(self, symbols: list[str] = None,
                      lookback_days: int = 30, time_bucket: str = "1 day") -> pd.DataFrame:
    """Calculate price momentum across multiple timeframes."""
```

#### 5. Correlation Matrix

```python
def get_correlation_matrix(self, symbols: list[str],
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Calculate correlation matrix between multiple symbols."""
```

#### 6. Volatility Analysis

```python
def get_volatility_analysis(self, symbols: list[str] = None,
                           window_days: int = 30) -> pd.DataFrame:
    """Calculate rolling volatility metrics."""
```

#### 7. Seasonal Patterns

```python
def get_seasonal_patterns(self, symbols: list[str] = None,
                         years_back: int = 5) -> pd.DataFrame:
    """Analyze seasonal price patterns."""
```

#### 8. Top Performers

```python
def get_top_performers(self, limit: int = 10, timeframe: str = "1M") -> pd.DataFrame:
    """Find top performing stocks by return percentage."""
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

1. **Query Response Time**

   - P95 response time < 500ms for all standard operations
   - P99 response time < 2000ms for complex analytics queries

1. **Throughput Metrics**

   - Minimum 1000 price records/second insertion rate
   - Minimum 100 concurrent read operations/second

1. **Resource Efficiency**

   - CPU utilization < 70% during peak operations
   - Memory usage < 2GB for typical workloads
   - Disk I/O < 100 MB/s sustained

### Performance Testing Requirements

#### Load Testing Scenarios

1. **Daily data ingestion**: 500 symbols × 1440 minutes = 720,000 records
1. **Historical backfill**: 1 symbol × 5 years × 1440 minutes = 2.6M records
1. **Analytics workload**: 100 concurrent volume profile calculations
1. **Mixed workload**: 70% read, 30% write operations

#### Benchmark Baselines

- **PostgreSQL baseline**: Within 20% of raw PostgreSQL performance
- **TimescaleDB optimizations**: 3x improvement for time-series aggregations

## ETL Module Structure

### Clean SQL Separation

```
src/stockula/etl/
├── timescale_manager.py    # ETL operations manager with SQL file loading
└── sql/                    # External SQL files for maintainability
    ├── tables.sql          # Table creation statements
    ├── hypertables.sql     # Hypertable setup and configuration
    ├── indexes.sql         # Performance indexes
    ├── policies.sql        # Compression and retention policies
    └── aggregates.sql      # Continuous aggregates for analytics
```

### ETL Manager Features

- **SQL File Loading**: Dynamic loading of SQL from external files
- **Schema Initialization**: Automated TimescaleDB setup
- **Bulk Operations**: High-performance data insertion with COPY
- **Quality Metrics**: Comprehensive data quality monitoring
- **Performance Analytics**: Database statistics and health checks

## Data Management Guidelines

### Data Operations Performance

- **Bulk Operations**: < 2 hours for 1M price records
- **Query Response**: < 5 seconds for complex analytics
- **Validation**: 100% data integrity verification for all operations
- **Backup Recovery**: Complete backup restoration within 15 minutes if needed

### Data Quality Benchmarks

- **Data Accuracy**: 100% validation for all incoming data
- **Data Integrity**: Checksum validation for all critical operations
- **Temporal Consistency**: Chronological ordering maintained
- **Reference Integrity**: All foreign key relationships enforced

## Error Handling Standards

### Performance Degradation Response

1. **Query timeout**: Fallback to cached results within 5 seconds
1. **Connection exhaustion**: Queue operations with 30-second timeout
1. **Database unavailable**: Automatic failover mechanisms

### Monitoring and Alerting

- **Query performance**: Alert if P95 > 1000ms for 5 minutes
- **Connection pool**: Alert if utilization > 80% for 10 minutes
- **Error rate**: Alert if error rate > 5% for 5 minutes

## Best Practices

### Development Guidelines

1. **Use Context Managers**: Always use `with session:` for automatic cleanup
1. **Batch Operations**: Use `session.add_all()` for bulk inserts
1. **Lazy Loading**: Be aware of N+1 query problems with relationships
1. **Indexes**: Define indexes in the model for performance
1. **Validation**: Leverage Pydantic validators for data integrity

### Query Optimization

```python
# Efficient querying with SQLModel
stmt = select(PriceHistory).where(
    PriceHistory.symbol == symbol,
    PriceHistory.timestamp >= start_date
).order_by(PriceHistory.timestamp)

results = session.exec(stmt).all()
```

### Bulk Operations

```python
# Bulk insert with SQLModel
with session:
    price_records = [
        PriceHistory(**record) for record in price_data_list
    ]
    session.add_all(price_records)
    session.commit()
```

## Configuration

### TimescaleDB Connection

```yaml
# Required: TimescaleDB connection settings
database:
  host: localhost
  port: 5432
  name: stockula
  user: stockula_user
  password: ${STOCKULA_DB_PASSWORD}

# TimescaleDB-specific settings
timescale:
  chunk_time_interval: "7 days"
  compression_after: "30 days"
  retention_policy: "2 years"
  enable_hypertables: true
```

### Environment Variables

```bash
# Required for database services
POSTGRES_PASSWORD=your_secure_password
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=stockula
TIMESCALEDB_USER=stockula_user

# Optional TimescaleDB tuning
TS_TUNE_MEMORY=4GB
TS_TUNE_NUM_CPUS=4
```

## Docker Integration

### TimescaleDB Service

```yaml
services:
  stockula-timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: stockula
      POSTGRES_USER: stockula_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      TS_TUNE_MEMORY: ${TS_TUNE_MEMORY:-4GB}
      TS_TUNE_NUM_CPUS: ${TS_TUNE_NUM_CPUS:-4}
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data
```

### Connection Pooling

```yaml
  stockula-pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      DATABASES_HOST: stockula-timescaledb
      DATABASES_PORT: 5432
      DATABASES_USER: stockula_user
      DATABASES_PASSWORD: ${POSTGRES_PASSWORD}
      DATABASES_DBNAME: stockula
    ports:
      - "6432:5432"
```

## Conclusion

The consolidated TimescaleDB implementation provides:

- ✅ **Simplified Architecture**: 3 core files instead of 8+ components
- ✅ **Type Safety**: Full SQLModel integration with comprehensive validation
- ✅ **High Performance**: Optimized time-series operations with hypertables
- ✅ **Comprehensive Analytics**: 8 built-in methods for technical analysis
- ✅ **Production Ready**: Connection pooling, error handling, and monitoring
- ✅ **Future Proof**: Scalable architecture supporting growth to millions of records

This architecture ensures Stockula can handle enterprise-scale time-series data while maintaining developer productivity
and code quality.

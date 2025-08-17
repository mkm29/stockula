# ETL Pipeline Design for Stockula TimescaleDB Migration

## Executive Summary

This document outlines a comprehensive ETL pipeline architecture for migrating existing SQLite data to TimescaleDB and
implementing real-time/batch data ingestion from Yahoo Finance API. The solution emphasizes reliability, scalability,
and data quality while maintaining the existing Stockula architecture patterns.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
1. [Current State Analysis](#current-state-analysis)
1. [ETL Pipeline Architecture](#etl-pipeline-architecture)
1. [Data Migration Strategy](#data-migration-strategy)
1. [Real-time Data Ingestion](#real-time-data-ingestion)
1. [Data Quality Framework](#data-quality-framework)
1. [Monitoring and Alerting](#monitoring-and-alerting)
1. [Implementation Roadmap](#implementation-roadmap)
1. [Technology Stack](#technology-stack)

## Architecture Overview

### High-Level Data Flow

```mermaid
graph TB
    subgraph "Data Sources"
        A[Yahoo Finance API]
        B[Existing SQLite DB]
        C[External Market Feeds]
    end

    subgraph "ETL Pipeline"
        D[Data Ingestion Layer]
        E[Data Validation]
        F[Data Transformation]
        G[Data Quality Checks]
        H[Error Handling & Retry]
    end

    subgraph "TimescaleDB"
        I[Time-Series Tables]
        J[Hypertables]
        K[Continuous Aggregates]
        L[Retention Policies]
    end

    subgraph "Monitoring"
        M[Metrics Collection]
        N[Alerting System]
        O[Data Quality Dashboard]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> I
    I --> J
    J --> K
    K --> L

    G --> M
    M --> N
    M --> O

    H -.-> D
```

## Current State Analysis

### Existing Data Models

The current Stockula platform uses SQLModel with the following core entities:

1. **Stock** - Basic stock metadata (symbol, name, sector, industry)
1. **PriceHistory** - OHLCV time-series data with interval support
1. **Dividend** - Dividend payment history
1. **Split** - Stock split history
1. **OptionsCall/OptionsPut** - Options chain data
1. **StockInfo** - Raw yfinance JSON data
1. **Strategy/StrategyPreset** - Trading strategy configurations
1. **AutoTSModel/AutoTSPreset** - Time-series forecasting models

### Current Challenges

1. **Scalability**: SQLite limitations for concurrent access and large datasets
1. **Performance**: Single-threaded writes and query optimization
1. **Time-series Optimization**: No native time-series features
1. **Data Retention**: Manual cleanup processes
1. **Real-time Processing**: Limited streaming capabilities

## ETL Pipeline Architecture

### 1. Data Ingestion Layer

#### Batch Ingestion (Primary)

- **Schedule**: Configurable intervals (daily, hourly, real-time)
- **Source**: Yahoo Finance API via existing DataFetcher
- **Pattern**: Extract → Transform → Load with checkpointing
- **Volume**: 1000+ symbols with historical data

#### Streaming Ingestion (Future)

- **Real-time Feeds**: WebSocket connections for live quotes
- **Event Processing**: Apache Kafka for message queuing
- **Low Latency**: Sub-second data ingestion for active trading

#### Migration Ingestion (One-time)

- **Source**: Existing SQLite database
- **Strategy**: Bulk export → Transform → Bulk import
- **Validation**: Schema mapping and data integrity checks

### 2. Data Transformation Logic

#### Schema Adaptation for TimescaleDB

```sql
-- Hypertable for price history (partitioned by time)
CREATE TABLE price_history (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    interval_type TEXT NOT NULL,
    open_price NUMERIC(12,4),
    high_price NUMERIC(12,4),
    low_price NUMERIC(12,4),
    close_price NUMERIC(12,4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable partitioned by timestamp
SELECT create_hypertable('price_history', 'timestamp', chunk_time_interval => INTERVAL '1 week');

-- Create index for symbol-time queries
CREATE INDEX idx_price_history_symbol_time ON price_history (symbol, timestamp DESC);
```

#### Data Transformation Rules

1. **Timestamp Conversion**: Convert date fields to TIMESTAMPTZ
1. **Numeric Precision**: Use NUMERIC for financial data to avoid floating-point errors
1. **Symbol Normalization**: Uppercase and validate ticker symbols
1. **Data Deduplication**: UPSERT patterns with conflict resolution
1. **Data Enrichment**: Add calculated fields (returns, volatility)

### 3. Data Quality Framework

#### Validation Rules

```python
# Example validation schema using Pydantic
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
import re

class PriceHistoryValidation(BaseModel):
    symbol: str
    timestamp: datetime
    interval_type: str
    open_price: Optional[float]
    high_price: Optional[float]
    low_price: Optional[float]
    close_price: Optional[float]
    volume: Optional[int]

    @validator('symbol')
    def validate_symbol(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v):
            raise ValueError('Invalid symbol format')
        return v

    @validator('high_price', 'low_price', 'open_price', 'close_price')
    def validate_prices(cls, v, values):
        if v is not None and v < 0:
            raise ValueError('Price cannot be negative')
        return v

    @validator('volume')
    def validate_volume(cls, v):
        if v is not None and v < 0:
            raise ValueError('Volume cannot be negative')
        return v
```

#### Quality Checks

1. **Completeness**: Required fields validation
1. **Accuracy**: Price relationship validation (high >= low)
1. **Consistency**: Cross-table referential integrity
1. **Freshness**: Data staleness detection
1. **Uniqueness**: Duplicate detection and handling

### 4. Error Handling and Retry Logic

#### Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: logging.error(f"Failed after {retry_state.attempt_number} attempts")
)
def fetch_and_store_data(symbol: str, start_date: str, end_date: str):
    """Fetch data with exponential backoff retry."""
    # Implementation with error handling
    pass
```

#### Error Categories

1. **Transient Errors**: Network timeouts, API rate limits
1. **Data Errors**: Invalid data format, missing fields
1. **System Errors**: Database connection issues, disk space
1. **Business Logic Errors**: Data validation failures

## Data Migration Strategy

### Phase 1: Schema Migration

1. **TimescaleDB Setup**

   - Install TimescaleDB extension
   - Configure retention policies
   - Set up compression policies
   - Create hypertables and indexes

1. **Schema Mapping**

   - Map SQLModel classes to TimescaleDB tables
   - Handle data type conversions
   - Create migration scripts

### Phase 2: Data Migration

```python
# Migration script structure
class StockulaMigrator:
    def __init__(self, sqlite_path: str, timescale_uri: str):
        self.sqlite_db = DatabaseManager(sqlite_path)
        self.timescale_db = TimescaleDBManager(timescale_uri)

    def migrate_stocks(self):
        """Migrate stock metadata."""
        stocks = self.sqlite_db.get_all_stocks()
        self.timescale_db.bulk_insert_stocks(stocks)

    def migrate_price_history(self, batch_size: int = 10000):
        """Migrate price history in batches."""
        for batch in self.sqlite_db.get_price_history_batches(batch_size):
            transformed_batch = self.transform_price_data(batch)
            self.timescale_db.bulk_insert_prices(transformed_batch)

    def validate_migration(self):
        """Validate data integrity after migration."""
        sqlite_count = self.sqlite_db.count_price_records()
        timescale_count = self.timescale_db.count_price_records()
        assert sqlite_count == timescale_count, "Record count mismatch"
```

### Phase 3: Incremental Updates

1. **Change Data Capture**: Track changes in source systems
1. **Upsert Patterns**: Handle updates and new records
1. **Conflict Resolution**: Last-writer-wins or business rule-based

## Real-time Data Ingestion

### Streaming Architecture

```python
# Airflow DAG for batch processing
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'stockula',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'stockula_daily_etl',
    default_args=default_args,
    description='Daily stock data ETL pipeline',
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    catchup=False
)

def extract_stock_data(**context):
    """Extract data from Yahoo Finance."""
    symbols = get_active_symbols()
    fetcher = DataFetcher()
    return fetcher.get_stock_data_batch(symbols)

def transform_and_load(**context):
    """Transform and load data to TimescaleDB."""
    raw_data = context['task_instance'].xcom_pull(task_ids='extract')
    transformer = DataTransformer()
    loader = TimescaleDBLoader()

    for symbol, data in raw_data.items():
        transformed_data = transformer.transform_price_data(data)
        loader.upsert_price_data(symbol, transformed_data)

extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_stock_data,
    dag=dag
)

transform_load_task = PythonOperator(
    task_id='transform_load',
    python_callable=transform_and_load,
    dag=dag
)

extract_task >> transform_load_task
```

### Real-time Processing with Kafka

```python
# Kafka consumer for real-time data
from kafka import KafkaConsumer
import json

class StockDataConsumer:
    def __init__(self, bootstrap_servers: str, topic: str):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.loader = TimescaleDBLoader()

    def process_messages(self):
        for message in self.consumer:
            try:
                stock_data = message.value
                self.validate_and_load(stock_data)
            except Exception as e:
                logging.error(f"Error processing message: {e}")

    def validate_and_load(self, data):
        validator = PriceHistoryValidation(**data)
        self.loader.insert_price_data(validator.dict())
```

## Data Quality Framework

### Quality Metrics

1. **Data Completeness**: Percentage of required fields populated
1. **Data Accuracy**: Price validation rules compliance
1. **Data Freshness**: Time since last update per symbol
1. **Data Consistency**: Cross-table referential integrity
1. **Data Uniqueness**: Duplicate detection rates

### Quality Monitoring

```python
class DataQualityMonitor:
    def __init__(self, db_connection):
        self.db = db_connection
        self.metrics = {}

    def check_completeness(self, table_name: str, required_fields: list):
        """Check data completeness for required fields."""
        for field in required_fields:
            null_count = self.db.count_nulls(table_name, field)
            total_count = self.db.count_rows(table_name)
            completeness = (total_count - null_count) / total_count
            self.metrics[f"{table_name}_{field}_completeness"] = completeness

    def check_freshness(self, symbol: str, max_age_hours: int = 24):
        """Check data freshness for a symbol."""
        latest_timestamp = self.db.get_latest_timestamp(symbol)
        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
        is_fresh = age_hours <= max_age_hours
        self.metrics[f"{symbol}_freshness"] = is_fresh

    def generate_report(self):
        """Generate data quality report."""
        return {
            'timestamp': datetime.now(),
            'metrics': self.metrics,
            'overall_score': sum(self.metrics.values()) / len(self.metrics)
        }
```

## Monitoring and Alerting Strategy

### Key Metrics to Monitor

1. **Pipeline Performance**

   - Data ingestion rate (records/second)
   - Pipeline execution time
   - Error rates by source and type
   - Resource utilization (CPU, memory, disk)

1. **Data Quality**

   - Validation failure rates
   - Data freshness by symbol
   - Completeness scores
   - Anomaly detection alerts

1. **System Health**

   - Database connection status
   - API rate limit usage
   - Disk space and I/O performance
   - Network connectivity

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: stockula_etl
    rules:
      - alert: HighETLErrorRate
        expr: rate(etl_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "ETL error rate is high"
          description: "ETL pipeline error rate is {{ $value }} errors/second"

      - alert: DataFreshnessIssue
        expr: time() - stockula_latest_data_timestamp > 7200
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Data freshness issue detected"
          description: "Latest data is {{ $value }} seconds old"

      - alert: TimescaleDBDown
        expr: up{job="timescaledb"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TimescaleDB is down"
```

### Dashboard Components

1. **Pipeline Overview**

   - Real-time pipeline status
   - Throughput and latency metrics
   - Error rate trends

1. **Data Quality Dashboard**

   - Completeness scores by table
   - Data freshness heatmap
   - Validation error breakdown

1. **Performance Metrics**

   - Query performance trends
   - Resource utilization
   - Historical performance comparisons

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [ ] Set up TimescaleDB instance
- [ ] Create hypertables and indexes
- [ ] Implement basic ETL framework
- [ ] Set up monitoring infrastructure

### Phase 2: Migration (Weeks 3-4)

- [ ] Develop migration scripts
- [ ] Perform data migration with validation
- [ ] Implement incremental update mechanisms
- [ ] Test data integrity and performance

### Phase 3: Real-time Processing (Weeks 5-6)

- [ ] Implement Airflow DAGs for batch processing
- [ ] Set up Kafka infrastructure for streaming
- [ ] Develop real-time consumers
- [ ] Implement data quality monitoring

### Phase 4: Optimization (Weeks 7-8)

- [ ] Performance tuning and optimization
- [ ] Implement advanced monitoring and alerting
- [ ] Documentation and team training
- [ ] Production deployment and monitoring

## Technology Stack Recommendations

### Orchestration Framework: Apache Airflow

**Pros:**

- Rich ecosystem and community support
- Powerful scheduling and dependency management
- Built-in monitoring and logging
- Python-native, integrates well with existing codebase

**Configuration:**

```python
# Recommended Airflow configuration
AIRFLOW__CORE__EXECUTOR = 'CeleryExecutor'
AIRFLOW__CORE__SQL_ALCHEMY_CONN = 'postgresql://airflow:password@localhost/airflow'
AIRFLOW__CELERY__BROKER_URL = 'redis://localhost:6379/0'
AIRFLOW__CELERY__RESULT_BACKEND = 'db+postgresql://airflow:password@localhost/airflow'
```

### Alternative: Prefect 2.0

**Pros:**

- Modern Python-first design
- Better error handling and retry mechanisms
- Native support for async operations
- Simpler deployment and maintenance

### Message Queue: Apache Kafka

**Use Cases:**

- Real-time data streaming
- Event-driven architecture
- Decoupling of data producers and consumers

### Database: TimescaleDB

**Benefits:**

- PostgreSQL compatibility
- Native time-series optimizations
- Automatic partitioning and compression
- SQL familiarity for the team

### Monitoring Stack

- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and management
- **ELK Stack**: Log aggregation and analysis

## Performance Optimization Strategies

### TimescaleDB Optimizations

1. **Hypertable Configuration**

   ```sql
   -- Optimize chunk size based on data patterns
   SELECT set_chunk_time_interval('price_history', INTERVAL '1 week');

   -- Enable compression for older data
   ALTER TABLE price_history SET (
       timescaledb.compress,
       timescaledb.compress_segmentby = 'symbol',
       timescaledb.compress_orderby = 'timestamp DESC'
   );

   -- Set up compression policy
   SELECT add_compression_policy('price_history', INTERVAL '30 days');
   ```

1. **Indexing Strategy**

   ```sql
   -- Symbol-time composite index for fast queries
   CREATE INDEX idx_price_history_symbol_time ON price_history (symbol, timestamp DESC);

   -- Partial indexes for active symbols
   CREATE INDEX idx_price_history_active ON price_history (timestamp DESC)
   WHERE timestamp >= NOW() - INTERVAL '1 year';
   ```

1. **Continuous Aggregates**

   ```sql
   -- Pre-compute daily aggregates
   CREATE MATERIALIZED VIEW daily_prices
   WITH (timescaledb.continuous) AS
   SELECT symbol,
          time_bucket('1 day', timestamp) AS day,
          first(open_price, timestamp) AS open_price,
          max(high_price) AS high_price,
          min(low_price) AS low_price,
          last(close_price, timestamp) AS close_price,
          sum(volume) AS volume
   FROM price_history
   GROUP BY symbol, day;
   ```

### ETL Performance Optimizations

1. **Batch Processing**

   - Use bulk insert operations
   - Implement parallel processing for independent symbols
   - Optimize batch sizes based on memory constraints

1. **Connection Pooling**

   ```python
   from sqlalchemy.pool import QueuePool

   engine = create_engine(
       connection_string,
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True
   )
   ```

1. **Caching Strategy**

   - Implement Redis caching for frequently accessed data
   - Use ETags for API response caching
   - Cache static reference data (symbol metadata)

## Cost Optimization

### Infrastructure Costs

1. **Resource Right-sizing**: Monitor and optimize compute resources
1. **Data Retention**: Implement tiered storage strategies
1. **Compression**: Enable TimescaleDB compression for historical data
1. **Auto-scaling**: Use cloud auto-scaling for variable workloads

### Operational Costs

1. **API Usage**: Optimize Yahoo Finance API calls to minimize rate limiting
1. **Monitoring**: Use efficient monitoring strategies to avoid over-instrumentation
1. **Development**: Leverage existing Stockula patterns to minimize development time

## Risk Mitigation

### Data Loss Prevention

1. **Backup Strategy**: Automated daily backups with point-in-time recovery
1. **Replication**: Set up TimescaleDB streaming replication
1. **Data Validation**: Comprehensive validation before and after migration

### Security Considerations

1. **Access Control**: Implement role-based access control
1. **Encryption**: Enable encryption at rest and in transit
1. **API Security**: Secure API endpoints and implement rate limiting
1. **Audit Logging**: Comprehensive audit trails for data access

### Business Continuity

1. **Disaster Recovery**: Multi-region deployment strategy
1. **Fallback Mechanisms**: Maintain SQLite compatibility during transition
1. **Performance SLAs**: Define and monitor service level agreements

## Conclusion

This ETL pipeline design provides a robust, scalable solution for migrating Stockula's data infrastructure to
TimescaleDB while maintaining high data quality and operational reliability. The phased implementation approach allows
for gradual transition with minimal risk to existing operations.

The combination of Airflow for orchestration, TimescaleDB for time-series optimization, and comprehensive monitoring
ensures the platform can scale to handle increased data volumes and provide real-time analytics capabilities for
enhanced trading strategies.

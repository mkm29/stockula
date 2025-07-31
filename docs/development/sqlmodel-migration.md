# SQLModel Migration Guide

This document outlines the migration from raw SQL queries to SQLModel, a modern ORM that combines SQLAlchemy with Pydantic.

## Why SQLModel?

### Current Issues with Raw SQL

1. **No Type Safety**: Raw SQL queries return tuples or dictionaries without type checking
1. **SQL Injection Risk**: String concatenation in SQL queries can lead to security vulnerabilities
1. **Maintenance Burden**: SQL queries scattered throughout the code are hard to maintain
1. **No IDE Support**: No autocomplete or type hints for database operations
1. **Manual Mapping**: Need to manually map database results to Python objects

### Benefits of SQLModel

1. **Type Safety**: Full type checking with Pydantic models
1. **Security**: Automatic SQL injection prevention through parameterized queries
1. **Maintainability**: Database schema defined in one place
1. **Developer Experience**: Full IDE support with autocomplete and type hints
1. **Automatic Validation**: Pydantic validation for all database inputs
1. **Alembic Integration**: Works seamlessly with existing Alembic migrations

## Migration Overview

### 1. Install SQLModel

```bash
uv add sqlmodel
```

### 2. Model Comparison

#### Old Raw SQL Approach

```python
# Creating tables with raw SQL
conn.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        symbol TEXT PRIMARY KEY,
        name TEXT,
        sector TEXT,
        industry TEXT,
        market_cap REAL,
        exchange TEXT,
        currency TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Inserting data
conn.execute(
    """
    INSERT OR REPLACE INTO stocks
    (symbol, name, sector, industry, market_cap, exchange, currency, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """,
    (symbol, name, sector, industry, market_cap, exchange, currency),
)

# Querying data
cursor = conn.execute(
    "SELECT * FROM stocks WHERE symbol = ?", (symbol,)
)
row = cursor.fetchone()
```

#### New SQLModel Approach

```python
# Model definition
class Stock(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Creating/updating data
with session:
    stock = Stock(
        symbol=symbol,
        name=name,
        sector=sector,
        industry=industry,
        market_cap=market_cap,
        exchange=exchange,
        currency=currency
    )
    session.add(stock)
    session.commit()

# Querying data
with session:
    stock = session.get(Stock, symbol)
```

### 3. Key Differences

| Feature                  | Raw SQL         | SQLModel     |
| ------------------------ | --------------- | ------------ |
| Type Safety              | ❌ No           | ✅ Yes       |
| SQL Injection Protection | ⚠️ Manual       | ✅ Automatic |
| IDE Support              | ❌ Limited      | ✅ Full      |
| Relationships            | ❌ Manual joins | ✅ Automatic |
| Validation               | ❌ None         | ✅ Pydantic  |
| Migration Support        | ✅ Alembic      | ✅ Alembic   |

## DRY Base Models

SQLModel allows us to create base models with common fields, eliminating code duplication:

### Base Model Structure

```python
class TimestampMixin(SQLModel):
    """Mixin for adding created_at and updated_at timestamps to models."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLADateTime, server_default=func.current_timestamp()),
        description="Timestamp when the record was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            SQLADateTime,
            server_default=func.current_timestamp(),
            onupdate=func.current_timestamp()
        ),
        description="Timestamp when the record was last updated"
    )


class BaseModel(TimestampMixin):
    """Base model with common fields for all database models."""

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Primary key"
    )
```

### Benefits of Base Models

1. **No Repetition**: Common fields defined once
1. **Consistency**: All tables have the same timestamp behavior
1. **Maintainability**: Changes to common fields happen in one place
1. **Type Safety**: Inherited fields maintain full type checking
1. **Documentation**: Field descriptions are inherited

### Usage Pattern

```python
# For tables with auto-incrementing ID
class PriceHistory(BaseModel, table=True):
    symbol: str = Field(foreign_key="stocks.symbol")
    date: date
    # ... other fields

# For tables with custom primary key
class Stock(TimestampMixin, table=True):
    symbol: str = Field(primary_key=True)
    # ... other fields
```

## Implementation Details

### Database Manager Comparison

#### Old DatabaseManager (Raw SQL)

```python
def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d"):
    with self.get_connection() as conn:
        for date, row in data.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO price_history
                (symbol, date, open_price, high_price, low_price, close_price, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, date.strftime("%Y-%m-%d"), row.get("Open"),
                 row.get("High"), row.get("Low"), row.get("Close"),
                 row.get("Volume"), interval)
            )
```

#### New DatabaseManager (Using SQLModel)

```python
def store_price_history(self, symbol: str, data: pd.DataFrame, interval: str = "1d"):
    with self.get_session() as session:
        for date, row in data.iterrows():
            price_history = PriceHistory(
                symbol=symbol,
                date=date.date(),
                open_price=row.get("Open"),
                high_price=row.get("High"),
                low_price=row.get("Low"),
                close_price=row.get("Close"),
                volume=row.get("Volume"),
                interval=interval
            )
            session.add(price_history)
        session.commit()
```

### Relationship Handling

SQLModel automatically handles relationships:

```python
class Stock(SQLModel, table=True):
    symbol: str = Field(primary_key=True)
    # ... other fields ...

    # Automatic relationships
    price_history: List["PriceHistory"] = Relationship(back_populates="stock")
    dividends: List["Dividend"] = Relationship(back_populates="stock")

class PriceHistory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(foreign_key="stocks.symbol")
    # ... other fields ...

    # Back reference
    stock: Stock = Relationship(back_populates="price_history")
```

## Migration Status

✅ **Migration Complete** (2025-01-28)

The migration from raw SQL to SQLModel has been successfully completed:

1. ✅ Created SQLModel models for all database tables
1. ✅ Implemented type-safe DatabaseManager using SQLModel
1. ✅ Removed inheritance complexity - each model has its own timestamp fields
1. ✅ Updated all imports throughout the codebase
1. ✅ Removed old raw SQL implementation
1. ✅ Tests are passing with the new implementation

### Current Implementation

```python
from stockula.database.manager import DatabaseManager

# Uses SQLModel internally for type safety and better developer experience
db = DatabaseManager("stockula.db")

# All operations are now type-safe with full IDE support
stock_info = db.get_stock_info("AAPL")  # Returns Optional[Dict[str, Any]]
price_history = db.get_price_history("AAPL", "2024-01-01", "2024-12-31")  # Returns pd.DataFrame
```

### Next Steps

- Update all tests to ensure they work with the new SQLModel implementation
- Consider adding more type hints to method returns for better IDE support
- Add validation for input data using Pydantic validators

## Performance Considerations

### Batch Operations

SQLModel supports efficient batch operations:

```python
# Bulk insert with SQLModel
with session:
    price_records = [
        PriceHistory(**record) for record in price_data_list
    ]
    session.add_all(price_records)
    session.commit()
```

### Query Optimization

```python
# Efficient querying with SQLModel
stmt = select(PriceHistory).where(
    PriceHistory.symbol == symbol,
    PriceHistory.date >= start_date
).order_by(PriceHistory.date)

results = session.exec(stmt).all()
```

## Best Practices

1. **Use Context Managers**: Always use `with session:` for automatic cleanup
1. **Batch Operations**: Use `session.add_all()` for bulk inserts
1. **Lazy Loading**: Be aware of N+1 query problems with relationships
1. **Indexes**: Define indexes in the model for performance
1. **Validation**: Leverage Pydantic validators for data integrity

## Conclusion

Migrating from raw SQL to SQLModel provides:

- ✅ Type safety and better developer experience
- ✅ Automatic SQL injection protection
- ✅ Cleaner, more maintainable code
- ✅ Better integration with modern Python tooling
- ✅ Seamless Alembic migration support

The migration can be done gradually, allowing both implementations to coexist during the transition period.

# Migration Guide: AutoTSModelValidator to Database-Driven Validation

This guide helps you migrate from the old `AutoTSModelValidator` class to the new database-driven validation system.

## Overview of Changes

The `AutoTSModelValidator` class has been completely removed and replaced with:

- **AutoTSModel**: SQLModel-based database model with self-validation
- **AutoTSPreset**: Database model for preset configurations
- **AutoTSRepository**: Repository pattern for database operations

## Migration Steps

### 1. Remove Old Imports

**Before:**

```python
from stockula.forecasting.model_validator import AutoTSModelValidator
```

**After:**

```python
from stockula.database.models import AutoTSModel, AutoTSPreset
from stockula.data.autots_repository import AutoTSRepository
```

### 2. Update Validation Code

#### Validating Individual Models

**Before:**

```python
validator = AutoTSModelValidator()
is_valid = validator.validate_model("ARIMA")
```

**After:**

```python
# Direct validation (no database required)
is_valid = AutoTSModel.is_valid_model("ARIMA")

# Or with repository (if you need database features)
from sqlmodel import Session
with Session(engine) as session:
    repo = AutoTSRepository(session)
    model = repo.get_model("ARIMA")
    is_valid = model is not None
```

#### Validating Model Lists

**Before:**

```python
validator = AutoTSModelValidator()
is_valid, invalid = validator.validate_model_list(["ARIMA", "ETS", "VAR"])
```

**After:**

```python
# Direct validation
is_valid, invalid = AutoTSModel.validate_model_list(["ARIMA", "ETS", "VAR"])

# Or with repository
with Session(engine) as session:
    repo = AutoTSRepository(session)
    is_valid, invalid = repo.validate_model_list(["ARIMA", "ETS", "VAR"])
```

#### Getting Valid Models

**Before:**

```python
validator = AutoTSModelValidator()
all_models = validator.ALL_MODELS
```

**After:**

```python
# Get all valid model names
all_models = AutoTSModel.get_valid_models()

# Or from database
with Session(engine) as session:
    repo = AutoTSRepository(session)
    db_models = repo.get_all_models()
    model_names = [m.name for m in db_models]
```

#### Checking Presets

**Before:**

```python
validator = AutoTSModelValidator()
preset_models = validator.PRESETS["fast"]
```

**After:**

```python
# Check if preset is valid
is_valid = AutoTSPreset.is_valid_preset("fast")

# Get preset models from database
with Session(engine) as session:
    repo = AutoTSRepository(session)
    preset = repo.get_preset("fast")
    if preset:
        preset_models = preset.model_list
```

### 3. Update Dependency Injection

If you were injecting `AutoTSModelValidator`, update your container:

**Before:**

```python
from dependency_injector import containers, providers
from stockula.forecasting.model_validator import AutoTSModelValidator

class Container(containers.DeclarativeContainer):
    model_validator = providers.Singleton(AutoTSModelValidator)
```

**After:**

```python
from dependency_injector import containers, providers
from stockula.database.manager import DatabaseManager

class Container(containers.DeclarativeContainer):
    database_manager = providers.Singleton(DatabaseManager)
```

### 4. Update Forecaster Usage

The `StockForecaster` now uses dependency injection for `DatabaseManager`:

**Before:**

```python
from stockula.forecasting.forecaster import StockForecaster
from stockula.forecasting.model_validator import AutoTSModelValidator

validator = AutoTSModelValidator()
forecaster = StockForecaster(
    model_list="fast",
    model_validator=validator
)
```

**After:**

```python
from stockula.forecasting.forecaster import StockForecaster
from stockula.container import Container

container = Container()
forecaster = StockForecaster(
    model_list="fast",
    database_manager=container.database_manager()  # Optional, will be injected
)
```

## Benefits of the New System

1. **Dynamic Configuration**: Add or modify models without changing code
1. **Database Persistence**: Model definitions are stored and versioned
1. **Self-Validation**: Models validate themselves when saved
1. **Repository Pattern**: Clean separation of data access logic
1. **Better Testing**: Easier to mock and test with dependency injection

## Database Seeding

The database is automatically seeded directly from the AutoTS library when empty:

```python
from sqlmodel import Session, create_engine
from stockula.data.autots_repository import AutoTSRepository

engine = create_engine("sqlite:///stockula.db")
with Session(engine) as session:
    repo = AutoTSRepository(session)
    
    # Seed from AutoTS library if database is empty
    if not repo.get_all_models():
        models_count, presets_count = repo.seed_from_autots()
        print(f"Seeded {models_count} models and {presets_count} presets")
```

## Model Management

Models are validated against the AutoTS library's authoritative list:

```python
with Session(engine) as session:
    repo = AutoTSRepository(session)
    
    # Add a model (must be a valid AutoTS model)
    model_data = {
        "name": "ARIMA",  # Must be a valid AutoTS model
        "description": "ARIMA forecasting model",
        "categories": ["univariate"],
        "is_slow": True,
        "is_gpu_enabled": False,
        "requires_regressor": False,
        "min_data_points": 100,
    }
    
    # The validation ensures only valid AutoTS models are stored
    try:
        model = repo.create_or_update_model(model_data)
        print(f"Added model: {model.name}")
    except ValueError as e:
        print(f"Invalid model: {e}")
```

## Troubleshooting

### Models Not Found

If models are not being found:

1. Check that the database has been seeded using `seed_from_autots()`
1. Verify AutoTS is installed properly (`pip install autots`)
1. Ensure database tables are created

```python
from sqlmodel import SQLModel
SQLModel.metadata.create_all(engine)
```

### Validation Errors

The new system validates against AutoTS's authoritative model list:

- Models must exist in AutoTS's model_lists to be added to the database
- Presets validate that all their models are valid AutoTS models
- Validation happens automatically when saving

### Performance

The new system uses class-level caching:

- Models from AutoTS are loaded once and cached
- Database queries are optimized with indexes
- Repository pattern allows for efficient batch operations

## Example: Complete Migration

Here's a complete example showing the migration:

**Old Code:**

```python
from stockula.forecasting.model_validator import AutoTSModelValidator
from stockula.forecasting.forecaster import StockForecaster

# Create validator
validator = AutoTSModelValidator()

# Validate models
if validator.validate_model("ARIMA"):
    print("ARIMA is valid")

# Get preset models
fast_models = validator.PRESETS["fast"]

# Create forecaster
forecaster = StockForecaster(
    model_list=fast_models,
    model_validator=validator
)
```

**New Code:**

```python
from sqlmodel import Session, create_engine
from stockula.database.models import AutoTSModel, AutoTSPreset
from stockula.data.autots_repository import AutoTSRepository
from stockula.forecasting.forecaster import StockForecaster
from stockula.container import Container

# Setup database
engine = create_engine("sqlite:///stockula.db")

# Validate models (no database needed)
if AutoTSModel.is_valid_model("ARIMA"):
    print("ARIMA is valid")

# Get preset models from database
with Session(engine) as session:
    repo = AutoTSRepository(session)
    preset = repo.get_preset("fast")
    if preset:
        fast_models = preset.model_list

# Create forecaster with dependency injection
container = Container()
forecaster = StockForecaster(
    model_list="fast",  # Can use preset name directly
    database_manager=container.database_manager()
)
```

The new system is more flexible, maintainable, and testable while providing the same validation capabilities.

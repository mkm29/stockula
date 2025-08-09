#!/usr/bin/env python3
"""Standalone script to seed AutoTS models into the database from data/models.json."""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()


class AutoTSModel(Base):
    """AutoTS model definitions and metadata."""

    __tablename__ = "autots_models"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True, unique=True)
    categories = Column(String, default="[]")
    is_slow = Column(Boolean, default=False)
    is_gpu_enabled = Column(Boolean, default=False)
    requires_regressor = Column(Boolean, default=False)
    min_data_points = Column(Integer, default=10)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class AutoTSPreset(Base):
    """AutoTS model preset configurations."""

    __tablename__ = "autots_presets"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True, unique=True)
    models = Column(String)
    description = Column(String, nullable=True)
    use_case = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


def seed_from_json(session: Session, json_path: Path) -> tuple[int, int]:
    """Seed the database from a models.json file."""
    with open(json_path) as f:
        data = json.load(f)

    models_count = 0
    presets_count = 0

    # Get all unique models
    all_models = set(data.get("all", []))

    # Add models from presets
    for preset_name, preset_data in data.items():
        if preset_name == "all":
            continue
        if isinstance(preset_data, list):
            all_models.update(preset_data)
        elif isinstance(preset_data, dict):
            all_models.update(preset_data.keys())

    # Categorize models
    categories_map = {
        "univariate": set(data.get("univariate", [])),
        "multivariate": set(data.get("multivariate", [])),
        "probabilistic": set(data.get("probabilistic", [])),
        "slow": set(data.get("slow", [])),
        "gpu": set(data.get("gpu", [])),
        "regressor": set(data.get("regressor", [])),
        "motifs": set(data.get("motifs", [])),
        "regressions": set(data.get("regressions", [])),
    }

    # Create/update models
    for model_name in all_models:
        model_categories = []
        for category, models in categories_map.items():
            if model_name in models:
                model_categories.append(category)

        # Check if model exists
        existing = session.query(AutoTSModel).filter_by(name=model_name).first()

        if existing:
            # Update existing
            existing.categories = json.dumps(model_categories)
            existing.is_slow = model_name in categories_map["slow"]
            existing.is_gpu_enabled = model_name in categories_map["gpu"]
            existing.requires_regressor = model_name in categories_map["regressor"]
            existing.updated_at = datetime.utcnow()
        else:
            # Create new
            model = AutoTSModel(
                name=model_name,
                categories=json.dumps(model_categories),
                is_slow=model_name in categories_map["slow"],
                is_gpu_enabled=model_name in categories_map["gpu"],
                requires_regressor=model_name in categories_map["regressor"],
            )

            # Add descriptions for some common models
            descriptions = {
                "ARIMA": "AutoRegressive Integrated Moving Average - classic time series model",
                "ETS": "Exponential Smoothing State Space Model",
                "FBProphet": "Facebook's Prophet model for time series with seasonality",
                "LastValueNaive": "Simple naive forecast using last observed value",
                "AverageValueNaive": "Forecast using average of historical values",
                "SeasonalNaive": "Naive forecast with seasonal patterns",
                "VAR": "Vector AutoRegression for multivariate time series",
                "GluonTS": "Deep learning models from Amazon's GluonTS library",
                "Theta": "Theta method for forecasting",
                "UnivariateMotif": "Pattern-based forecasting for single series",
                "MultivariateMotif": "Pattern-based forecasting for multiple series",
                "WindowRegression": "Regression using rolling windows",
                "DatepartRegression": "Regression using date components",
                "NVAR": "Neural Vector AutoRegression",
                "DynamicFactor": "Dynamic Factor Model",
                "UnobservedComponents": "State space model with unobserved components",
            }
            if model_name in descriptions:
                model.description = descriptions[model_name]

            session.add(model)
        models_count += 1

    # Create/update presets
    preset_descriptions = {
        "default": "Balanced selection of models for general use",
        "fast": "Fast models for quick results",
        "superfast": "Ultra-fast models for immediate results",
        "parallel": "Models that can run in parallel",
        "fast_parallel": "Fast models that can run in parallel",
        "scalable": "Models that scale well with large datasets",
        "probabilistic": "Models that provide probability distributions",
        "multivariate": "Models for multiple time series",
        "univariate": "Models for single time series",
        "best": "Best performing models (slower but more accurate)",
        "slow": "Computationally expensive models",
        "gpu": "Models that can utilize GPU acceleration",
        "regressor": "Models that support external regressors",
        "motifs": "Pattern-based models",
        "regressions": "Regression-based models",
    }

    use_cases = {
        "fast": "Quick analysis and prototyping",
        "superfast": "Real-time forecasting",
        "parallel": "Large-scale batch processing",
        "probabilistic": "Risk assessment and confidence intervals",
        "multivariate": "Multiple correlated time series",
        "best": "Production forecasting with high accuracy requirements",
        "gpu": "High-performance computing with GPU acceleration",
        "regressor": "Forecasting with external variables",
    }

    for preset_name, preset_data in data.items():
        if (
            preset_name == "all"
            or preset_name.startswith("all_")
            or preset_name.startswith("no_")
            or preset_name == "experimental"
        ):
            continue  # Skip some special categories

        # Check if preset exists
        existing = session.query(AutoTSPreset).filter_by(name=preset_name).first()

        if existing:
            # Update existing
            existing.models = json.dumps(preset_data)
            existing.description = preset_descriptions.get(preset_name)
            existing.use_case = use_cases.get(preset_name)
            existing.updated_at = datetime.utcnow()
        else:
            # Create new
            preset = AutoTSPreset(
                name=preset_name,
                models=json.dumps(preset_data),
                description=preset_descriptions.get(preset_name),
                use_case=use_cases.get(preset_name),
            )
            session.add(preset)
        presets_count += 1

    session.commit()
    return models_count, presets_count


def main():
    """Main function to seed AutoTS models."""
    # Path to models.json
    models_json_path = Path(__file__).parent.parent / "data" / "models.json"

    if not models_json_path.exists():
        print(f"Error: models.json not found at {models_json_path}")
        return

    # Database path
    db_path = Path(__file__).parent.parent / "stockula.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables if they don't exist
    Base.metadata.create_all(engine)

    # Seed the database
    with Session(engine) as session:
        models_count, presets_count = seed_from_json(session, models_json_path)
        print(f"Successfully seeded {models_count} models and {presets_count} presets")

        # Show some statistics
        print("\nDatabase contents:")
        total_models = session.query(AutoTSModel).count()
        total_presets = session.query(AutoTSPreset).count()
        print(f"- Total models: {total_models}")
        print(f"- Total presets: {total_presets}")

        # Show categories
        categories = ["univariate", "multivariate", "slow", "gpu", "probabilistic"]
        for category in categories:
            # Count models with this category
            all_models = session.query(AutoTSModel).all()
            count = sum(1 for m in all_models if category in json.loads(m.categories))
            print(f"- {category.capitalize()} models: {count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example of using database-backed AutoTS model validation."""

# Import from parent directory
import sys
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stockula.data.autots_repository import AutoTSRepository
from stockula.database.models import AutoTSModel, AutoTSPreset


def main():
    """Demonstrate database-backed model validation."""
    # Connect to database
    db_path = Path(__file__).parent.parent / "stockula.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Ensure tables exist
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create repository
        repo = AutoTSRepository(session)

        print("=" * 60)
        print("AutoTS Model Database Validation Example")
        print("=" * 60)

        # Ensure database is seeded
        models_json_path = Path(__file__).parent.parent / "data" / "models.json"
        if models_json_path.exists():
            all_models = repo.get_all_models()
            if not all_models:
                print("\nSeeding database from models.json...")
                models_count, presets_count = repo.seed_from_json(models_json_path)
                print(f"Seeded {models_count} models and {presets_count} presets")

        # 1. Validate individual models directly using AutoTSModel
        print("\n1. Validating individual models:")
        test_models = ["ARIMA", "ETS", "InvalidModel", "VAR", "LastValueNaive"]
        for model_name in test_models:
            is_valid = AutoTSModel.is_valid_model(model_name)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"  {model_name}: {status}")

            # Check if it exists in database
            if is_valid:
                db_model = repo.get_model(model_name)
                if db_model:
                    print("    → In database: Yes")
                else:
                    print("    → In database: No (but valid in models.json)")

        # 2. Check model characteristics from database
        print("\n2. Model characteristics from database:")
        characteristic_models = ["ARIMA", "ETS", "LastValueNaive", "VAR"]
        for model_name in characteristic_models:
            model = repo.get_model(model_name)
            if model:
                print(f"  {model_name}:")
                print(f"    - Slow: {'Yes' if model.is_slow else 'No'}")
                print(f"    - GPU: {'Yes' if model.is_gpu_enabled else 'No'}")
                print(f"    - Requires Regressor: {'Yes' if model.requires_regressor else 'No'}")
                print(f"    - Categories: {', '.join(model.category_list) if model.category_list else 'None'}")
                if model.description:
                    print(f"    - Description: {model.description}")

        # 3. Validate presets using AutoTSPreset
        print("\n3. Validating presets:")
        presets = ["fast", "superfast", "probabilistic", "invalid_preset", "default"]
        for preset_name in presets:
            is_valid = AutoTSPreset.is_valid_preset(preset_name)
            if is_valid:
                preset = repo.get_preset(preset_name)
                if preset:
                    models = preset.model_list
                    if isinstance(models, dict):
                        count = len(models)
                        weighted = "weighted"
                    else:
                        count = len(models)
                        weighted = "unweighted"
                    print(f"  {preset_name}: ✓ Valid ({count} models, {weighted})")
                else:
                    print(f"  {preset_name}: ✓ Valid (built-in AutoTS preset)")
            else:
                print(f"  {preset_name}: ✗ Invalid preset")

        # 4. Get preset details from database
        print("\n4. Preset details from database:")
        preset_name = "fast"
        preset = repo.get_preset(preset_name)
        if preset:
            print(f"  Preset: {preset.name}")
            if preset.description:
                print(f"  Description: {preset.description}")
            if preset.use_case:
                print(f"  Use case: {preset.use_case}")
            models = preset.model_list
            if isinstance(models, dict):
                print("  Models (with weights):")
                for model, weight in list(models.items())[:5]:
                    print(f"    - {model}: {weight}")
                if len(models) > 5:
                    print(f"    ... and {len(models) - 5} more")
            else:
                print(f"  Models: {', '.join(models[:5]) if len(models) > 5 else ', '.join(models)}")
                if len(models) > 5:
                    print(f"    ... and {len(models) - 5} more")

        # 5. Find models by category
        print("\n5. Models by category:")
        categories = ["gpu", "probabilistic", "slow", "univariate", "multivariate"]
        for category in categories:
            models = repo.get_models_by_category(category)
            print(f"  {category.capitalize()}: {len(models)} models")
            if models:
                sample = [m.name for m in models[:3]]
                print(f"    Examples: {', '.join(sample)}")

        # 6. Validate a custom model list using repository
        print("\n6. Validating custom model list:")
        custom_list = ["ARIMA", "ETS", "VAR", "FakeModel", "NotRealModel"]
        is_valid, invalid = repo.validate_model_list(custom_list)
        if is_valid:
            print("  ✓ All models valid")
        else:
            print(f"  ✗ Invalid models found: {', '.join(invalid)}")

        # 7. Get all valid models
        print("\n7. All valid models summary:")
        valid_models = AutoTSModel.get_valid_models()
        print(f"  Total valid models: {len(valid_models)}")
        print(f"  Sample models: {', '.join(sorted(list(valid_models))[:10])}")

        # 8. Create or update a model programmatically
        print("\n8. Creating/updating models:")
        try:
            # Try to create/update a model
            model_data = {
                "name": "ARIMA",
                "description": "AutoRegressive Integrated Moving Average - updated",
                "categories": ["univariate", "statistical"],
                "is_slow": False,
                "is_gpu_enabled": False,
                "requires_regressor": False,
                "min_data_points": 30,
            }
            model = repo.create_or_update_model(model_data)
            print(f"  ✓ Updated model: {model.name}")
            print(f"    Description: {model.description}")
            print(f"    Categories: {', '.join(model.category_list)}")
        except ValueError as e:
            print(f"  ✗ Error: {e}")

        # 9. Validate model list including presets
        print("\n9. Mixed validation (models and presets):")
        mixed_list = "fast"  # This is a preset
        is_valid, invalid = repo.validate_model_list(mixed_list)
        print(f"  Preset 'fast': {'✓ Valid' if is_valid else '✗ Invalid'}")

        mixed_list = ["ARIMA", "ETS", "fast"]  # Mix of models and preset name
        for item in mixed_list:
            # Check if it's a model
            if AutoTSModel.is_valid_model(item):
                print(f"  {item}: ✓ Valid model")
            # Check if it's a preset
            elif repo.get_preset(item):
                print(f"  {item}: ✓ Valid preset")
            else:
                print(f"  {item}: ✗ Invalid")


if __name__ == "__main__":
    main()

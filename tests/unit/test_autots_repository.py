"""Tests for AutoTS repository."""

from unittest.mock import MagicMock

import pytest
from sqlmodel import Session, create_engine
from sqlmodel.pool import StaticPool

from stockula.data.autots_repository import AutoTSRepository
from stockula.database.models import AutoTSModel, AutoTSPreset, SQLModel


class TestAutoTSRepository:
    """Test AutoTS repository functionality."""

    @pytest.fixture
    def session(self):
        """Create an in-memory database session for testing."""
        # Create an in-memory database
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            yield session

    @pytest.fixture
    def logger(self):
        """Create a mock logger."""
        logger = MagicMock()
        return logger

    @pytest.fixture
    def repository(self, session, logger):
        """Create a repository instance."""
        return AutoTSRepository(session, logger)

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset the model cache before each test."""
        AutoTSModel._valid_models = None
        AutoTSModel._models_data = None
        AutoTSPreset._valid_presets = None
        yield
        AutoTSModel._valid_models = None
        AutoTSModel._models_data = None
        AutoTSPreset._valid_presets = None

    def test_init(self, session, logger):
        """Test repository initialization."""
        repo = AutoTSRepository(session, logger)
        assert repo.session == session
        assert repo.logger == logger

    def test_init_without_logger(self, session):
        """Test repository initialization without logger."""
        repo = AutoTSRepository(session)
        assert repo.session == session
        assert repo.logger is None

    def test_get_model_exists(self, repository, session):
        """Test getting an existing model."""
        # Create a model
        model = AutoTSModel(name="ARIMA", description="Test ARIMA model")
        session.add(model)
        session.commit()

        # Get the model
        result = repository.get_model("ARIMA")
        assert result is not None
        assert result.name == "ARIMA"
        assert result.description == "Test ARIMA model"

    def test_get_model_not_exists(self, repository):
        """Test getting a non-existent model."""
        result = repository.get_model("NonExistent")
        assert result is None

    def test_get_all_models_empty(self, repository):
        """Test getting all models when database is empty."""
        result = repository.get_all_models()
        assert result == []

    def test_get_all_models_with_data(self, repository, session):
        """Test getting all models when database has data."""
        # Create models
        model1 = AutoTSModel(name="ARIMA")
        model2 = AutoTSModel(name="ETS")
        model3 = AutoTSModel(name="VAR")
        session.add_all([model1, model2, model3])
        session.commit()

        # Get all models
        result = repository.get_all_models()
        assert len(result) == 3
        assert {m.name for m in result} == {"ARIMA", "ETS", "VAR"}

    def test_get_models_by_category(self, repository, session):
        """Test getting models by category."""
        # Create models with categories
        model1 = AutoTSModel(name="ARIMA", categories='["univariate"]')
        model2 = AutoTSModel(name="VAR", categories='["multivariate"]')
        model3 = AutoTSModel(name="ETS", categories='["univariate", "probabilistic"]')
        session.add_all([model1, model2, model3])
        session.commit()

        # Get univariate models
        result = repository.get_models_by_category("univariate")
        assert len(result) == 2
        assert {m.name for m in result} == {"ARIMA", "ETS"}

        # Get multivariate models
        result = repository.get_models_by_category("multivariate")
        assert len(result) == 1
        assert result[0].name == "VAR"

        # Get models from non-existent category
        result = repository.get_models_by_category("nonexistent")
        assert result == []

    def test_get_preset_exists(self, repository, session):
        """Test getting an existing preset."""
        # Create a preset
        preset = AutoTSPreset(name="fast", models='["ARIMA", "ETS"]', description="Fast preset")
        session.add(preset)
        session.commit()

        # Get the preset
        result = repository.get_preset("fast")
        assert result is not None
        assert result.name == "fast"
        assert result.description == "Fast preset"
        assert result.model_list == ["ARIMA", "ETS"]

    def test_get_preset_not_exists(self, repository):
        """Test getting a non-existent preset."""
        result = repository.get_preset("NonExistent")
        assert result is None

    def test_get_all_presets_empty(self, repository):
        """Test getting all presets when database is empty."""
        result = repository.get_all_presets()
        assert result == []

    def test_get_all_presets_with_data(self, repository, session):
        """Test getting all presets when database has data."""
        # Create presets
        preset1 = AutoTSPreset(name="fast", models='["ARIMA"]')
        preset2 = AutoTSPreset(name="slow", models='["VAR"]')
        preset3 = AutoTSPreset(name="default", models='["ETS"]')
        session.add_all([preset1, preset2, preset3])
        session.commit()

        # Get all presets
        result = repository.get_all_presets()
        assert len(result) == 3
        assert {p.name for p in result} == {"fast", "slow", "default"}

    def test_create_or_update_model_create(self, repository, session):
        """Test creating a new model."""
        model_data = {
            "name": "ARIMA",
            "description": "ARIMA model",
            "categories": ["univariate", "probabilistic"],
            "is_slow": False,
            "is_gpu_enabled": False,
            "requires_regressor": False,
        }

        result = repository.create_or_update_model(model_data)

        assert result.name == "ARIMA"
        assert result.description == "ARIMA model"
        assert result.category_list == ["univariate", "probabilistic"]
        assert result.is_slow is False

        # Verify it was saved to database
        saved = repository.get_model("ARIMA")
        assert saved is not None
        assert saved.name == "ARIMA"

    def test_create_or_update_model_update(self, repository, session):
        """Test updating an existing model."""
        # Create initial model
        model = AutoTSModel(name="ARIMA", description="Old description")
        session.add(model)
        session.commit()

        # Update the model
        model_data = {
            "name": "ARIMA",
            "description": "New description",
            "is_slow": True,
            "categories": ["univariate"],
        }

        result = repository.create_or_update_model(model_data)

        assert result.name == "ARIMA"
        assert result.description == "New description"
        assert result.is_slow is True
        assert result.category_list == ["univariate"]

        # Verify there's still only one ARIMA model
        all_models = repository.get_all_models()
        arima_models = [m for m in all_models if m.name == "ARIMA"]
        assert len(arima_models) == 1

    def test_create_or_update_model_invalid(self, repository):
        """Test creating a model with invalid name."""
        model_data = {
            "name": "InvalidModel",
            "description": "This should fail",
        }

        with pytest.raises(ValueError, match="is not a valid AutoTS model"):
            repository.create_or_update_model(model_data)

    def test_create_or_update_preset_create(self, repository, session):
        """Test creating a new preset."""
        preset_data = {
            "name": "test_preset",
            "models": ["ARIMA", "ETS"],
            "description": "Test preset",
            "use_case": "Testing",
        }

        result = repository.create_or_update_preset(preset_data)

        assert result.name == "test_preset"
        assert result.description == "Test preset"
        assert result.use_case == "Testing"
        assert result.model_list == ["ARIMA", "ETS"]

        # Verify it was saved to database
        saved = repository.get_preset("test_preset")
        assert saved is not None
        assert saved.name == "test_preset"

    def test_create_or_update_preset_with_dict_models(self, repository, session):
        """Test creating a preset with weighted models."""
        preset_data = {
            "name": "weighted_preset",
            "models": {"ARIMA": 0.5, "ETS": 0.3, "VAR": 0.2},
            "description": "Weighted preset",
        }

        result = repository.create_or_update_preset(preset_data)

        assert result.name == "weighted_preset"
        assert result.model_list == {"ARIMA": 0.5, "ETS": 0.3, "VAR": 0.2}

    def test_create_or_update_preset_update(self, repository, session):
        """Test updating an existing preset."""
        # Create initial preset
        preset = AutoTSPreset(name="fast", models='["ARIMA"]')
        session.add(preset)
        session.commit()

        # Update the preset
        preset_data = {
            "name": "fast",
            "models": ["ARIMA", "ETS", "VAR"],
            "description": "Updated fast preset",
        }

        result = repository.create_or_update_preset(preset_data)

        assert result.name == "fast"
        assert result.description == "Updated fast preset"
        assert result.model_list == ["ARIMA", "ETS", "VAR"]

        # Verify there's still only one fast preset
        all_presets = repository.get_all_presets()
        fast_presets = [p for p in all_presets if p.name == "fast"]
        assert len(fast_presets) == 1

    def test_create_or_update_preset_invalid_models(self, repository):
        """Test creating a preset with invalid models."""
        preset_data = {
            "name": "bad_preset",
            "models": ["ARIMA", "InvalidModel"],
            "description": "This should fail",
        }

        with pytest.raises(ValueError, match="contains invalid models"):
            repository.create_or_update_preset(preset_data)

    def test_seed_from_autots(self, repository):
        """Test seeding database from AutoTS library."""
        # Seed the database directly from AutoTS
        models_count, presets_count = repository.seed_from_autots()

        # We should have at least the core models and presets
        assert models_count >= 40  # AutoTS has around 49 models
        assert presets_count >= 10  # AutoTS has various presets

        # Verify key models were created
        all_models = repository.get_all_models()
        assert len(all_models) >= 40
        model_names = {m.name for m in all_models}
        # Check for some core models
        assert "ARIMA" in model_names
        assert "ETS" in model_names
        assert "VAR" in model_names
        assert "VECM" in model_names

        # Verify model categories
        univariate_models = repository.get_models_by_category("univariate")
        assert len(univariate_models) > 0
        univariate_names = {m.name for m in univariate_models}
        # ARIMA and ETS should be in univariate
        assert "ARIMA" in univariate_names or "ETS" in univariate_names

        # Verify presets were created
        all_presets = repository.get_all_presets()
        assert len(all_presets) >= 10
        preset_names = {p.name for p in all_presets}
        # Should include core presets
        assert "fast" in preset_names
        assert "default" in preset_names
        assert "superfast" in preset_names

        # Verify preset content
        fast_preset = repository.get_preset("fast")
        assert fast_preset is not None
        assert len(fast_preset.model_list) > 0  # Should have models

        default_preset = repository.get_preset("default")
        assert default_preset is not None
        assert len(default_preset.model_list) > 0  # Should have models

    def test_seed_from_json_deprecated(self, repository, logger):
        """Test that seed_from_json is deprecated and calls seed_from_autots."""
        # The deprecated method should still work but warn
        models_count, presets_count = repository.seed_from_json("/any/path.json")

        # Should have seeded from AutoTS, not the file
        assert models_count >= 40
        assert presets_count >= 10

        # Should have logged a deprecation warning
        logger.warning.assert_called_once_with("seed_from_json is deprecated. Using seed_from_autots() instead.")

    def test_seed_from_autots_with_logger(self, repository, logger):
        """Test that seeding logs the result."""
        # Seed the database from AutoTS
        models_count, presets_count = repository.seed_from_autots()

        # Verify logging (should log the actual counts)
        logger.info.assert_called_once()
        call_args = logger.info.call_args[0][0]
        assert "Seeded" in call_args
        assert "models" in call_args
        assert "presets" in call_args

    def test_validate_model_list_with_string_preset(self, repository, session):
        """Test validating a preset name."""
        # Create a preset
        preset = AutoTSPreset(name="fast", models='["ARIMA", "ETS"]')
        session.add(preset)
        session.commit()

        # Validate the preset name
        is_valid, invalid = repository.validate_model_list("fast")
        assert is_valid is True
        assert invalid == []

    def test_validate_model_list_with_string_model(self, repository, session):
        """Test validating a single model name."""
        # Create a model
        model = AutoTSModel(name="ARIMA")
        session.add(model)
        session.commit()

        # Validate the model name
        is_valid, invalid = repository.validate_model_list("ARIMA")
        assert is_valid is True
        assert invalid == []

    def test_validate_model_list_with_string_invalid(self, repository):
        """Test validating an invalid string."""
        is_valid, invalid = repository.validate_model_list("InvalidName")
        assert is_valid is False
        assert invalid == ["InvalidName"]

    def test_validate_model_list_with_list_valid(self, repository, session):
        """Test validating a list of valid models."""
        # Create models
        model1 = AutoTSModel(name="ARIMA")
        model2 = AutoTSModel(name="ETS")
        model3 = AutoTSModel(name="VAR")
        session.add_all([model1, model2, model3])
        session.commit()

        # Validate the model list
        is_valid, invalid = repository.validate_model_list(["ARIMA", "ETS", "VAR"])
        assert is_valid is True
        assert invalid == []

    def test_validate_model_list_with_list_mixed(self, repository, session):
        """Test validating a list with some invalid models."""
        # Create models
        model1 = AutoTSModel(name="ARIMA")
        model2 = AutoTSModel(name="ETS")
        session.add_all([model1, model2])
        session.commit()

        # Validate the model list
        is_valid, invalid = repository.validate_model_list(["ARIMA", "InvalidModel", "ETS", "FakeModel"])
        assert is_valid is False
        assert set(invalid) == {"InvalidModel", "FakeModel"}

    def test_validate_model_list_with_empty_list(self, repository):
        """Test validating an empty list."""
        is_valid, invalid = repository.validate_model_list([])
        assert is_valid is True
        assert invalid == []

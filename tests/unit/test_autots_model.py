"""Tests for AutoTS model validation."""

import json

import pytest

from stockula.database.models import AutoTSModel, AutoTSPreset


class TestAutoTSModel:
    """Test AutoTS model validation functionality."""

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

    def test_validate_valid_model(self):
        """Test validation of valid model names."""
        assert AutoTSModel.is_valid_model("ARIMA") is True
        assert AutoTSModel.is_valid_model("ETS") is True
        assert AutoTSModel.is_valid_model("LastValueNaive") is True
        assert AutoTSModel.is_valid_model("UnivariateMotif") is True

    def test_validate_invalid_model(self):
        """Test validation of invalid model names."""
        assert AutoTSModel.is_valid_model("InvalidModel") is False
        assert AutoTSModel.is_valid_model("RandomModel") is False
        assert AutoTSModel.is_valid_model("") is False

    def test_validate_model_list_valid(self):
        """Test validation of valid model lists."""
        valid_list = ["ARIMA", "ETS", "Theta"]
        is_valid, invalid = AutoTSModel.validate_model_list(valid_list)
        assert is_valid is True
        assert invalid == []

    def test_validate_model_list_invalid(self):
        """Test validation of model lists with invalid models."""
        mixed_list = ["ARIMA", "InvalidModel", "ETS", "FakeModel"]
        is_valid, invalid = AutoTSModel.validate_model_list(mixed_list)
        assert is_valid is False
        assert set(invalid) == {"InvalidModel", "FakeModel"}

    def test_model_validation_on_creation(self):
        """Test that model validation occurs when creating an AutoTSModel."""
        # Valid model should work
        valid_model = AutoTSModel(name="ARIMA")
        valid_model.validate()  # Should not raise

        # Invalid model should raise
        invalid_model = AutoTSModel(name="InvalidModel")
        with pytest.raises(ValueError, match="is not a valid AutoTS model"):
            invalid_model.validate()

    def test_get_valid_models(self):
        """Test getting all valid models."""
        valid_models = AutoTSModel.get_valid_models()
        assert isinstance(valid_models, set)
        assert len(valid_models) > 0
        assert "ARIMA" in valid_models
        assert "ETS" in valid_models

    def test_case_sensitivity(self):
        """Test that model validation is case-sensitive."""
        # AutoTS models are case-sensitive
        assert AutoTSModel.is_valid_model("ARIMA") is True
        assert AutoTSModel.is_valid_model("arima") is False
        assert AutoTSModel.is_valid_model("Arima") is False

    def test_load_valid_models_idempotent(self):
        """Test that loading models multiple times is idempotent."""
        AutoTSModel.load_valid_models()
        first_set = AutoTSModel._valid_models.copy() if AutoTSModel._valid_models else set()

        AutoTSModel.load_valid_models()
        second_set = AutoTSModel._valid_models.copy() if AutoTSModel._valid_models else set()

        assert first_set == second_set

    def test_load_valid_models_force_reload(self):
        """Test force reload of models."""
        AutoTSModel.load_valid_models()
        AutoTSModel._valid_models = {"TestModel"}  # Modify the cache

        AutoTSModel.load_valid_models()  # Should not reload
        assert AutoTSModel._valid_models == {"TestModel"}

        AutoTSModel.load_valid_models(force_reload=True)  # Should reload
        assert "TestModel" not in AutoTSModel._valid_models
        assert "ARIMA" in AutoTSModel._valid_models


class TestAutoTSPreset:
    """Test AutoTS preset validation functionality."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset the preset cache before each test."""
        AutoTSPreset._valid_presets = None
        AutoTSModel._valid_models = None
        AutoTSModel._models_data = None
        yield
        AutoTSPreset._valid_presets = None
        AutoTSModel._valid_models = None
        AutoTSModel._models_data = None

    def test_validate_preset_names(self):
        """Test validation of preset names."""
        # Test known presets
        assert AutoTSPreset.is_valid_preset("fast") is True
        assert AutoTSPreset.is_valid_preset("superfast") is True
        assert AutoTSPreset.is_valid_preset("probabilistic") is True

        # AutoTS built-in presets should also be valid
        assert AutoTSPreset.is_valid_preset("all") is True
        assert AutoTSPreset.is_valid_preset("no_shared") is True

    def test_preset_validation_on_creation(self):
        """Test that preset validation occurs when creating an AutoTSPreset."""
        # Valid preset with valid models should work
        valid_preset = AutoTSPreset(name="test_preset", models='["ARIMA", "ETS"]')
        valid_preset.validate()  # Should not raise

        # Preset with invalid models should raise
        invalid_preset = AutoTSPreset(name="bad_preset", models='["ARIMA", "InvalidModel"]')
        with pytest.raises(ValueError, match="contains invalid models"):
            invalid_preset.validate()

    def test_preset_model_list_property(self):
        """Test the model_list property of AutoTSPreset."""
        # Test list format
        preset_list = AutoTSPreset(name="test_list", models='["ARIMA", "ETS", "VAR"]')
        assert preset_list.model_list == ["ARIMA", "ETS", "VAR"]

        # Test dict format
        preset_dict = AutoTSPreset(name="test_dict", models='{"ARIMA": 0.5, "ETS": 0.3, "VAR": 0.2}')
        assert preset_dict.model_list == {"ARIMA": 0.5, "ETS": 0.3, "VAR": 0.2}

    def test_preset_set_models_method(self):
        """Test the set_models method of AutoTSPreset."""
        preset = AutoTSPreset(name="test_preset", models="[]")

        # Set as list
        preset.set_models(["ARIMA", "ETS"])
        assert json.loads(preset.models) == ["ARIMA", "ETS"]

        # Set as dict
        preset.set_models({"VAR": 0.6, "VECM": 0.4})
        assert json.loads(preset.models) == {"VAR": 0.6, "VECM": 0.4}

"""Repository for AutoTS model management."""

from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from ..database.models import AutoTSModel, AutoTSPreset
from ..interfaces import ILoggingManager


class AutoTSRepository:
    """Repository for managing AutoTS models and presets in the database."""

    def __init__(self, session: Session, logger: ILoggingManager | None = None):
        """Initialize the repository.

        Args:
            session: Database session
            logger: Optional logging manager
        """
        self.session = session
        self.logger = logger

    def get_model(self, name: str) -> AutoTSModel | None:
        """Get a model by name.

        Args:
            name: Model name

        Returns:
            AutoTSModel or None if not found
        """
        statement = select(AutoTSModel).where(AutoTSModel.name == name)
        return self.session.exec(statement).first()

    def get_all_models(self) -> list[AutoTSModel]:
        """Get all models.

        Returns:
            List of all AutoTS models
        """
        statement = select(AutoTSModel)
        return list(self.session.exec(statement).all())

    def get_models_by_category(self, category: str) -> list[AutoTSModel]:
        """Get models by category.

        Args:
            category: Category name (e.g., 'univariate', 'multivariate')

        Returns:
            List of models in the category
        """
        # This requires checking the JSON field
        all_models = self.get_all_models()
        return [m for m in all_models if category in m.category_list]

    def get_preset(self, name: str) -> AutoTSPreset | None:
        """Get a preset by name.

        Args:
            name: Preset name

        Returns:
            AutoTSPreset or None if not found
        """
        statement = select(AutoTSPreset).where(AutoTSPreset.name == name)
        return self.session.exec(statement).first()

    def get_all_presets(self) -> list[AutoTSPreset]:
        """Get all presets.

        Returns:
            List of all presets
        """
        statement = select(AutoTSPreset)
        return list(self.session.exec(statement).all())

    def create_or_update_model(self, model_data: dict[str, Any]) -> AutoTSModel:
        """Create or update a model.

        Args:
            model_data: Model data dictionary

        Returns:
            Created or updated model

        Raises:
            ValueError: If the model name is not valid
        """
        name = model_data["name"]

        # Validate the model name before proceeding
        if not AutoTSModel.is_valid_model(name):
            raise ValueError(f"'{name}' is not a valid AutoTS model")

        existing = self.get_model(name)

        if existing:
            # Update existing
            for key, value in model_data.items():
                if key == "categories" and isinstance(value, list):
                    existing.set_categories(value)
                elif hasattr(existing, key):
                    setattr(existing, key, value)
            model = existing
        else:
            # Create new
            if "categories" in model_data and isinstance(model_data["categories"], list):
                categories = model_data.pop("categories")
                model = AutoTSModel(**model_data)
                model.set_categories(categories)
            else:
                model = AutoTSModel(**model_data)

            # Validate before adding to session
            model.validate()
            self.session.add(model)

        self.session.commit()
        return model

    def create_or_update_preset(self, preset_data: dict[str, Any]) -> AutoTSPreset:
        """Create or update a preset.

        Args:
            preset_data: Preset data dictionary

        Returns:
            Created or updated preset

        Raises:
            ValueError: If the preset contains invalid models
        """
        name = preset_data["name"]
        existing = self.get_preset(name)

        if existing:
            # Update existing
            for key, value in preset_data.items():
                if key == "models" and isinstance(value, list | dict):
                    existing.set_models(value)
                elif hasattr(existing, key):
                    setattr(existing, key, value)
            # Validate after update
            existing.validate()
            preset = existing
        else:
            # Create new
            if "models" in preset_data and isinstance(preset_data["models"], list | dict):
                models = preset_data.pop("models")
                preset = AutoTSPreset(**preset_data)
                preset.set_models(models)
            else:
                preset = AutoTSPreset(**preset_data)

            # Validate before adding to session
            preset.validate()
            self.session.add(preset)

        self.session.commit()
        return preset

    def seed_from_autots(self) -> tuple[int, int]:
        """Seed the database directly from AutoTS library.

        This method uses AutoTS's model_lists as the single source of truth
        for models and presets.

        Returns:
            Tuple of (models_count, presets_count) created/updated
        """
        try:
            from autots.models.model_list import model_lists
        except ImportError:
            raise ImportError("AutoTS is required to seed the database") from None

        models_count = 0
        presets_count = 0

        # Get all unique models from AutoTS
        # Only use models from the "all" list as it's the authoritative list
        all_models = set(model_lists.get("all", []))

        # Build categories map from AutoTS data
        categories_map = {}
        for key in model_lists.keys():
            if isinstance(model_lists[key], list):
                categories_map[key] = set(model_lists[key])
            elif isinstance(model_lists[key], dict):
                categories_map[key] = set(model_lists[key].keys())

        # Create/update models based on AutoTS data
        for model_name in all_models:
            model_categories = []
            for category, models in categories_map.items():
                if model_name in models:
                    model_categories.append(category)

            model_data = {
                "name": model_name,
                "categories": model_categories,
                "is_slow": model_name in categories_map.get("slow", set()),
                "is_gpu_enabled": model_name in categories_map.get("gpu", set()),
                "requires_regressor": model_name in categories_map.get("regressor", set()),
            }

            # Note: Descriptions would ideally come from AutoTS documentation
            # For now, we'll skip them rather than hardcode
            # In the future, these could be extracted from AutoTS docstrings

            self.create_or_update_model(model_data)
            models_count += 1

        # Create/update presets - these are the actual presets for AutoTS
        # We'll store the main presets that are commonly used
        preset_keys = {
            "default",
            "fast",
            "superfast",
            "parallel",
            "fast_parallel",
            "fast_parallel_no_arima",
            "scalable",
            "probabilistic",
            "multivariate",
            "univariate",
            "best",
            "slow",
            "gpu",
            "regressor",
            "motifs",
            "regressions",
            "experimental",
        }

        for preset_name in preset_keys:
            if preset_name not in model_lists:
                continue

            preset_data = model_lists[preset_name]

            # Filter preset models to only include valid models
            if isinstance(preset_data, list):
                # Filter out any models not in the "all" list
                filtered_models = [m for m in preset_data if m in all_models]
                if not filtered_models:
                    continue  # Skip empty presets
                preset_data = filtered_models
            elif isinstance(preset_data, dict):
                # Filter out any models not in the "all" list
                filtered_models = {k: v for k, v in preset_data.items() if k in all_models}
                if not filtered_models:
                    continue  # Skip empty presets
                preset_data = filtered_models

            preset_info = {
                "name": preset_name,
                "models": preset_data,
                # Let descriptions be None rather than hardcoding them
                # They can be added later from AutoTS documentation if needed
            }

            self.create_or_update_preset(preset_info)
            presets_count += 1

        if self.logger:
            self.logger.info(f"Seeded {models_count} models and {presets_count} presets")

        return models_count, presets_count

    def seed_from_json(self, json_path: str | Path) -> tuple[int, int]:
        """Deprecated: Use seed_from_autots() instead.

        This method is kept for backward compatibility but now just calls
        seed_from_autots() directly, ignoring the json_path parameter.

        Args:
            json_path: Path to models.json file (ignored)

        Returns:
            Tuple of (models_count, presets_count) created/updated
        """
        if self.logger:
            self.logger.warning("seed_from_json is deprecated. Using seed_from_autots() instead.")
        return self.seed_from_autots()

    def validate_model_list(self, model_list: list[str] | str) -> tuple[bool, list[str]]:
        """Validate a model list against the database.

        Args:
            model_list: List of model names or preset name

        Returns:
            Tuple of (is_valid, invalid_models)
        """
        if isinstance(model_list, str):
            # Check if it's a preset
            preset = self.get_preset(model_list)
            if preset:
                return True, []
            # Check if it's a single model
            model = self.get_model(model_list)
            return model is not None, [] if model else [model_list]

        # Validate list of models
        all_models = {m.name for m in self.get_all_models()}
        invalid_models = [m for m in model_list if m not in all_models]
        return len(invalid_models) == 0, invalid_models

"""Settings management using pydantic-settings."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import MACDConfig, RSIConfig, SMACrossConfig, StockulaConfig, StrategyConfig


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="STOCKULA_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    config_file: str | None = Field(default=None, description="Path to YAML configuration file")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration data
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_strategy_config(strategy_dict: dict[str, Any]) -> StrategyConfig:
    """Parse strategy configuration from dictionary.

    Args:
        strategy_dict: Dictionary containing strategy configuration

    Returns:
        StrategyConfig instance
    """
    strategy_name = strategy_dict.get("name", "").lower()
    params = strategy_dict.get("parameters", {})

    # Map strategy names to specific config classes
    if strategy_name == "smacross":
        params_model = SMACrossConfig(**params)
        params = params_model.model_dump()
    elif strategy_name == "rsi":
        params_model = RSIConfig(**params)
        params = params_model.model_dump()
    elif strategy_name == "macd":
        params_model = MACDConfig(**params)
        params = params_model.model_dump()

    return StrategyConfig(name=strategy_dict["name"], parameters=params)


def load_config(config_path: str | Path | None = None) -> StockulaConfig:
    """Load configuration from file or environment.

    Args:
        config_path: Optional path to configuration file.
                    If not provided, checks for:
                    1. STOCKULA_CONFIG_FILE env var
                    2. .config.yaml in current directory
                    3. .config.yml in current directory
                    4. stockula.yaml in current directory
                    5. stockula.yml in current directory

    Returns:
        StockulaConfig instance
    """
    settings = Settings()

    # Determine config file path
    if config_path is None:
        config_path = settings.config_file

        # If no env var, check for default files
        if config_path is None:
            default_files = [
                ".config.yaml",
                ".config.yml",
                "stockula.yaml",
                "stockula.yml",
            ]
            for filename in default_files:
                if Path(filename).exists():
                    config_path = filename
                    break

    # If no config file specified or found, return default configuration
    if config_path is None:
        return StockulaConfig()

    # Load YAML configuration
    config_data = load_yaml_config(config_path)

    # Parse strategies if present
    if "backtest" in config_data and "strategies" in config_data["backtest"]:
        strategies = []
        for strategy_dict in config_data["backtest"]["strategies"]:
            strategies.append(parse_strategy_config(strategy_dict))
        config_data["backtest"]["strategies"] = strategies

    # Create and return configuration object
    return StockulaConfig(**config_data)


def save_config(config: StockulaConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary for YAML serialization
    config_dict = config.model_dump(mode="json")

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

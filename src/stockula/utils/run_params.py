"""Simple parameter classes for reducing function complexity."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DateOverrides:
    """Simple container for date override parameters."""

    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    def apply_to_config(self, config) -> None:
        """Apply date overrides to configuration object."""
        if self.train_start:
            config.forecast.train_start_date = datetime.strptime(self.train_start, "%Y-%m-%d").date()
        if self.train_end:
            config.forecast.train_end_date = datetime.strptime(self.train_end, "%Y-%m-%d").date()
        if self.test_start:
            config.forecast.test_start_date = datetime.strptime(self.test_start, "%Y-%m-%d").date()
        if self.test_end:
            config.forecast.test_end_date = datetime.strptime(self.test_end, "%Y-%m-%d").date()


@dataclass
class SavePaths:
    """Simple container for save path parameters."""

    config_path: str | None = None
    optimized_config_path: str | None = None

    def get_save_path_for_optimization(self) -> str | None:
        """Get the appropriate save path for optimization mode."""
        return self.optimized_config_path or self.config_path


@dataclass
class RunConfig:
    """Simple container for run configuration parameters."""

    config_file: str | None = None
    ticker: str | None = None
    mode: str = "all"
    output: str = "console"
    save_paths: SavePaths | None = None
    date_overrides: DateOverrides | None = None

    def __post_init__(self):
        """Initialize nested objects if not provided."""
        if self.save_paths is None:
            self.save_paths = SavePaths()
        if self.date_overrides is None:
            self.date_overrides = DateOverrides()

    def should_save_config(self) -> bool:
        """Check if configuration should be saved."""
        return (
            self.save_paths is not None
            and self.save_paths.config_path is not None
            and self.mode != "optimize-allocation"
        )

    def is_optimization_mode(self) -> bool:
        """Check if this is optimization mode."""
        return self.mode == "optimize-allocation"


@dataclass
class PipelineConfig:
    """Simple container for pipeline configuration parameters."""

    base_config: str = ".stockula.yaml"
    optimized_config: str | None = None
    output: str | None = None
    skip_optimization: bool = False
    skip_backtest: bool = False
    verbose: bool = False

    def should_skip_both(self) -> bool:
        """Check if both optimization and backtest should be skipped."""
        return self.skip_optimization and self.skip_backtest

    def should_run_full_pipeline(self) -> bool:
        """Check if full pipeline should be run."""
        return not self.skip_optimization and not self.skip_backtest

    def get_output_format(self) -> str:
        """Determine output format from file extension."""
        if not self.output:
            return "json"
        if self.output.endswith((".yaml", ".yml")):
            return "yaml"
        elif self.output.endswith(".csv"):
            return "csv"
        return "json"

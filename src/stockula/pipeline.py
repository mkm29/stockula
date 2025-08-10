"""Stockula Pipeline - Orchestrates optimization and backtesting workflows."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import StockulaConfig
from .container import Container
from .display import ResultsDisplay
from .manager import StockulaManager


class StockulaPipeline:
    """Orchestrates the complete Stockula workflow from optimization to backtesting.

    This class provides a high-level interface for:
    1. Loading and validating configurations
    2. Running portfolio optimization
    3. Saving optimized configurations
    4. Running backtests with optimized allocations
    5. Comparing results and generating reports
    """

    def __init__(
        self,
        base_config_path: str | Path | None = None,
        verbose: bool = False,
        console: Console | None = None,
    ):
        """Initialize the Stockula Pipeline.

        Args:
            base_config_path: Path to the base configuration file
            verbose: Enable verbose output
            console: Rich console for output (creates one if not provided)
        """
        self.base_config_path = Path(base_config_path) if base_config_path else None
        self.verbose = verbose
        self.console = console or Console()

        # Initialize container and manager
        self.container = Container()
        self.manager: StockulaManager | None = None
        self.display = ResultsDisplay()

        # Store results
        self.optimization_results: dict[str, Any] = {}
        self.backtest_results: dict[str, Any] = {}
        self.optimized_config: StockulaConfig | None = None

    def load_configuration(self, config_path: str | Path | None = None) -> StockulaConfig:
        """Load and validate configuration.

        Args:
            config_path: Path to configuration file (uses base_config_path if not provided)

        Returns:
            Validated StockulaConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config is invalid
        """
        path = Path(config_path) if config_path else self.base_config_path

        if not path or not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Load YAML configuration
        with open(path) as f:
            config_data = yaml.safe_load(f)

        # Configure container with the config path
        if path:
            self.container.config_path.override(str(path))

        # Create StockulaConfig from the data
        config = StockulaConfig.model_validate(config_data)

        if self.verbose:
            self.console.print(f"[green]✓[/green] Loaded configuration from {path}")

        return cast(StockulaConfig, config)

    def run_optimization(
        self,
        config: StockulaConfig | None = None,
        save_config_path: str | Path | None = None,
        **optimization_params: Any,
    ) -> tuple[StockulaConfig, dict[str, Any]]:
        """Run portfolio optimization and optionally save the optimized configuration.

        Args:
            config: Configuration to use (loads from base_config_path if not provided)
            save_config_path: Path to save the optimized configuration
            **optimization_params: Additional parameters for optimization

        Returns:
            Tuple of (optimized_config, optimization_results)
        """
        # Load configuration if not provided
        if config is None:
            config = self.load_configuration()

        # Initialize manager with the configuration
        self._initialize_manager(config)

        # Show optimization start
        self.console.print("\n[bold blue]🎯 Starting Portfolio Optimization[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            # Run optimization
            task = progress.add_task("Optimizing portfolio allocation...", total=None)

            # Execute optimization mode
            if not self.manager:
                raise RuntimeError("Manager not initialized. Call _initialize_manager first.")

            # Call the specific optimization method
            save_path = optimization_params.get("save_path") if optimization_params else None
            exit_code = self.manager.run_optimize_allocation(save_path)

            if exit_code != 0:
                raise RuntimeError(f"Optimization failed with exit code {exit_code}")

            # Extract results from the updated configuration
            # The optimization modifies the config in-place
            results: dict[str, Any] = {"optimized_allocations": {}, "metrics": {}}

            # Get the optimized quantities from the updated config
            for ticker_config in self.manager.config.portfolio.tickers:
                results["optimized_allocations"][ticker_config.symbol] = ticker_config.quantity

            progress.update(task, completed=True)

        # Extract optimized configuration from results
        self.optimized_config = self._extract_optimized_config(config, results)
        self.optimization_results = results

        # Save optimized configuration if path provided
        if save_config_path:
            self.save_optimized_config(save_config_path)

        # Display optimization results
        self._display_optimization_results(results)

        return self.optimized_config, results

    def run_backtest(
        self,
        config: StockulaConfig | None = None,
        use_optimized: bool = True,
        **backtest_params: Any,
    ) -> dict[str, Any]:
        """Run backtest with either provided or optimized configuration.

        Args:
            config: Configuration to use (uses optimized_config if use_optimized=True)
            use_optimized: Whether to use the optimized configuration
            **backtest_params: Additional parameters for backtesting

        Returns:
            Backtest results dictionary
        """
        # Determine which configuration to use
        if use_optimized and self.optimized_config:
            config = self.optimized_config
            self.console.print("[green]Using optimized configuration for backtesting[/green]")
        elif config is None:
            config = self.load_configuration()

        # Initialize manager with the configuration
        self._initialize_manager(config)

        # Show backtest start
        self.console.print("\n[bold blue]📊 Starting Backtest Analysis[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            # Run backtest
            task = progress.add_task("Running backtest strategies...", total=None)

            # Execute backtest mode
            if not self.manager:
                raise RuntimeError("Manager not initialized. Call _initialize_manager first.")

            # Create portfolio for backtesting
            portfolio = self.manager.create_portfolio()

            # Run backtesting through the main processing method
            results = self.manager.run_main_processing("backtest", portfolio)

            progress.update(task, completed=True)

        self.backtest_results = results

        # Display backtest results
        self._display_backtest_results(results)

        return cast(dict[str, Any], results)

    def run_full_pipeline(
        self,
        base_config_path: str | Path | None = None,
        optimized_config_path: str | Path | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Run the complete pipeline: optimization followed by backtesting.

        Args:
            base_config_path: Path to base configuration
            optimized_config_path: Path to save optimized configuration
            **params: Additional parameters for both optimization and backtesting

        Returns:
            Combined results dictionary with optimization and backtest results
        """
        # Load base configuration
        if base_config_path:
            self.base_config_path = Path(base_config_path)
        config = self.load_configuration()

        # Run optimization
        self.console.rule("[bold]Step 1: Portfolio Optimization[/bold]")
        optimized_config, opt_results = self.run_optimization(
            config=config,
            save_config_path=optimized_config_path,
            **params.get("optimization", {}),
        )

        # Run backtest with optimized configuration
        self.console.rule("[bold]Step 2: Backtesting with Optimized Allocation[/bold]")
        self.run_backtest(
            config=optimized_config,
            use_optimized=True,
            **params.get("backtest", {}),
        )

        # Combine and compare results
        combined_results = self._combine_results()

        # Display comparison
        self.console.rule("[bold]Pipeline Results Summary[/bold]")
        self._display_comparison()

        return combined_results

    def save_optimized_config(self, path: str | Path) -> None:
        """Save the optimized configuration to a YAML file.

        Args:
            path: Path to save the configuration
        """
        if not self.optimized_config:
            raise ValueError("No optimized configuration available. Run optimization first.")

        path = Path(path)

        # Convert config to dict
        config_dict = self.optimized_config.model_dump()

        # Save to YAML
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.console.print(f"[green]✓[/green] Saved optimized configuration to {path}")

    def save_results(self, path: str | Path, format: str = "json") -> None:
        """Save pipeline results to file.

        Args:
            path: Path to save results
            format: Output format ('json', 'yaml', or 'csv')
        """
        path = Path(path)
        combined_results = self._combine_results()

        if format == "json":
            with open(path, "w") as f:
                json.dump(combined_results, f, indent=2, default=str)
        elif format == "yaml":
            with open(path, "w") as f:
                yaml.dump(combined_results, f, default_flow_style=False)
        elif format == "csv":
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(combined_results.get("backtest", {}).get("results", []))
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.console.print(f"[green]✓[/green] Saved results to {path}")

    def _initialize_manager(self, config: StockulaConfig) -> None:
        """Initialize the StockulaManager with configuration.

        Args:
            config: Configuration to use
        """
        # Wire the container
        self.container.wire(
            modules=[
                "stockula.main",
                "stockula.manager",
                "stockula.allocation.manager",
                "stockula.allocation.allocator",
                "stockula.allocation.backtest_allocator",
                "stockula.backtesting.manager",
                "stockula.data.fetcher",
                "stockula.data.manager",
                "stockula.domain.factory",
                "stockula.domain.portfolio",
                "stockula.forecasting.manager",
                "stockula.technical_analysis.manager",
            ]
        )

        # Override configuration
        self.container.stockula_config.override(config)

        # Create manager
        self.manager = StockulaManager(
            config=config,
            container=self.container,
            console=self.console,
        )

    def _extract_optimized_config(self, base_config: StockulaConfig, results: dict[str, Any]) -> StockulaConfig:
        """Extract optimized configuration from optimization results.

        Args:
            base_config: Base configuration
            results: Optimization results

        Returns:
            Updated configuration with optimized allocations
        """
        # Create a copy of the base configuration
        config_dict = base_config.model_dump()

        # Extract optimized allocations from results
        if "optimized_allocations" in results:
            allocations = results["optimized_allocations"]

            # Update ticker quantities in the configuration
            for ticker_config in config_dict["portfolio"]["tickers"]:
                symbol = ticker_config["symbol"]
                if symbol in allocations:
                    ticker_config["quantity"] = allocations[symbol]

            # Mark as optimized
            config_dict["portfolio"]["allocation_method"] = "optimized"

        # Create new config from updated dict
        return cast(StockulaConfig, StockulaConfig.model_validate(config_dict))

    def _combine_results(self) -> dict[str, Any]:
        """Combine optimization and backtest results.

        Returns:
            Combined results dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "optimization": self.optimization_results,
            "backtest": self.backtest_results,
            "config": {
                "base": str(self.base_config_path) if self.base_config_path else None,
                "optimized": self.optimized_config.model_dump() if self.optimized_config else None,
            },
        }

    def _display_optimization_results(self, results: dict[str, Any]) -> None:
        """Display optimization results in a formatted table.

        Args:
            results: Optimization results
        """
        if not results.get("optimized_allocations"):
            return

        table = Table(title="Optimized Portfolio Allocation")
        table.add_column("Symbol", style="cyan")
        table.add_column("Quantity", justify="right")
        table.add_column("Allocation %", justify="right")

        allocations = results["optimized_allocations"]
        total = sum(allocations.values())

        for symbol, quantity in sorted(allocations.items()):
            percentage = (quantity / total * 100) if total > 0 else 0
            table.add_row(symbol, f"{quantity:.2f}", f"{percentage:.1f}%")

        self.console.print(table)

    def _display_backtest_results(self, results: dict[str, Any]) -> None:
        """Display backtest results using ResultsDisplay.

        Args:
            results: Backtest results
        """
        if "results" in results and results["results"]:
            # Format results for display using print_results
            # Extract the list of results for the display
            display_results = {"backtesting": results["results"]}
            self.display.print_results(display_results, "console")

    def _display_comparison(self) -> None:
        """Display comparison between optimization and backtest results."""
        if not self.optimization_results or not self.backtest_results:
            return

        # Create comparison panel
        comparison_text = []

        # Add optimization metrics
        if "metrics" in self.optimization_results:
            comparison_text.append("[bold]Optimization Metrics:[/bold]")
            for key, value in self.optimization_results["metrics"].items():
                comparison_text.append(f"  • {key}: {value}")

        # Add backtest performance
        if "summary" in self.backtest_results:
            comparison_text.append("\n[bold]Backtest Performance:[/bold]")
            summary = self.backtest_results["summary"]
            comparison_text.append(f"  • Total Return: {summary.get('total_return', 'N/A')}")
            comparison_text.append(f"  • Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A')}")
            comparison_text.append(f"  • Max Drawdown: {summary.get('max_drawdown', 'N/A')}")

        panel = Panel("\n".join(comparison_text), title="Pipeline Summary", border_style="green")
        self.console.print(panel)


# Convenience functions for CLI integration
def run_pipeline(
    base_config: str,
    optimized_config: str | None = None,
    output: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the complete Stockula pipeline.

    Args:
        base_config: Path to base configuration file
        optimized_config: Path to save optimized configuration
        output: Path to save results
        verbose: Enable verbose output

    Returns:
        Pipeline results
    """
    pipeline = StockulaPipeline(base_config_path=base_config, verbose=verbose)

    results = pipeline.run_full_pipeline(
        optimized_config_path=optimized_config,
    )

    if output:
        format = "json"
        if output.endswith(".yaml") or output.endswith(".yml"):
            format = "yaml"
        elif output.endswith(".csv"):
            format = "csv"
        pipeline.save_results(output, format=format)

    return results

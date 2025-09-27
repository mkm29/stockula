"""
Report Service following SRP.
Single Responsibility: Report generation and file operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console

from ..config import StockulaConfig
from ..config.models import BacktestResult, PortfolioBacktestResults, StrategyBacktestSummary
from ..container import Container
from ..utils import get_console

logger = logging.getLogger(__name__)


class ReportService:
    """Manages report generation - Single Responsibility: Report Generation."""

    def __init__(
        self,
        config: StockulaConfig,
        container: Container,
        console: Optional[Console] = None,
    ):
        """Initialize report service.

        Args:
            config: Configuration object
            container: Dependency injection container
            console: Rich console for output (optional)
        """
        self.config = config
        self.container = container
        self.console = get_console(console)
        self.log_manager = container.logging_manager()

    def save_detailed_report(
        self,
        strategy_name: str,
        strategy_results: List[Dict],
        results: Dict[str, Any],
        portfolio_results: Optional[PortfolioBacktestResults] = None,
    ) -> str:
        """Save detailed strategy report to file.

        Args:
            strategy_name: Name of the strategy
            strategy_results: List of backtest results for this strategy
            results: Overall results dictionary
            portfolio_results: Portfolio backtest results (optional)

        Returns:
            Path to the saved report file
        """
        # Create reports directory if it doesn't exist
        reports_dir = Path(self.config.output.get("results_dir", "./results")) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"strategy_report_{strategy_name}_{timestamp}.json"

        # Prepare detailed report data
        report_data = {
            "strategy": strategy_name,
            "timestamp": timestamp,
            "date_range": {
                "start": self._date_to_string(self.config.data.start_date),
                "end": self._date_to_string(self.config.data.end_date),
            },
            "portfolio": {
                "initial_value": results.get("initial_portfolio_value", 0),
                "initial_capital": results.get("initial_capital", 0),
            },
            "broker_config": self._get_broker_config_dict(),
            "detailed_results": strategy_results,
            "summary": self._calculate_strategy_summary(strategy_results),
        }

        # Save report
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Also save structured results if provided
        if portfolio_results:
            structured_file = reports_dir / f"portfolio_backtest_{timestamp}.json"
            with open(structured_file, "w") as f:
                # Convert to dict using model_dump
                json.dump(portfolio_results.model_dump(), f, indent=2, default=str)

        return str(report_file)

    def save_optimized_config(self, save_path: str, optimized_quantities: Dict[str, float]) -> None:
        """Save optimized configuration to file.

        Args:
            save_path: Path to save the configuration
            optimized_quantities: Dictionary of symbol to quantity
        """
        # Update the config with optimized quantities
        for ticker_config in self.config.portfolio.tickers:
            if ticker_config.symbol in optimized_quantities:
                # Convert numpy types to native Python types
                quantity = optimized_quantities[ticker_config.symbol]
                if hasattr(quantity, "item"):
                    # Convert numpy scalar to Python type
                    ticker_config.quantity = float(quantity.item())
                else:
                    # Keep as integer if it's already an integer (from backtest_optimized)
                    if isinstance(quantity, int):
                        ticker_config.quantity = float(quantity)
                    else:
                        ticker_config.quantity = float(quantity)
                # Clear allocation_pct and allocation_amount since we now have quantities
                ticker_config.allocation_pct = None
                ticker_config.allocation_amount = None

        # Change allocation method to custom since we now have fixed quantities
        self.config.portfolio.allocation_method = "custom"
        self.config.portfolio.dynamic_allocation = False
        self.config.portfolio.auto_allocate = False

        # Save to file
        config_dict = self.config.model_dump(exclude_none=True)

        # Normalize strategy names in backtest configuration
        if "backtest" in config_dict and "strategies" in config_dict["backtest"]:
            for strategy in config_dict["backtest"]["strategies"]:
                if "name" in strategy:
                    strategy["name"] = self._normalize_strategy_name(strategy["name"])

        # Convert dates to strings for YAML serialization
        config_dict = self._convert_dates(config_dict)

        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.console.print(f"\n[green]✓ Optimized configuration saved to: {save_path}[/green]")
        self.console.print(
            f"[dim]You can now run backtest with: uv run python -m stockula --config {save_path} --mode backtest[/dim]"
        )

    def create_portfolio_backtest_results(
        self,
        results: Dict[str, Any],
        strategy_results: Dict[str, List[Dict]],
    ) -> PortfolioBacktestResults:
        """Create structured backtest results.

        Args:
            results: Main results dictionary with initial values
            strategy_results: Raw backtest results grouped by strategy

        Returns:
            Structured portfolio backtest results
        """
        # Build strategy summaries
        strategy_summaries = []

        for strategy_name, backtests in strategy_results.items():
            # Create BacktestResult objects
            detailed_results = []
            for backtest in backtests:
                detailed_results.append(
                    BacktestResult(
                        ticker=backtest["ticker"],
                        strategy=backtest["strategy"],
                        parameters=backtest.get("parameters", {}),
                        return_pct=backtest["return_pct"],
                        sharpe_ratio=backtest["sharpe_ratio"],
                        max_drawdown_pct=backtest["max_drawdown_pct"],
                        num_trades=backtest["num_trades"],
                        win_rate=backtest.get("win_rate"),
                    )
                )

            # Calculate summary metrics
            summary_metrics = self._calculate_portfolio_summary_metrics(detailed_results, results)

            # Get strategy parameters from first result
            strategy_params = detailed_results[0].parameters if detailed_results else {}

            # Create strategy summary
            summary = StrategyBacktestSummary(
                strategy_name=strategy_name,
                parameters=strategy_params,
                **summary_metrics,
                detailed_results=detailed_results,
            )

            strategy_summaries.append(summary)

        # Create broker config dict
        broker_config = self._get_full_broker_config_dict()

        # Get date range from config or results
        date_start, date_end = self._get_date_range(results)

        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=results.get("initial_portfolio_value", 0),
            initial_capital=results.get("initial_capital", 0),
            date_range={
                "start": date_start,
                "end": date_end,
            },
            broker_config=broker_config,
            strategy_summaries=strategy_summaries,
        )

        return portfolio_results

    def _calculate_strategy_summary(self, strategy_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for strategy results.

        Args:
            strategy_results: List of strategy results

        Returns:
            Summary statistics dictionary
        """
        if not strategy_results:
            return {
                "total_trades": 0,
                "winning_stocks": 0,
                "losing_stocks": 0,
                "average_return": 0,
                "average_sharpe": 0,
            }

        return {
            "total_trades": sum(r.get("num_trades", 0) for r in strategy_results),
            "winning_stocks": sum(1 for r in strategy_results if r.get("return_pct", 0) > 0),
            "losing_stocks": sum(1 for r in strategy_results if r.get("return_pct", 0) < 0),
            "average_return": sum(r.get("return_pct", 0) for r in strategy_results) / len(strategy_results),
            "average_sharpe": sum(r.get("sharpe_ratio", 0) for r in strategy_results) / len(strategy_results),
        }

    def _calculate_portfolio_summary_metrics(
        self, detailed_results: List[BacktestResult], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate portfolio summary metrics.

        Args:
            detailed_results: List of detailed backtest results
            results: Overall results dictionary

        Returns:
            Portfolio summary metrics
        """
        if not detailed_results:
            return {
                "initial_portfolio_value": results["initial_portfolio_value"],
                "final_portfolio_value": results["initial_portfolio_value"],
                "total_return_pct": 0,
                "total_trades": 0,
                "winning_stocks": 0,
                "losing_stocks": 0,
                "average_return_pct": 0,
                "average_sharpe_ratio": 0,
            }

        total_return = sum(r.return_pct for r in detailed_results)
        avg_return = total_return / len(detailed_results)
        avg_sharpe = sum(r.sharpe_ratio for r in detailed_results) / len(detailed_results)
        total_trades = sum(r.num_trades for r in detailed_results)
        winning_stocks = sum(1 for r in detailed_results if r.return_pct > 0)
        losing_stocks = sum(1 for r in detailed_results if r.return_pct < 0)

        # Calculate approximate final portfolio value
        final_value = results["initial_portfolio_value"] * (1 + avg_return / 100)

        return {
            "initial_portfolio_value": results["initial_portfolio_value"],
            "final_portfolio_value": final_value,
            "total_return_pct": avg_return,
            "total_trades": total_trades,
            "winning_stocks": winning_stocks,
            "losing_stocks": losing_stocks,
            "average_return_pct": avg_return,
            "average_sharpe_ratio": avg_sharpe,
        }

    def _get_broker_config_dict(self) -> Dict[str, Any]:
        """Get broker configuration as dictionary.

        Returns:
            Dictionary with broker configuration
        """
        if self.config.backtest.broker_config:
            return {
                "name": self.config.backtest.broker_config.name,
                "commission_type": self.config.backtest.broker_config.commission_type,
                "commission_value": self.config.backtest.broker_config.commission_value,
                "min_commission": self.config.backtest.broker_config.min_commission,
                "regulatory_fees": self.config.backtest.broker_config.regulatory_fees,
            }
        else:
            return {
                "name": "legacy",
                "commission_type": "percentage",
                "commission_value": self.config.backtest.commission,
                "min_commission": None,
                "regulatory_fees": 0,
            }

    def _get_full_broker_config_dict(self) -> Dict[str, Any]:
        """Get full broker configuration as dictionary.

        Returns:
            Dictionary with full broker configuration
        """
        if self.config.backtest.broker_config:
            return {
                "name": self.config.backtest.broker_config.name,
                "commission_type": self.config.backtest.broker_config.commission_type,
                "commission_value": self.config.backtest.broker_config.commission_value,
                "min_commission": self.config.backtest.broker_config.min_commission,
                "regulatory_fees": self.config.backtest.broker_config.regulatory_fees,
                "exchange_fees": getattr(self.config.backtest.broker_config, "exchange_fees", 0),
            }
        else:
            return {
                "name": "legacy",
                "commission_type": "percentage",
                "commission_value": self.config.backtest.commission,
                "min_commission": None,
                "regulatory_fees": 0,
                "exchange_fees": 0,
            }

    def _get_date_range(self, results: Dict[str, Any]) -> tuple[str, str]:
        """Get date range from configuration or results.

        Args:
            results: Results dictionary

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        date_start: str = "N/A"
        date_end: str = "N/A"

        # First try backtest dates, then data dates
        if self.config.backtest.start_date:
            date_start_val = self._date_to_string(self.config.backtest.start_date)
            if date_start_val is not None:
                date_start = date_start_val
        elif self.config.data.start_date:
            date_start_val = self._date_to_string(self.config.data.start_date)
            if date_start_val is not None:
                date_start = date_start_val

        if self.config.backtest.end_date:
            date_end_val = self._date_to_string(self.config.backtest.end_date)
            if date_end_val is not None:
                date_end = date_end_val
        elif self.config.data.end_date:
            date_end_val = self._date_to_string(self.config.data.end_date)
            if date_end_val is not None:
                date_end = date_end_val

        # If dates not in config, try to get from backtest results
        if (
            (date_start == "N/A" or date_end == "N/A")
            and results.get("backtesting")
            and len(results["backtesting"]) > 0
        ):
            # Look through all results to find one with dates
            for backtest_result in results["backtesting"]:
                if date_start == "N/A" and "start_date" in backtest_result:
                    date_start = backtest_result["start_date"]
                if date_end == "N/A" and "end_date" in backtest_result:
                    date_end = backtest_result["end_date"]
                # Stop if we found both dates
                if date_start != "N/A" and date_end != "N/A":
                    break

        return date_start, date_end

    def _convert_dates(self, obj: Any) -> Any:
        """Recursively convert date objects to strings.

        Args:
            obj: Object to convert

        Returns:
            Converted object
        """
        if isinstance(obj, dict):
            return {k: self._convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_dates(item) for item in obj]
        elif hasattr(obj, "strftime"):  # date/datetime objects
            return obj.strftime("%Y-%m-%d")
        return obj

    def _date_to_string(self, date_value: Any) -> Optional[str]:
        """Convert date or string to string format.

        Args:
            date_value: Date value to convert

        Returns:
            String representation or None
        """
        if date_value is None:
            return None
        if isinstance(date_value, str):
            return date_value
        return str(date_value.strftime("%Y-%m-%d"))

    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """Normalize strategy name to snake_case format.

        Args:
            strategy_name: Strategy name in any format

        Returns:
            Normalized strategy name in snake_case
        """
        # Import here to avoid circular imports
        from ..backtesting import strategy_registry

        return strategy_registry.normalize_strategy_name(strategy_name)

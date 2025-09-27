"""Backtest results display service for Single Responsibility Principle compliance."""

from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stockula.config.models import StockulaConfig
from stockula.utils.console_factory import get_console


class BacktestResultsDisplay:
    """Handles display and formatting of backtesting results only."""

    def __init__(self, console: Console | None = None):
        """Initialize the display handler.

        Args:
            console: Rich console for output (optional)
        """
        self.console = get_console(console)

    def display_backtesting_results(self, results: dict[str, Any], config: StockulaConfig | None, container):
        """Display backtesting results.

        Args:
            results: Results dictionary
            config: Optional configuration object
            container: Optional DI container
        """
        # Display general portfolio information
        self.console.print("\n[bold green]=== Backtesting Results ===[/bold green]")

        # Create portfolio information panel
        portfolio_info = []

        # Extract portfolio info from results metadata
        if "portfolio" in results:
            portfolio_data = results["portfolio"]
            if "initial_capital" in portfolio_data:
                portfolio_info.append(f"[cyan]Initial Capital:[/cyan] ${portfolio_data['initial_capital']:,.2f}")
            if "start" in portfolio_data and portfolio_data["start"]:
                portfolio_info.append(f"[cyan]Start Date:[/cyan] {portfolio_data['start']}")
            if "end" in portfolio_data and portfolio_data["end"]:
                portfolio_info.append(f"[cyan]End Date:[/cyan] {portfolio_data['end']}")

        # If portfolio info not in metadata, try to extract from backtest results
        if not portfolio_info and results.get("backtesting"):
            # Get portfolio information from the first backtest result
            first_backtest = results["backtesting"][0] if results["backtesting"] else {}

            if "initial_cash" in first_backtest:
                portfolio_info.append(f"[cyan]Initial Capital:[/cyan] ${first_backtest['initial_cash']:,.2f}")
            if "start_date" in first_backtest:
                portfolio_info.append(f"[cyan]Start Date:[/cyan] {first_backtest['start_date']}")
            if "end_date" in first_backtest:
                portfolio_info.append(f"[cyan]End Date:[/cyan] {first_backtest['end_date']}")

        # Display portfolio information if available
        if portfolio_info:
            self.console.print("[bold blue]Portfolio Information:[/bold blue]")
            for info in portfolio_info:
                self.console.print(f"  {info}")
            self.console.print()  # Add blank line

        # Display portfolio composition table (only if config and container are provided)
        if config and container:
            self.display_portfolio_composition(config, container)

        # Show ticker-level backtest results
        self.display_backtest_ticker_results(results["backtesting"])

        # Show strategy average returns summary
        self.display_strategy_average_returns(results["backtesting"])

    def display_portfolio_composition(self, config: StockulaConfig, container):
        """Display portfolio composition table.

        Args:
            config: Configuration object
            container: DI container
        """
        from stockula.domain.models import Category

        table = Table(title="Portfolio Composition")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Category", style="yellow")
        table.add_column("Quantity", style="white", justify="right")
        table.add_column("Allocation %", style="green", justify="right")
        table.add_column("Value", style="blue", justify="right")
        table.add_column("Status", style="magenta")

        # Get portfolio composition information
        portfolio = container.domain_factory().create_portfolio(config)
        all_assets = portfolio.get_all_assets()

        # Get hold-only categories from config
        hold_only_category_names = set(config.backtest.hold_only_categories)
        hold_only_categories = set()
        for category_name in hold_only_category_names:
            try:
                hold_only_categories.add(Category[category_name])
            except KeyError:
                pass  # Skip unknown categories

        # Get current prices for calculation
        fetcher = container.data_fetcher()
        symbols = [asset.symbol for asset in all_assets]
        try:
            current_prices = fetcher.get_current_prices(symbols, show_progress=False)
            total_portfolio_value = sum(asset.quantity * current_prices.get(asset.symbol, 0) for asset in all_assets)

            for asset in all_assets:
                current_price = current_prices.get(asset.symbol, 0)
                asset_value = asset.quantity * current_price
                allocation_pct = (asset_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

                # Determine status
                status = "Hold Only" if asset.category in hold_only_categories else "Tradeable"
                status_color = "yellow" if status == "Hold Only" else "green"

                table.add_row(
                    asset.symbol,
                    asset.category.name if asset.category and hasattr(asset.category, "name") else str(asset.category),
                    f"{asset.quantity:.2f}",
                    f"{allocation_pct:.1f}%",
                    f"${asset_value:,.2f}",
                    f"[{status_color}]{status}[/{status_color}]",
                )
        except Exception:
            # Fallback if we can't get prices
            for asset in all_assets:
                status = "Hold Only" if asset.category in hold_only_categories else "Tradeable"
                status_color = "yellow" if status == "Hold Only" else "green"

                table.add_row(
                    asset.symbol,
                    asset.category.name if asset.category and hasattr(asset.category, "name") else str(asset.category),
                    f"{asset.quantity:.2f}",
                    "N/A",
                    "N/A",
                    f"[{status_color}]{status}[/{status_color}]",
                )

        self.console.print(table)
        self.console.print()  # Add blank line

    def display_backtest_ticker_results(self, backtest_results: list[dict[str, Any]]):
        """Display ticker-level backtest results.

        Args:
            backtest_results: List of backtest results
        """
        # Check if we have multiple strategies
        strategies = {b["strategy"] for b in backtest_results}

        # Show ticker-level backtest results in a table
        self.console.print("\n[bold green]Ticker-Level Backtest Results[/bold green]")

        # Check if we have train/test results
        has_train_test = any("train_results" in backtest for backtest in backtest_results)

        if has_train_test:
            self.display_train_test_results(backtest_results)
        else:
            self.display_standard_backtest_results(backtest_results)

        # Show summary message about strategies and stocks
        unique_tickers = {b["ticker"] for b in backtest_results}
        self.console.print(
            f"Running [bold]{len(strategies)}[/bold] strategies across [bold]{len(unique_tickers)}[/bold] stocks..."
        )
        if len(strategies) > 1:
            self.console.print("Detailed results will be shown per strategy below.")

    def display_train_test_results(self, backtest_results: list[dict[str, Any]]):
        """Display train/test split results.

        Args:
            backtest_results: List of backtest results
        """
        import pandas as pd

        table = Table(title="Ticker-Level Backtest Results (Train/Test Split)")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Strategy", style="yellow", no_wrap=True)
        table.add_column("Train Return", style="green", justify="right")
        table.add_column("Test Return", style="green", justify="right")
        table.add_column("Train Sharpe", style="blue", justify="right")
        table.add_column("Test Sharpe", style="blue", justify="right")
        table.add_column("Test Trades", style="white", justify="right")
        table.add_column("Test Win Rate", style="magenta", justify="right")

        for backtest in backtest_results:
            if "train_results" in backtest:
                train_return_str = f"{backtest['train_results']['return_pct']:+.2f}%"
                test_return_str = f"{backtest['test_results']['return_pct']:+.2f}%"
                train_sharpe_str = f"{backtest['train_results']['sharpe_ratio']:.2f}"
                test_sharpe_str = f"{backtest['test_results']['sharpe_ratio']:.2f}"
                test_trades_str = str(backtest["test_results"]["num_trades"])

                if backtest["test_results"]["win_rate"] is None or pd.isna(backtest["test_results"]["win_rate"]):
                    test_win_rate_str = "N/A"
                else:
                    test_win_rate_str = f"{backtest['test_results']['win_rate']:.1f}%"

                table.add_row(
                    backtest["ticker"],
                    backtest["strategy"].upper(),
                    train_return_str,
                    test_return_str,
                    train_sharpe_str,
                    test_sharpe_str,
                    test_trades_str,
                    test_win_rate_str,
                )
            else:
                # Fallback for strategies without train/test split
                return_str = f"{backtest['return_pct']:+.2f}%"
                sharpe_str = f"{backtest['sharpe_ratio']:.2f}"
                trades_str = str(backtest["num_trades"])

                if backtest["win_rate"] is None:
                    win_rate_str = "N/A"
                else:
                    win_rate_str = f"{backtest['win_rate']:.1f}%"

                table.add_row(
                    backtest["ticker"],
                    backtest["strategy"].upper(),
                    return_str,
                    return_str,  # Same for both train/test
                    sharpe_str,
                    sharpe_str,  # Same for both train/test
                    trades_str,
                    win_rate_str,
                )

        self.console.print(table)
        self.console.print()  # Add blank line

        # Show train/test periods
        first_with_split = next((b for b in backtest_results if "train_period" in b), None)
        if first_with_split:
            self.console.print("[bold cyan]Data Periods:[/bold cyan]")
            train_start = first_with_split["train_period"]["start"]
            train_end = first_with_split["train_period"]["end"]
            train_days = first_with_split["train_period"]["days"]
            test_start = first_with_split["test_period"]["start"]
            test_end = first_with_split["test_period"]["end"]
            test_days = first_with_split["test_period"]["days"]

            self.console.print(f"  Training: {train_start} to {train_end} ({train_days} days)")
            self.console.print(f"  Testing:  {test_start} to {test_end} ({test_days} days)")
            self.console.print()

    def display_standard_backtest_results(self, backtest_results: list[dict[str, Any]]):
        """Display standard backtest results without train/test split.

        Args:
            backtest_results: List of backtest results
        """
        table = Table(title="Ticker-Level Backtest Results")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Strategy", style="yellow", no_wrap=True)
        table.add_column("Return", style="green", justify="right")
        table.add_column("Sharpe Ratio", style="blue", justify="right")
        table.add_column("Max Drawdown", style="red", justify="right")
        table.add_column("Trades", style="white", justify="right")
        table.add_column("Win Rate", style="magenta", justify="right")

        for backtest in backtest_results:
            return_str = f"{backtest['return_pct']:+.2f}%"
            sharpe_str = f"{backtest['sharpe_ratio']:.2f}"
            drawdown_str = f"{backtest['max_drawdown_pct']:.2f}%"
            trades_str = str(backtest["num_trades"])

            if backtest["win_rate"] is None:
                win_rate_str = "N/A"
            else:
                win_rate_str = f"{backtest['win_rate']:.1f}%"

            table.add_row(
                backtest["ticker"],
                backtest["strategy"].upper(),
                return_str,
                sharpe_str,
                drawdown_str,
                trades_str,
                win_rate_str,
            )

        self.console.print(table)
        self.console.print()  # Add blank line

    def display_strategy_average_returns(self, backtest_results: list[dict[str, Any]]):
        """Display average returns for each strategy across all tickers.

        Args:
            backtest_results: List of backtest results
        """
        strategy_returns = defaultdict(list)

        for backtest in backtest_results:
            strategy = backtest["strategy"]

            # Get the return percentage based on result type
            if "test_results" in backtest:
                # Use test period returns for train/test split results
                return_pct = backtest["test_results"]["return_pct"]
            else:
                # Use standard return percentage
                return_pct = backtest["return_pct"]

            strategy_returns[strategy].append(return_pct)

        # Calculate averages and sort by highest to lowest
        strategy_averages = []
        for strategy, returns in strategy_returns.items():
            avg_return = sum(returns) / len(returns)
            strategy_averages.append({"strategy": strategy, "avg_return": avg_return, "num_tickers": len(returns)})

        # Sort by average return (highest to lowest)
        strategy_averages.sort(key=lambda x: x["avg_return"], reverse=True)

        # Display the summary table
        self.console.print("\n[bold cyan]=== Strategy Average Returns Summary ===[/bold cyan]")

        table = Table(title="Average Returns by Strategy (Sorted Highest to Lowest)")
        table.add_column("Strategy", style="yellow", no_wrap=True)
        table.add_column("Average Return %", style="green", justify="right")
        table.add_column("# Tickers Tested", style="white", justify="right")

        for strategy_data in strategy_averages:
            avg_return = strategy_data["avg_return"]
            # Color code the return
            if avg_return >= 0:
                return_style = "green"
            else:
                return_style = "red"

            table.add_row(
                strategy_data["strategy"].upper(),
                f"[{return_style}]{avg_return:+.2f}%[/{return_style}]",
                str(strategy_data["num_tickers"]),
            )

        self.console.print(table)
        self.console.print()  # Add blank line

        # Show the best performing strategy
        if strategy_averages:
            best_strategy = strategy_averages[0]
            self.console.print(
                f"[bold green]Best Performing Strategy:[/bold green] "
                f"[yellow]{best_strategy['strategy'].upper()}[/yellow] "
                f"with average return of [green]{best_strategy['avg_return']:+.2f}%[/green]"
            )
            self.console.print()

    def show_strategy_summaries(self, manager, config: StockulaConfig, results: dict[str, Any]):
        """Show strategy-specific summaries.

        Args:
            manager: StockulaManager instance
            config: Configuration object
            results: Results dictionary
        """
        # Group results by strategy
        strategy_results = defaultdict(list)

        for backtest in results["backtesting"]:
            strategy_results[backtest["strategy"]].append(backtest)

        # Only proceed if we have results
        if not strategy_results:
            self.console.print("\n[red]No backtesting results to display.[/red]")
            return

        # Create structured backtest results
        portfolio_backtest_results = manager.create_portfolio_backtest_results(results, strategy_results)

        # Sort strategy summaries by return during period (highest to lowest)
        sorted_summaries = sorted(
            portfolio_backtest_results.strategy_summaries,
            key=lambda s: (s.final_portfolio_value - s.initial_portfolio_value),
            reverse=True,  # Highest returns first
        )

        # Show summary for each strategy using structured data
        for strategy_summary in sorted_summaries:
            # Get broker config info
            broker_info = self._get_broker_info(config)

            # Create rich panel for strategy summary
            period_return = strategy_summary.final_portfolio_value - strategy_summary.initial_portfolio_value
            period_return_color = "green" if period_return > 0 else "red" if period_return < 0 else "white"

            # Format dates
            start_date, end_date = self._get_strategy_dates(config, portfolio_backtest_results)

            summary_content = f"""Start: {start_date}
End:   {end_date}

Parameters: {strategy_summary.parameters if strategy_summary.parameters else "Default"}
{broker_info}

Portfolio Value at {start_date}: ${strategy_summary.initial_portfolio_value:,.2f}
Portfolio Value at {end_date}: ${strategy_summary.final_portfolio_value:,.2f}

Strategy Performance:
  Average Return: [{period_return_color}]{strategy_summary.average_return_pct:+.2f}%[/{period_return_color}]
  Winning Stocks: {strategy_summary.winning_stocks}
  Losing Stocks: {strategy_summary.losing_stocks}
  Total Trades: {strategy_summary.total_trades}

Return During Period: [{period_return_color}]${period_return:,.2f} \\
({strategy_summary.total_return_pct:+.2f}%)[/{period_return_color}]

Detailed report saved to: {
                manager.save_detailed_report(
                    strategy_summary.strategy_name,
                    [r.model_dump() for r in strategy_summary.detailed_results],
                    results,
                )
            }"""

            self.console.print(
                Panel(
                    summary_content,
                    title=f" STRATEGY: {strategy_summary.strategy_name.upper()} ",
                    border_style="white",
                    padding=(0, 1),
                )
            )

    def _get_broker_info(self, config: StockulaConfig) -> str:
        """Get broker information string.

        Args:
            config: Configuration object

        Returns:
            Formatted broker information string
        """
        if config.backtest.broker_config:
            broker_config = config.backtest.broker_config
            if broker_config.name in [
                "td_ameritrade",
                "etrade",
                "robinhood",
                "fidelity",
                "schwab",
            ]:
                broker_info = f"Broker: {broker_config.name} (zero-commission)"
            elif broker_config.commission_type == "percentage":
                commission_val = broker_config.commission_value
                if isinstance(commission_val, dict):
                    # If it's a dict, use the first value or default
                    commission_val = next(iter(commission_val.values())) if commission_val else 0.0
                broker_info = f"Broker: {broker_config.name} ({commission_val * 100:.1f}% commission"
                if broker_config.min_commission:
                    broker_info += f", ${broker_config.min_commission:.2f} min"
                broker_info += ")"
            elif broker_config.commission_type == "per_share":
                per_share_comm = broker_config.per_share_commission or broker_config.commission_value
                broker_info = f"Broker: {broker_config.name} (${per_share_comm:.3f}/share"
                if broker_config.min_commission:
                    broker_info += f", ${broker_config.min_commission:.2f} min"
                broker_info += ")"
            elif broker_config.commission_type == "tiered":
                broker_info = f"Broker: {broker_config.name} (tiered pricing"
                if broker_config.min_commission:
                    broker_info += f", ${broker_config.min_commission:.2f} min"
                broker_info += ")"
            elif broker_config.commission_type == "fixed":
                broker_info = f"Broker: {broker_config.name} (${broker_config.commission_value:.2f}/trade)"
            else:
                broker_info = f"Broker: {broker_config.name} ({broker_config.commission_type})"
        else:
            broker_info = f"Commission: {config.backtest.commission * 100:.1f}%"

        return broker_info

    def _get_strategy_dates(self, config: StockulaConfig, portfolio_backtest_results) -> tuple[str, str]:
        """Get strategy date range.

        Args:
            config: Configuration object
            portfolio_backtest_results: Portfolio backtest results

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        from datetime import date

        start_date: str = "N/A"
        end_date: str = "N/A"

        # First try backtest dates, then data dates, then results
        if config.backtest.start_date:
            start_date = (
                config.backtest.start_date.strftime("%Y-%m-%d")
                if isinstance(config.backtest.start_date, date)
                else str(config.backtest.start_date)
            )
        elif config.data.start_date:
            start_date = (
                config.data.start_date.strftime("%Y-%m-%d")
                if isinstance(config.data.start_date, date)
                else str(config.data.start_date)
            )
        elif portfolio_backtest_results.date_range and portfolio_backtest_results.date_range.get("start"):
            start_date = portfolio_backtest_results.date_range["start"]

        if config.backtest.end_date:
            end_date = (
                config.backtest.end_date.strftime("%Y-%m-%d")
                if isinstance(config.backtest.end_date, date)
                else str(config.backtest.end_date)
            )
        elif config.data.end_date:
            end_date = (
                config.data.end_date.strftime("%Y-%m-%d")
                if isinstance(config.data.end_date, date)
                else str(config.data.end_date)
            )
        elif portfolio_backtest_results.date_range and portfolio_backtest_results.date_range.get("end"):
            end_date = portfolio_backtest_results.date_range["end"]

        return start_date, end_date

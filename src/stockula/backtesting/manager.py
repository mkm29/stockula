"""BacktestingManager - Coordinate different backtesting strategies and provide unified interface."""

from typing import TYPE_CHECKING, Any, Optional

from ..interfaces import IDataFetcher, ILoggingManager

if TYPE_CHECKING:
    from ..config.models import StockulaConfig
    from ..data.strategy_repository import StrategyRepository
    from .runner import BacktestRunner


class BacktestingManager:
    """Manages different backtesting strategies and provides unified interface."""

    def __init__(
        self,
        data_fetcher: "IDataFetcher",
        logging_manager: ILoggingManager,
        strategy_repository: "StrategyRepository",
    ):
        """Initialize BacktestingManager.

        Args:
            data_fetcher: Data fetcher interface for retrieving market data
            logging_manager: Logging manager for structured logging
            strategy_repository: Repository of strategies and presets
        """
        self.data_fetcher = data_fetcher
        self.logger = logging_manager

        # Initialize BacktestRunner (will be set with proper configuration)
        self._runner: BacktestRunner | None = None

        # Use injected strategy repository/registry
        self.strategy_registry = strategy_repository

    def set_runner(self, runner: "BacktestRunner") -> None:
        """Set the BacktestRunner instance.

        Args:
            runner: Configured BacktestRunner instance
        """
        self._runner = runner

    def run_single_strategy(
        self,
        ticker: str,
        strategy_name: str,
        config: Optional["StockulaConfig"] = None,
        strategy_params: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Run backtest for a single strategy on a single ticker.

        Args:
            ticker: Stock ticker symbol
            strategy_name: Name of the strategy to test
            config: Configuration object (optional)
            strategy_params: Custom strategy parameters (optional)
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Dictionary containing backtest results
        """
        if not self._runner:
            raise ValueError("BacktestRunner not initialized. Call set_runner() first.")

        try:
            self.logger.info(f"Running {strategy_name} strategy backtest for {ticker}")

            # Use provided parameters or fall back to presets
            params = strategy_params or self.strategy_registry.get_strategy_preset(strategy_name)

            # Get the strategy class using the registry
            strategy_class = self.strategy_registry.get_strategy_class(strategy_name)
            if not strategy_class:
                available_strategies = self.strategy_registry.get_available_strategy_names()
                raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available_strategies}")

            # Run backtest using the runner
            result = self._runner.run_from_symbol(
                symbol=ticker,
                strategy=strategy_class,
                start_date=start_date,
                end_date=end_date,
                **params,
            )

            self.logger.info(f"Completed {strategy_name} backtest for {ticker}")
            return result

        except Exception as e:
            self.logger.error(f"Error running {strategy_name} backtest for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker, "strategy": strategy_name}

    def run_multiple_strategies(
        self,
        ticker: str,
        strategy_group: str = "basic",
        config: Optional["StockulaConfig"] = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run backtest for multiple strategies on a single ticker.

        Args:
            ticker: Stock ticker symbol
            strategy_group: Group of strategies to test ('basic', 'momentum', 'trend', 'advanced', 'comprehensive')
            config: Configuration object (optional)
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Dictionary with strategy names as keys and results as values
        """
        if not self.strategy_registry.is_valid_strategy_group(strategy_group):
            available_groups = list(self.strategy_registry.get_strategy_groups().keys())
            raise ValueError(f"Unknown strategy group: {strategy_group}. Available: {available_groups}")

        self.logger.info(f"Running {strategy_group} strategy group backtest for {ticker}")

        results = {}
        strategies = self.strategy_registry.get_strategies_in_group(strategy_group)

        for strategy_name in strategies:
            result = self.run_single_strategy(
                ticker=ticker,
                strategy_name=strategy_name,
                config=config,
                start_date=start_date,
                end_date=end_date,
            )
            results[strategy_name] = result

        self.logger.info(f"Completed {strategy_group} strategy group backtest for {ticker}")
        return results

    def run_portfolio_backtest(
        self,
        tickers: list[str],
        strategy_name: str,
        config: Optional["StockulaConfig"] = None,
        strategy_params: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run backtest for a single strategy across multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            strategy_name: Name of the strategy to test
            config: Configuration object (optional)
            strategy_params: Custom strategy parameters (optional)
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Dictionary with tickers as keys and results as values
        """
        self.logger.info(f"Running portfolio backtest with {strategy_name} strategy for {len(tickers)} tickers")

        results = {}
        for ticker in tickers:
            result = self.run_single_strategy(
                ticker=ticker,
                strategy_name=strategy_name,
                config=config,
                strategy_params=strategy_params,
                start_date=start_date,
                end_date=end_date,
            )
            results[ticker] = result

        self.logger.info(f"Completed portfolio backtest with {strategy_name} strategy")
        return results

    def run_comprehensive_backtest(
        self,
        tickers: list[str],
        strategy_group: str = "comprehensive",
        config: Optional["StockulaConfig"] = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run comprehensive backtest across multiple tickers and strategies.

        Args:
            tickers: List of stock ticker symbols
            strategy_group: Group of strategies to test
            config: Configuration object (optional)
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Nested dictionary: {ticker: {strategy: results}}
        """
        self.logger.info(f"Running comprehensive backtest for {len(tickers)} tickers with {strategy_group} strategies")

        all_results = {}
        for ticker in tickers:
            ticker_results = self.run_multiple_strategies(
                ticker=ticker,
                strategy_group=strategy_group,
                config=config,
                start_date=start_date,
                end_date=end_date,
            )
            all_results[ticker] = ticker_results

        self.logger.info("Completed comprehensive backtest")
        return all_results

    def run_with_train_test_split(
        self,
        ticker: str,
        strategy_name: str,
        train_ratio: float = 0.7,
        config: Optional["StockulaConfig"] = None,
        strategy_params: dict[str, Any] | None = None,
        optimize_on_train: bool = True,
        param_ranges: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run backtest with train/test split for out-of-sample validation.

        Args:
            ticker: Stock ticker symbol
            strategy_name: Name of the strategy to test
            train_ratio: Ratio of data to use for training (0.0-1.0)
            config: Configuration object (optional)
            strategy_params: Custom strategy parameters (optional)
            optimize_on_train: Whether to optimize parameters on training data
            param_ranges: Parameter ranges for optimization (optional)

        Returns:
            Dictionary containing train/test results and performance degradation metrics
        """
        if not self._runner:
            raise ValueError("BacktestRunner not initialized. Call set_runner() first.")

        try:
            self.logger.info(f"Running train/test split backtest for {ticker} with {strategy_name}")

            # Use provided parameters or fall back to presets
            params = strategy_params or self.strategy_registry.get_strategy_preset(strategy_name)

            # Get the strategy class using the registry
            strategy_class = self.strategy_registry.get_strategy_class(strategy_name)
            if not strategy_class:
                available_strategies = self.strategy_registry.get_available_strategy_names()
                raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available_strategies}")

            # Run train/test split backtest using the runner
            result = self._runner.run_with_train_test_split(
                symbol=ticker,
                strategy=strategy_class,
                train_ratio=train_ratio,
                optimize_on_train=optimize_on_train,
                param_ranges=param_ranges,
                **params,
            )

            self.logger.info(f"Completed train/test split backtest for {ticker}")
            return result

        except Exception as e:
            self.logger.error(f"Error running train/test split backtest for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker, "strategy": strategy_name}

    def quick_backtest(
        self,
        ticker: str,
        strategy_name: str = "smacross",
        config: Optional["StockulaConfig"] = None,
    ) -> dict[str, Any]:
        """Run a quick backtest with default parameters for rapid testing.

        Args:
            ticker: Stock ticker symbol
            strategy_name: Name of the strategy to test (default: 'smacross')
            config: Configuration object (optional)

        Returns:
            Dictionary containing essential backtest metrics
        """
        self.logger.info(f"Running quick backtest for {ticker} with {strategy_name}")

        result = self.run_single_strategy(
            ticker=ticker,
            strategy_name=strategy_name,
            config=config,
        )

        # Extract essential metrics for quick overview
        if "error" not in result:
            quick_metrics = {
                "ticker": ticker,
                "strategy": strategy_name,
                "return_pct": result.get("Return [%]", 0),
                "sharpe_ratio": result.get("Sharpe Ratio", 0),
                "max_drawdown_pct": result.get("Max. Drawdown [%]", 0),
                "num_trades": result.get("# Trades", 0),
                "win_rate": result.get("Win Rate [%]", 0),
            }
            return quick_metrics

        return result

    def get_available_strategies(self) -> list[str]:
        """Get list of all available strategies.

        Returns:
            List of strategy names
        """
        return self.strategy_registry.get_available_strategy_names()

    def get_strategy_groups(self) -> dict[str, list[str]]:
        """Get all available strategy groups.

        Returns:
            Dictionary of strategy groups and their constituent strategies
        """
        return self.strategy_registry.get_strategy_groups()

    def get_strategy_presets(self) -> dict[str, dict[str, Any]]:
        """Get default parameter presets for all strategies.

        Returns:
            Dictionary of strategy names and their default parameters
        """
        return self.strategy_registry.get_strategy_presets()

    def customize_strategy_parameters(
        self,
        strategy_name: str,
        new_params: dict[str, Any],
    ) -> None:
        """Customize default parameters for a strategy.

        Args:
            strategy_name: Name of the strategy
            new_params: New parameter values to set
        """
        if not self.strategy_registry.is_valid_strategy(strategy_name):
            available_strategies = self.strategy_registry.get_available_strategy_names()
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available_strategies}")

        # Note: This modifies the class-level registry, affecting all instances
        self.strategy_registry.update_strategy_preset(strategy_name, new_params)
        normalized_name = self.strategy_registry.normalize_strategy_name(strategy_name)
        self.logger.info(f"Updated {normalized_name} parameters: {new_params}")

    def create_custom_strategy_group(
        self,
        group_name: str,
        strategies: list[str],
    ) -> None:
        """Create a custom strategy group.

        Args:
            group_name: Name for the new strategy group
            strategies: List of strategy names to include

        Raises:
            ValueError: If any strategy name is not available
        """
        # Validate and normalize strategy names
        valid_strategies, invalid_strategies = self.strategy_registry.validate_strategies(strategies)

        if invalid_strategies:
            available_strategies = self.strategy_registry.get_available_strategy_names()
            raise ValueError(f"Invalid strategies: {invalid_strategies}. Available: {available_strategies}")

        # Note: This modifies the class-level registry, affecting all instances
        self.strategy_registry.add_strategy_group(group_name, valid_strategies)
        self.logger.info(f"Created custom strategy group '{group_name}' with strategies: {valid_strategies}")

"""Backtest-optimized asset allocation strategy."""

import pandas as pd

from ..backtesting.runner import BacktestRunner
from ..backtesting.strategies import (
    BaseStrategy,
    DoubleEMACrossStrategy,
    FRAMAStrategy,
    KAMAStrategy,
    KaufmanEfficiencyStrategy,
    MACDStrategy,
    RSIStrategy,
    SMACrossStrategy,
    TRIMACrossStrategy,
    TripleEMACrossStrategy,
    VAMAStrategy,
    VIDYAStrategy,
)
from ..config import StockulaConfig, TickerConfig
from ..data.fetcher import DataFetcher
from ..interfaces import ILoggingManager
from .base_allocator import BaseAllocator


class BacktestOptimizedAllocator(BaseAllocator):
    """Allocator that uses backtest results to optimize asset allocation.

    This allocator:
    1. Starts with equal allocations
    2. Runs backtests on training data to find the best strategy per asset
    3. Runs backtests on test data with the best strategies
    4. Uses the performance metrics to determine final allocations
    """

    # Available strategies to test
    AVAILABLE_STRATEGIES = [
        SMACrossStrategy,
        RSIStrategy,
        MACDStrategy,
        DoubleEMACrossStrategy,
        VIDYAStrategy,
        KAMAStrategy,
        FRAMAStrategy,
        TripleEMACrossStrategy,
        TRIMACrossStrategy,
        VAMAStrategy,
        KaufmanEfficiencyStrategy,
    ]

    # Default performance metric to use for ranking
    DEFAULT_RANKING_METRIC = "Return [%]"  # Changed from "Sharpe Ratio" to "Return [%]"

    # Default allocation constraints
    DEFAULT_MIN_ALLOCATION_PCT = 2.0
    DEFAULT_MAX_ALLOCATION_PCT = 25.0

    def __init__(
        self,
        fetcher: DataFetcher,
        logging_manager: ILoggingManager,
        backtest_runner: BacktestRunner | None = None,
    ):
        """Initialize allocator with data fetcher, logging manager, and backtest runner.

        Args:
            fetcher: Data fetcher instance for price lookups
            logging_manager: Logging manager
            backtest_runner: Optional backtest runner (will create if not provided)
        """
        super().__init__(fetcher, logging_manager)
        self.backtest_runner = backtest_runner or BacktestRunner(data_fetcher=fetcher)
        self._strategy_cache: dict[str, type[BaseStrategy]] = {}
        self._performance_cache: dict[str, float] = {}

    def calculate_backtest_optimized_quantities(
        self,
        config: StockulaConfig,
        tickers_to_add: list[TickerConfig],
        train_start_date: str | None = None,
        train_end_date: str | None = None,
        test_start_date: str | None = None,
        test_end_date: str | None = None,
        initial_allocation_pct: float | None = None,
    ) -> dict[str, float]:
        """Calculate quantities using backtest optimization.

        Args:
            config: Stockula configuration
            tickers_to_add: List of ticker configurations
            train_start_date: Start date for training period (YYYY-MM-DD), overrides config
            train_end_date: End date for training period (YYYY-MM-DD), overrides config
            test_start_date: Start date for testing period (YYYY-MM-DD), overrides config
            test_end_date: End date for testing period (YYYY-MM-DD), overrides config
            initial_allocation_pct: Initial allocation percentage per asset, overrides config

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        self._validate_fetcher()

        # Get configuration from config file or use defaults
        opt_config = config.backtest_optimization
        if opt_config:
            # Use config values if not overridden by parameters
            train_start = train_start_date or (
                opt_config.train_start_date.strftime("%Y-%m-%d") if opt_config.train_start_date else None
            )
            train_end = train_end_date or (
                opt_config.train_end_date.strftime("%Y-%m-%d") if opt_config.train_end_date else None
            )
            test_start = test_start_date or (
                opt_config.test_start_date.strftime("%Y-%m-%d") if opt_config.test_start_date else None
            )
            test_end = test_end_date or (
                opt_config.test_end_date.strftime("%Y-%m-%d") if opt_config.test_end_date else None
            )
            # Note: initial_allocation_pct is stored in config but not used in current implementation
            # It could be used for setting initial cash per asset in backtesting
            self.ranking_metric = opt_config.ranking_metric
            self.min_allocation_pct = opt_config.min_allocation_pct
            self.max_allocation_pct = opt_config.max_allocation_pct
        else:
            # Use parameters or defaults
            train_start = train_start_date
            train_end = train_end_date
            test_start = test_start_date
            test_end = test_end_date
            # Note: initial_allocation_pct is not used in current implementation
            self.ranking_metric = self.DEFAULT_RANKING_METRIC
            self.min_allocation_pct = self.DEFAULT_MIN_ALLOCATION_PCT
            self.max_allocation_pct = self.DEFAULT_MAX_ALLOCATION_PCT

        # Validate dates are provided
        if not all([train_start, train_end, test_start, test_end]):
            raise ValueError("All date parameters must be provided either in config or as arguments")

        symbols = [ticker.symbol for ticker in tickers_to_add]

        # Step 1: Find best strategy for each asset using training data
        self.logger.info("Step 1: Finding optimal strategy for each asset on training data...")
        self.logger.info(f"Using ranking metric: {self.ranking_metric}")
        best_strategies = self._find_best_strategies(symbols, train_start, train_end)

        # Step 2: Run backtests on test data with best strategies
        self.logger.info("Step 2: Evaluating performance on test data with optimal strategies...")
        test_performances = self._evaluate_test_performance(symbols, best_strategies, test_start, test_end)

        # Step 3: Calculate allocations based on performance
        self.logger.info("Step 3: Calculating allocations based on test performance...")
        allocation_percentages = self._calculate_performance_based_allocations(
            symbols, test_performances, config.portfolio.initial_capital
        )

        # Step 4: Convert allocations to quantities
        self.logger.info("Step 4: Converting allocations to quantities...")
        calculated_quantities = self._convert_allocations_to_quantities(
            config, symbols, allocation_percentages, test_end_date
        )

        # Log summary
        self._log_optimization_summary(
            symbols, best_strategies, test_performances, allocation_percentages, calculated_quantities
        )

        return calculated_quantities

    def _find_best_strategies(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, type[BaseStrategy]]:
        """Find the best strategy for each symbol using training data.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for training (YYYY-MM-DD)
            end_date: End date for training (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to their best strategies
        """
        best_strategies = {}

        for symbol in symbols:
            self.logger.debug(f"\nTesting strategies for {symbol}...")
            best_strategy = None
            best_metric = float("-inf")

            # Fetch data once for this symbol
            try:
                data = self.fetcher.get_stock_data(symbol, start_date, end_date)
                if data.empty:
                    self.logger.warning(f"No data available for {symbol} in training period")
                    best_strategies[symbol] = SMACrossStrategy  # Default strategy
                    continue

            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                best_strategies[symbol] = SMACrossStrategy  # Default strategy
                continue

            # Test each strategy
            for strategy_class in self.AVAILABLE_STRATEGIES:
                try:
                    # Run backtest
                    results = self.backtest_runner.run(data, strategy_class)

                    # Get the ranking metric
                    metric_value = results.get(self.ranking_metric, float("-inf"))

                    self.logger.debug(f"  {strategy_class.__name__}: {self.ranking_metric} = {metric_value:.4f}")

                    # Update best if this is better
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_strategy = strategy_class

                except Exception as e:
                    self.logger.debug(f"  {strategy_class.__name__}: Failed - {e}")
                    continue

            # Store the best strategy
            if best_strategy:
                best_strategies[symbol] = best_strategy
                self.logger.info(
                    f"{symbol}: Best strategy = {best_strategy.__name__} ({self.ranking_metric} = {best_metric:.4f})"
                )
            else:
                # Fallback to default strategy
                best_strategies[symbol] = SMACrossStrategy
                self.logger.warning(f"{symbol}: No successful strategy, using default SMACrossStrategy")

        return best_strategies

    def _evaluate_test_performance(
        self,
        symbols: list[str],
        best_strategies: dict[str, type[BaseStrategy]],
        start_date: str,
        end_date: str,
    ) -> dict[str, float]:
        """Evaluate performance on test data using the best strategies.

        Args:
            symbols: List of ticker symbols
            best_strategies: Dictionary mapping symbols to their best strategies
            start_date: Start date for testing (YYYY-MM-DD)
            end_date: End date for testing (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to their test performance metric values
        """
        test_performances = {}

        for symbol in symbols:
            strategy_class = best_strategies.get(symbol, SMACrossStrategy)

            try:
                # Fetch test data
                data = self.fetcher.get_stock_data(symbol, start_date, end_date)
                if data.empty:
                    self.logger.warning(f"No data available for {symbol} in test period")
                    test_performances[symbol] = 0.0
                    continue

                # Run backtest with the best strategy
                results = self.backtest_runner.run(data, strategy_class)

                # Get the performance metric
                metric_value = results.get(self.ranking_metric, 0.0)
                test_performances[symbol] = metric_value

                self.logger.info(
                    f"{symbol}: Test {self.ranking_metric} = {metric_value:.4f} using {strategy_class.__name__}"
                )

            except Exception as e:
                self.logger.error(f"Error evaluating {symbol} on test data: {e}")
                test_performances[symbol] = 0.0

        return test_performances

    def _calculate_performance_based_allocations(
        self, symbols: list[str], performances: dict[str, float], total_capital: float
    ) -> dict[str, float]:
        """Calculate allocation percentages based on performance metrics.

        Uses a weighted allocation approach where better-performing assets get
        larger allocations, subject to min/max constraints.

        Args:
            symbols: List of ticker symbols
            performances: Dictionary mapping symbols to performance metrics
            total_capital: Total portfolio capital

        Returns:
            Dictionary mapping symbols to allocation percentages
        """
        # Filter out assets with non-positive performance
        positive_performers = {symbol: perf for symbol, perf in performances.items() if perf > 0}

        if not positive_performers:
            # If no positive performers, use equal allocation
            self.logger.warning("No assets with positive performance, using equal allocation")
            equal_pct = 100.0 / len(symbols) if symbols else 0.0
            return dict.fromkeys(symbols, equal_pct)

        # Calculate raw allocations based on performance
        total_performance = sum(positive_performers.values())
        raw_allocations = {}

        for symbol in symbols:
            if symbol in positive_performers:
                # Allocate proportionally to performance
                raw_pct = (positive_performers[symbol] / total_performance) * 100.0
                raw_allocations[symbol] = raw_pct
            else:
                # Give minimum allocation to non-positive performers
                raw_allocations[symbol] = self.min_allocation_pct

        # Apply min/max constraints
        constrained_allocations = {}
        for symbol, raw_pct in raw_allocations.items():
            constrained_pct = max(self.min_allocation_pct, min(self.max_allocation_pct, raw_pct))
            constrained_allocations[symbol] = constrained_pct

        # Normalize to ensure total is 100%
        total_constrained = sum(constrained_allocations.values())
        normalized_allocations = {
            symbol: (pct / total_constrained) * 100.0 for symbol, pct in constrained_allocations.items()
        }

        return normalized_allocations

    def _convert_allocations_to_quantities(
        self,
        config: StockulaConfig,
        symbols: list[str],
        allocation_percentages: dict[str, float],
        price_date: str,
    ) -> dict[str, float]:
        """Convert allocation percentages to quantities.

        Args:
            config: Stockula configuration
            symbols: List of ticker symbols
            allocation_percentages: Dictionary mapping symbols to allocation percentages
            price_date: Date to use for price lookup (YYYY-MM-DD)

        Returns:
            Dictionary mapping symbols to quantities
        """
        # Get prices for the end of test period
        prices = {}
        for symbol in symbols:
            try:
                data = self.fetcher.get_stock_data(symbol, price_date, price_date)
                if not data.empty:
                    prices[symbol] = data["Close"].iloc[-1]
                else:
                    # Try a few days later
                    end_dt = pd.to_datetime(price_date)
                    extended_end = (end_dt + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                    data = self.fetcher.get_stock_data(symbol, price_date, extended_end)
                    if not data.empty:
                        prices[symbol] = data["Close"].iloc[0]
                    else:
                        # Fallback to current price
                        current_prices = self.fetcher.get_current_prices([symbol])
                        prices[symbol] = current_prices.get(symbol, 0.0)
            except Exception as e:
                self.logger.error(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0.0

        # Calculate quantities
        calculated_quantities = {}
        for symbol in symbols:
            if symbol not in prices or prices[symbol] <= 0:
                self.logger.warning(f"Invalid price for {symbol}, skipping")
                calculated_quantities[symbol] = 0.0
                continue

            allocation_pct = allocation_percentages.get(symbol, 0.0)
            allocation_amount = (allocation_pct / 100.0) * config.portfolio.initial_capital

            raw_quantity = allocation_amount / prices[symbol]

            # For backtest_optimized allocation, always use whole shares
            # This ensures the output can be used directly in portfolio construction
            calculated_quantities[symbol] = max(1, int(raw_quantity))

        return calculated_quantities

    def _log_optimization_summary(
        self,
        symbols: list[str],
        best_strategies: dict[str, type[BaseStrategy]],
        test_performances: dict[str, float],
        allocation_percentages: dict[str, float],
        calculated_quantities: dict[str, float],
    ):
        """Log a summary of the optimization results.

        Args:
            symbols: List of ticker symbols
            best_strategies: Dictionary mapping symbols to their best strategies
            test_performances: Dictionary mapping symbols to test performance metrics
            allocation_percentages: Dictionary mapping symbols to allocation percentages
            calculated_quantities: Dictionary mapping symbols to quantities
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BACKTEST OPTIMIZATION SUMMARY")
        self.logger.info("=" * 80)

        for symbol in symbols:
            strategy = best_strategies.get(symbol, SMACrossStrategy).__name__
            performance = test_performances.get(symbol, 0.0)
            allocation_pct = allocation_percentages.get(symbol, 0.0)
            quantity = calculated_quantities.get(symbol, 0.0)

            self.logger.info(
                f"{symbol:8} | Strategy: {strategy:25} | "
                f"{self.ranking_metric}: {performance:7.4f} | "
                f"Allocation: {allocation_pct:5.2f}% | "
                f"Quantity: {quantity:8.0f}"
            )

        self.logger.info("=" * 80)

        return calculated_quantities

    def calculate_quantities(
        self,
        config: StockulaConfig,
        tickers: list[TickerConfig],
        **kwargs,
    ) -> dict[str, float]:
        """Calculate quantities using backtest optimization.

        This is the main entry point that conforms to the BaseAllocator interface.

        Args:
            config: Stockula configuration
            tickers: List of ticker configurations
            **kwargs: Additional parameters passed to calculate_backtest_optimized_quantities

        Returns:
            Dictionary mapping ticker symbols to calculated quantities
        """
        return self.calculate_backtest_optimized_quantities(config=config, tickers_to_add=tickers, **kwargs)

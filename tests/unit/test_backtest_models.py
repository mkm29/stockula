"""Unit tests for backtest data models."""

from datetime import datetime

from stockula.config.models import BacktestResult, PortfolioBacktestResults, StrategyBacktestSummary


class TestBacktestResult:
    """Test BacktestResult model."""

    def test_backtest_result_creation(self):
        """Test creating a BacktestResult with all fields."""
        result = BacktestResult(
            ticker="AAPL",
            strategy="SMACross",
            parameters={"fast_period": 10, "slow_period": 20},
            return_pct=15.5,
            sharpe_ratio=1.2,
            max_drawdown_pct=-10.5,
            num_trades=25,
            win_rate=60.0,
        )

        assert result.ticker == "AAPL"
        assert result.strategy == "SMACross"
        assert result.parameters == {"fast_period": 10, "slow_period": 20}
        assert result.return_pct == 15.5
        assert result.sharpe_ratio == 1.2
        assert result.max_drawdown_pct == -10.5
        assert result.num_trades == 25
        assert result.win_rate == 60.0

    def test_backtest_result_no_trades(self):
        """Test BacktestResult with no trades (win_rate should be None)."""
        result = BacktestResult(
            ticker="NVDA",
            strategy="VIDYA",
            parameters={},
            return_pct=30.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=-5.0,
            num_trades=0,
            win_rate=None,
        )

        assert result.num_trades == 0
        assert result.win_rate is None

    def test_backtest_result_negative_return(self):
        """Test BacktestResult with negative returns."""
        result = BacktestResult(
            ticker="TSLA",
            strategy="RSI",
            parameters={"period": 14},
            return_pct=-25.5,
            sharpe_ratio=-0.8,
            max_drawdown_pct=-35.0,
            num_trades=10,
            win_rate=20.0,
        )

        assert result.return_pct == -25.5
        assert result.sharpe_ratio == -0.8
        assert result.win_rate == 20.0

    def test_backtest_result_serialization(self):
        """Test BacktestResult serialization to dict."""
        result = BacktestResult(
            ticker="GOOGL",
            strategy="MACD",
            parameters={"fast": 12, "slow": 26, "signal": 9},
            return_pct=8.5,
            sharpe_ratio=0.9,
            max_drawdown_pct=-12.0,
            num_trades=15,
            win_rate=53.33,
        )

        data = result.model_dump()
        assert data["ticker"] == "GOOGL"
        assert data["strategy"] == "MACD"
        assert data["parameters"]["fast"] == 12
        assert data["return_pct"] == 8.5
        assert data["num_trades"] == 15


class TestStrategyBacktestSummary:
    """Test StrategyBacktestSummary model."""

    def test_strategy_summary_creation(self):
        """Test creating a complete strategy summary."""
        # Create some backtest results
        results = [
            BacktestResult(
                ticker="AAPL",
                strategy="SMACross",
                parameters={},
                return_pct=10.0,
                sharpe_ratio=1.0,
                max_drawdown_pct=-5.0,
                num_trades=10,
                win_rate=60.0,
            ),
            BacktestResult(
                ticker="GOOGL",
                strategy="SMACross",
                parameters={},
                return_pct=-5.0,
                sharpe_ratio=-0.5,
                max_drawdown_pct=-15.0,
                num_trades=8,
                win_rate=25.0,
            ),
            BacktestResult(
                ticker="MSFT",
                strategy="SMACross",
                parameters={},
                return_pct=15.0,
                sharpe_ratio=1.2,
                max_drawdown_pct=-8.0,
                num_trades=12,
                win_rate=66.67,
            ),
        ]

        summary = StrategyBacktestSummary(
            strategy_name="SMACross",
            parameters={"fast_period": 10, "slow_period": 20},
            initial_portfolio_value=10000.0,
            final_portfolio_value=10667.0,
            total_return_pct=6.67,
            total_trades=30,
            winning_stocks=2,
            losing_stocks=1,
            average_return_pct=6.67,
            average_sharpe_ratio=0.567,
            detailed_results=results,
        )

        assert summary.strategy_name == "SMACross"
        assert summary.initial_portfolio_value == 10000.0
        assert summary.final_portfolio_value == 10667.0
        assert summary.total_return_pct == 6.67
        assert summary.total_trades == 30
        assert summary.winning_stocks == 2
        assert summary.losing_stocks == 1
        assert len(summary.detailed_results) == 3

    def test_strategy_summary_no_results(self):
        """Test strategy summary with no results."""
        summary = StrategyBacktestSummary(
            strategy_name="EmptyStrategy",
            parameters={},
            initial_portfolio_value=10000.0,
            final_portfolio_value=10000.0,
            total_return_pct=0.0,
            total_trades=0,
            winning_stocks=0,
            losing_stocks=0,
            average_return_pct=0.0,
            average_sharpe_ratio=0.0,
            detailed_results=[],
        )

        assert summary.total_trades == 0
        assert summary.winning_stocks == 0
        assert summary.losing_stocks == 0
        assert len(summary.detailed_results) == 0
        assert summary.total_return_pct == 0.0

    def test_strategy_summary_all_winning(self):
        """Test strategy summary with all winning trades."""
        results = [
            BacktestResult(
                ticker=ticker,
                strategy="Winner",
                parameters={},
                return_pct=20.0,
                sharpe_ratio=1.5,
                max_drawdown_pct=-3.0,
                num_trades=5,
                win_rate=100.0,
            )
            for ticker in ["AAPL", "GOOGL", "MSFT"]
        ]

        summary = StrategyBacktestSummary(
            strategy_name="Winner",
            parameters={},
            initial_portfolio_value=10000.0,
            final_portfolio_value=12000.0,
            total_return_pct=20.0,
            total_trades=15,
            winning_stocks=3,
            losing_stocks=0,
            average_return_pct=20.0,
            average_sharpe_ratio=1.5,
            detailed_results=results,
        )

        assert summary.winning_stocks == 3
        assert summary.losing_stocks == 0
        assert summary.average_return_pct == 20.0


class TestPortfolioBacktestResults:
    """Test PortfolioBacktestResults model."""

    def test_portfolio_results_creation(self):
        """Test creating complete portfolio backtest results."""
        # Create strategy summaries
        strategy1 = StrategyBacktestSummary(
            strategy_name="SMACross",
            parameters={},
            initial_portfolio_value=10000.0,
            final_portfolio_value=11000.0,
            total_return_pct=10.0,
            total_trades=50,
            winning_stocks=8,
            losing_stocks=2,
            average_return_pct=10.0,
            average_sharpe_ratio=1.0,
            detailed_results=[],
        )

        strategy2 = StrategyBacktestSummary(
            strategy_name="RSI",
            parameters={"period": 14},
            initial_portfolio_value=10000.0,
            final_portfolio_value=10500.0,
            total_return_pct=5.0,
            total_trades=30,
            winning_stocks=6,
            losing_stocks=4,
            average_return_pct=5.0,
            average_sharpe_ratio=0.7,
            detailed_results=[],
        )

        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=10000.0,
            initial_capital=10000.0,
            date_range={"start": "2024-01-01", "end": "2025-07-25"},
            broker_config={
                "name": "robinhood",
                "commission_type": "fixed",
                "commission_value": 0.0,
            },
            strategy_summaries=[strategy1, strategy2],
        )

        assert portfolio_results.initial_portfolio_value == 10000.0
        assert portfolio_results.initial_capital == 10000.0
        assert portfolio_results.date_range["start"] == "2024-01-01"
        assert portfolio_results.date_range["end"] == "2025-07-25"
        assert portfolio_results.broker_config["name"] == "robinhood"
        assert len(portfolio_results.strategy_summaries) == 2
        assert portfolio_results.strategy_summaries[0].strategy_name == "SMACross"
        assert portfolio_results.strategy_summaries[1].strategy_name == "RSI"

    def test_portfolio_results_timestamp(self):
        """Test that portfolio results have a timestamp."""
        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=10000.0,
            initial_capital=10000.0,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},
            broker_config={"name": "custom"},
            strategy_summaries=[],
        )

        assert isinstance(portfolio_results.timestamp, datetime)
        assert portfolio_results.timestamp <= datetime.now()

    def test_portfolio_results_serialization(self):
        """Test serializing portfolio results to dict/JSON."""
        result = BacktestResult(
            ticker="AAPL",
            strategy="VIDYA",
            parameters={},
            return_pct=25.0,
            sharpe_ratio=1.8,
            max_drawdown_pct=-7.0,
            num_trades=15,
            win_rate=73.33,
        )

        strategy = StrategyBacktestSummary(
            strategy_name="VIDYA",
            parameters={},
            initial_portfolio_value=20000.0,
            final_portfolio_value=25000.0,
            total_return_pct=25.0,
            total_trades=15,
            winning_stocks=1,
            losing_stocks=0,
            average_return_pct=25.0,
            average_sharpe_ratio=1.8,
            detailed_results=[result],
        )

        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=20000.0,
            initial_capital=20000.0,
            date_range={"start": "2024-01-01", "end": "2025-07-25"},
            broker_config={
                "name": "robinhood",
                "commission_type": "fixed",
                "commission_value": 0.0,
                "regulatory_fees": 0.0,
                "exchange_fees": 0.000166,
            },
            strategy_summaries=[strategy],
        )

        # Serialize to dict
        data = portfolio_results.model_dump()

        assert data["initial_portfolio_value"] == 20000.0
        assert data["broker_config"]["exchange_fees"] == 0.000166
        assert len(data["strategy_summaries"]) == 1
        assert data["strategy_summaries"][0]["strategy_name"] == "VIDYA"
        assert data["strategy_summaries"][0]["detailed_results"][0]["ticker"] == "AAPL"

    def test_portfolio_results_empty_strategies(self):
        """Test portfolio results with no strategies."""
        portfolio_results = PortfolioBacktestResults(
            initial_portfolio_value=10000.0,
            initial_capital=10000.0,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},
            broker_config={"name": "custom"},
            strategy_summaries=[],
        )

        assert len(portfolio_results.strategy_summaries) == 0
        assert portfolio_results.initial_portfolio_value == portfolio_results.initial_capital

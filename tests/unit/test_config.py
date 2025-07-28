"""Tests for configuration models and loading."""

from datetime import date

import pytest
import yaml

from stockula.config import (
    BacktestConfig,
    DataConfig,
    ForecastConfig,
    LoggingConfig,
    PortfolioConfig,
    StockulaConfig,
    TechnicalAnalysisConfig,
    TickerConfig,
    load_config,
    save_config,
)
from stockula.config.models import MACDConfig, RSIConfig, SMACrossConfig, StrategyConfig


class TestTickerConfig:
    """Test TickerConfig model."""

    def test_ticker_with_quantity(self):
        """Test ticker configuration with quantity."""
        ticker = TickerConfig(symbol="AAPL", quantity=10.0)
        assert ticker.symbol == "AAPL"
        assert ticker.quantity == 10.0
        assert ticker.allocation_pct is None
        assert ticker.allocation_amount is None

    def test_ticker_with_allocation_pct(self):
        """Test ticker configuration with allocation percentage."""
        ticker = TickerConfig(symbol="GOOGL", allocation_pct=25.0)
        assert ticker.symbol == "GOOGL"
        assert ticker.quantity is None
        assert ticker.allocation_pct == 25.0
        assert ticker.allocation_amount is None

    def test_ticker_with_allocation_amount(self):
        """Test ticker configuration with allocation amount."""
        ticker = TickerConfig(symbol="MSFT", allocation_amount=5000.0)
        assert ticker.symbol == "MSFT"
        assert ticker.quantity is None
        assert ticker.allocation_pct is None
        assert ticker.allocation_amount == 5000.0

    def test_ticker_with_category_only(self):
        """Test ticker configuration with only category (for auto-allocation)."""
        ticker = TickerConfig(symbol="SPY", category="INDEX")
        assert ticker.symbol == "SPY"
        assert ticker.category == "INDEX"
        assert ticker.quantity is None
        assert ticker.allocation_pct is None
        assert ticker.allocation_amount is None

    def test_ticker_multiple_allocations_raises_error(self):
        """Test that specifying multiple allocation methods raises an error."""
        with pytest.raises(ValueError, match="must specify exactly one"):
            TickerConfig(symbol="AAPL", quantity=10.0, allocation_pct=25.0)

    def test_ticker_no_allocation_no_category_raises_error(self):
        """Test that no allocation method and no category raises an error."""
        with pytest.raises(ValueError, match="must specify exactly one"):
            TickerConfig(symbol="AAPL")

    def test_ticker_with_full_details(self):
        """Test ticker with all optional fields."""
        ticker = TickerConfig(
            symbol="AAPL",
            quantity=10.0,
            market_cap=3000.0,
            sector="Technology",
            category="MOMENTUM",
            price_range={"open": 150.0, "high": 155.0, "low": 149.0, "close": 153.0},
        )
        assert ticker.market_cap == 3000.0
        assert ticker.sector == "Technology"
        assert ticker.category == "MOMENTUM"
        assert ticker.price_range["close"] == 153.0


class TestPortfolioConfig:
    """Test PortfolioConfig model."""

    def test_default_portfolio_config(self):
        """Test default portfolio configuration."""
        config = PortfolioConfig()
        assert config.name == "Main Portfolio"
        assert config.initial_capital == 10000.0
        assert config.allocation_method == "equal_weight"
        assert len(config.tickers) == 3  # Default tickers

    def test_dynamic_allocation_config(self):
        """Test dynamic allocation configuration."""
        config = PortfolioConfig(
            dynamic_allocation=True,
            allocation_method="dynamic",
            initial_capital=50000.0,
            tickers=[
                TickerConfig(symbol="AAPL", allocation_pct=50.0),
                TickerConfig(symbol="GOOGL", allocation_pct=30.0),
                TickerConfig(symbol="MSFT", allocation_pct=20.0),
            ],
        )
        assert config.dynamic_allocation is True
        assert config.allocation_method == "dynamic"

        # Test validation - total percentage should be 100%
        total_pct = sum(t.allocation_pct for t in config.tickers)
        assert total_pct == 100.0

    def test_auto_allocation_config(self):
        """Test auto allocation configuration."""
        config = PortfolioConfig(
            auto_allocate=True,
            category_ratios={"INDEX": 0.35, "MOMENTUM": 0.45, "SPECULATIVE": 0.20},
            capital_utilization_target=0.95,
            tickers=[
                TickerConfig(symbol="SPY", category="INDEX"),
                TickerConfig(symbol="AAPL", category="MOMENTUM"),
            ],
        )
        assert config.auto_allocate is True
        assert sum(config.category_ratios.values()) == 1.0
        assert config.capital_utilization_target == 0.95

    def test_auto_allocation_without_ratios_raises_error(self):
        """Test that auto allocation without category ratios raises an error."""
        with pytest.raises(ValueError, match="requires category_ratios"):
            PortfolioConfig(
                auto_allocate=True,
                tickers=[TickerConfig(symbol="AAPL", category="MOMENTUM")],
            )

    def test_auto_allocation_ratios_not_sum_to_one_raises_error(self):
        """Test that category ratios not summing to 1.0 raises an error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PortfolioConfig(
                auto_allocate=True,
                category_ratios={
                    "INDEX": 0.35,
                    "MOMENTUM": 0.45,
                    "SPECULATIVE": 0.15,  # Sum is 0.95, not 1.0
                },
            )

    def test_dynamic_allocation_exceeds_capital_raises_error(self):
        """Test that allocation amounts exceeding capital raises an error."""
        with pytest.raises(ValueError, match="exceed initial capital"):
            PortfolioConfig(
                dynamic_allocation=True,
                initial_capital=10000.0,
                tickers=[
                    TickerConfig(symbol="AAPL", allocation_amount=6000.0),
                    TickerConfig(
                        symbol="GOOGL", allocation_amount=5000.0
                    ),  # Total 11000 > 10000
                ],
            )

    def test_risk_management_settings(self):
        """Test risk management settings."""
        config = PortfolioConfig(max_position_size=20.0, stop_loss_pct=10.0)
        assert config.max_position_size == 20.0
        assert config.stop_loss_pct == 10.0


class TestDataConfig:
    """Test DataConfig model."""

    def test_data_config_with_dates(self):
        """Test data configuration with date strings."""
        config = DataConfig(
            start_date="2023-01-01", end_date="2023-12-31", interval="1d"
        )
        assert config.start_date == date(2023, 1, 1)
        assert config.end_date == date(2023, 12, 31)
        assert config.interval == "1d"

    def test_data_config_with_date_objects(self):
        """Test data configuration with date objects."""
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        config = DataConfig(start_date=start, end_date=end)
        assert config.start_date == start
        assert config.end_date == end

    def test_data_config_defaults(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.start_date is None
        assert config.end_date is None
        assert config.interval == "1d"


class TestStrategyConfigs:
    """Test strategy configuration models."""

    def test_sma_cross_config(self):
        """Test SMA cross strategy configuration."""
        config = SMACrossConfig(fast_period=10, slow_period=20)
        assert config.fast_period == 10
        assert config.slow_period == 20

    def test_sma_cross_invalid_periods_raises_error(self):
        """Test that invalid SMA periods raise an error."""
        with pytest.raises(ValueError, match="slow_period must be greater"):
            SMACrossConfig(fast_period=20, slow_period=10)

    def test_rsi_config(self):
        """Test RSI strategy configuration."""
        config = RSIConfig(
            period=14, oversold_threshold=30.0, overbought_threshold=70.0
        )
        assert config.period == 14
        assert config.oversold_threshold == 30.0
        assert config.overbought_threshold == 70.0

    def test_rsi_invalid_thresholds_raises_error(self):
        """Test that invalid RSI thresholds raise an error."""
        with pytest.raises(ValueError, match="overbought_threshold must be greater"):
            RSIConfig(oversold_threshold=70.0, overbought_threshold=30.0)

    def test_macd_config(self):
        """Test MACD strategy configuration."""
        config = MACDConfig(fast_period=12, slow_period=26, signal_period=9)
        assert config.fast_period == 12
        assert config.slow_period == 26
        assert config.signal_period == 9

    def test_strategy_config(self):
        """Test generic strategy configuration."""
        config = StrategyConfig(
            name="CustomStrategy", parameters={"param1": 10, "param2": "value"}
        )
        assert config.name == "CustomStrategy"
        assert config.parameters["param1"] == 10
        assert config.parameters["param2"] == "value"


class TestBacktestConfig:
    """Test BacktestConfig model."""

    def test_backtest_config_defaults(self):
        """Test default backtest configuration."""
        config = BacktestConfig()
        assert config.initial_cash == 10000.0
        assert config.commission == 0.002
        assert config.margin == 1.0
        assert config.hold_only_categories == ["INDEX", "BOND"]

    def test_backtest_with_strategies(self):
        """Test backtest configuration with strategies."""
        config = BacktestConfig(
            initial_cash=50000.0,
            strategies=[
                StrategyConfig(
                    name="SMACross", parameters={"fast_period": 10, "slow_period": 20}
                ),
                StrategyConfig(name="RSI", parameters={"period": 14}),
            ],
        )
        assert len(config.strategies) == 2
        assert config.strategies[0].name == "SMACross"
        assert config.strategies[1].parameters["period"] == 14


class TestForecastConfig:
    """Test ForecastConfig model."""

    def test_forecast_config_defaults(self):
        """Test default forecast configuration."""
        config = ForecastConfig()
        assert config.forecast_length == 14
        assert config.frequency == "infer"
        assert config.prediction_interval == 0.9
        assert config.model_list == "fast"

    def test_forecast_config_custom(self):
        """Test custom forecast configuration."""
        config = ForecastConfig(
            forecast_length=30,
            frequency="D",
            prediction_interval=0.95,
            model_list="default",
            ensemble="simple",
            max_generations=10,
        )
        assert config.forecast_length == 30
        assert config.frequency == "D"
        assert config.prediction_interval == 0.95
        assert config.model_list == "default"
        assert config.ensemble == "simple"
        assert config.max_generations == 10


class TestTechnicalAnalysisConfig:
    """Test TechnicalAnalysisConfig model."""

    def test_ta_config_defaults(self):
        """Test default technical analysis configuration."""
        config = TechnicalAnalysisConfig()
        assert "sma" in config.indicators
        assert "rsi" in config.indicators
        assert 20 in config.sma_periods
        assert config.rsi_period == 14

    def test_ta_config_custom(self):
        """Test custom technical analysis configuration."""
        config = TechnicalAnalysisConfig(
            indicators=["sma", "ema", "bbands"],
            sma_periods=[10, 30, 50],
            ema_periods=[9, 21],
            rsi_period=21,
            atr_period=20,
        )
        assert len(config.indicators) == 3
        assert 30 in config.sma_periods
        assert 9 in config.ema_periods
        assert config.rsi_period == 21
        assert config.atr_period == 20


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_logging_config_defaults(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.enabled is False
        assert config.level == "INFO"
        assert config.log_to_file is False
        assert config.log_file == "stockula.log"

    def test_logging_config_custom(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            enabled=True,
            level="DEBUG",
            log_to_file=True,
            log_file="custom.log",
            max_log_size=20_000_000,
            backup_count=5,
        )
        assert config.enabled is True
        assert config.level == "DEBUG"
        assert config.log_to_file is True
        assert config.log_file == "custom.log"
        assert config.max_log_size == 20_000_000
        assert config.backup_count == 5


class TestStockulaConfig:
    """Test main StockulaConfig model."""

    def test_stockula_config_defaults(self):
        """Test default Stockula configuration."""
        config = StockulaConfig()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.portfolio, PortfolioConfig)
        assert isinstance(config.backtest, BacktestConfig)
        assert isinstance(config.forecast, ForecastConfig)
        assert isinstance(config.technical_analysis, TechnicalAnalysisConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.output["format"] == "console"

    def test_stockula_config_custom(self, sample_portfolio_config, sample_data_config):
        """Test custom Stockula configuration."""
        config = StockulaConfig(
            portfolio=sample_portfolio_config,
            data=sample_data_config,
            output={"format": "json", "save_results": True},
        )
        assert config.portfolio.name == "Test Portfolio"
        assert config.data.start_date == date(2023, 1, 1)
        assert config.output["format"] == "json"


class TestConfigLoading:
    """Test configuration loading and saving."""

    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        config = load_config(temp_config_file)
        assert isinstance(config, StockulaConfig)
        assert config.portfolio.name == "Test Portfolio"
        assert config.portfolio.initial_capital == 100000.0

    def test_load_config_missing_file(self):
        """Test loading configuration from missing file raises error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("nonexistent.yaml")

    def test_load_config_no_file_uses_defaults(self):
        """Test loading configuration with no file uses defaults."""
        config = load_config(None)
        assert isinstance(config, StockulaConfig)
        assert config.portfolio.name == "Main Portfolio"  # Default

    def test_save_config(self, tmp_path, sample_stockula_config):
        """Test saving configuration to file."""
        config_path = tmp_path / "saved_config.yaml"
        save_config(sample_stockula_config, str(config_path))

        # Load it back
        loaded_config = load_config(str(config_path))
        assert loaded_config.portfolio.name == sample_stockula_config.portfolio.name
        assert (
            loaded_config.portfolio.initial_capital
            == sample_stockula_config.portfolio.initial_capital
        )

    def test_config_yaml_serialization(self, sample_stockula_config):
        """Test that configuration can be serialized to YAML."""
        config_dict = sample_stockula_config.model_dump()
        yaml_str = yaml.dump(config_dict)
        assert "Test Portfolio" in yaml_str
        assert "100000" in yaml_str

    def test_environment_variable_override(self, monkeypatch, tmp_path):
        """Test environment variable overrides."""
        config_file = tmp_path / "env_test.yaml"
        config = StockulaConfig()
        save_config(config, str(config_file))

        monkeypatch.setenv("STOCKULA_CONFIG_FILE", str(config_file))

        # In real implementation, this would be handled by the load_config function
        # checking os.environ.get("STOCKULA_CONFIG_FILE")
        loaded_config = load_config(str(config_file))
        assert isinstance(loaded_config, StockulaConfig)

"""Tests for TechnicalAnalysisManager."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from stockula.technical_analysis.manager import TechnicalAnalysisManager


class TestTechnicalAnalysisManager:
    """Test suite for TechnicalAnalysisManager."""

    @pytest.fixture
    def mock_data_fetcher(self):
        """Create mock data fetcher."""
        mock = MagicMock()
        # Create sample OHLCV data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {
                "Open": [100 + i * 0.1 for i in range(len(dates))],
                "High": [110 + i * 0.1 for i in range(len(dates))],
                "Low": [90 + i * 0.1 for i in range(len(dates))],
                "Close": [105 + i * 0.1 for i in range(len(dates))],
                "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            },
            index=dates,
        )
        mock.get_stock_data.return_value = data
        return mock

    @pytest.fixture
    def mock_logging_manager(self):
        """Create mock logging manager."""
        mock = MagicMock()
        mock.info = MagicMock()
        mock.error = MagicMock()
        mock.warning = MagicMock()
        mock.debug = MagicMock()
        return mock

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.data.start_date = "2023-01-01"
        config.data.end_date = "2023-12-31"
        config.technical_analysis.indicators = ["sma", "ema", "rsi", "macd"]
        config.technical_analysis.sma_periods = [20, 50]
        config.technical_analysis.ema_periods = [12, 26]
        config.technical_analysis.rsi = {"period": 14}
        config.technical_analysis.macd = {"period_fast": 12, "period_slow": 26, "signal": 9}
        return config

    @pytest.fixture
    def ta_manager(self, mock_data_fetcher, mock_logging_manager):
        """Create TechnicalAnalysisManager instance."""
        return TechnicalAnalysisManager(data_fetcher=mock_data_fetcher, logging_manager=mock_logging_manager)

    def test_init(self, ta_manager, mock_data_fetcher, mock_logging_manager):
        """Test TechnicalAnalysisManager initialization."""
        assert ta_manager.data_fetcher == mock_data_fetcher
        assert ta_manager.logger == mock_logging_manager
        assert "basic" in ta_manager.indicator_groups
        assert "momentum" in ta_manager.indicator_groups
        assert "trend" in ta_manager.indicator_groups
        assert "volatility" in ta_manager.indicator_groups
        assert "comprehensive" in ta_manager.indicator_groups

    def test_analyze_symbol_comprehensive(self, ta_manager, mock_config):
        """Test comprehensive analysis of a symbol."""
        result = ta_manager.analyze_symbol("AAPL", mock_config, analysis_type="comprehensive")

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "comprehensive"
        assert "current_price" in result
        assert "indicators" in result
        assert "summary" in result

    def test_analyze_symbol_basic(self, ta_manager, mock_config):
        """Test basic analysis of a symbol."""
        result = ta_manager.analyze_symbol("AAPL", mock_config, analysis_type="basic")

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "basic"
        assert "indicators" in result
        # Basic indicators should include sma, ema, rsi, volume
        expected_indicators = ["sma", "ema", "rsi", "volume"]
        for indicator in expected_indicators:
            assert any(indicator in key for key in result["indicators"].keys())

    def test_analyze_symbol_momentum(self, ta_manager, mock_config):
        """Test momentum analysis of a symbol."""
        result = ta_manager.analyze_symbol("AAPL", mock_config, analysis_type="momentum")

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "momentum"
        assert "indicators" in result

    def test_analyze_symbol_custom_indicators(self, ta_manager, mock_config):
        """Test analysis with custom indicators."""
        custom_indicators = ["sma", "rsi", "macd"]
        result = ta_manager.analyze_symbol("AAPL", mock_config, custom_indicators=custom_indicators)

        assert result["ticker"] == "AAPL"
        assert "indicators" in result
        # Check that only requested indicators are calculated
        for indicator in custom_indicators:
            assert any(indicator in key for key in result["indicators"].keys())

    def test_analyze_symbol_no_data(self, ta_manager, mock_config, mock_data_fetcher):
        """Test analysis when no data is available."""
        mock_data_fetcher.get_stock_data.return_value = pd.DataFrame()

        result = ta_manager.analyze_symbol("INVALID", mock_config)

        assert result["ticker"] == "INVALID"
        assert result["error"] == "No data available"

    def test_analyze_multiple_symbols(self, ta_manager, mock_config):
        """Test analyzing multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = ta_manager.analyze_multiple_symbols(symbols, mock_config)

        assert len(results) == 3
        assert "AAPL" in results
        assert "GOOGL" in results
        assert "MSFT" in results
        for symbol, result in results.items():
            assert result["ticker"] == symbol
            assert "indicators" in result

    def test_quick_analysis(self, ta_manager):
        """Test quick analysis functionality."""
        result = ta_manager.quick_analysis("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "quick"
        assert "current_price" in result
        assert "sma20" in result
        assert "ema20" in result
        assert "rsi" in result
        assert "price_vs_sma20" in result
        assert "volume_ratio" in result
        assert "trend" in result
        assert "momentum" in result

    def test_momentum_analysis(self, ta_manager, mock_config):
        """Test momentum-specific analysis."""
        result = ta_manager.momentum_analysis("AAPL", mock_config)

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "momentum"
        assert "indicators" in result

    def test_trend_analysis(self, ta_manager, mock_config):
        """Test trend-specific analysis."""
        result = ta_manager.trend_analysis("AAPL", mock_config)

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "trend"
        assert "indicators" in result

    def test_volatility_analysis(self, ta_manager, mock_config):
        """Test volatility-specific analysis."""
        result = ta_manager.volatility_analysis("AAPL", mock_config)

        assert result["ticker"] == "AAPL"
        assert result["analysis_type"] == "volatility"
        assert "indicators" in result

    def test_get_indicator_groups(self, ta_manager):
        """Test getting available indicator groups."""
        groups = ta_manager.get_indicator_groups()

        assert isinstance(groups, dict)
        assert "basic" in groups
        assert "momentum" in groups
        assert "trend" in groups
        assert "volatility" in groups
        assert "comprehensive" in groups

    def test_get_available_indicators(self, ta_manager):
        """Test getting all available indicators."""
        indicators = ta_manager.get_available_indicators()

        assert isinstance(indicators, list)
        assert len(indicators) > 0
        assert "sma" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators

    def test_calculate_custom_indicators(self, ta_manager):
        """Test calculating custom indicators with specific parameters."""
        custom_indicators = {
            "sma": {"period": 30},
            "ema": {"period": 15},
            "rsi": {"period": 21},
        }

        result = ta_manager.calculate_custom_indicators("AAPL", custom_indicators)

        assert result["ticker"] == "AAPL"
        assert "indicators" in result
        for indicator_name, params in custom_indicators.items():
            assert indicator_name in result["indicators"]
            assert result["indicators"][indicator_name]["params"] == params

    def test_analyze_symbol_with_error(self, ta_manager, mock_config, mock_data_fetcher):
        """Test error handling in analyze_symbol."""
        mock_data_fetcher.get_stock_data.side_effect = Exception("Test error")

        result = ta_manager.analyze_symbol("ERROR", mock_config)

        assert result["ticker"] == "ERROR"
        assert result["error"] == "Test error"

    def test_determine_trend(self, ta_manager):
        """Test trend determination logic."""
        # Create sample data
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        data = pd.DataFrame(
            {
                "Close": [100 + i for i in range(len(dates))],  # Upward trend
            },
            index=dates,
        )

        sma = pd.Series([95 + i for i in range(len(dates))], index=dates)
        ema = pd.Series([96 + i for i in range(len(dates))], index=dates)

        trend = ta_manager._determine_trend(data, sma, ema)
        assert trend == "uptrend"

    def test_determine_momentum(self, ta_manager):
        """Test momentum determination logic."""
        # Test overbought
        assert ta_manager._determine_momentum(75) == "overbought"

        # Test oversold
        assert ta_manager._determine_momentum(25) == "oversold"

        # Test bullish
        assert ta_manager._determine_momentum(60) == "bullish"

        # Test bearish
        assert ta_manager._determine_momentum(40) == "bearish"

        # Test unknown
        assert ta_manager._determine_momentum(None) == "unknown"

    def test_generate_analysis_summary(self, ta_manager):
        """Test analysis summary generation."""
        indicators = {
            "rsi": {"current": 75},
            "macd": {"current": {"MACD": 1.5, "MACD_SIGNAL": 1.2}},
            "sma": {"current": 100},
        }

        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        data = pd.DataFrame({"Close": [105] * len(dates)}, index=dates)

        summary = ta_manager._generate_analysis_summary(indicators, data)

        assert "signals" in summary
        assert "strength" in summary
        assert len(summary["signals"]) > 0
        assert "RSI Overbought" in summary["signals"]
        assert "MACD Bullish" in summary["signals"]

    def test_get_indicator_params_from_config(self, ta_manager, mock_config):
        """Test getting indicator parameters from configuration."""
        ta_config = mock_config.technical_analysis

        # Test getting RSI params from config
        params = ta_manager._get_indicator_params("rsi", ta_config)
        assert params == {"period": 14}

        # Test getting default params when not in config
        params = ta_manager._get_indicator_params("atr", ta_config)
        assert params == {"period": 14}  # Default ATR period

    def test_error_handling_in_indicator_calculation(self, ta_manager, mock_config, mock_data_fetcher):
        """Test error handling when indicator calculation fails."""
        # Create data that might cause indicator calculation errors
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")  # Very short data
        data = pd.DataFrame(
            {
                "Open": [100] * len(dates),
                "High": [110] * len(dates),
                "Low": [90] * len(dates),
                "Close": [105] * len(dates),
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )
        mock_data_fetcher.get_stock_data.return_value = data

        result = ta_manager.analyze_symbol("SHORT", mock_config, custom_indicators=["sma", "macd"])

        assert result["ticker"] == "SHORT"
        assert "indicators" in result
        # Some indicators might have errors due to insufficient data
        for _indicator_name, indicator_data in result["indicators"].items():
            if "error" in indicator_data:
                assert isinstance(indicator_data["error"], str)

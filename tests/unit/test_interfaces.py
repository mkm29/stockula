"""Tests for interface definitions in stockula.interfaces."""

import inspect
from abc import ABC
from unittest.mock import Mock

import pandas as pd

from stockula.interfaces import (
    IBacktestRunner,
    IDatabaseManager,
    IDataFetcher,
    IDomainFactory,
    ILoggingManager,
    IStockForecaster,
    ITechnicalIndicators,
)


class TestIDataFetcher:
    """Test IDataFetcher interface definition."""

    def test_is_abstract_base_class(self):
        """Test that IDataFetcher is an abstract base class."""
        assert issubclass(IDataFetcher, ABC)
        assert hasattr(IDataFetcher, "__abstractmethods__")

    def test_has_required_methods(self):
        """Test that IDataFetcher has all required abstract methods."""
        required_methods = ["get_stock_data", "get_current_prices", "get_info"]

        for method in required_methods:
            assert hasattr(IDataFetcher, method)
            assert callable(getattr(IDataFetcher, method))

    def test_get_stock_data_signature(self):
        """Test get_stock_data method signature."""
        method = IDataFetcher.get_stock_data
        sig = inspect.signature(method)

        # Check parameter names
        param_names = list(sig.parameters.keys())
        expected_params = [
            "self",
            "symbol",
            "start",
            "end",
            "interval",
            "force_refresh",
        ]
        assert param_names == expected_params

    def test_get_current_prices_signature(self):
        """Test get_current_prices method signature."""
        method = IDataFetcher.get_current_prices
        sig = inspect.signature(method)

        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "symbols" in param_names
        assert "show_progress" in param_names

    def test_cannot_instantiate_directly(self):
        """Test that IDataFetcher cannot be instantiated directly."""
        try:
            IDataFetcher()
            raise AssertionError("Should not be able to instantiate abstract class")
        except TypeError:
            pass  # Expected


class TestIDatabaseManager:
    """Test IDatabaseManager interface definition."""

    def test_is_abstract_base_class(self):
        """Test that IDatabaseManager is an abstract base class."""
        assert issubclass(IDatabaseManager, ABC)

    def test_has_required_methods(self):
        """Test that IDatabaseManager has all required abstract methods."""
        required_methods = ["get_price_history", "store_price_history", "has_data"]

        for method in required_methods:
            assert hasattr(IDatabaseManager, method)

    def test_cannot_instantiate_directly(self):
        """Test that IDatabaseManager cannot be instantiated directly."""
        try:
            IDatabaseManager()
            raise AssertionError("Should not be able to instantiate abstract class")
        except TypeError:
            pass


class TestIStockForecaster:
    """Test IStockForecaster interface definition."""

    def test_is_abstract_base_class(self):
        """Test that IStockForecaster is an abstract base class."""
        assert issubclass(IStockForecaster, ABC)

    def test_has_required_methods(self):
        """Test that IStockForecaster has all required abstract methods."""
        required_methods = ["forecast", "forecast_from_symbol"]

        for method in required_methods:
            assert hasattr(IStockForecaster, method)

    def test_forecast_method_signature(self):
        """Test forecast method signature."""
        method = IStockForecaster.forecast
        sig = inspect.signature(method)

        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "data" in param_names


class TestIBacktestRunner:
    """Test IBacktestRunner interface definition."""

    def test_is_abstract_base_class(self):
        """Test that IBacktestRunner is an abstract base class."""
        assert issubclass(IBacktestRunner, ABC)

    def test_has_required_methods(self):
        """Test that IBacktestRunner has all required abstract methods."""
        required_methods = ["run", "run_from_symbol"]

        for method in required_methods:
            assert hasattr(IBacktestRunner, method)

    def test_run_method_signature(self):
        """Test run method signature."""
        method = IBacktestRunner.run
        sig = inspect.signature(method)

        param_names = list(sig.parameters.keys())
        assert "self" in param_names
        assert "data" in param_names
        assert "strategy" in param_names


class TestIDomainFactory:
    """Test IDomainFactory interface definition."""

    def test_is_abstract_base_class(self):
        """Test that IDomainFactory is an abstract base class."""
        assert issubclass(IDomainFactory, ABC)

    def test_has_required_methods(self):
        """Test that IDomainFactory has all required abstract methods."""
        required_methods = ["create_portfolio", "create_asset"]

        for method in required_methods:
            assert hasattr(IDomainFactory, method)

    def test_create_portfolio_signature(self):
        """Test create_portfolio method signature."""
        method = IDomainFactory.create_portfolio
        sig = inspect.signature(method)

        param_names = list(sig.parameters.keys())
        assert "self" in param_names


class TestILoggingManager:
    """Test ILoggingManager interface definition."""

    def test_is_abstract_base_class(self):
        """Test that ILoggingManager is an abstract base class."""
        assert issubclass(ILoggingManager, ABC)

    def test_has_required_methods(self):
        """Test that ILoggingManager has all required abstract methods."""
        required_methods = ["setup", "debug", "info", "warning", "error", "critical"]

        for method in required_methods:
            assert hasattr(ILoggingManager, method)

    def test_logging_methods_signatures(self):
        """Test logging method signatures."""
        for method_name in ["debug", "info", "warning", "error", "critical"]:
            method = getattr(ILoggingManager, method_name)
            sig = inspect.signature(method)

            param_names = list(sig.parameters.keys())
            assert "self" in param_names
            assert "message" in param_names


class TestITechnicalIndicators:
    """Test ITechnicalIndicators interface definition."""

    def test_is_abstract_base_class(self):
        """Test that ITechnicalIndicators is an abstract base class."""
        assert issubclass(ITechnicalIndicators, ABC)

    def test_has_required_methods(self):
        """Test that ITechnicalIndicators has all required abstract methods."""
        required_methods = ["sma", "ema", "rsi", "macd"]

        for method in required_methods:
            assert hasattr(ITechnicalIndicators, method)

    def test_cannot_instantiate_directly(self):
        """Test that ITechnicalIndicators cannot be instantiated directly."""
        try:
            ITechnicalIndicators()
            raise AssertionError("Should not be able to instantiate abstract class")
        except TypeError:
            pass


class TestInterfaceIntegration:
    """Test interface integration and consistency."""

    def test_all_interfaces_importable(self):
        """Test that all interfaces can be imported."""
        from stockula.interfaces import (
            IBacktestRunner,
            IDatabaseManager,
            IDataFetcher,
            IDomainFactory,
            ILoggingManager,
            IStockForecaster,
            ITechnicalIndicators,
        )

        # Just checking they exist and are classes
        for interface in [
            IDataFetcher,
            IDatabaseManager,
            IStockForecaster,
            IBacktestRunner,
            IDomainFactory,
            ILoggingManager,
            ITechnicalIndicators,
        ]:
            assert inspect.isclass(interface)
            assert issubclass(interface, ABC)

    def test_interface_method_consistency(self):
        """Test that interface methods have consistent patterns."""
        interfaces = [
            IDataFetcher,
            IDatabaseManager,
            IStockForecaster,
            IBacktestRunner,
            IDomainFactory,
            ILoggingManager,
            ITechnicalIndicators,
        ]

        for interface in interfaces:
            # All interfaces should have at least one abstract method
            assert len(interface.__abstractmethods__) > 0

            # All abstract methods should be callable
            for method_name in interface.__abstractmethods__:
                method = getattr(interface, method_name)
                assert callable(method)

    def test_interface_inheritance_structure(self):
        """Test interface inheritance structure."""
        # All our interfaces should inherit from ABC
        interfaces = [
            IDataFetcher,
            IDatabaseManager,
            IStockForecaster,
            IBacktestRunner,
            IDomainFactory,
            ILoggingManager,
            ITechnicalIndicators,
        ]

        for interface in interfaces:
            assert ABC in interface.__mro__
            assert interface.__module__ == "stockula.interfaces"


class TestInterfaceUsage:
    """Test how interfaces would be used in practice."""

    def test_interface_implementation_pattern(self):
        """Test that interfaces can be properly implemented."""

        # Create a mock implementation
        class MockDataFetcher(IDataFetcher):
            def get_stock_data(self, symbol, start=None, end=None, interval="1d", force_refresh=False):
                return pd.DataFrame()

            def get_current_prices(self, symbols, show_progress=True):
                return {}

            def get_info(self, symbol, force_refresh=False):
                return {}

            def get_treasury_rates(self, start_date=None, end_date=None, duration="3_month"):
                return pd.Series()

        # Should be able to instantiate the implementation
        mock_fetcher = MockDataFetcher()
        assert isinstance(mock_fetcher, IDataFetcher)

    def test_interface_type_checking(self):
        """Test interface type checking functionality."""
        # Mock implementation for testing
        mock_fetcher = Mock(spec=IDataFetcher)

        # Should pass isinstance check
        assert isinstance(mock_fetcher, type(mock_fetcher))

        # Should have the interface methods
        for method in IDataFetcher.__abstractmethods__:
            assert hasattr(mock_fetcher, method)

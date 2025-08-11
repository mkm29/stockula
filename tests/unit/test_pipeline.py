"""Unit tests for the Stockula Pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from stockula.config import StockulaConfig
from stockula.pipeline import StockulaPipeline, run_pipeline


class TestStockulaPipeline:
    """Test cases for StockulaPipeline class."""

    @pytest.fixture
    def sample_config_dict(self):
        """Provide a sample configuration dictionary."""
        return {
            "portfolio": {
                "initial_capital": 100000,
                "allocation_method": "equal_weight",
                "tickers": [
                    {"symbol": "AAPL", "quantity": 10},
                    {"symbol": "GOOGL", "quantity": 5},
                    {"symbol": "MSFT", "quantity": 8},
                ],
            },
            "backtest": {
                "strategies": [
                    {"name": "rsi", "enabled": True},
                    {"name": "macd", "enabled": True},
                ],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
            },
        }

    @pytest.fixture
    def sample_config(self, sample_config_dict):
        """Provide a sample StockulaConfig instance."""
        return StockulaConfig.model_validate(sample_config_dict)

    @pytest.fixture
    def pipeline(self):
        """Create a StockulaPipeline instance."""
        return StockulaPipeline(verbose=False)

    def test_initialization(self):
        """Test pipeline initialization."""
        # Test with no arguments
        pipeline = StockulaPipeline()
        assert pipeline.base_config_path is None
        assert pipeline.verbose is False
        assert pipeline.console is not None
        assert pipeline.container is not None
        assert pipeline.manager is None
        assert pipeline.optimization_results == {}
        assert pipeline.backtest_results == {}
        assert pipeline.optimized_config is None

        # Test with base config path
        pipeline = StockulaPipeline(base_config_path="config.yaml", verbose=True)
        assert pipeline.base_config_path == Path("config.yaml")
        assert pipeline.verbose is True

    def test_load_configuration(self, pipeline, sample_config_dict, tmp_path):
        """Test configuration loading."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        # Test loading configuration
        config = pipeline.load_configuration(config_file)
        assert isinstance(config, StockulaConfig)
        assert config.portfolio.initial_capital == 100000
        assert len(config.portfolio.tickers) == 3

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.load_configuration("non_existent.yaml")

    @patch("stockula.pipeline.StockulaManager")
    def test_run_optimization(self, mock_manager_class, pipeline, sample_config):
        """Test optimization workflow."""
        # Setup mock manager
        mock_manager = MagicMock()
        # Mock run_optimize_allocation to return success (0)
        mock_manager.run_optimize_allocation.return_value = 0

        # Create a mock config with updated tickers
        mock_config = MagicMock()
        mock_config.portfolio.tickers = [
            MagicMock(symbol="AAPL", quantity=15.0),
            MagicMock(symbol="GOOGL", quantity=10.0),
            MagicMock(symbol="MSFT", quantity=12.0),
        ]
        mock_manager.config = mock_config

        mock_manager_class.return_value = mock_manager

        # Run optimization
        optimized_config, results = pipeline.run_optimization(config=sample_config)

        # Verify results
        assert optimized_config is not None
        assert "optimized_allocations" in results
        assert results["optimized_allocations"]["AAPL"] == 15.0
        assert pipeline.optimized_config is not None
        assert pipeline.optimization_results == results

    @patch("stockula.pipeline.StockulaManager")
    def test_run_backtest(self, mock_manager_class, pipeline, sample_config):
        """Test backtest workflow."""
        # Setup mock manager
        mock_manager = MagicMock()
        # Mock create_portfolio
        mock_portfolio = MagicMock()
        mock_manager.create_portfolio.return_value = mock_portfolio

        # Mock run_main_processing to return expected results
        mock_manager.run_main_processing.return_value = {
            "results": [
                {
                    "ticker": "AAPL",
                    "strategy": "rsi",
                    "return": 0.15,
                    "sharpe_ratio": 1.2,
                },
                {
                    "ticker": "GOOGL",
                    "strategy": "macd",
                    "return": 0.18,
                    "sharpe_ratio": 1.4,
                },
            ],
            "summary": {
                "total_return": 0.165,
                "sharpe_ratio": 1.3,
                "max_drawdown": -0.08,
            },
        }
        mock_manager_class.return_value = mock_manager

        # Mock the display method to avoid display issues in tests
        pipeline._display_backtest_results = MagicMock()

        # Run backtest
        results = pipeline.run_backtest(config=sample_config, use_optimized=False)

        # Verify results
        assert "results" in results
        assert len(results["results"]) == 2
        assert results["summary"]["total_return"] == 0.165
        assert pipeline.backtest_results == results

    @patch("stockula.pipeline.StockulaManager")
    def test_run_backtest_with_optimized(self, mock_manager_class, pipeline, sample_config):
        """Test backtest with optimized configuration."""
        # Set optimized config
        pipeline.optimized_config = sample_config

        # Setup mock manager
        mock_manager = MagicMock()
        # Mock create_portfolio
        mock_portfolio = MagicMock()
        mock_manager.create_portfolio.return_value = mock_portfolio

        # Mock run_main_processing to return expected results
        mock_manager.run_main_processing.return_value = {
            "results": [],
            "summary": {"total_return": 0.20},
        }
        mock_manager_class.return_value = mock_manager

        # Run backtest with optimized config
        results = pipeline.run_backtest(use_optimized=True)

        # Verify optimized config was used
        assert results["summary"]["total_return"] == 0.20

    @patch("stockula.pipeline.StockulaPipeline.run_backtest")
    @patch("stockula.pipeline.StockulaPipeline.run_optimization")
    @patch("stockula.pipeline.StockulaPipeline.load_configuration")
    def test_run_full_pipeline(
        self, mock_load_config, mock_run_optimization, mock_run_backtest, pipeline, sample_config
    ):
        """Test full pipeline execution."""
        # Setup mocks
        mock_load_config.return_value = sample_config
        mock_run_optimization.return_value = (
            sample_config,
            {"optimized_allocations": {"AAPL": 20}},
        )
        mock_run_backtest.return_value = {"summary": {"total_return": 0.25}}

        # Run full pipeline
        results = pipeline.run_full_pipeline(base_config_path="config.yaml")

        # Verify methods were called
        mock_load_config.assert_called_once()
        mock_run_optimization.assert_called_once()
        mock_run_backtest.assert_called_once()

        # Verify results structure
        assert "optimization" in results
        assert "backtest" in results
        assert "config" in results
        assert "timestamp" in results

    def test_save_optimized_config(self, pipeline, sample_config, tmp_path):
        """Test saving optimized configuration."""
        # Set optimized config
        pipeline.optimized_config = sample_config

        # Save to file
        config_path = tmp_path / "optimized.yaml"
        pipeline.save_optimized_config(config_path)

        # Verify file was created
        assert config_path.exists()

        # Load and verify content
        with open(config_path) as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data["portfolio"]["initial_capital"] == 100000

        # Test without optimized config
        pipeline.optimized_config = None
        with pytest.raises(ValueError, match="No optimized configuration"):
            pipeline.save_optimized_config("test.yaml")

    def test_save_results(self, pipeline, tmp_path):
        """Test saving results in different formats."""
        # Set some results
        pipeline.optimization_results = {"test": "optimization"}
        pipeline.backtest_results = {"test": "backtest"}

        # Test JSON format
        json_path = tmp_path / "results.json"
        pipeline.save_results(json_path, format="json")
        assert json_path.exists()

        # Test YAML format
        yaml_path = tmp_path / "results.yaml"
        pipeline.save_results(yaml_path, format="yaml")
        assert yaml_path.exists()

        # Test CSV format
        pipeline.backtest_results = {"results": [{"a": 1, "b": 2}]}
        csv_path = tmp_path / "results.csv"
        pipeline.save_results(csv_path, format="csv")
        assert csv_path.exists()

        # Test unsupported format
        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline.save_results("test.txt", format="txt")

    def test_extract_optimized_config(self, pipeline, sample_config):
        """Test extracting optimized configuration from results."""
        results = {
            "optimized_allocations": {
                "AAPL": 25.0,
                "GOOGL": 15.0,
                "MSFT": 20.0,
            }
        }

        optimized = pipeline._extract_optimized_config(sample_config, results)

        # Verify quantities were updated
        for ticker in optimized.portfolio.tickers:
            if ticker.symbol == "AAPL":
                assert ticker.quantity == 25.0
            elif ticker.symbol == "GOOGL":
                assert ticker.quantity == 15.0
            elif ticker.symbol == "MSFT":
                assert ticker.quantity == 20.0

        # Verify allocation method was set to optimized
        assert optimized.portfolio.allocation_method == "optimized"

    def test_combine_results(self, pipeline):
        """Test combining optimization and backtest results."""
        pipeline.optimization_results = {"opt": "test"}
        pipeline.backtest_results = {"back": "test"}
        pipeline.optimized_config = MagicMock()
        pipeline.optimized_config.model_dump.return_value = {"config": "test"}

        combined = pipeline._combine_results()

        assert "timestamp" in combined
        assert combined["optimization"] == {"opt": "test"}
        assert combined["backtest"] == {"back": "test"}
        assert combined["config"]["optimized"] == {"config": "test"}

    @patch("stockula.pipeline.ResultsDisplay")
    def test_display_methods(self, mock_display_class, pipeline):
        """Test display methods."""
        # Test optimization results display
        results = {
            "optimized_allocations": {
                "AAPL": 100,
                "GOOGL": 50,
            }
        }
        pipeline._display_optimization_results(results)

        # Test backtest results display
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        pipeline.display = mock_display

        results = {"results": [{"test": "data"}]}
        pipeline._display_backtest_results(results)
        mock_display.print_results.assert_called_once()

        # Test comparison display
        pipeline.optimization_results = {"metrics": {"sharpe": 1.5}}
        pipeline.backtest_results = {"summary": {"total_return": 0.15}}
        pipeline._display_comparison()


class TestPipelineConvenienceFunctions:
    """Test convenience functions for CLI integration."""

    @patch("stockula.pipeline.StockulaPipeline")
    def test_run_pipeline_function(self, mock_pipeline_class, tmp_path):
        """Test the run_pipeline convenience function."""
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_pipeline.return_value = {"test": "result"}
        mock_pipeline_class.return_value = mock_pipeline

        # Create a config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("portfolio:\n  initial_capital: 100000\n")

        # Run pipeline
        results = run_pipeline(
            base_config=str(config_file),
            optimized_config="opt.yaml",
            output=str(tmp_path / "results.json"),
            verbose=True,
        )

        # Verify calls
        mock_pipeline_class.assert_called_once_with(
            base_config_path=str(config_file),
            verbose=True,
        )
        mock_pipeline.run_full_pipeline.assert_called_once()
        mock_pipeline.save_results.assert_called_once()

        assert results == {"test": "result"}

    @patch("stockula.pipeline.StockulaPipeline")
    def test_run_pipeline_with_yaml_output(self, mock_pipeline_class, tmp_path):
        """Test run_pipeline with YAML output format."""
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_pipeline.return_value = {"test": "result"}
        mock_pipeline_class.return_value = mock_pipeline

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Run with YAML output
        run_pipeline(
            base_config=str(config_file),
            output=str(tmp_path / "results.yaml"),
        )

        # Verify YAML format was used
        mock_pipeline.save_results.assert_called_with(
            str(tmp_path / "results.yaml"),
            format="yaml",
        )

    @patch("stockula.pipeline.StockulaPipeline")
    def test_run_pipeline_with_csv_output(self, mock_pipeline_class, tmp_path):
        """Test run_pipeline with CSV output format."""
        mock_pipeline = MagicMock()
        mock_pipeline.run_full_pipeline.return_value = {"test": "result"}
        mock_pipeline_class.return_value = mock_pipeline

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: config")

        # Run with CSV output
        run_pipeline(
            base_config=str(config_file),
            output=str(tmp_path / "results.csv"),
        )

        # Verify CSV format was used
        mock_pipeline.save_results.assert_called_with(
            str(tmp_path / "results.csv"),
            format="csv",
        )

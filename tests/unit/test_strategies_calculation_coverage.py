"""Tests specifically targeting calculation function coverage in strategies.py."""

import warnings
import numpy as np
from unittest.mock import Mock, patch

from stockula.backtesting.strategies import (
    VIDYAStrategy,
    KAMAStrategy,
    FRAMAStrategy,
    TripleEMACrossStrategy,
    TRIMACrossStrategy,
    DoubleEMACrossStrategy,
)


class TestCalculationFunctionCoverage:
    """Tests targeting specific calculation functions for coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.sample_data = [100 + i + np.random.randn() * 0.5 for i in range(100)]
    
    def create_test_strategy(self, strategy_class, data_length=100):
        """Create a test strategy with proper setup."""
        strategy = Mock()
        strategy.data = Mock()
        strategy.data.__len__ = Mock(return_value=data_length)
        strategy.data.Close = self.sample_data[:data_length]
        strategy.data.High = [x + 2 for x in self.sample_data[:data_length]]
        strategy.data.Low = [x - 2 for x in self.sample_data[:data_length]]
        
        # Set required attributes for each strategy
        if strategy_class == VIDYAStrategy:
            strategy.cmo_period = 9
            strategy.smoothing_period = 12
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == KAMAStrategy:
            strategy.er_period = 10
            strategy.slow_period = 30
            strategy.fast_period = 2
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == FRAMAStrategy:
            strategy.frama_period = 16
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == TripleEMACrossStrategy:
            strategy.fast_period = 8
            strategy.medium_period = 21
            strategy.slow_period = 55
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == TRIMACrossStrategy:
            strategy.fast_period = 14
            strategy.slow_period = 28
            strategy.min_trading_days_buffer = 20
            strategy.atr_period = 14
        elif strategy_class == DoubleEMACrossStrategy:
            strategy.fast_period = 34
            strategy.slow_period = 55
            strategy.momentum_atr_multiple = 1.25
            strategy.speculative_atr_multiple = 1.0
            strategy.atr_period = 14
            strategy.min_trading_days_buffer = 20
        
        return strategy
    
    def test_vidya_cmo_calculation_complete(self):
        """Test complete VIDYA CMO calculation - targets lines 271-336."""
        strategy = self.create_test_strategy(VIDYAStrategy, 100)
        
        # Create data that will exercise CMO calculation logic
        volatile_data = []
        for i in range(100):
            if i % 10 < 5:
                volatile_data.append(100 + i)  # Upward moves
            else:
                volatile_data.append(100 + i - 5)  # Downward moves
        
        strategy.data.Close = volatile_data
        
        # Execute actual calculation functions
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # This should execute the complete VIDYA calculation including CMO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            VIDYAStrategy.init(strategy)
        
        # Verify calculation completed
        assert hasattr(strategy, 'vidya')
        assert hasattr(strategy, 'atr')
    
    def test_kama_efficiency_ratio_complete(self):
        """Test complete KAMA efficiency ratio calculation - targets lines 465-509."""
        strategy = self.create_test_strategy(KAMAStrategy, 100)
        
        # Create trending data to test efficiency ratio calculation
        trending_data = list(range(100, 200))  # Strong uptrend
        strategy.data.Close = trending_data
        
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # Execute KAMA calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            KAMAStrategy.init(strategy)
        
        assert hasattr(strategy, 'kama')
        assert hasattr(strategy, 'atr')
    
    def test_frama_fractal_dimension_complete(self):
        """Test complete FRAMA fractal dimension calculation - targets lines 625-686."""
        strategy = self.create_test_strategy(FRAMAStrategy, 100)
        
        # Create fractal-like data
        fractal_data = []
        for i in range(100):
            fractal_data.append(100 + 10 * np.sin(i / 10) + np.random.randn() * 0.1)
        
        strategy.data.Close = fractal_data
        strategy.data.High = [x + 1 for x in fractal_data]
        strategy.data.Low = [x - 1 for x in fractal_data]
        
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # Execute FRAMA calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            FRAMAStrategy.init(strategy)
        
        assert hasattr(strategy, 'frama')
        assert hasattr(strategy, 'atr')
    
    def test_triple_ema_tema_calculation_complete(self):
        """Test complete TripleEMA TEMA calculation - targets lines 786-847."""
        strategy = self.create_test_strategy(TripleEMACrossStrategy, 200)
        
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # Execute TripleEMA calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TripleEMACrossStrategy.init(strategy)
        
        assert hasattr(strategy, 'ema_fast')
        assert hasattr(strategy, 'ema_medium')
        assert hasattr(strategy, 'ema_slow')
    
    def test_trima_calculation_complete(self):
        """Test complete TRIMA calculation - targets lines 1437-1499."""
        strategy = self.create_test_strategy(TRIMACrossStrategy, 100)
        
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # Execute TRIMA calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TRIMACrossStrategy.init(strategy)
        
        assert hasattr(strategy, 'trima_fast')
        assert hasattr(strategy, 'trima_slow')
    
    def test_next_method_trading_logic_comprehensive(self):
        """Test comprehensive next method trading logic across strategies."""
        
        # Test VIDYA next method with buy/sell signals
        strategy = self.create_test_strategy(VIDYAStrategy, 100)
        strategy.vidya = [100, 101, 102]  # Upward trend
        strategy.data.Close = [103]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()
        
        with patch('stockula.backtesting.strategies.crossover', return_value=True):
            VIDYAStrategy.next(strategy)
        strategy.buy.assert_called_once()
        
        # Test KAMA next method with sell signal
        strategy = self.create_test_strategy(KAMAStrategy, 100)
        strategy.kama = [102, 101, 100]  # Downward trend
        strategy.data.Close = [99]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()
        
        with patch('stockula.backtesting.strategies.crossover', side_effect=[False, True]):
            KAMAStrategy.next(strategy)
        strategy.position.close.assert_called_once()
        
        # Test FRAMA next method
        strategy = self.create_test_strategy(FRAMAStrategy, 100)
        strategy.frama = [100, 101, 102]
        strategy.data.Close = [103]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()
        
        with patch('stockula.backtesting.strategies.crossover', return_value=True):
            FRAMAStrategy.next(strategy)
        strategy.buy.assert_called_once()
    
    def test_atr_calculation_variants(self):
        """Test ATR calculation with different price patterns."""
        
        # Test with gaps and volatility
        strategy = self.create_test_strategy(DoubleEMACrossStrategy, 100)
        
        # Create price data with gaps
        gap_data = []
        for i in range(100):
            if i == 20:
                gap_data.append(120)  # Gap up
            elif i == 50:
                gap_data.append(80)   # Gap down
            else:
                gap_data.append(100 + i * 0.5)
        
        strategy.data.Close = gap_data
        strategy.data.High = [x + 3 for x in gap_data]
        strategy.data.Low = [x - 3 for x in gap_data]
        
        def execute_actual_calculation(func, *args, **kwargs):
            return func(*args, **kwargs)
        
        strategy.I = execute_actual_calculation
        
        # Execute ATR calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DoubleEMACrossStrategy.init(strategy)
        
        assert hasattr(strategy, 'atr')
    
    def test_stop_loss_calculation_variants(self):
        """Test stop loss calculations with different scenarios."""
        
        # Test momentum stop loss
        strategy = self.create_test_strategy(DoubleEMACrossStrategy, 100)
        strategy.data.Close = [95]  # Current price
        strategy.atr = [2.0]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()
        
        # Mock recent trade
        mock_trade = Mock()
        mock_trade.entry_price = 100
        strategy.trades = [mock_trade]
        
        # No crossover, should trigger stop loss
        with patch('stockula.backtesting.strategies.crossover', return_value=False):
            DoubleEMACrossStrategy.next(strategy)
        
        strategy.position.close.assert_called_once()
        
        # Test speculative stop loss
        strategy = self.create_test_strategy(DoubleEMACrossStrategy, 100)
        strategy.data.Close = [98]  # Current price
        strategy.atr = [1.5]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()
        
        mock_trade = Mock()
        mock_trade.entry_price = 100
        strategy.trades = [mock_trade]
        
        with patch('stockula.backtesting.strategies.crossover', return_value=False):
            DoubleEMACrossStrategy.next(strategy)
        
        strategy.position.close.assert_called_once()
    
    def test_edge_case_price_scenarios(self):
        """Test calculation functions with edge case price scenarios."""
        
        # Test with constant prices (no volatility)
        strategy = self.create_test_strategy(VIDYAStrategy, 50)
        strategy.data.Close = [100.0] * 50
        strategy.data.High = [100.1] * 50
        strategy.data.Low = [99.9] * 50
        
        def execute_actual_calculation(func, *args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ZeroDivisionError, ValueError):
                # Handle division by zero gracefully
                return [0.0] * len(strategy.data.Close)
        
        strategy.I = execute_actual_calculation
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            VIDYAStrategy.init(strategy)
        
        # Test with extreme volatility
        strategy = self.create_test_strategy(KAMAStrategy, 80)
        extreme_data = []
        for i in range(80):
            extreme_data.append(100 + (-1) ** i * i * 2)  # Alternating extreme moves
        
        strategy.data.Close = extreme_data
        strategy.data.High = [x + 5 for x in extreme_data]
        strategy.data.Low = [x - 5 for x in extreme_data]
        
        strategy.I = execute_actual_calculation
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            KAMAStrategy.init(strategy)
    
    def test_crossover_signal_combinations(self):
        """Test different crossover signal combinations."""
        
        # Test TripleEMA with all combinations
        strategy = self.create_test_strategy(TripleEMACrossStrategy, 200)
        
        # Test fast > medium > slow (strong buy)
        strategy.ema_fast = [103]
        strategy.ema_medium = [102]
        strategy.ema_slow = [101]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()
        
        with patch('stockula.backtesting.strategies.crossover', side_effect=[True, True]):
            TripleEMACrossStrategy.next(strategy)
        strategy.buy.assert_called_once()
        
        # Test fast < medium < slow (strong sell)
        strategy = self.create_test_strategy(TripleEMACrossStrategy, 200)
        strategy.ema_fast = [99]
        strategy.ema_medium = [100]
        strategy.ema_slow = [101]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=True)
        strategy.position.close = Mock()
        
        # Mock trades list
        mock_trade = Mock()
        mock_trade.entry_price = 102
        strategy.trades = [mock_trade]
        strategy.atr = [1.5]
        strategy.atr_multiple = 1.5  # Add missing attribute
        strategy.data.Close = [98]
        
        with patch('stockula.backtesting.strategies.crossover', side_effect=[False, False, False, True]):
            TripleEMACrossStrategy.next(strategy)
        strategy.position.close.assert_called_once()
        
        # Test TRIMA crossover
        strategy = self.create_test_strategy(TRIMACrossStrategy, 100)
        strategy.trima_fast = [102]
        strategy.trima_slow = [101]
        strategy.position = Mock()
        strategy.position.__bool__ = Mock(return_value=False)
        strategy.buy = Mock()
        
        with patch('stockula.backtesting.strategies.crossover', return_value=True):
            TRIMACrossStrategy.next(strategy)
        strategy.buy.assert_called_once()
    
    def test_warning_generation_comprehensive(self):
        """Test warning generation for insufficient data across all strategies."""
        
        warning_cases = [
            (VIDYAStrategy, 25),
            (KAMAStrategy, 40), 
            (FRAMAStrategy, 35),
            (TripleEMACrossStrategy, 100),
            (TRIMACrossStrategy, 45),
        ]
        
        for strategy_class, insufficient_length in warning_cases:
            strategy = self.create_test_strategy(strategy_class, insufficient_length)
            
            def mock_calculation(func, *args, **kwargs):
                return [1.0] * insufficient_length
            
            strategy.I = mock_calculation
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                strategy_class.init(strategy)
                
                # Should generate insufficient data warning
                assert len(w) > 0
                warning_msg = str(w[0].message)
                assert "Insufficient data" in warning_msg or "need at least" in warning_msg
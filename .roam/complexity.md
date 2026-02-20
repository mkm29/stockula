Cognitive complexity (2358 functions analyzed, avg=2.9, p90=5.0, 49 critical, 30 high):

!! 171 DataFetcher.get_stock_data_batch meth src/stockula/data/fetcher.py:435 (nest=7, bool=3, params=5, density=2.01,
H.vol=1163) !! 119 Allocator.calculate_auto_allocation_quantities meth src/stockula/allocation/allocator.py:81 (nest=5,
bool=5, cb=2, ret=4, density=0.52, H.vol=3843) !! 111 StockulaManager.run_main_processing meth
src/stockula/manager.py:1045 (nest=6, bool=7, density=0.91, H.vol=1862) !! 99 BaseAllocator.\_get_calculation_prices
meth src/stockula/allocation/base_allocator.py:53 (nest=7, bool=4, density=1.30, H.vol=1016) !! 99
TechnicalAnalysisManager.analyze_symbol meth src/stockula/technical_analysis/manager.py:65 (nest=6, params=7,
density=1.08, H.vol=1500) !! 92 DataFetcher.get_current_prices meth src/stockula/data/fetcher.py:180 (nest=6,
density=1.23, H.vol=979) !! 88 StockulaManager.run_technical_analysis meth src/stockula/manager.py:239 (nest=4, bool=8,
density=0.72, H.vol=2355) !! 85 ResultsDisplay.show_portfolio_forecast_value meth src/stockula/display.py:768 (nest=5,
bool=7, density=0.62, H.vol=2209) !! 73 ResultsDisplay.\_display_forecast_results meth src/stockula/display.py:533
(nest=6, density=0.48, H.vol=2437) !! 65 pytest_sessionfinish fn tests/conftest.py:20 (nest=6, density=1.81) !! 61
StockulaManager.\_compute_indicators meth src/stockula/manager.py:362 (bool=14, params=7, density=0.76, H.vol=1547) !!
60 main fn utils/lint.py:114 (nest=4, bool=8, density=0.65, H.vol=1501) !! 59
BacktestingManager.run_with_train_test_split meth src/stockula/backtesting/manager.py:216 (nest=4, bool=3, params=8,
ret=4, density=0.71, H.vol=1180) !! 58 TechnicalAnalysisManager.calculate_custom_indicators meth
src/stockula/technical_analysis/manager.py:280 (nest=6, params=5, density=1.11, H.vol=806) !! 56
ChronosBackend.\_load_pipeline meth src/stockula/forecasting/backends/chronos.py:72 (nest=4, density=1.37) !! 56
ForecastingManager.forecast_symbol meth src/stockula/forecasting/manager.py:57 (nest=5, density=0.56, H.vol=1834) !! 55
ResultsDisplay.\_display_portfolio_composition meth src/stockula/display.py:133 (nest=4, density=0.80, H.vol=1397) !! 50
DataFetcher.get_current_prices_batch meth src/stockula/data/fetcher.py:521 (nest=5, density=1.32) !! 45
ResultsDisplay.show_portfolio_holdings meth src/stockula/display.py:220 (nest=4, bool=5, params=5, density=0.75,
H.vol=1297) !! 43 AutoGluonBackend.fit meth src/stockula/forecasting/backends/autogluon.py:218 (bool=4, params=8,
density=0.28, H.vol=2356)

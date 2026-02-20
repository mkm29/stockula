Fitness check: 2 rules

[FAIL] No circular imports in core (1241 violations) -- Reason: Production code should not import test modules [PASS]
Complexity threshold

Violations (1241):

No circular imports in core: config -> run_migrations_offline at alembic/env.py:96 No circular imports in core: config
-> run_migrations_online at alembic/env.py:98 No circular imports in core: revision -> exception at
alembic/versions/c1ceaae14e9f_add_performance_indexes.py:32 No circular imports in core: main -> BacktestRunner at
examples/automatic_dynamic_rates_example.py:4 No circular imports in core: main -> SMACrossStrategy at
examples/automatic_dynamic_rates_example.py:4 No circular imports in core: main -> create_container at
examples/automatic_dynamic_rates_example.py:5 No circular imports in core: main -> print at
examples/automatic_dynamic_rates_example.py:10 No circular imports in core: main -> create_container at
examples/automatic_dynamic_rates_example.py:15 No circular imports in core: main -> BacktestRunner at
examples/automatic_dynamic_rates_example.py:19 No circular imports in core: main -> BacktestOptimizationConfig at
examples/backtest_optimized_allocation_example.py:22 No circular imports in core: main -> PortfolioConfig at
examples/backtest_optimized_allocation_example.py:23 No circular imports in core: main -> StockulaConfig at
examples/backtest_optimized_allocation_example.py:24 No circular imports in core: main -> TickerConfig at
examples/backtest_optimized_allocation_example.py:25 No circular imports in core: main -> print at
examples/backtest_optimized_allocation_example.py:31 No circular imports in core: main -> StockulaConfig at
examples/backtest_optimized_allocation_example.py:40 No circular imports in core: main -> PortfolioConfig at
examples/backtest_optimized_allocation_example.py:41 No circular imports in core: main -> BacktestOptimizationConfig at
examples/backtest_optimized_allocation_example.py:46 No circular imports in core: main -> TickerConfig at
examples/backtest_optimized_allocation_example.py:61 No circular imports in core: main -> BacktestRunner at
examples/dynamic_sharpe_example.py:4 No circular imports in core: main -> DataFetcher at
examples/dynamic_sharpe_example.py:4 No circular imports in core: main -> SMACrossStrategy at
examples/dynamic_sharpe_example.py:4 No circular imports in core: main -> DataFetcher at
examples/dynamic_sharpe_example.py:10 No circular imports in core: main -> print at
examples/dynamic_sharpe_example.py:17 No circular imports in core: main -> BacktestRunner at
examples/dynamic_sharpe_example.py:25 No circular imports in core: main -> calculate_rolling_sharpe_ratio at
examples/dynamic_sharpe_example.py:86 No circular imports in core: main -> calculate_rolling_sharpe_ratio at
examples/dynamic_sharpe_example.py:93 No circular imports in core: main -> run_stockula at
examples/forecast_chronos.py:14 No circular imports in core: main -> run_stockula at examples/forecast_chronos.py:27 No
circular imports in core: main -> StockulaPipeline at examples/pipeline_example.py:16 No circular imports in core: main
-> print at examples/pipeline_example.py:32

... and 1211 more

1 passed, 1 failed

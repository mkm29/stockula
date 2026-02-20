VERDICT: Unhealthy codebase (34/100) — 7 critical, 10 warnings

Health Score: 34/100 | Tangle: 1.9% (51/2683 symbols in cycles) Propagation Cost: 0.1% | Algebraic Connectivity: 0.0000

Health: 39 issues — 7 CRITICAL, 10 WARNING, 22 INFO (12 cycles, 12 god components (8 actionable, 4 expected utilities),
15 bottlenecks (13 actionable, 2 expected utilities))

=== Cycles === [INFO] cycle 1 (12 symbols, 1 dir): run_command, check_docker_installation, check_docker_compose,
test_docker_build, test_basic_functionality, test_uv_functionality, test_python_version, test_security,
test_volume_functionality, test_docker_compose (+2 more) files: utils/validate_docker.py [INFO] cycle 2 (7 symbols, 1
dir): Stock, PriceHistory, Dividend, Split, OptionsCall, OptionsPut, StockInfo files: src/stockula/database/models.py
[INFO] cycle 3 (6 symbols, 1 dir): run_command, check_docker_installed, check_nvidia_docker, build_docker_image,
test_docker_image, main files: utils/verify_docker.py [INFO] cycle 4 (5 symbols, 1 dir): run_command, check_markdown,
fix_markdown, apply_fixes, main files: utils/lint.py [INFO] cycle 5 (4 symbols, 1 dir): run_command, format_yaml_files,
validate_yaml_files, main files: utils/format_yaml.py [INFO] cycle 6 (4 symbols, 1 dir): check_file_exists,
verify_gpu_build, verify_standard_build, main files: utils/verify_build.py [INFO] cycle 7 (3 symbols, 1 dir):
run_command, format_markdown, main files: utils/format_markdown.py [WARNING] cycle 8 (2 symbols, 2 dirs): Portfolio,
IDataFetcher files: src/stockula/domain/portfolio.py, src/stockula/interfaces.py [INFO] cycle 9 (2 symbols, 1 dir):
Strategy, StrategyPreset files: src/stockula/database/models.py [INFO] cycle 10 (2 symbols, 1 dir): load_dataset, main
files: scripts/chronos_batch_infer.py [INFO] cycle 11 (2 symbols, 1 dir): check_python_version, main files:
utils/check_python.py [INFO] cycle 12 (2 symbols, 1 dir): check_package, main files: utils/verify_gpu.py total: 12
cycle(s)

Cycle break suggestions: Break: remove dependency run_command -> main (highest edge betweenness in cycle (0.841)) Break:
remove dependency PriceHistory -> Stock (highest edge betweenness in cycle (0.143)) Break: remove dependency run_command
-> main (highest edge betweenness in cycle (0.700)) Break: remove dependency run_command -> main (highest edge
betweenness in cycle (0.600)) Break: remove dependency run_command -> main (highest edge betweenness in cycle (0.583))
Break: remove dependency check_file_exists -> main (highest edge betweenness in cycle (0.583)) Break: remove dependency
format_markdown -> run_command (highest edge betweenness in cycle (0.500))

=== God Components (degree > 20) === Sev Name Kind Degree Cat File

______________________________________________________________________

CRITICAL print meth 67 act src/stockula/cli_manager.py INFO BacktestOptimizedAllocator cls 26 act
src/stockula/allocation/backtest_allocator.py INFO DatabaseManager cls 24 act src/stockula/database/manager.py INFO
StrategyRepository cls 23 act src/stockula/data/strategy_repository.py INFO StockulaManager cls 23 act
src/stockula/manager.py INFO BaseStrategy cls 22 act src/stockula/backtesting/strategies.py INFO Container cls 22 act
src/stockula/container.py INFO StrategyRegistry var 21 act src/stockula/backtesting/__init__.py INFO StockulaConfig cls
44 util src/stockula/config/models.py INFO exception meth 38 util src/stockula/utils/logging_manager.py INFO field prop
33 util src/stockula/config/exceptions.py INFO TickerConfig cls 26 util src/stockula/config/models.py

=== Bottlenecks (high betweenness) === Sev Name Kind Betweenness Cat File

______________________________________________________________________

CRITICAL Container cls 1962 act src/stockula/container.py CRITICAL app var 1299 act src/stockula/cli.py CRITICAL
run_stockula fn 1081 act src/stockula/cli.py CRITICAL StockulaManager cls 848 act src/stockula/manager.py CRITICAL
BacktestOptimizedAllocator cls 605 act src/stockula/allocation/backtest_allocator.py CRITICAL ResultsDisplay cls 598 act
src/stockula/display.py WARNING DatabaseManager cls 541 act src/stockula/database/manager.py WARNING create_container fn
373 act src/stockula/container.py WARNING ForecastingManager cls 356 act src/stockula/forecasting/manager.py WARNING
BacktestRunner cls 210 act src/stockula/backtesting/runner.py WARNING StockulaPipeline cls 210 act
src/stockula/pipeline.py WARNING BaseStrategy cls 175 act src/stockula/backtesting/strategies.py WARNING DataManager cls
172 act src/stockula/data/manager.py WARNING StockulaConfig cls 560 util src/stockula/config/models.py WARNING
load_config fn 269 util src/stockula/config/settings.py

=== Layer Violations (0) === (none)

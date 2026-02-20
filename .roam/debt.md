=== Technical Debt (hotspot-weighted) ===

Project: 213 files, total debt = 52.3, mean = 0.245, median = 0.078 Worst quartile: 53 files hold 37.9 debt Signals: 12
files in cycles, 9 with god components, 141 hotspots

Suggestions: - Refactor hot complex files first: fetcher.py, manager.py, display.py (high churn + high complexity =
maximum debt leverage) - 4 hotspot file(s) participate in dependency cycles -- breaking these cycles reduces cascading
change cost - 527 dead export(s) in 19 hotspot file(s) -- removing them reduces cognitive load in frequently changed
code - Worst quartile (53 files) holds 72% of total debt -- focus refactoring budget here

Debt Health Hotspot Heat Breakdown File

______________________________________________________________________

1.782 0.65 2.8x HOT cyc god dead=17 src/stockula/database/models.py 1.484 0.59 2.5x HOT cx=16 cyc dead=21
src/stockula/domain/portfolio.py 1.256 0.43 2.9x HOT cx=14 god dead=35 src/stockula/backtesting/strategies.py 1.154 0.57
2.0x warm cyc god utils/validate_docker.py 1.115 0.40 2.8x HOT cx=26 dead=21 src/stockula/data/fetcher.py 1.114 0.39
2.9x HOT cx=25 dead=13 src/stockula/manager.py 1.055 0.38 2.8x HOT cx=12 god dead=14 src/stockula/config/models.py 1.041
0.39 2.7x HOT cx=25 dead=7 src/stockula/display.py 1.035 0.39 2.6x HOT god dead=31 tests/conftest.py 1.016 0.35 2.9x HOT
cx=22 dead=27 src/stockula/database/manager.py 0.902 0.31 2.9x HOT cx=18 dead=120 tests/unit/test_fetcher.py 0.887 0.30
3.0x HOT cx=17 dead=60 tests/unit/test_main.py 0.856 0.31 2.8x HOT cx=19 dead=6 src/stockula/forecasting/manager.py
0.847 0.31 2.7x HOT cx=18 dead=15 tests/integration/test_main.py 0.830 0.38 2.2x warm cx=24 dead=9
src/stockula/technical_analysis/manager.py 0.814 0.31 2.6x HOT cx=19 dead=7 src/stockula/backtesting/runner.py 0.789
0.28 2.9x HOT cx=15 dead=81 tests/unit/test_domain.py 0.743 0.41 1.8x warm cyc dead=27 src/stockula/interfaces.py 0.716
0.32 2.2x warm god dead=3 src/stockula/cli.py 0.709 0.42 1.7x warm cx=12 god dead=13
src/stockula/utils/logging_manager.py

(+193 more files, use --limit to show more)

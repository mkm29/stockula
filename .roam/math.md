VERDICT: 30 algorithmic improvements found (20 medium, 10 low)

I/O call in loop (N+1 query) (1): meth StrategyRepository.sync_to_database src/stockula/data/strategy_repository.py:150
[medium] Current: Per-item query in loop -- O(n) round trips Better: Batch query / bulk I/O -- O(1) round trips Tip: Use
WHERE IN (...) / bulk API / batch fetch instead of per-item queries

Branching recursion without memoization (6): meth DataFetchException.__init__ src/stockula/config/exceptions.py:15
[medium] Current: Naive branching recursion -- O(2^n) Better: Memoized / iterative DP -- O(n) Tip: Add @cache /
@lru_cache, or convert to iterative with a table meth DatabaseException.__init__ src/stockula/config/exceptions.py:79
[medium] Current: Naive branching recursion -- O(2^n) Better: Memoized / iterative DP -- O(n) Tip: Add @cache /
@lru_cache, or convert to iterative with a table meth ConfigurationException.__init__
src/stockula/config/exceptions.py:105 [medium] Current: Naive branching recursion -- O(2^n) Better: Memoized / iterative
DP -- O(n) Tip: Add @cache / @lru_cache, or convert to iterative with a table meth ValidationException.__init__
src/stockula/config/exceptions.py:125 [medium] Current: Naive branching recursion -- O(2^n) Better: Memoized / iterative
DP -- O(n) Tip: Add @cache / @lru_cache, or convert to iterative with a table meth AutoGluonBackend.fit
src/stockula/forecasting/backends/autogluon.py:218 [medium] Current: Naive branching recursion -- O(2^n) Better:
Memoized / iterative DP -- O(n) Tip: Add @cache / @lru_cache, or convert to iterative with a table meth
StockulaManager.\_convert_dates src/stockula/manager.py:208 [medium] Current: Naive branching recursion -- O(2^n)
Better: Memoized / iterative DP -- O(n) Tip: Add @cache / @lru_cache, or convert to iterative with a table

Loop-invariant call inside loop (13): fn load_dataset scripts/chronos_batch_infer.py:23 [medium] Current: Repeated call
per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the call before the loop
and store the result in a variable fn main scripts/chronos_batch_infer.py:75 [medium] Current: Repeated call per
iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the call before the loop and
store the result in a variable fn to_records scripts/export_to_gluonts_file_dataset.py:41 [medium] Current: Repeated
call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the call before the
loop and store the result in a variable fn calculate_vidya src/stockula/backtesting/indicators.py:148 [medium] Current:
Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the call
before the loop and store the result in a variable fn calculate_kama src/stockula/backtesting/indicators.py:191 [medium]
Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the
call before the loop and store the result in a variable fn calculate_frama src/stockula/backtesting/indicators.py:301
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn handle_validation_error src/stockula/cli.py:358
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn load_config src/stockula/config/settings.py:71
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn check_package_compatibility utils/check_python.py:26
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn check_installed_packages utils/check_python.py:73
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn get_markdown_files utils/format_markdown.py:51
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn check_required_files utils/validate_docker.py:111
[medium] Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip:
Move the call before the loop and store the result in a variable fn check_package utils/verify_gpu.py:14 [medium]
Current: Repeated call per iteration -- O(f(x)) per iter Better: Call hoisted before loop -- O(1) per iter Tip: Move the
call before the loop and store the result in a variable

Collection membership test (2): meth BacktestOptimizedAllocator.\_find_best_strategies
src/stockula/allocation/backtest_allocator.py:202 [low] Current: List linear scan -- O(n) per lookup Better: Set/hash
lookup -- O(1) amortized Tip: Convert to set for repeated lookups meth StockulaManager.run_main_processing
src/stockula/manager.py:1045 [low] Current: List linear scan -- O(n) per lookup Better: Set/hash lookup -- O(1)
amortized Tip: Convert to set for repeated lookups

String building (2): meth Allocator.calculate_auto_allocation_quantities src/stockula/allocation/allocator.py:81 [low]
Current: Loop concatenation -- O(n^2) Better: Join / StringBuilder -- O(n) Tip: Collect parts in a list, join once at
the end meth Portfolio.get_allocation_by_category src/stockula/domain/portfolio.py:229 [low] Current: Loop concatenation
-- O(n^2) Better: Join / StringBuilder -- O(n) Tip: Collect parts in a list, join once at the end

Nested loop lookup (4): meth Allocator.calculate_auto_allocation_quantities src/stockula/allocation/allocator.py:81
[low] Current: Nested iteration -- O(n*m) Better: Hash-map join -- O(n+m) Tip: Build a dict/set from one collection,
iterate the other meth BacktestOptimizedAllocator.\_find_best_strategies
src/stockula/allocation/backtest_allocator.py:202 [low] Current: Nested iteration -- O(n*m) Better: Hash-map join --
O(n+m) Tip: Build a dict/set from one collection, iterate the other meth
StockulaManager.create_portfolio_backtest_results src/stockula/manager.py:811 [low] Current: Nested iteration -- O(n*m)
Better: Hash-map join -- O(n+m) Tip: Build a dict/set from one collection, iterate the other meth
StockulaManager.run_main_processing src/stockula/manager.py:1045 [low] Current: Nested iteration -- O(n*m) Better:
Hash-map join -- O(n+m) Tip: Build a dict/set from one collection, iterate the other

Group by key (2): meth StrategyRepository.add_strategy_group src/stockula/data/strategy_repository.py:392 [low] Current:
Manual key-existence check -- O(n) Better: defaultdict / Collectors.groupingBy -- O(n) Tip: Use defaultdict(list) /
setdefault() / Collectors.groupingBy() meth StockulaManager.categorize_assets src/stockula/manager.py:1011 [low]
Current: Manual key-existence check -- O(n) Better: defaultdict / Collectors.groupingBy -- O(n) Tip: Use
defaultdict(list) / setdefault() / Collectors.groupingBy()

(showing 30 of more findings, use --limit to see more)

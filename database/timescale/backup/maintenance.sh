#!/bin/bash
# TimescaleDB Maintenance Script for Stockula Financial Data
# Performs regular maintenance tasks for optimal performance

set -euo pipefail

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-timescaledb}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stockula}"
POSTGRES_USER="${POSTGRES_USER:-stockula}"
LOG_FILE="${LOG_FILE:-/var/log/maintenance.log}"
DRY_RUN="${DRY_RUN:-false}"

# Logging function
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
	log "ERROR: Maintenance failed on line $1"
	exit 1
}
trap 'handle_error $LINENO' ERR

# Execute SQL command with logging
execute_sql() {
	local sql_command="$1"
	local description="${2:-SQL command}"

	log "Executing: $description"

	if [[ "$DRY_RUN" == "true" ]]; then
		log "DRY RUN: $sql_command"
		return
	fi

	psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--command="$sql_command" \
		2>&1 | tee -a "$LOG_FILE"
}

# Update table statistics
update_statistics() {
	log "Starting statistics update..."

	# Analyze all tables
	execute_sql "ANALYZE;" "Analyze all tables"

	# Update specific high-traffic tables more frequently
	local high_traffic_tables=(
		"price_history"
		"options_calls"
		"options_puts"
		"dividends"
		"splits"
	)

	for table in "${high_traffic_tables[@]}"; do
		execute_sql "ANALYZE $table;" "Analyze $table"
	done

	log "Statistics update completed"
}

# Vacuum maintenance
vacuum_maintenance() {
	log "Starting vacuum maintenance..."

	# Regular vacuum for all tables
	execute_sql "VACUUM;" "Vacuum all tables"

	# Full vacuum for small reference tables
	local reference_tables=(
		"stocks"
		"strategies"
		"strategy_presets"
		"autots_models"
		"autots_presets"
	)

	for table in "${reference_tables[@]}"; do
		execute_sql "VACUUM FULL $table;" "Full vacuum $table"
	done

	log "Vacuum maintenance completed"
}

# Reindex maintenance
reindex_maintenance() {
	log "Starting reindex maintenance..."

	# Reindex tables that might have index bloat
	local tables_to_reindex=(
		"price_history"
		"stock_info"
	)

	for table in "${tables_to_reindex[@]}"; do
		execute_sql "REINDEX TABLE $table;" "Reindex $table"
	done

	# Reindex indexes on JSONB columns which can fragment
	execute_sql "REINDEX INDEX idx_stock_info_json_gin;" "Reindex stock_info JSONB index"
	execute_sql "REINDEX INDEX idx_autots_models_categories;" "Reindex autots_models categories index"

	log "Reindex maintenance completed"
}

# TimescaleDB specific maintenance
timescaledb_maintenance() {
	log "Starting TimescaleDB specific maintenance..."

	# Refresh continuous aggregates
	local aggregates=(
		"price_history_daily"
		"price_history_weekly"
		"price_history_monthly"
		"market_volume_hourly"
		"options_volume_daily"
	)

	for aggregate in "${aggregates[@]}"; do
		execute_sql "CALL refresh_continuous_aggregate('$aggregate', NULL, NULL);" "Refresh $aggregate"
	done

	# Check and apply compression policies
	execute_sql "SELECT compress_chunk(i) FROM show_chunks('price_history', older_than => INTERVAL '2 years') i;" "Apply compression to old price_history chunks"
	execute_sql "SELECT compress_chunk(i) FROM show_chunks('options_calls', older_than => INTERVAL '3 months') i;" "Apply compression to old options_calls chunks"
	execute_sql "SELECT compress_chunk(i) FROM show_chunks('options_puts', older_than => INTERVAL '3 months') i;" "Apply compression to old options_puts chunks"

	log "TimescaleDB maintenance completed"
}

# Check database health
check_database_health() {
	log "Checking database health..."

	# Check for long-running queries
	local long_queries
	long_queries=$(psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--tuples-only \
		--command="SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active' AND now() - query_start > INTERVAL '1 hour';" | xargs)

	if [[ "$long_queries" -gt 0 ]]; then
		log "WARNING: $long_queries long-running queries detected"
	else
		log "✓ No long-running queries"
	fi

	# Check for blocked queries
	local blocked_queries
	blocked_queries=$(psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--tuples-only \
		--command="SELECT COUNT(*) FROM pg_stat_activity WHERE waiting;" | xargs)

	if [[ "$blocked_queries" -gt 0 ]]; then
		log "WARNING: $blocked_queries blocked queries detected"
	else
		log "✓ No blocked queries"
	fi

	# Check database size
	local db_size
	db_size=$(psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--tuples-only \
		--command="SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'));" | xargs)

	log "✓ Database size: $db_size"

	# Check connection count
	local connection_count
	connection_count=$(psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--tuples-only \
		--command="SELECT COUNT(*) FROM pg_stat_activity WHERE datname = '$POSTGRES_DB';" | xargs)

	log "✓ Active connections: $connection_count"

	log "Database health check completed"
}

# Performance tuning checks
performance_checks() {
	log "Running performance checks..."

	# Check for missing indexes on foreign keys
	execute_sql "
        SELECT
            c.conrelid::regclass AS table_name,
            string_agg(a.attname, ', ') AS column_names
        FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
        WHERE c.contype = 'f'
        AND NOT EXISTS (
            SELECT 1 FROM pg_index i
            WHERE i.indrelid = c.conrelid
            AND c.conkey <@ i.indkey
        )
        GROUP BY c.conrelid, c.conname;" "Check for missing foreign key indexes"

	# Check for unused indexes
	execute_sql "
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_tup_read,
            idx_tup_fetch
        FROM pg_stat_user_indexes
        WHERE idx_tup_read = 0 AND idx_tup_fetch = 0
        ORDER BY schemaname, tablename;" "Check for unused indexes"

	# Check table bloat estimation
	execute_sql "
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        LIMIT 10;" "Check largest tables"

	log "Performance checks completed"
}

# Generate maintenance report
generate_report() {
	log "Generating maintenance report..."

	local report_file="/tmp/maintenance_report_$(date +%Y%m%d_%H%M%S).txt"

	cat <<EOF >"$report_file"
TimescaleDB Maintenance Report
=============================
Date: $(date)
Database: $POSTGRES_DB
Host: $POSTGRES_HOST:$POSTGRES_PORT

Database Statistics:
-------------------
EOF

	# Add database statistics to report
	psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--command="
            SELECT
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC;" >>"$report_file"

	echo "" >>"$report_file"
	echo "Hypertable Information:" >>"$report_file"
	echo "----------------------" >>"$report_file"

	psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--command="
            SELECT
                hypertable_name,
                num_chunks,
                total_size_pretty,
                compression_enabled
            FROM timescaledb_information.hypertables;" >>"$report_file"

	log "Maintenance report generated: $report_file"
}

# Main maintenance function
main() {
	local task="${1:-all}"

	log "Starting maintenance task: $task"

	# Check database connectivity
	if ! pg_isready --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --quiet; then
		log "ERROR: Database is not ready for maintenance"
		exit 1
	fi

	case "$task" in
	"statistics" | "stats")
		update_statistics
		;;
	"vacuum")
		vacuum_maintenance
		;;
	"reindex")
		reindex_maintenance
		;;
	"timescaledb" | "ts")
		timescaledb_maintenance
		;;
	"health")
		check_database_health
		;;
	"performance" | "perf")
		performance_checks
		;;
	"report")
		generate_report
		;;
	"all")
		update_statistics
		vacuum_maintenance
		reindex_maintenance
		timescaledb_maintenance
		check_database_health
		performance_checks
		generate_report
		;;
	*)
		log "ERROR: Unknown maintenance task: $task"
		echo "Available tasks: statistics, vacuum, reindex, timescaledb, health, performance, report, all"
		exit 1
		;;
	esac

	log "Maintenance task completed: $task"
}

# Show usage
usage() {
	cat <<EOF
Usage: $0 [TASK]

Perform TimescaleDB maintenance tasks for Stockula

TASKS:
    statistics    Update table statistics
    vacuum        Perform vacuum maintenance
    reindex       Reindex tables and indexes
    timescaledb   TimescaleDB specific maintenance
    health        Check database health
    performance   Run performance checks
    report        Generate maintenance report
    all           Run all maintenance tasks (default)

ENVIRONMENT VARIABLES:
    POSTGRES_HOST     Database host (default: timescaledb)
    POSTGRES_PORT     Database port (default: 5432)
    POSTGRES_DB       Database name (default: stockula)
    POSTGRES_USER     Database user (default: stockula)
    LOG_FILE          Log file path (default: /var/log/maintenance.log)
    DRY_RUN           Set to 'true' for dry run (default: false)

EXAMPLES:
    $0                    # Run all maintenance tasks
    $0 vacuum             # Run only vacuum maintenance
    DRY_RUN=true $0       # Dry run of all tasks

EOF
}

# Check command line arguments
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
	usage
	exit 0
fi

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	main "${1:-all}"
fi

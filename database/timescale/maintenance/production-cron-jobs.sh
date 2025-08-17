#!/bin/bash
# Production Cron Jobs for TimescaleDB Maintenance
# Add these to your system crontab for automated maintenance

set -euo pipefail

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-timescaledb}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stockula}"
POSTGRES_USER="${POSTGRES_USER:-stockula}"
LOG_DIR="/var/log/stockula"
MAINTENANCE_LOG="$LOG_DIR/maintenance.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAINTENANCE_LOG"
}

# Execute SQL function with logging
execute_maintenance_function() {
	local function_name="$1"
	local description="$2"

	log "Starting: $description"

	if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT $function_name();" >>"$MAINTENANCE_LOG" 2>&1; then
		log "SUCCESS: $description completed"
		return 0
	else
		log "ERROR: $description failed"
		return 1
	fi
}

# Health check with alerting
health_check() {
	log "Running production health check"

	# Execute health check and capture results
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT metric, value, threshold, status FROM check_production_health();" \
		-t -A -F'|' 2>/dev/null | while IFS='|' read -r metric value threshold status; do

		if [[ "$status" == "CRITICAL" ]]; then
			log "CRITICAL ALERT: $metric = $value (threshold: $threshold)"
			# Add your alerting mechanism here (email, Slack, PagerDuty, etc.)
			# Example: send_alert "CRITICAL" "$metric = $value exceeds threshold $threshold"
		elif [[ "$status" == "WARNING" ]]; then
			log "WARNING: $metric = $value (threshold: $threshold)"
		else
			log "OK: $metric = $value"
		fi
	done
}

# Data freshness check
freshness_check() {
	log "Checking data freshness"

	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT table_name, latest_timestamp, age_hours, status FROM check_data_freshness();" \
		-t -A -F'|' 2>/dev/null | while IFS='|' read -r table_name latest_timestamp age_hours status; do

		if [[ "$status" == "STALE" ]]; then
			log "DATA ALERT: $table_name is stale (last update: $latest_timestamp, age: ${age_hours}h)"
			# Add alerting for stale data
		elif [[ "$status" == "WARNING" ]]; then
			log "DATA WARNING: $table_name data is aging (age: ${age_hours}h)"
		else
			log "DATA OK: $table_name is fresh"
		fi
	done
}

# Slow query analysis
slow_query_analysis() {
	log "Analyzing slow queries from last hour"

	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT calls, total_time, avg_time, LEFT(query, 100) as query_preview
             FROM get_slow_queries(60)
             WHERE calls > 10;" >>"$MAINTENANCE_LOG" 2>&1
}

# Index usage monitoring
index_monitoring() {
	log "Monitoring index usage"

	# Find unused indexes
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT table_name, index_name, scans
             FROM monitor_index_usage()
             WHERE scans < 10
             ORDER BY scans;" >>"$MAINTENANCE_LOG" 2>&1
}

# Connection monitoring
connection_monitoring() {
	log "Monitoring database connections"

	# Check for long-running queries
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT pid, usename, application_name, state,
                    query_start, now() - query_start as duration,
                    LEFT(query, 100) as query_preview
             FROM pg_stat_activity
             WHERE state = 'active'
             AND now() - query_start > interval '5 minutes'
             ORDER BY query_start;" >>"$MAINTENANCE_LOG" 2>&1

	# Check for idle in transaction
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT pid, usename, application_name, state,
                    state_change, now() - state_change as idle_duration
             FROM pg_stat_activity
             WHERE state = 'idle in transaction'
             AND now() - state_change > interval '10 minutes'
             ORDER BY state_change;" >>"$MAINTENANCE_LOG" 2>&1
}

# Backup verification
verify_backups() {
	local backup_dir="/backups"
	local today=$(date +%Y%m%d)

	log "Verifying recent backups"

	# Check if today's backup exists
	if ls "$backup_dir"/full/stockula_backup_${today}_* 1>/dev/null 2>&1 ||
		ls "$backup_dir"/incremental/stockula_backup_${today}_* 1>/dev/null 2>&1; then
		log "✓ Backup found for today"
	else
		log "✗ WARNING: No backup found for today"
		# Add alerting for missing backups
	fi

	# Check backup sizes (should be reasonable)
	find "$backup_dir" -name "stockula_backup_*" -mtime -1 -exec du -h {} \; |
		while read -r size file; do
			log "Recent backup: $file ($size)"
		done
}

# Disk space monitoring
disk_monitoring() {
	log "Monitoring disk space"

	# Check database size
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB')) as database_size;" \
		>>"$MAINTENANCE_LOG" 2>&1

	# Check largest tables
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
             FROM pg_tables
             WHERE schemaname = 'public'
             ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
             LIMIT 10;" >>"$MAINTENANCE_LOG" 2>&1

	# Check WAL directory size
	if [[ -d "/backups/archived_wal" ]]; then
		local wal_size=$(du -sh /backups/archived_wal 2>/dev/null | cut -f1)
		log "WAL archive size: $wal_size"
	fi
}

# TimescaleDB specific monitoring
timescale_monitoring() {
	log "TimescaleDB specific monitoring"

	# Check hypertable stats
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT hypertable_name, num_chunks,
                    pg_size_pretty(total_bytes) as total_size,
                    pg_size_pretty(compressed_total_bytes) as compressed_size
             FROM timescaledb_information.hypertables h
             LEFT JOIN timescaledb_information.chunks c ON h.hypertable_name = c.hypertable_name
             GROUP BY hypertable_name, total_bytes, compressed_total_bytes;" \
		>>"$MAINTENANCE_LOG" 2>&1

	# Check compression status
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT chunk_name, compression_status,
                    pg_size_pretty(before_compression_total_bytes) as before_compression,
                    pg_size_pretty(after_compression_total_bytes) as after_compression
             FROM timescaledb_information.chunks
             WHERE compression_status = 'Compressed'
             ORDER BY before_compression_total_bytes DESC
             LIMIT 10;" >>"$MAINTENANCE_LOG" 2>&1
}

# Main execution functions for cron jobs

# Every 5 minutes - Critical monitoring
critical_monitoring() {
	log "=== CRITICAL MONITORING (5min) ==="
	health_check
	connection_monitoring
}

# Every 15 minutes - Performance monitoring
performance_monitoring() {
	log "=== PERFORMANCE MONITORING (15min) ==="
	freshness_check
	slow_query_analysis
}

# Every hour - General monitoring
hourly_monitoring() {
	log "=== HOURLY MONITORING ==="
	index_monitoring
	disk_monitoring
	timescale_monitoring
}

# Daily maintenance
daily_maintenance() {
	log "=== DAILY MAINTENANCE ==="

	# Run daily ANALYZE
	execute_maintenance_function "production_vacuum_maintenance" "Daily vacuum and analyze"

	# Backup verification
	verify_backups

	# Cleanup old logs (keep 30 days)
	find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
}

# Weekly maintenance
weekly_maintenance() {
	log "=== WEEKLY MAINTENANCE ==="

	# Reindex maintenance
	execute_maintenance_function "production_reindex_maintenance" "Weekly reindex maintenance"

	# Data cleanup (be very careful with this!)
	# execute_maintenance_function "cleanup_old_data" "Data cleanup"
}

# Emergency health check - can be called manually
emergency_check() {
	log "=== EMERGENCY HEALTH CHECK ==="
	health_check
	connection_monitoring
	disk_monitoring

	# Check for blocking queries
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
		-c "SELECT blocked_locks.pid AS blocked_pid,
                    blocked_activity.usename AS blocked_user,
                    blocking_locks.pid AS blocking_pid,
                    blocking_activity.usename AS blocking_user,
                    blocked_activity.query AS blocked_statement,
                    blocking_activity.query AS current_statement_in_blocking_process
             FROM pg_catalog.pg_locks blocked_locks
             JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
             JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
                 AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
                 AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
                 AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
                 AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
                 AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
                 AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
                 AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
                 AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
                 AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
                 AND blocking_locks.pid != blocked_locks.pid
             JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
             WHERE NOT blocked_locks.GRANTED;" >>"$MAINTENANCE_LOG" 2>&1
}

# Parse command line arguments
case "${1:-}" in
"critical")
	critical_monitoring
	;;
"performance")
	performance_monitoring
	;;
"hourly")
	hourly_monitoring
	;;
"daily")
	daily_maintenance
	;;
"weekly")
	weekly_maintenance
	;;
"emergency")
	emergency_check
	;;
"health")
	health_check
	;;
*)
	echo "Usage: $0 {critical|performance|hourly|daily|weekly|emergency|health}"
	echo ""
	echo "Cron schedule recommendations:"
	echo "# Critical monitoring every 5 minutes"
	echo "*/5 * * * * /path/to/production-cron-jobs.sh critical"
	echo ""
	echo "# Performance monitoring every 15 minutes"
	echo "*/15 * * * * /path/to/production-cron-jobs.sh performance"
	echo ""
	echo "# Hourly monitoring"
	echo "0 * * * * /path/to/production-cron-jobs.sh hourly"
	echo ""
	echo "# Daily maintenance at 2 AM"
	echo "0 2 * * * /path/to/production-cron-jobs.sh daily"
	echo ""
	echo "# Weekly maintenance on Sunday at 3 AM"
	echo "0 3 * * 0 /path/to/production-cron-jobs.sh weekly"
	exit 1
	;;
esac

log "Completed: $1"

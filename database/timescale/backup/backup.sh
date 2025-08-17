#!/bin/bash
# PRODUCTION TimescaleDB Backup Script for Stockula Financial Data
# Hardened for immediate production deployment with comprehensive error handling
# Performs full and incremental backups with retention policies and verification

set -euo pipefail

# PRODUCTION Configuration - Tuned for reliability
POSTGRES_HOST="${POSTGRES_HOST:-timescaledb}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stockula}"
POSTGRES_USER="${POSTGRES_USER:-stockula}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-9}" # Maximum compression for production
LOG_FILE="${BACKUP_DIR}/backup.log"
MAX_BACKUP_TIME="7200" # 2 hours timeout
MIN_FREE_SPACE_GB="10" # Minimum free space required
VERIFY_BACKUPS="${VERIFY_BACKUPS:-true}"
SEND_ALERTS="${SEND_ALERTS:-true}"
ALERT_EMAIL="${ALERT_EMAIL:-admin@stockula.com}"

# Derived variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="stockula_backup_${TIMESTAMP}"
FULL_BACKUP_PATH="${BACKUP_DIR}/full/${BACKUP_NAME}"
INCREMENTAL_BACKUP_PATH="${BACKUP_DIR}/incremental/${BACKUP_NAME}"
WAL_ARCHIVE_DIR="${BACKUP_DIR}/archived_wal"

# Create backup directories
mkdir -p "${BACKUP_DIR}/full"
mkdir -p "${BACKUP_DIR}/incremental"
mkdir -p "${WAL_ARCHIVE_DIR}"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Error handling
handle_error() {
	log "ERROR: Backup failed on line $1"
	exit 1
}
trap 'handle_error $LINENO' ERR

# Check if this is a full backup day (Sunday) or incremental
is_full_backup_day() {
	[[ $(date +%u) -eq 7 ]] # Sunday = 7
}

# Check available disk space
check_disk_space() {
	local required_gb="$1"
	local available_gb

	available_gb=$(df "$BACKUP_DIR" | tail -1 | awk '{print int($4/1024/1024)}')

	if [ "$available_gb" -lt "$required_gb" ]; then
		log "ERROR: Insufficient disk space. Available: ${available_gb}GB, Required: ${required_gb}GB"
		send_alert "CRITICAL" "Backup failed: Insufficient disk space (${available_gb}GB available, ${required_gb}GB required)"
		exit 1
	fi

	log "Disk space check passed: ${available_gb}GB available"
}

# Send alert notifications
send_alert() {
	local severity="$1"
	local message="$2"

	if [[ "$SEND_ALERTS" == "true" ]]; then
		# Log the alert
		log "ALERT [$severity]: $message"

		# Send email alert (if mail is configured)
		if command -v mail >/dev/null 2>&1 && [[ -n "$ALERT_EMAIL" ]]; then
			echo "$message" | mail -s "[Stockula Backup $severity] $message" "$ALERT_EMAIL" || true
		fi

		# Send to syslog
		logger -t stockula-backup "[$severity] $message" || true

		# Add webhook support for Slack/Teams (customize as needed)
		if [[ -n "${WEBHOOK_URL:-}" ]]; then
			curl -X POST "$WEBHOOK_URL" \
				-H 'Content-Type: application/json' \
				-d "{\"text\": \"[$severity] Stockula Backup: $message\"}" \
				>/dev/null 2>&1 || true
		fi
	fi
}

# Rotate log files to prevent disk space issues
rotate_logs() {
	if [[ -f "$LOG_FILE" ]] && [[ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 104857600 ]]; then # 100MB
		mv "$LOG_FILE" "${LOG_FILE}.old"
		touch "$LOG_FILE"
		log "Log file rotated"
	fi
}

# Database connectivity check with timeout
check_database_connectivity() {
	log "Checking database connectivity..."

	if ! timeout 30 pg_isready --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --quiet; then
		log "ERROR: Database connectivity check failed"
		send_alert "CRITICAL" "Backup failed: Cannot connect to database $POSTGRES_HOST:$POSTGRES_PORT"
		exit 1
	fi

	# Check if database is accepting connections
	if ! timeout 30 psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" >/dev/null 2>&1; then
		log "ERROR: Database is not accepting connections"
		send_alert "CRITICAL" "Backup failed: Database not accepting connections"
		exit 1
	fi

	log "Database connectivity verified"
}

# Enhanced cleanup with safety checks
cleanup_old_backups() {
	log "Cleaning up backups older than ${RETENTION_DAYS} days..."

	local deleted_count=0
	local total_size_freed=0

	# Clean full backups with size tracking
	while IFS= read -r -d '' file; do
		local size
		size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
		if rm "$file" 2>/dev/null; then
			deleted_count=$((deleted_count + 1))
			total_size_freed=$((total_size_freed + size))
		fi
	done < <(find "${BACKUP_DIR}/full" -name "stockula_backup_*" -type f -mtime +"${RETENTION_DAYS}" -print0 2>/dev/null || true)

	# Clean incremental backups
	while IFS= read -r -d '' file; do
		local size
		size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
		if rm "$file" 2>/dev/null; then
			deleted_count=$((deleted_count + 1))
			total_size_freed=$((total_size_freed + size))
		fi
	done < <(find "${BACKUP_DIR}/incremental" -name "stockula_backup_*" -type f -mtime +"${RETENTION_DAYS}" -print0 2>/dev/null || true)

	# Clean WAL files (more conservative - keep extra days)
	local wal_retention_days=$((RETENTION_DAYS + 7))
	while IFS= read -r -d '' file; do
		local size
		size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
		if rm "$file" 2>/dev/null; then
			deleted_count=$((deleted_count + 1))
			total_size_freed=$((total_size_freed + size))
		fi
	done < <(find "${WAL_ARCHIVE_DIR}" -name "*.gz" -type f -mtime +"${wal_retention_days}" -print0 2>/dev/null || true)

	local size_mb=$((total_size_freed / 1024 / 1024))
	log "Cleanup completed: Removed $deleted_count files, freed ${size_mb}MB"
}

# Perform full backup using pg_dump
perform_full_backup() {
	log "Starting full backup: ${BACKUP_NAME}"

	# Custom format backup with compression
	pg_dump \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--format=custom \
		--compress="$COMPRESSION_LEVEL" \
		--no-password \
		--verbose \
		--file="${FULL_BACKUP_PATH}.dump" \
		2>&1 | tee -a "$LOG_FILE"

	# Create a plain SQL backup for easier inspection
	pg_dump \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--format=plain \
		--no-password \
		--file="${FULL_BACKUP_PATH}.sql" \
		2>&1 | tee -a "$LOG_FILE"

	# Compress the SQL file
	gzip "${FULL_BACKUP_PATH}.sql"

	# Generate checksums and metadata
	cd "${BACKUP_DIR}/full"
	sha256sum "${BACKUP_NAME}.dump" >"${BACKUP_NAME}.dump.sha256"
	sha256sum "${BACKUP_NAME}.sql.gz" >"${BACKUP_NAME}.sql.gz.sha256"

	# Create backup metadata file
	cat >"${BACKUP_NAME}.meta" <<EOF
Backup Metadata
===============
Timestamp: $(date -Iseconds)
Database: $POSTGRES_DB
Host: $POSTGRES_HOST:$POSTGRES_PORT
Backup Type: Full
Format: Custom + SQL (gzipped)
Compression Level: $COMPRESSION_LEVEL
Pg_dump Version: $(pg_dump --version | head -1)
Database Version: $(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT version();" | xargs)
EOF

	# Get backup size
	BACKUP_SIZE=$(du -h "${FULL_BACKUP_PATH}.dump" | cut -f1)
	log "Full backup completed: ${BACKUP_NAME}.dump (${BACKUP_SIZE})"
}

# Perform incremental backup using WAL shipping
perform_incremental_backup() {
	log "Starting incremental backup: ${BACKUP_NAME}"

	# Force a WAL switch to ensure recent changes are archived
	psql \
		--host="$POSTGRES_HOST" \
		--port="$POSTGRES_PORT" \
		--username="$POSTGRES_USER" \
		--dbname="$POSTGRES_DB" \
		--no-password \
		--command="SELECT pg_switch_wal();" \
		2>&1 | tee -a "$LOG_FILE"

	# Create incremental backup by copying recent WAL files
	INCREMENTAL_DIR="${INCREMENTAL_BACKUP_PATH}"
	mkdir -p "$INCREMENTAL_DIR"

	# Copy WAL files from the last 24 hours
	find "${WAL_ARCHIVE_DIR}" -name "*.gz" -type f -mtime -1 -exec cp {} "$INCREMENTAL_DIR/" \; 2>/dev/null || true

	# Create manifest
	ls -la "$INCREMENTAL_DIR" >"${INCREMENTAL_DIR}/manifest.txt"

	# Create tarball of incremental backup
	cd "${BACKUP_DIR}/incremental"
	tar -czf "${BACKUP_NAME}.tar.gz" -C "$INCREMENTAL_DIR" .
	rm -rf "$INCREMENTAL_DIR"

	# Generate checksum
	sha256sum "${BACKUP_NAME}.tar.gz" >"${BACKUP_NAME}.tar.gz.sha256"

	BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
	log "Incremental backup completed: ${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})"
}

# Enhanced backup verification
verify_backup() {
	local backup_file="$1"
	local backup_type="$2"

	if [[ "$VERIFY_BACKUPS" != "true" ]]; then
		log "Backup verification skipped (VERIFY_BACKUPS=false)"
		return 0
	fi

	log "Verifying ${backup_type} backup integrity..."

	if [[ "$backup_type" == "full" ]]; then
		# Verify pg_dump file structure
		if ! pg_restore --list "${backup_file}" >/dev/null 2>&1; then
			log "ERROR: Full backup verification failed - corrupted file"
			send_alert "CRITICAL" "Full backup verification failed: corrupted file ${backup_file}"
			return 1
		fi

		# Verify checksums
		local checksum_file="${backup_file}.sha256"
		if [[ -f "$checksum_file" ]]; then
			if ! (cd "$(dirname "$backup_file")" && sha256sum -c "$(basename "$checksum_file")") >/dev/null 2>&1; then
				log "ERROR: Checksum verification failed for ${backup_file}"
				send_alert "CRITICAL" "Backup checksum verification failed: ${backup_file}"
				return 1
			fi
			log "✓ Checksum verification passed"
		fi

		# Check backup size is reasonable
		local backup_size
		backup_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file" 2>/dev/null || echo 0)
		local min_size=$((10 * 1024 * 1024)) # 10MB minimum

		if [[ $backup_size -lt $min_size ]]; then
			log "ERROR: Backup file suspiciously small: $((backup_size / 1024 / 1024))MB"
			send_alert "CRITICAL" "Backup file suspiciously small: ${backup_file}"
			return 1
		fi

		log "✓ Full backup verification passed ($((backup_size / 1024 / 1024))MB)"
	else
		# Verify incremental backup (tarball)
		if ! tar -tzf "${backup_file}" >/dev/null 2>&1; then
			log "ERROR: Incremental backup verification failed - corrupted tarball"
			send_alert "CRITICAL" "Incremental backup verification failed: ${backup_file}"
			return 1
		fi
		log "✓ Incremental backup verification passed"
	fi

	return 0
}

# Generate backup report
generate_report() {
	local backup_type="$1"
	local backup_file="$2"

	cat <<EOF >>"${BACKUP_DIR}/backup_report.txt"
Backup Report - $(date)
====================================
Type: ${backup_type}
File: ${backup_file}
Size: $(du -h "$backup_file" | cut -f1)
Timestamp: ${TIMESTAMP}
Database: ${POSTGRES_DB}
Host: ${POSTGRES_HOST}
Status: SUCCESS
====================================

EOF
}

# Send notifications (placeholder for integration with monitoring)
send_notification() {
	local status="$1"
	local message="$2"

	# This could be extended to send to Slack, email, etc.
	log "NOTIFICATION [$status]: $message"
}

# Production-ready main backup function
main() {
	local start_time
	start_time=$(date +%s)

	log "Starting PRODUCTION backup process..."

	# Rotate logs first
	rotate_logs

	# Check disk space (require at least 10GB free)
	check_disk_space "$MIN_FREE_SPACE_GB"

	# Enhanced database connectivity check
	check_database_connectivity

	# Set timeout for entire backup process
	(
		# Perform backup based on schedule with timeout
		if is_full_backup_day || [[ "${FORCE_FULL_BACKUP:-}" == "true" ]]; then
			log "Performing FULL backup (production mode)"
			perform_full_backup
			BACKUP_FILE="${FULL_BACKUP_PATH}.dump"
			BACKUP_TYPE="full"

			if ! verify_backup "$BACKUP_FILE" "$BACKUP_TYPE"; then
				send_alert "CRITICAL" "Full backup verification failed: ${BACKUP_FILE}"
				exit 1
			fi
		else
			log "Performing INCREMENTAL backup (production mode)"
			perform_incremental_backup
			BACKUP_FILE="${INCREMENTAL_BACKUP_PATH}.tar.gz"
			BACKUP_TYPE="incremental"

			if ! verify_backup "$BACKUP_FILE" "$BACKUP_TYPE"; then
				send_alert "CRITICAL" "Incremental backup verification failed: ${BACKUP_FILE}"
				exit 1
			fi
		fi

		# Generate report
		generate_report "$BACKUP_TYPE" "$BACKUP_FILE"

		# Cleanup old backups
		cleanup_old_backups

		# Calculate total time
		local end_time
		end_time=$(date +%s)
		local duration=$((end_time - start_time))

		# Success notification
		send_alert "SUCCESS" "${BACKUP_TYPE} backup completed successfully in ${duration}s: $(basename "$BACKUP_FILE")"

		log "PRODUCTION backup process completed successfully in ${duration} seconds"

	) &

	local backup_pid=$!

	# Wait for backup with timeout
	if ! timeout "$MAX_BACKUP_TIME" wait $backup_pid; then
		kill $backup_pid 2>/dev/null || true
		log "ERROR: Backup process timed out after ${MAX_BACKUP_TIME} seconds"
		send_alert "CRITICAL" "Backup process timed out after ${MAX_BACKUP_TIME} seconds"
		exit 1
	fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	main "$@"
fi

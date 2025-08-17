#!/bin/bash
# Production Backup Verification Script for TimescaleDB
# Comprehensive verification and restoration testing

set -euo pipefail

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-timescaledb}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_DB="${POSTGRES_DB:-stockula}"
POSTGRES_USER="${POSTGRES_USER:-stockula}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
TEST_DB="${TEST_DB:-stockula_restore_test}"
LOG_FILE="${BACKUP_DIR}/verify.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
	echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
	echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
	echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
	echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Find the most recent backup
find_latest_backup() {
	local backup_type="$1" # full or incremental
	local backup_path

	if [[ "$backup_type" == "full" ]]; then
		backup_path=$(find "${BACKUP_DIR}/full" -name "stockula_backup_*.dump" -type f | sort | tail -1)
	else
		backup_path=$(find "${BACKUP_DIR}/incremental" -name "stockula_backup_*.tar.gz" -type f | sort | tail -1)
	fi

	if [[ -z "$backup_path" ]]; then
		log_error "No $backup_type backup found"
		return 1
	fi

	echo "$backup_path"
}

# Verify backup file integrity
verify_file_integrity() {
	local backup_file="$1"
	local backup_type="$2"

	log_info "Verifying file integrity for $backup_file"

	# Check if file exists and is readable
	if [[ ! -f "$backup_file" ]]; then
		log_error "Backup file does not exist: $backup_file"
		return 1
	fi

	if [[ ! -r "$backup_file" ]]; then
		log_error "Backup file is not readable: $backup_file"
		return 1
	fi

	# Check file size
	local file_size=$(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file")
	local size_mb=$((file_size / 1024 / 1024))

	if [[ $file_size -lt 1048576 ]]; then # Less than 1MB
		log_error "Backup file is suspiciously small: ${size_mb}MB"
		return 1
	fi

	log_info "File size: ${size_mb}MB"

	# Verify checksum if available
	local checksum_file="${backup_file}.sha256"
	if [[ -f "$checksum_file" ]]; then
		log_info "Verifying SHA256 checksum..."
		if (cd "$(dirname "$backup_file")" && sha256sum -c "$(basename "$checksum_file")") >/dev/null 2>&1; then
			log "✓ Checksum verification passed"
		else
			log_error "✗ Checksum verification failed"
			return 1
		fi
	else
		log_warn "No checksum file found, skipping checksum verification"
	fi

	# Type-specific verification
	if [[ "$backup_type" == "full" ]]; then
		# Verify pg_dump file structure
		log_info "Verifying pg_dump file structure..."
		if pg_restore --list "$backup_file" >/dev/null 2>&1; then
			log "✓ pg_dump file structure is valid"
		else
			log_error "✗ pg_dump file structure is invalid"
			return 1
		fi

		# Check table count in backup
		local table_count=$(pg_restore --list "$backup_file" | grep -c "TABLE DATA" || echo 0)
		log_info "Backup contains $table_count tables"

		if [[ $table_count -lt 5 ]]; then
			log_warn "Low table count in backup: $table_count"
		fi

	elif [[ "$backup_type" == "incremental" ]]; then
		# Verify tarball integrity
		log_info "Verifying tarball integrity..."
		if tar -tzf "$backup_file" >/dev/null 2>&1; then
			log "✓ Tarball integrity verified"
		else
			log_error "✗ Tarball is corrupted"
			return 1
		fi

		# Check WAL file count
		local wal_count=$(tar -tzf "$backup_file" | wc -l || echo 0)
		log_info "Incremental backup contains $wal_count WAL files"
	fi

	return 0
}

# Test database connectivity
test_connectivity() {
	log_info "Testing database connectivity..."

	if pg_isready --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --quiet; then
		log "✓ Database connectivity verified"
	else
		log_error "✗ Cannot connect to database"
		return 1
	fi

	# Test actual query execution
	if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" >/dev/null 2>&1; then
		log "✓ Database query execution verified"
	else
		log_error "✗ Cannot execute queries on database"
		return 1
	fi

	return 0
}

# Create test database for restoration
create_test_database() {
	log_info "Creating test database: $TEST_DB"

	# Drop test database if it exists
	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "postgres" \
		-c "DROP DATABASE IF EXISTS $TEST_DB;" >/dev/null 2>&1 || true

	# Create test database
	if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "postgres" \
		-c "CREATE DATABASE $TEST_DB;" >/dev/null 2>&1; then
		log "✓ Test database created: $TEST_DB"
	else
		log_error "✗ Failed to create test database"
		return 1
	fi

	# Enable TimescaleDB extension
	if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
		-c "CREATE EXTENSION IF NOT EXISTS timescaledb;" >/dev/null 2>&1; then
		log "✓ TimescaleDB extension enabled on test database"
	else
		log_warn "Could not enable TimescaleDB extension on test database"
	fi

	return 0
}

# Test backup restoration
test_restore() {
	local backup_file="$1"
	local backup_type="$2"

	log_info "Testing backup restoration from $backup_file"

	if [[ "$backup_type" != "full" ]]; then
		log_warn "Restoration test only supports full backups, skipping incremental"
		return 0
	fi

	# Create test database
	if ! create_test_database; then
		return 1
	fi

	# Restore backup to test database
	log_info "Restoring backup to test database..."
	if pg_restore --host="$POSTGRES_HOST" --port="$POSTGRES_PORT" --username="$POSTGRES_USER" \
		--dbname="$TEST_DB" --verbose --clean --if-exists \
		--no-owner --no-privileges "$backup_file" >/dev/null 2>&1; then
		log "✓ Backup restoration completed"
	else
		log_error "✗ Backup restoration failed"
		return 1
	fi

	# Verify restored data
	log_info "Verifying restored data..."

	# Check if main tables exist
	local tables=("stocks" "price_history" "dividends" "splits")
	for table in "${tables[@]}"; do
		local exists=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
			-t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');" 2>/dev/null | xargs)

		if [[ "$exists" == "t" ]]; then
			# Get row count
			local row_count=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
				-t -c "SELECT COUNT(*) FROM $table;" 2>/dev/null | xargs || echo 0)
			log "✓ Table $table exists with $row_count rows"
		else
			log_warn "Table $table does not exist in restored database"
		fi
	done

	# Check TimescaleDB hypertables
	local hypertables=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
		-t -c "SELECT COUNT(*) FROM timescaledb_information.hypertables;" 2>/dev/null | xargs || echo 0)

	if [[ $hypertables -gt 0 ]]; then
		log "✓ TimescaleDB hypertables restored: $hypertables tables"
	else
		log_warn "No TimescaleDB hypertables found in restored database"
	fi

	# Verify data integrity with sample queries
	log_info "Running data integrity checks..."

	# Check for recent price data
	local recent_prices=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
		-t -c "SELECT COUNT(*) FROM price_history WHERE time >= CURRENT_DATE - INTERVAL '7 days';" 2>/dev/null | xargs || echo 0)

	if [[ $recent_prices -gt 0 ]]; then
		log "✓ Recent price data found: $recent_prices records"
	else
		log_warn "No recent price data found (may be expected for test data)"
	fi

	# Check for data consistency
	local price_nulls=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
		-t -c "SELECT COUNT(*) FROM price_history WHERE close_price IS NULL;" 2>/dev/null | xargs || echo 0)

	local total_prices=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$TEST_DB" \
		-t -c "SELECT COUNT(*) FROM price_history;" 2>/dev/null | xargs || echo 0)

	if [[ $total_prices -gt 0 ]]; then
		local null_percentage=$((price_nulls * 100 / total_prices))
		if [[ $null_percentage -lt 10 ]]; then
			log "✓ Data quality check passed: ${null_percentage}% null prices"
		else
			log_warn "High percentage of null prices: ${null_percentage}%"
		fi
	fi

	log "✓ Backup restoration test completed successfully"
	return 0
}

# Cleanup test database
cleanup_test_database() {
	log_info "Cleaning up test database..."

	psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "postgres" \
		-c "DROP DATABASE IF EXISTS $TEST_DB;" >/dev/null 2>&1 || true

	log "✓ Test database cleaned up"
}

# Generate verification report
generate_report() {
	local backup_file="$1"
	local backup_type="$2"
	local status="$3"

	local report_file="${BACKUP_DIR}/verification_report_$(date +%Y%m%d_%H%M%S).txt"

	cat >"$report_file" <<EOF
Stockula Backup Verification Report
===================================
Date: $(date)
Backup File: $backup_file
Backup Type: $backup_type
Verification Status: $status

File Details:
- Size: $(stat -f%z "$backup_file" 2>/dev/null || stat -c%s "$backup_file") bytes
- Modified: $(stat -f%m "$backup_file" 2>/dev/null || stat -c%Y "$backup_file" | xargs date -d@)

Verification Steps:
$(tail -50 "$LOG_FILE" | grep -E "(✓|✗|WARN)" || echo "No verification steps found")

Database Connection:
- Host: $POSTGRES_HOST:$POSTGRES_PORT
- Database: $POSTGRES_DB
- User: $POSTGRES_USER

EOF

	log "Verification report generated: $report_file"
}

# Main verification function
main() {
	local backup_type="${1:-full}" # Default to full backup
	local specific_file="${2:-}"   # Optional specific file

	log "Starting backup verification process..."
	log "Backup type: $backup_type"

	# Test database connectivity first
	if ! test_connectivity; then
		log_error "Database connectivity test failed"
		exit 1
	fi

	# Find backup file
	local backup_file
	if [[ -n "$specific_file" ]]; then
		backup_file="$specific_file"
		log_info "Using specified backup file: $backup_file"
	else
		backup_file=$(find_latest_backup "$backup_type")
		if [[ $? -ne 0 ]]; then
			exit 1
		fi
		log_info "Using latest $backup_type backup: $backup_file"
	fi

	# Verify file integrity
	if ! verify_file_integrity "$backup_file" "$backup_type"; then
		log_error "File integrity verification failed"
		generate_report "$backup_file" "$backup_type" "FAILED"
		exit 1
	fi

	# Test restoration (only for full backups)
	if [[ "$backup_type" == "full" ]]; then
		if ! test_restore "$backup_file" "$backup_type"; then
			log_error "Restoration test failed"
			cleanup_test_database
			generate_report "$backup_file" "$backup_type" "FAILED"
			exit 1
		fi
		cleanup_test_database
	fi

	log "✓ All verification tests passed!"
	generate_report "$backup_file" "$backup_type" "PASSED"

	# Send success notification
	if command -v logger >/dev/null 2>&1; then
		logger -t stockula-backup-verify "Backup verification PASSED: $backup_file"
	fi
}

# Show usage
usage() {
	cat <<EOF
Usage: $0 [backup_type] [specific_file]

Arguments:
  backup_type    Type of backup to verify (full|incremental) [default: full]
  specific_file  Specific backup file to verify (optional)

Examples:
  $0                          # Verify latest full backup
  $0 full                     # Verify latest full backup
  $0 incremental             # Verify latest incremental backup
  $0 full /backups/full/stockula_backup_20240101_120000.dump

Environment Variables:
  POSTGRES_HOST              PostgreSQL host [default: timescaledb]
  POSTGRES_PORT              PostgreSQL port [default: 5432]
  POSTGRES_DB                Database name [default: stockula]
  POSTGRES_USER              Database user [default: stockula]
  BACKUP_DIR                 Backup directory [default: /backups]
  TEST_DB                    Test database name [default: stockula_restore_test]

EOF
}

# Handle command line arguments
case "${1:-}" in
-h | --help)
	usage
	exit 0
	;;
"")
	main "full"
	;;
*)
	main "$@"
	;;
esac

#!/bin/bash
# TimescaleDB Setup Script for Stockula
# Automates the complete deployment of TimescaleDB with monitoring and backup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_POSTGRES_PASSWORD="SuperSecret12"
DEFAULT_GRAFANA_PASSWORD="admin"
DEFAULT_ENVIRONMENT="production"
DEFAULT_MEMORY="4GB"
DEFAULT_CPUS="4"

# Configuration variables
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$DEFAULT_POSTGRES_PASSWORD}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$DEFAULT_GRAFANA_PASSWORD}"
ENVIRONMENT="${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}"
TS_TUNE_MEMORY="${TS_TUNE_MEMORY:-$DEFAULT_MEMORY}"
TS_TUNE_NUM_CPUS="${TS_TUNE_NUM_CPUS:-$DEFAULT_CPUS}"
DOCKER_COMPOSE_FILE="docker-compose.timescale.yml"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Logging functions
log_info() {
	echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
	echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
	echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
	log_step "Checking prerequisites..."

	# Check if Docker is installed and running
	if ! command -v docker &>/dev/null; then
		log_error "Docker is not installed. Please install Docker first."
		exit 1
	fi

	if ! docker info &>/dev/null; then
		log_error "Docker is not running. Please start Docker first."
		exit 1
	fi

	# Check if Docker Compose is installed
	if ! command -v docker-compose &>/dev/null; then
		log_error "Docker Compose is not installed. Please install Docker Compose first."
		exit 1
	fi

	# Check available disk space (minimum 10GB)
	available_space=$(df . | tail -1 | awk '{print $4}')
	min_space=$((10 * 1024 * 1024)) # 10GB in KB

	if [ "$available_space" -lt "$min_space" ]; then
		log_warn "Available disk space is less than 10GB. Consider freeing up space."
	fi

	log_info "Prerequisites check passed"
}

# Generate environment file
generate_env_file() {
	log_step "Generating environment configuration..."

	cat >.env <<EOF
# TimescaleDB Configuration for Stockula
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=stockula
POSTGRES_USER=stockula

# Grafana Configuration
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Performance Tuning
TS_TUNE_MEMORY=${TS_TUNE_MEMORY}
TS_TUNE_NUM_CPUS=${TS_TUNE_NUM_CPUS}
TS_TUNE_MAX_CONNS=200
TS_TUNE_MAX_BG_WORKERS=16

# Backup Configuration
RETENTION_DAYS=${BACKUP_RETENTION_DAYS}
BACKUP_DIR=/backups

# Environment
STOCKULA_ENV=${ENVIRONMENT}
COMPOSE_PROJECT_NAME=stockula-timescale
EOF

	log_info "Environment file created: .env"
}

# Create required directories
create_directories() {
	log_step "Creating required directories..."

	# Create backup directories
	mkdir -p ./backups/{full,incremental,archived_wal}
	mkdir -p ./logs

	# Set permissions
	chmod 755 ./backups
	chmod 755 ./logs

	log_info "Directories created successfully"
}

# Validate configuration files
validate_config() {
	log_step "Validating configuration files..."

	# Check if Docker Compose file exists
	if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
		log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
		exit 1
	fi

	# Check if database initialization scripts exist
	if [ ! -d "./database/timescale/init" ]; then
		log_error "Database initialization scripts not found"
		exit 1
	fi

	# Validate Docker Compose file
	if ! docker-compose -f "$DOCKER_COMPOSE_FILE" config >/dev/null 2>&1; then
		log_error "Invalid Docker Compose configuration"
		exit 1
	fi

	log_info "Configuration validation passed"
}

# Start TimescaleDB services
start_services() {
	log_step "Starting TimescaleDB services..."

	# Start core database services first
	log_info "Starting core database services..."
	docker-compose -f "$DOCKER_COMPOSE_FILE" up -d timescaledb redis

	# Wait for database to be ready
	log_info "Waiting for database to be ready..."
	local max_attempts=30
	local attempt=0

	while [ $attempt -lt $max_attempts ]; do
		if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T timescaledb pg_isready -U stockula -d stockula >/dev/null 2>&1; then
			log_info "Database is ready"
			break
		fi

		attempt=$((attempt + 1))
		log_info "Attempt $attempt/$max_attempts - waiting for database..."
		sleep 5
	done

	if [ $attempt -eq $max_attempts ]; then
		log_error "Database failed to start within timeout period"
		exit 1
	fi

	# Start connection pooling
	log_info "Starting PgBouncer..."
	docker-compose -f "$DOCKER_COMPOSE_FILE" up -d pgbouncer

	# Wait for PgBouncer
	sleep 10

	log_info "Core services started successfully"
}

# Start monitoring services
start_monitoring() {
	log_step "Starting monitoring services..."

	# Start Prometheus and exporters
	log_info "Starting Prometheus and exporters..."
	docker-compose -f "$DOCKER_COMPOSE_FILE" up -d prometheus postgres_exporter

	# Start Grafana
	log_info "Starting Grafana..."
	docker-compose -f "$DOCKER_COMPOSE_FILE" up -d grafana

	# Wait for services to be ready
	sleep 15

	log_info "Monitoring services started successfully"
}

# Verify installation
verify_installation() {
	log_step "Verifying installation..."

	# Check database connectivity
	log_info "Testing database connectivity..."
	if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T timescaledb psql -U stockula -d stockula -c "SELECT version();" >/dev/null 2>&1; then
		log_info "✓ Database connectivity verified"
	else
		log_error "✗ Database connectivity failed"
		return 1
	fi

	# Check TimescaleDB extension
	log_info "Checking TimescaleDB extension..."
	if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T timescaledb psql -U stockula -d stockula -c "SELECT extversion FROM pg_extension WHERE extname='timescaledb';" | grep -q "2\|1"; then
		log_info "✓ TimescaleDB extension verified"
	else
		log_error "✗ TimescaleDB extension not found"
		return 1
	fi

	# Check hypertables
	log_info "Checking hypertables..."
	local hypertables_count
	hypertables_count=$(docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T timescaledb psql -U stockula -d stockula -t -c "SELECT COUNT(*) FROM timescaledb_information.hypertables;" | xargs)

	if [ "$hypertables_count" -gt 0 ]; then
		log_info "✓ Hypertables created: $hypertables_count tables"
	else
		log_warn "⚠ No hypertables found (this may be normal for a fresh installation)"
	fi

	# Check PgBouncer connectivity
	log_info "Testing PgBouncer connectivity..."
	if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T pgbouncer psql -h localhost -p 5432 -U stockula -d stockula -c "SELECT 1;" >/dev/null 2>&1; then
		log_info "✓ PgBouncer connectivity verified"
	else
		log_warn "⚠ PgBouncer connectivity failed"
	fi

	# Check monitoring services
	log_info "Checking monitoring services..."
	if curl -s http://localhost:9090/-/healthy >/dev/null 2>&1; then
		log_info "✓ Prometheus is healthy"
	else
		log_warn "⚠ Prometheus is not accessible"
	fi

	if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
		log_info "✓ Grafana is healthy"
	else
		log_warn "⚠ Grafana is not accessible"
	fi

	log_info "Installation verification completed"
}

# Display connection information
show_connection_info() {
	log_step "Connection Information"

	echo
	echo "TimescaleDB is now running! Here are the connection details:"
	echo
	echo "Database Connections:"
	echo "  Direct:     postgresql://stockula:${POSTGRES_PASSWORD}@localhost:5432/stockula"
	echo "  PgBouncer:  postgresql://stockula:${POSTGRES_PASSWORD}@localhost:6432/stockula"
	echo
	echo "Monitoring:"
	echo "  Grafana:    http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
	echo "  Prometheus: http://localhost:9090"
	echo
	echo "Redis:"
	echo "  Connection: redis://localhost:6379"
	echo
	echo "Useful Commands:"
	echo "  Status:     docker-compose -f ${DOCKER_COMPOSE_FILE} ps"
	echo "  Logs:       docker-compose -f ${DOCKER_COMPOSE_FILE} logs -f"
	echo "  Stop:       docker-compose -f ${DOCKER_COMPOSE_FILE} down"
	echo "  Backup:     docker-compose -f ${DOCKER_COMPOSE_FILE} run --rm backup"
	echo "  Maintenance: ./database/timescale/backup/maintenance.sh"
	echo
}

# Setup cron jobs for automation
setup_automation() {
	log_step "Setting up automation..."

	# Create backup script wrapper
	cat >./backup-cron.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose -f docker-compose.timescale.yml run --rm backup
EOF
	chmod +x ./backup-cron.sh

	# Create maintenance script wrapper
	cat >./maintenance-cron.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./database/timescale/backup/maintenance.sh
EOF
	chmod +x ./maintenance-cron.sh

	log_info "Automation scripts created:"
	log_info "  ./backup-cron.sh - Daily backup script"
	log_info "  ./maintenance-cron.sh - Maintenance script"
	log_info ""
	log_info "To set up automatic backups, add to crontab:"
	log_info "  # Daily backup at 2 AM"
	log_info "  0 2 * * * $(pwd)/backup-cron.sh"
	log_info "  # Weekly maintenance on Sunday at 3 AM"
	log_info "  0 3 * * 0 $(pwd)/maintenance-cron.sh"
}

# Main function
main() {
	local start_monitoring_flag=false
	local start_app_flag=false
	local skip_verification=false

	# Parse command line arguments
	while [[ $# -gt 0 ]]; do
		case $1 in
		--monitoring)
			start_monitoring_flag=true
			shift
			;;
		--app)
			start_app_flag=true
			shift
			;;
		--skip-verification)
			skip_verification=true
			shift
			;;
		--help)
			cat <<EOF
Usage: $0 [OPTIONS]

OPTIONS:
    --monitoring        Start monitoring services (Prometheus, Grafana)
    --app              Start Stockula application
    --skip-verification Skip installation verification
    --help             Show this help message

ENVIRONMENT VARIABLES:
    POSTGRES_PASSWORD   Database password (default: $DEFAULT_POSTGRES_PASSWORD)
    GRAFANA_PASSWORD    Grafana admin password (default: $DEFAULT_GRAFANA_PASSWORD)
    TS_TUNE_MEMORY      Memory allocation (default: $DEFAULT_MEMORY)
    TS_TUNE_NUM_CPUS    CPU count (default: $DEFAULT_CPUS)
    ENVIRONMENT         Environment type (default: $DEFAULT_ENVIRONMENT)
    BACKUP_RETENTION_DAYS Backup retention period (default: 30)

EXAMPLES:
    $0                  # Start core services only
    $0 --monitoring     # Start with monitoring
    $0 --monitoring --app # Start everything

EOF
			exit 0
			;;
		*)
			log_error "Unknown option: $1"
			exit 1
			;;
		esac
	done

	log_info "Starting TimescaleDB setup for Stockula..."
	log_info "Environment: $ENVIRONMENT"
	log_info "Memory allocation: $TS_TUNE_MEMORY"
	log_info "CPU count: $TS_TUNE_NUM_CPUS"
	echo

	# Run setup steps
	check_prerequisites
	generate_env_file
	create_directories
	validate_config
	start_services

	if [ "$start_monitoring_flag" = true ]; then
		start_monitoring
	fi

	if [ "$start_app_flag" = true ]; then
		log_step "Starting Stockula application..."
		docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --profile app
	fi

	if [ "$skip_verification" = false ]; then
		verify_installation
	fi

	setup_automation
	show_connection_info

	log_info "TimescaleDB setup completed successfully! 🎉"
}

# Error handling
handle_error() {
	log_error "Setup failed on line $1"
	log_error "Check the logs above for details"
	exit 1
}
trap 'handle_error $LINENO' ERR

# Run main function
main "$@"

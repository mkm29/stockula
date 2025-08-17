#!/bin/bash
set -euo pipefail

# Stockula ETL Pipeline Deployment Script
# This script deploys the complete ETL infrastructure for immediate production use

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"
DATA_DIR="$DEPLOY_DIR/data"
LOGS_DIR="$DEPLOY_DIR/logs"
CONFIG_DIR="$DEPLOY_DIR/config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi

    # Check if Docker Compose is available
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi

    # Check available disk space (need at least 10GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    required_space=10485760  # 10GB in KB

    if [ "$available_space" -lt "$required_space" ]; then
        warn "Available disk space is less than 10GB. This may cause issues during deployment."
    fi

    success "Prerequisites check completed"
}

# Setup directory structure
setup_directories() {
    log "Setting up directory structure..."

    mkdir -p "$DATA_DIR/timescaledb"
    mkdir -p "$DATA_DIR/sqlite"
    mkdir -p "$LOGS_DIR/timescaledb"
    mkdir -p "$LOGS_DIR/etl"
    mkdir -p "$LOGS_DIR/monitoring"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DEPLOY_DIR/backups"

    # Set appropriate permissions
    chmod 755 "$DATA_DIR"
    chmod 755 "$LOGS_DIR"
    chmod 755 "$CONFIG_DIR"

    success "Directory structure created"
}

# Generate configuration files
generate_configs() {
    log "Generating configuration files..."

    # Generate ETL configuration
    cat > "$CONFIG_DIR/etl_config.yaml" << 'EOF'
# Stockula ETL Pipeline Configuration - Production Ready

# TimescaleDB connection settings
timescaledb:
  host: timescaledb
  port: 5432
  database: stockula
  username: stockula_etl
  password: ${ETL_PASSWORD:-etl_secure_password_change_me}
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

# Data migration settings (optimized for speed)
migration:
  sqlite_path: /data/stockula.db
  batch_size: 50000
  parallel_workers: 8
  chunk_size: 5000
  disable_foreign_keys: true
  use_copy_from: true
  vacuum_analyze: true
  verify_counts: true
  verify_checksums: true
  max_retries: 5
  retry_delay: 2.0
  exponential_backoff: true

# Real-time streaming settings (production optimized)
streaming:
  yfinance_enabled: true
  fetch_interval: 30
  batch_size: 2000
  batch_timeout: 3.0
  max_batch_age: 15.0
  worker_count: 4
  worker_queue_size: 5000
  max_retries: 5
  retry_delay: 1.0
  dead_letter_queue: true
  enable_metrics: true
  metrics_interval: 30

# Data validation settings (strict for production)
validation:
  check_null_values: true
  check_data_types: true
  check_ranges: true
  check_duplicates: true
  null_threshold: 0.02
  duplicate_threshold: 0.005
  min_price: 0.01
  max_price: 500000.0
  max_price_change: 0.3
  min_volume: 0
  max_volume: 50000000000
  generate_reports: true
  report_format: json

# Global settings
log_level: INFO
log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
environment: production
debug: false
EOF

    # Generate environment file
    cat > "$DEPLOY_DIR/.env" << 'EOF'
# Stockula ETL Environment Configuration
# IMPORTANT: Change all default passwords before production deployment!

# Database passwords
POSTGRES_PASSWORD=stockula_secure_password_change_me
ETL_PASSWORD=etl_secure_password_change_me
ANALYST_PASSWORD=analyst_secure_password_change_me

# Admin interface passwords
GRAFANA_PASSWORD=grafana_secure_password_change_me
PGADMIN_PASSWORD=pgadmin_secure_password_change_me

# Deployment settings
COMPOSE_PROJECT_NAME=stockula-etl
COMPOSE_FILE=docker-compose.yml
EOF

    # Generate Docker Compose override for production
    cat > "$DEPLOY_DIR/docker-compose.prod.yml" << 'EOF'
version: '3.8'

# Production overrides for Stockula ETL
services:
  timescaledb:
    restart: always
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8.0'
        reservations:
          memory: 8G
          cpus: '4.0'

  etl-streaming:
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  etl-monitoring:
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
EOF

    success "Configuration files generated"
}

# Copy SQLite database if it exists
copy_sqlite_data() {
    log "Checking for existing SQLite database..."

    SQLITE_SOURCE="$PROJECT_ROOT/stockula.db"
    SQLITE_DEST="$DATA_DIR/sqlite/stockula.db"

    if [ -f "$SQLITE_SOURCE" ]; then
        log "Found SQLite database, copying to deployment directory..."
        cp "$SQLITE_SOURCE" "$SQLITE_DEST"
        chmod 644 "$SQLITE_DEST"
        success "SQLite database copied"
    else
        warn "No SQLite database found at $SQLITE_SOURCE"
        warn "Migration will be skipped. You can add the database later and run migration manually."
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..."

    cd "$DEPLOY_DIR"

    # Build TimescaleDB image
    docker compose build timescaledb

    # Build ETL images (if Dockerfiles exist)
    if [ -f "$PROJECT_ROOT/deploy/docker/Dockerfile" ]; then
        docker compose build etl-migration etl-streaming etl-monitoring
    fi

    success "Docker images built"
}

# Start core services
start_core_services() {
    log "Starting core services..."

    cd "$DEPLOY_DIR"

    # Start TimescaleDB first
    docker compose up -d timescaledb

    # Wait for TimescaleDB to be ready
    log "Waiting for TimescaleDB to be ready..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if docker compose exec -T timescaledb pg_isready -U postgres -d stockula; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        error "TimescaleDB failed to start within 2 minutes"
        exit 1
    fi

    success "TimescaleDB is ready"

    # Start Redis
    docker compose up -d redis

    success "Core services started"
}

# Run data migration
run_migration() {
    log "Running data migration..."

    cd "$DEPLOY_DIR"

    if [ -f "$DATA_DIR/sqlite/stockula.db" ]; then
        # Run migration using the CLI
        docker compose run --rm \
            -v "$DATA_DIR/sqlite:/data" \
            -v "$CONFIG_DIR:/app/config" \
            etl-migration \
            python -m stockula.etl.cli migrate \
            --config /app/config/etl_config.yaml \
            --log-level INFO

        success "Data migration completed"
    else
        warn "No SQLite database found, skipping migration"
    fi
}

# Start streaming services
start_streaming() {
    log "Starting streaming services..."

    cd "$DEPLOY_DIR"

    # Start streaming pipeline
    docker compose up -d etl-streaming

    # Start monitoring
    docker compose up -d etl-monitoring

    success "Streaming services started"
}

# Run optimization
run_optimization() {
    log "Running database optimization..."

    cd "$DEPLOY_DIR"

    # Run optimization using the CLI
    docker compose run --rm \
        -v "$CONFIG_DIR:/app/config" \
        etl-migration \
        python -m stockula.etl.cli optimize \
        --config /app/config/etl_config.yaml \
        --full \
        --log-level INFO

    success "Database optimization completed"
}

# Setup monitoring dashboards (optional)
setup_monitoring() {
    log "Setting up monitoring dashboards..."

    read -p "Do you want to set up Grafana and Prometheus monitoring? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$DEPLOY_DIR"
        docker compose --profile monitoring up -d grafana prometheus
        success "Monitoring dashboards started"
        success "Grafana: http://localhost:3000 (admin/grafana_secure_password_change_me)"
        success "Prometheus: http://localhost:9090"
    fi
}

# Setup admin tools (optional)
setup_admin_tools() {
    log "Setting up admin tools..."

    read -p "Do you want to set up pgAdmin for database administration? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$DEPLOY_DIR"
        docker compose --profile admin up -d pgadmin
        success "pgAdmin started: http://localhost:8081"
    fi
}

# Health check
health_check() {
    log "Performing health check..."

    cd "$DEPLOY_DIR"

    # Check TimescaleDB
    if docker compose exec -T timescaledb pg_isready -U postgres -d stockula; then
        success "TimescaleDB: Healthy"
    else
        error "TimescaleDB: Unhealthy"
    fi

    # Check Redis
    if docker compose exec -T redis redis-cli ping | grep -q "PONG"; then
        success "Redis: Healthy"
    else
        error "Redis: Unhealthy"
    fi

    # Check ETL services
    if docker compose ps etl-streaming | grep -q "Up"; then
        success "ETL Streaming: Running"
    else
        warn "ETL Streaming: Not running"
    fi

    if docker compose ps etl-monitoring | grep -q "Up"; then
        success "ETL Monitoring: Running"
        success "Monitoring dashboard: http://localhost:8080"
    else
        warn "ETL Monitoring: Not running"
    fi
}

# Show status and next steps
show_status() {
    log "Deployment completed!"
    echo
    echo "=== Stockula ETL Pipeline Status ==="
    echo
    success "Core Services:"
    success "  - TimescaleDB: localhost:5432"
    success "  - Redis: localhost:6379"
    success "  - ETL Monitoring: http://localhost:8080"
    echo
    success "Data Directory: $DATA_DIR"
    success "Logs Directory: $LOGS_DIR"
    success "Config Directory: $CONFIG_DIR"
    echo
    success "Next Steps:"
    success "  1. Change default passwords in $DEPLOY_DIR/.env"
    success "  2. Review configuration in $CONFIG_DIR/etl_config.yaml"
    success "  3. Monitor services with: docker compose logs -f"
    success "  4. Check ETL status with CLI commands"
    echo
    success "CLI Examples:"
    success "  - Check status: docker compose run --rm etl-migration python -m stockula.etl.cli status"
    success "  - Monitor metrics: curl http://localhost:8080/health"
    success "  - View logs: docker compose logs etl-streaming"
}

# Cleanup function for errors
cleanup_on_error() {
    error "Deployment failed. Cleaning up..."
    cd "$DEPLOY_DIR"
    docker compose down
    exit 1
}

# Main deployment function
main() {
    echo "===================================================="
    echo "  Stockula ETL Pipeline Deployment"
    echo "  Production-Ready Data Migration & Streaming"
    echo "===================================================="
    echo

    # Set up error handling
    trap cleanup_on_error ERR

    # Run deployment steps
    check_prerequisites
    setup_directories
    generate_configs
    copy_sqlite_data
    build_images
    start_core_services

    # Run migration if data exists
    if [ -f "$DATA_DIR/sqlite/stockula.db" ]; then
        run_migration
        run_optimization
    fi

    start_streaming
    setup_monitoring
    setup_admin_tools
    health_check
    show_status
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "start")
        cd "$DEPLOY_DIR"
        docker compose up -d
        health_check
        ;;
    "stop")
        cd "$DEPLOY_DIR"
        docker compose down
        ;;
    "restart")
        cd "$DEPLOY_DIR"
        docker compose restart
        health_check
        ;;
    "logs")
        cd "$DEPLOY_DIR"
        docker compose logs -f "${2:-}"
        ;;
    "status")
        cd "$DEPLOY_DIR"
        docker compose ps
        health_check
        ;;
    "clean")
        read -p "This will remove all containers and volumes. Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$DEPLOY_DIR"
            docker compose down -v
            docker system prune -f
            success "Cleanup completed"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  deploy   - Full deployment (default)"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs (optionally specify service)"
        echo "  status   - Show service status"
        echo "  clean    - Remove all containers and volumes"
        echo "  help     - Show this help"
        ;;
    *)
        error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac

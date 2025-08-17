#!/bin/bash
set -e

# TimescaleDB optimized entrypoint script for Stockula ETL

# Source the original entrypoint functions
source /usr/local/bin/docker-entrypoint.sh

# Custom initialization function
stockula_init() {
    echo "Initializing TimescaleDB for Stockula ETL..."

    # Copy optimized configuration files
    if [ -f /tmp/postgresql.conf ]; then
        echo "Applying custom PostgreSQL configuration..."
        cp /tmp/postgresql.conf "$PGDATA/postgresql.conf"
    fi

    if [ -f /tmp/pg_hba.conf ]; then
        echo "Applying custom pg_hba configuration..."
        cp /tmp/pg_hba.conf "$PGDATA/pg_hba.conf"
    fi

    # Set optimal kernel parameters if running with privileges
    if [ "$(id -u)" = "0" ]; then
        echo "Setting kernel parameters for database performance..."

        # Shared memory settings
        echo "kernel.shmmax = 68719476736" >> /etc/sysctl.conf  # 64GB
        echo "kernel.shmall = 4294967296" >> /etc/sysctl.conf   # 16TB

        # Network settings for high connection loads
        echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
        echo "net.ipv4.tcp_max_syn_backlog = 1024" >> /etc/sysctl.conf

        # Apply settings (may fail in containers, that's OK)
        sysctl -p 2>/dev/null || true
    fi
}

# Custom post-init function for database setup
stockula_post_init() {
    echo "Running Stockula-specific database setup..."

    # Wait for PostgreSQL to be ready
    until pg_isready -U postgres -d postgres; do
        echo "Waiting for PostgreSQL to be ready..."
        sleep 2
    done

    # Run optimization scripts if they exist
    if [ -d /opt/optimization ]; then
        echo "Running optimization scripts..."
        for script in /opt/optimization/*.sql; do
            if [ -f "$script" ]; then
                echo "Executing $(basename "$script")..."
                psql -U postgres -d stockula -f "$script" || echo "Warning: Script $(basename "$script") failed"
            fi
        done
    fi

    echo "Stockula TimescaleDB initialization completed!"
}

# Override the main function to include our custom setup
main() {
    # If we're not the postgres user, delegate to it
    if [ "$1" = 'postgres' ] && [ "$(id -u)" = '0' ]; then
        stockula_init
        chown -R postgres:postgres "$PGDATA"
        exec gosu postgres "$BASH_SOURCE" "$@"
    fi

    # If we're the postgres user (or were delegated to postgres)
    if [ "$1" = 'postgres' ]; then
        # Run the original initialization
        docker_setup_env
        docker_create_db_directories

        # Initialize the database if needed
        if [ ! -s "$PGDATA/PG_VERSION" ]; then
            docker_verify_minimum_env
            docker_init_database_dir
            pg_setup_hba_conf

            # Start postgres temporarily for setup
            docker_temp_server_start "$@"
            docker_setup_db
            docker_process_init_files /docker-entrypoint-initdb.d/*

            # Run our custom post-init
            stockula_post_init

            docker_temp_server_stop
        fi

        # Start the permanent server
        exec "$@"
    fi

    # For non-postgres commands, just execute them
    exec "$@"
}

# Run the main function with all arguments
main "$@"

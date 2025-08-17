# TimescaleDB Docker image optimized for Stockula ETL
FROM timescale/timescaledb:latest-pg15

# Install additional extensions and tools
RUN apt-get update && apt-get install -y \
    postgresql-15-cron \
    postgresql-15-postgis-3 \
    postgresql-contrib-15 \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Copy custom PostgreSQL configuration
COPY postgresql.conf /tmp/postgresql.conf
COPY pg_hba.conf /tmp/pg_hba.conf

# Copy initialization scripts
COPY init-scripts/ /docker-entrypoint-initdb.d/

# Copy optimization scripts
COPY optimization/ /opt/optimization/

# Set environment variables for better performance
ENV POSTGRES_INITDB_ARGS="--data-checksums"
ENV POSTGRES_HOST_AUTH_METHOD=md5
ENV TIMESCALEDB_TELEMETRY=off

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres -d stockula

# Use custom entrypoint for additional setup
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["postgres"]

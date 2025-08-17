# TimescaleDB Quick Start Guide for Stockula

This guide provides a quick setup and reference for the TimescaleDB deployment.

## 🚀 Quick Setup

### 1. Basic Setup (Database Only)

```bash
./scripts/setup-timescaledb.sh
```

### 2. Full Setup (Database + Monitoring)

```bash
./scripts/setup-timescaledb.sh --monitoring
```

### 3. Complete Setup (Everything)

```bash
./scripts/setup-timescaledb.sh --monitoring --app
```

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- 10GB free disk space

## 🔧 Configuration

Create `.env` file or set environment variables:

```bash
# Required
POSTGRES_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_grafana_password

# Optional (with defaults)
TS_TUNE_MEMORY=4GB
TS_TUNE_NUM_CPUS=4
BACKUP_RETENTION_DAYS=30
```

## 📊 Access Points

| Service           | URL                                                      | Credentials       |
| ----------------- | -------------------------------------------------------- | ----------------- |
| Database (Direct) | `postgresql://stockula:password@localhost:5432/stockula` | stockula/password |
| Database (Pooled) | `postgresql://stockula:password@localhost:6432/stockula` | stockula/password |
| Grafana           | http://localhost:3000                                    | admin/password    |
| Prometheus        | http://localhost:9090                                    | No auth           |
| Redis             | redis://localhost:6379                                   | No auth           |

## 🗂 File Structure

```
database/timescale/
├── init/              # Database initialization scripts
│   ├── 001-extensions.sql
│   ├── 002-schema.sql
│   ├── 003-indexes.sql
│   └── 004-policies.sql
├── config/            # PostgreSQL configuration
│   └── postgresql.conf
├── pgbouncer/         # Connection pooling config
│   ├── pgbouncer.ini
│   └── userlist.txt
├── backup/            # Backup and maintenance scripts
│   ├── backup.sh
│   ├── restore.sh
│   └── maintenance.sh
├── prometheus/        # Monitoring configuration
│   ├── prometheus.yml
│   └── alert_rules.yml
├── grafana/           # Dashboard configuration
│   ├── provisioning/
│   └── dashboards/
└── README.md         # Detailed documentation
```

## 🛠 Common Operations

### Service Management

```bash
# Start all services
docker-compose -f docker-compose.timescale.yml up -d

# Stop all services
docker-compose -f docker-compose.timescale.yml down

# View logs
docker-compose -f docker-compose.timescale.yml logs -f

# Check status
docker-compose -f docker-compose.timescale.yml ps
```

### Database Operations

```bash
# Connect to database
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula

# Run SQL file
docker-compose -f docker-compose.timescale.yml exec -T timescaledb psql -U stockula -d stockula < your_file.sql

# Check hypertables
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "SELECT * FROM timescaledb_information.hypertables;"
```

### Backup Operations

```bash
# Manual backup
docker-compose -f docker-compose.timescale.yml run --rm backup

# Force full backup
FORCE_FULL_BACKUP=true docker-compose -f docker-compose.timescale.yml run --rm backup

# Restore backup
./database/timescale/backup/restore.sh /backups/full/stockula_backup_YYYYMMDD_HHMMSS.dump

# Verify backup
./database/timescale/backup/restore.sh --verify-only /path/to/backup.dump
```

### Maintenance Operations

```bash
# Run all maintenance
./database/timescale/backup/maintenance.sh

# Specific tasks
./database/timescale/backup/maintenance.sh vacuum
./database/timescale/backup/maintenance.sh statistics
./database/timescale/backup/maintenance.sh health
```

## 📈 Key Features

### Hypertables (Auto-partitioned)

- `price_history` - Partitioned by time (1 month chunks)
- `dividends` - Partitioned by time (1 year chunks)
- `splits` - Partitioned by time (1 year chunks)
- `options_calls` - Partitioned by time (3 month chunks)
- `options_puts` - Partitioned by time (3 month chunks)

### Compression (Automatic)

- Price data: Compressed after 2 years
- Options data: Compressed after 3 months
- Dividends/Splits: Compressed after 1 year

### Continuous Aggregates (Pre-computed)

- `price_history_daily` - Daily OHLCV
- `price_history_weekly` - Weekly OHLCV
- `price_history_monthly` - Monthly OHLCV
- `market_volume_hourly` - Market volume analysis
- `options_volume_daily` - Options activity

### Data Retention

- Price data: 10 years
- Options data: 3 years
- Dividends/Splits: 20 years

## 🚨 Troubleshooting

### Common Issues

**Connection refused:**

```bash
# Check if services are running
docker-compose -f docker-compose.timescale.yml ps

# Check logs
docker-compose -f docker-compose.timescale.yml logs timescaledb
```

**Out of memory:**

```bash
# Reduce memory settings in .env
TS_TUNE_MEMORY=2GB

# Restart services
docker-compose -f docker-compose.timescale.yml restart
```

**Disk space issues:**

```bash
# Check disk usage
df -h

# Clean old backups
find ./backups -name "*.dump" -mtime +30 -delete

# Compress old chunks
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "SELECT compress_chunk(i) FROM show_chunks('price_history', older_than => INTERVAL '1 month') i;"
```

### Health Checks

```bash
# Database connectivity
docker-compose -f docker-compose.timescale.yml exec timescaledb pg_isready -U stockula -d stockula

# Performance check
./database/timescale/backup/maintenance.sh health

# Monitor metrics
curl http://localhost:9090/metrics
```

## 🔐 Security Notes

### Production Checklist

- [ ] Change default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Set up proper authentication
- [ ] Enable audit logging
- [ ] Configure backup encryption
- [ ] Review user permissions

### Connection Security

```bash
# Create read-only user
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula -c "
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE stockula TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
"
```

## 📚 Additional Resources

- [Complete Documentation](database/timescale/README.md)
- [TimescaleDB Docs](https://docs.timescale.com/)
- [PostgreSQL Performance](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)

## 🆘 Support

For issues and questions:

1. Check the logs: `docker-compose -f docker-compose.timescale.yml logs`
1. Review the detailed documentation: `database/timescale/README.md`
1. Run health checks: `./database/timescale/backup/maintenance.sh health`
1. Check system resources: `docker stats`

______________________________________________________________________

**Quick Commands Reference:**

```bash
# Setup
./scripts/setup-timescaledb.sh --monitoring

# Status
docker-compose -f docker-compose.timescale.yml ps

# Backup
docker-compose -f docker-compose.timescale.yml run --rm backup

# Maintenance
./database/timescale/backup/maintenance.sh

# Connect
docker-compose -f docker-compose.timescale.yml exec timescaledb psql -U stockula -d stockula
```

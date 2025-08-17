# TimescaleDB Production Runbook for Stockula

## 🚨 Emergency Contact Information

- **Database Administrator**: admin@stockula.com
- **Platform Engineer**: ops@stockula.com
- **On-Call Phone**: +1-XXX-XXX-XXXX
- **Slack Channel**: #stockula-production-alerts

## 📋 Quick Reference

### Essential Connection Information

```bash
# Database Direct Connection
postgresql://stockula:${POSTGRES_PASSWORD}@localhost:5432/stockula

# PgBouncer Connection (RECOMMENDED for applications)
postgresql://stockula:${POSTGRES_PASSWORD}@localhost:6432/stockula

# Monitoring
Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})
Prometheus: http://localhost:9090
```

### Critical File Locations

```bash
/backups/                     # Backup storage
/var/log/stockula/           # Application logs
./database/timescale/config/ # Configuration files
./database/timescale/backup/ # Backup scripts
```

## 🚨 Critical Emergency Procedures

### 1. Database Down - CRITICAL

**Symptoms:**

- Applications cannot connect to database
- No response from port 5432
- Prometheus alerts: `TimescaleDBDown`

**Immediate Actions:**

```bash
# Check container status
docker-compose -f docker-compose.timescale.yml ps

# Check database logs
docker-compose -f docker-compose.timescale.yml logs timescaledb

# Restart database if needed
docker-compose -f docker-compose.timescale.yml restart timescaledb

# Monitor startup
docker-compose -f docker-compose.timescale.yml logs -f timescaledb
```

**Verification:**

```bash
# Test connectivity
pg_isready -h localhost -p 5432 -U stockula -d stockula

# Test query execution
psql -h localhost -p 5432 -U stockula -d stockula -c "SELECT 1;"
```

### 2. Connection Pool Exhaustion - CRITICAL

**Symptoms:**

- Applications getting connection refused errors
- Alert: `DatabaseConnectionsCritical`
- PgBouncer showing max connections

**Immediate Actions:**

```bash
# Check current connections
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT count(*), state
  FROM pg_stat_activity
  GROUP BY state;"

# Kill idle connections
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE state = 'idle'
  AND now() - state_change > interval '30 minutes';"

# Restart PgBouncer if needed
docker-compose -f docker-compose.timescale.yml restart pgbouncer
```

### 3. Disk Space Full - CRITICAL

**Symptoms:**

- Database writes failing
- Backup failures
- WAL archive failures

**Immediate Actions:**

```bash
# Check disk usage
df -h /
du -sh /backups/*

# Emergency cleanup - OLD WAL files
find /backups/archived_wal -name "*.gz" -mtime +7 -delete

# Emergency cleanup - OLD backups (keep last 7 days)
find /backups/full -name "stockula_backup_*" -mtime +7 -delete
find /backups/incremental -name "stockula_backup_*" -mtime +7 -delete

# Check database size
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT pg_size_pretty(pg_database_size('stockula'));"
```

### 4. Performance Degradation - HIGH

**Symptoms:**

- Slow query alerts
- High CPU/memory usage
- Application timeouts

**Immediate Actions:**

```bash
# Check active queries
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT pid, now() - pg_stat_activity.query_start AS duration, query
  FROM pg_stat_activity
  WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# Kill long-running queries if needed
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE now() - pg_stat_activity.query_start > interval '15 minutes'
  AND state = 'active';"

# Check for locks
psql -h localhost -p 5432 -U stockula -d stockula -c "
  SELECT * FROM pg_locks WHERE NOT granted;"

# Run immediate maintenance
./database/timescale/maintenance/production-cron-jobs.sh emergency
```

## 📊 Health Checks

### Daily Health Check Script

```bash
#!/bin/bash
# Run this every morning to check system health

echo "=== Stockula Database Health Check ==="
echo "Date: $(date)"
echo

# Database connectivity
echo "Database Connectivity:"
if pg_isready -h localhost -p 5432 -U stockula -d stockula; then
    echo "✓ Database is accepting connections"
else
    echo "✗ Database connection failed"
fi

# Connection count
CONNECTIONS=$(psql -h localhost -p 5432 -U stockula -d stockula -t -c "SELECT count(*) FROM pg_stat_activity;")
echo "Active connections: $CONNECTIONS"

# Database size
DB_SIZE=$(psql -h localhost -p 5432 -U stockula -d stockula -t -c "SELECT pg_size_pretty(pg_database_size('stockula'));")
echo "Database size: $DB_SIZE"

# Recent data
RECENT_DATA=$(psql -h localhost -p 5432 -U stockula -d stockula -t -c "SELECT count(*) FROM price_history WHERE time >= CURRENT_DATE;")
echo "Today's price records: $RECENT_DATA"

# Backup status
LAST_BACKUP=$(find /backups -name "stockula_backup_*" -type f | sort | tail -1)
if [[ -n "$LAST_BACKUP" ]]; then
    echo "Last backup: $(basename "$LAST_BACKUP")"
else
    echo "✗ No backups found"
fi
```

### Performance Monitoring Queries

**Top 10 Slowest Queries:**

```sql
SELECT
  calls,
  mean_exec_time,
  total_exec_time,
  LEFT(query, 100) as query_preview
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Connection Analysis:**

```sql
SELECT
  application_name,
  state,
  count(*) as connection_count
FROM pg_stat_activity
GROUP BY application_name, state
ORDER BY connection_count DESC;
```

**Table Sizes:**

```sql
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
```

## 🔄 Backup and Recovery

### Daily Backup Verification

```bash
# Verify latest backup
./database/timescale/backup/verify-backup.sh full

# Test restore (non-destructive)
./database/timescale/backup/verify-backup.sh full
```

### Emergency Restore Procedure

**WARNING: This will replace the current database!**

```bash
# 1. Stop all applications
docker-compose -f docker-compose.timescale.yml stop stockula-app

# 2. Create backup of current state
pg_dump -h localhost -p 5432 -U stockula -d stockula > emergency_backup_$(date +%Y%m%d_%H%M%S).sql

# 3. Find restore file
RESTORE_FILE=$(find /backups/full -name "stockula_backup_*.dump" | sort | tail -1)

# 4. Drop and recreate database
psql -h localhost -p 5432 -U stockula -d postgres -c "DROP DATABASE stockula;"
psql -h localhost -p 5432 -U stockula -d postgres -c "CREATE DATABASE stockula;"

# 5. Enable extensions
psql -h localhost -p 5432 -U stockula -d stockula -c "CREATE EXTENSION timescaledb;"

# 6. Restore data
pg_restore -h localhost -p 5432 -U stockula -d stockula --verbose "$RESTORE_FILE"

# 7. Verify restore
psql -h localhost -p 5432 -U stockula -d stockula -c "SELECT count(*) FROM price_history;"

# 8. Start applications
docker-compose -f docker-compose.timescale.yml start stockula-app
```

## 🔧 Maintenance Procedures

### Weekly Maintenance Checklist

**Every Sunday at 3 AM (automated):**

- [ ] Full database backup
- [ ] VACUUM and ANALYZE all tables
- [ ] Reindex critical tables
- [ ] Clean up old backups (>30 days)
- [ ] Review slow query log
- [ ] Check TimescaleDB compression status

**Manual Monthly Tasks:**

- [ ] Review and rotate logs
- [ ] Update monitoring dashboards
- [ ] Test disaster recovery procedure
- [ ] Review capacity planning metrics
- [ ] Update documentation

### Manual Maintenance Commands

**Force Full Backup:**

```bash
FORCE_FULL_BACKUP=true ./database/timescale/backup/backup.sh
```

**Manual VACUUM:**

```bash
./database/timescale/maintenance/production-cron-jobs.sh daily
```

**Reindex All Tables:**

```bash
./database/timescale/maintenance/production-cron-jobs.sh weekly
```

**Emergency Maintenance:**

```bash
./database/timescale/maintenance/production-cron-jobs.sh emergency
```

## 📈 Capacity Planning

### Growth Monitoring

**Daily Growth Rate:**

```sql
SELECT
  pg_size_pretty(pg_database_size('stockula')) as current_size,
  pg_size_pretty(
    pg_database_size('stockula') -
    lag(pg_database_size('stockula')) OVER (ORDER BY current_date)
  ) as daily_growth;
```

**Table Growth Analysis:**

```sql
SELECT
  schemaname,
  tablename,
  n_tup_ins as inserts_today,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_tup_ins DESC;
```

### Scaling Triggers

**Scale Up When:**

- Database size > 500GB
- Daily growth > 10GB/day
- Connection utilization > 80%
- Average query time > 1 second

**Scale Out When:**

- Read replicas needed for analytics
- Geographic distribution required
- Hot data separation needed

## 🚨 Alert Response Procedures

### Critical Alerts (Page Immediately)

#### TimescaleDBDown

1. Check container status
1. Review database logs
1. Restart if needed
1. Verify connectivity
1. Check for data corruption

#### DatabaseConnectionsCritical

1. Kill idle connections
1. Restart PgBouncer
1. Scale connection limits if needed
1. Review application connection patterns

#### CriticallyStaleStockData

1. Check data pipeline status
1. Review API limits/errors
1. Manual data fetch if needed
1. Verify data sources

### Warning Alerts (Review within 1 hour)

#### DataFreshnessSLAViolation

1. Check ETL pipeline logs
1. Verify external data sources
1. Review API rate limits
1. Consider manual intervention

#### HighZeroVolumeData

1. Validate data source quality
1. Check market hours/holidays
1. Review data transformation logic

## 🔍 Troubleshooting Guide

### Common Issues

#### "Too many connections" Error

```bash
# Current connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Kill idle connections
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle';"

# Increase max_connections if needed (requires restart)
# Edit postgresql.conf: max_connections = 500
```

#### Slow Queries

```bash
# Find slow queries
psql -c "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5;"

# Check for missing indexes
psql -c "SELECT schemaname, tablename, attname, n_distinct, correlation FROM pg_stats WHERE schemaname = 'public';"
```

#### High Disk Usage

```bash
# Find largest tables
psql -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Clean up old data (BE CAREFUL!)
# Only run after confirming with team
psql -c "DELETE FROM price_history WHERE time < '2020-01-01';"
```

### Recovery Time Objectives (RTO)

- **Database restart**: < 5 minutes
- **Full restore from backup**: < 2 hours
- **Point-in-time recovery**: < 4 hours
- **Disaster recovery**: < 24 hours

### Recovery Point Objectives (RPO)

- **Regular operations**: < 15 minutes (incremental backups)
- **Disaster scenarios**: < 24 hours (daily full backups)

## 📞 Escalation Matrix

| Severity | First Contact | Escalation (15 min) | Escalation (30 min) |
| -------- | ------------- | ------------------- | ------------------- |
| Critical | On-call DBA   | Platform Engineer   | CTO                 |
| High     | Team Lead     | Platform Engineer   | Engineering Manager |
| Medium   | Developer     | Team Lead           | Platform Engineer   |
| Low      | Ticket        | Team Lead           | -                   |

## 📚 Additional Resources

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [PgBouncer Documentation](https://www.pgbouncer.org/)
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)

______________________________________________________________________

**Last Updated**: $(date) **Version**: 1.0 **Review Date**: Monthly

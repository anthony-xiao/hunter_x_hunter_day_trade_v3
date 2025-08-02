# PostgreSQL UTC Configuration Guide

This guide provides three comprehensive methods to permanently configure PostgreSQL to use UTC timezone instead of local time. This is essential for applications that handle time-sensitive data across different timezones.

## 🎯 Overview

The timezone mismatch issue occurs when:
- PostgreSQL database uses local timezone (e.g., Asia/Shanghai)
- Application stores UTC timestamps
- Results in 8-hour shifts and data inconsistencies

## 📁 Files Created

- `configure_postgresql_utc.py` - Comprehensive Python script implementing all methods
- `set_postgresql_timezone_utc.sql` - SQL commands for immediate timezone changes
- `set_postgresql_env.sh` - Environment variable setup script
- `postgresql-utc.service` - Systemd service file (Linux)
- `.env.postgresql` - Docker environment file
- `com.postgresql.utc.plist` - macOS launchd configuration

## 🚀 Quick Start

### Option 1: Run All Methods (Recommended)
```bash
cd backend
python3 configure_postgresql_utc.py
```

### Option 2: Run Individual Methods
```bash
# Method 1: Configuration file
python3 configure_postgresql_utc.py 1

# Method 2: SQL commands
python3 configure_postgresql_utc.py 2

# Method 3: Environment variables
python3 configure_postgresql_utc.py 3

# Verify configuration
python3 configure_postgresql_utc.py verify
```

## 📋 Method Details

### Method 1: Configuration File Method

**Modifying postgresql.conf with timezone = 'UTC'**

```bash
# Automatic configuration
python3 configure_postgresql_utc.py 1
```

**Manual Steps:**
1. Locate postgresql.conf file:
   ```bash
   # Common locations:
   # macOS (Homebrew): /usr/local/var/postgres/postgresql.conf
   # macOS (Apple Silicon): /opt/homebrew/var/postgres/postgresql.conf
   # Linux: /var/lib/postgresql/data/postgresql.conf
   # Ubuntu/Debian: /etc/postgresql/*/main/postgresql.conf
   ```

2. Edit the file:
   ```bash
   sudo nano /path/to/postgresql.conf
   ```

3. Add or modify the timezone setting:
   ```
   timezone = 'UTC'
   ```

4. Restart PostgreSQL:
   ```bash
   # macOS (Homebrew)
   brew services restart postgresql
   
   # Linux (systemd)
   sudo systemctl restart postgresql
   
   # Manual restart
   pg_ctl restart -D /path/to/data/directory
   ```

### Method 2: SQL Commands

**Using ALTER SYSTEM SET timezone = 'UTC' for immediate changes**

```bash
# Run SQL script
psql -f set_postgresql_timezone_utc.sql
```

**Manual SQL Commands:**
```sql
-- Check current timezone
SHOW timezone;

-- Set timezone to UTC
ALTER SYSTEM SET timezone = 'UTC';

-- Reload configuration (no restart required)
SELECT pg_reload_conf();

-- Verify change
SHOW timezone;
```

### Method 3: Environment Variables

**Setting PGTZ=UTC and TZ=UTC**

```bash
# Run environment setup script
./set_postgresql_env.sh
```

**Manual Steps:**

1. **Current Session:**
   ```bash
   export PGTZ=UTC
   export TZ=UTC
   ```

2. **Permanent (Shell Profile):**
   ```bash
   # For Zsh
   echo 'export PGTZ=UTC' >> ~/.zshrc
   echo 'export TZ=UTC' >> ~/.zshrc
   source ~/.zshrc
   
   # For Bash
   echo 'export PGTZ=UTC' >> ~/.bashrc
   echo 'export TZ=UTC' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **System-wide (Linux):**
   ```bash
   echo 'PGTZ=UTC' | sudo tee -a /etc/environment
   echo 'TZ=UTC' | sudo tee -a /etc/environment
   ```

4. **Docker:**
   ```bash
   # Use environment file
   docker run --env-file .env.postgresql postgres
   
   # Or in docker-compose.yml
   services:
     postgres:
       image: postgres:latest
       environment:
         - PGTZ=UTC
         - TZ=UTC
         - POSTGRES_INITDB_ARGS=--timezone=UTC
   ```

5. **macOS (launchd):**
   ```bash
   cp com.postgresql.utc.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.postgresql.utc.plist
   ```

6. **Linux (systemd):**
   ```bash
   sudo cp postgresql-utc.service /etc/systemd/system/
   sudo systemctl enable postgresql-utc.service
   sudo systemctl start postgresql-utc.service
   ```

## ✅ Verification

### Check PostgreSQL Timezone
```sql
-- Check current timezone setting
SHOW timezone;

-- Check current timestamps
SELECT 
    NOW() as local_time,
    NOW() AT TIME ZONE 'UTC' as utc_time;

-- Check timezone offset
SELECT EXTRACT(timezone FROM NOW()) as timezone_offset_seconds;
```

### Check Environment Variables
```bash
echo $PGTZ
echo $TZ
```

### Test Timestamp Storage
```sql
-- Create test table
CREATE TEMPORARY TABLE test_timestamps (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert test data
INSERT INTO test_timestamps DEFAULT VALUES;

-- Check stored timestamp
SELECT created_at FROM test_timestamps;
```

## 🔧 Troubleshooting

### Common Issues

1. **Permission Denied (postgresql.conf)**
   ```bash
   # Run with appropriate permissions
   sudo python3 configure_postgresql_utc.py 1
   ```

2. **PostgreSQL Connection Failed**
   ```bash
   # Check PostgreSQL is running
   pg_isready
   
   # Check connection parameters
   export DB_USER=your_username
   export DB_PASSWORD=your_password
   ```

3. **Configuration Not Taking Effect**
   ```bash
   # Restart PostgreSQL
   brew services restart postgresql  # macOS
   sudo systemctl restart postgresql  # Linux
   ```

4. **Environment Variables Not Persistent**
   ```bash
   # Check shell profile
   cat ~/.zshrc | grep -E '(PGTZ|TZ)'
   
   # Reload profile
   source ~/.zshrc
   ```

### Verification Commands

```bash
# Check PostgreSQL timezone
psql -c "SHOW timezone;"

# Check environment variables
env | grep -E '(PGTZ|TZ)'

# Check PostgreSQL configuration
psql -c "SELECT name, setting FROM pg_settings WHERE name LIKE '%timezone%';"

# Test timestamp consistency
psql -c "SELECT NOW(), NOW() AT TIME ZONE 'UTC';"
```

## 🎯 Best Practices

### Application-Level Configuration

1. **Python/SQLAlchemy:**
   ```python
   from datetime import datetime, timezone
   import os
   
   # Set environment variables
   os.environ['TZ'] = 'UTC'
   
   # Use UTC timestamps
   utc_now = datetime.now(timezone.utc)
   
   # SQLAlchemy engine with timezone
   engine = create_engine(
       'postgresql://user:pass@host/db',
       connect_args={'options': '-c timezone=utc'}
   )
   ```

2. **Database Connections:**
   ```python
   # Always specify timezone in connection
   conn = psycopg2.connect(
       host='localhost',
       database='mydb',
       user='user',
       password='pass',
       options='-c timezone=utc'
   )
   ```

3. **Timestamp Handling:**
   ```python
   # Always use timezone-aware timestamps
   from datetime import datetime, timezone
   
   # Store timestamps in UTC
   timestamp = datetime.now(timezone.utc)
   
   # Convert to UTC before storing
   if timestamp.tzinfo is None:
       timestamp = timestamp.replace(tzinfo=timezone.utc)
   elif timestamp.tzinfo != timezone.utc:
       timestamp = timestamp.astimezone(timezone.utc)
   ```

### Data Migration

If you have existing data with timezone issues:

```sql
-- Backup existing data
CREATE TABLE market_data_backup AS SELECT * FROM market_data;

-- Update timestamps to UTC (adjust offset as needed)
UPDATE market_data 
SET timestamp = timestamp AT TIME ZONE 'Asia/Shanghai' AT TIME ZONE 'UTC'
WHERE timestamp IS NOT NULL;

-- Verify the conversion
SELECT 
    original.timestamp as original_timestamp,
    updated.timestamp as updated_timestamp
FROM market_data_backup original
JOIN market_data updated ON original.id = updated.id
LIMIT 10;
```

## 📊 Expected Results

After successful configuration:

1. **PostgreSQL timezone setting:** `UTC`
2. **Environment variables:** `PGTZ=UTC`, `TZ=UTC`
3. **Timestamp consistency:** No 8-hour offset between stored and displayed times
4. **Market data alignment:** US market hours (14:30-21:00 UTC) display correctly

## 🔄 Maintenance

### Regular Checks
```bash
# Weekly verification script
#!/bin/bash
echo "PostgreSQL Timezone Check - $(date)"
psql -c "SHOW timezone;"
psql -c "SELECT NOW(), NOW() AT TIME ZONE 'UTC';"
echo "Environment: PGTZ=$PGTZ, TZ=$TZ"
```

### Monitoring
```sql
-- Monitor timezone consistency
SELECT 
    'timezone_check' as check_type,
    current_setting('timezone') as current_timezone,
    CASE 
        WHEN current_setting('timezone') = 'UTC' THEN 'OK'
        ELSE 'ISSUE'
    END as status;
```

## 📞 Support

If you encounter issues:

1. Run the verification script: `python3 configure_postgresql_utc.py verify`
2. Check PostgreSQL logs for timezone-related errors
3. Verify environment variables are set correctly
4. Ensure PostgreSQL has been restarted after configuration changes

---

**Note:** This configuration ensures consistent UTC timezone handling across your PostgreSQL database, preventing timestamp mismatches and data inconsistencies in your trading system.
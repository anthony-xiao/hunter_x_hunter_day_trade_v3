# PostgreSQL UTC Timezone Configuration Guide

This guide shows how to permanently configure PostgreSQL to use UTC timezone instead of local time.

## Method 1: PostgreSQL Configuration File (Recommended)

### Step 1: Locate postgresql.conf

Find your PostgreSQL configuration file:

```bash
# Find the config file location
psql -c "SHOW config_file;"

# Common locations:
# macOS (Homebrew): /usr/local/var/postgres/postgresql.conf
# macOS (Postgres.app): ~/Library/Application Support/Postgres/var-XX/postgresql.conf
# Linux: /etc/postgresql/XX/main/postgresql.conf
# Windows: C:\Program Files\PostgreSQL\XX\data\postgresql.conf
```

### Step 2: Edit postgresql.conf

Open the configuration file and find the `timezone` setting:

```bash
# Edit the config file
sudo nano /path/to/postgresql.conf

# Or use your preferred editor
sudo vim /path/to/postgresql.conf
```

Find and modify these lines:

```conf
# LOCALE AND FORMATTING
#------------------------------------------------------------------------------

# These settings are initialized by initdb, but they can be changed.
timezone = 'UTC'                        # Set to UTC
#timezone_abbreviations = 'Default'     # Select the set of available time zone
#extra_float_digits = 1                 # min -15, max 3; any value >0 actually
#client_encoding = sql_ascii            # actually, defaults to database

# These settings are initialized by initdb and cannot be changed.
#lc_messages = 'en_US.UTF-8'            # locale for system error message
#lc_monetary = 'en_US.UTF-8'            # locale for monetary formatting
#lc_numeric = 'en_US.UTF-8'             # locale for number formatting
#lc_time = 'en_US.UTF-8'                # locale for time formatting

# default configuration for text search
#default_text_search_config = 'pg_catalog.english'

# Also set log timezone to UTC for consistency
log_timezone = 'UTC'
```

### Step 3: Restart PostgreSQL

```bash
# macOS (Homebrew)
brew services restart postgresql

# macOS (Postgres.app)
# Stop and start from the app interface

# Linux (systemd)
sudo systemctl restart postgresql

# Linux (older systems)
sudo service postgresql restart

# Windows
# Use Services.msc to restart PostgreSQL service
```

## Method 2: SQL Commands (Session/Database Level)

### Set for Current Session

```sql
-- Set timezone for current session
SET timezone = 'UTC';

-- Verify the change
SHOW timezone;
SELECT now();
```

### Set for Specific Database

```sql
-- Set timezone for a specific database
ALTER DATABASE your_database_name SET timezone = 'UTC';

-- Verify the change
\c your_database_name
SHOW timezone;
```

### Set for Specific User

```sql
-- Set timezone for a specific user
ALTER USER your_username SET timezone = 'UTC';

-- Verify the change
SHOW timezone;
```

### Set Globally for All New Connections

```sql
-- Set timezone globally (requires superuser privileges)
ALTER SYSTEM SET timezone = 'UTC';

-- Reload configuration
SELECT pg_reload_conf();

-- Verify the change
SHOW timezone;
```

## Method 3: Environment Variables

### Set PGTZ Environment Variable

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export PGTZ='UTC'

# Or set for specific application
PGTZ='UTC' your_application

# For systemd services, add to service file:
# Environment="PGTZ=UTC"
```

### Set TZ System Environment Variable

```bash
# Set system timezone to UTC (affects all applications)
export TZ='UTC'

# Add to /etc/environment for system-wide setting
echo 'TZ=UTC' | sudo tee -a /etc/environment
```

## Method 4: Docker/Container Setup

### Docker Compose

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=your_db
      - POSTGRES_USER=your_user
      - POSTGRES_PASSWORD=your_password
      - PGTZ=UTC
      - TZ=UTC
    command: [
      "postgres",
      "-c", "timezone=UTC",
      "-c", "log_timezone=UTC"
    ]
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
```

### Dockerfile

```dockerfile
FROM postgres:15

# Set timezone environment variables
ENV TZ=UTC
ENV PGTZ=UTC

# Copy custom postgresql.conf with UTC timezone
COPY postgresql.conf /etc/postgresql/postgresql.conf

# Set timezone in the config
RUN echo "timezone = 'UTC'" >> /etc/postgresql/postgresql.conf && \
    echo "log_timezone = 'UTC'" >> /etc/postgresql/postgresql.conf
```

## Verification Commands

### Check Current Timezone Settings

```sql
-- Show current timezone
SHOW timezone;

-- Show log timezone
SHOW log_timezone;

-- Show current timestamp with timezone
SELECT now();
SELECT current_timestamp;

-- Show timezone offset
SELECT extract(timezone from now());

-- List all available timezones
SELECT name FROM pg_timezone_names WHERE name LIKE '%UTC%';
```

### Test Timestamp Behavior

```sql
-- Create test table
CREATE TABLE timezone_test (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT now(),
    created_at_tz TIMESTAMPTZ DEFAULT now()
);

-- Insert test data
INSERT INTO timezone_test DEFAULT VALUES;

-- Check the results
SELECT 
    created_at,
    created_at_tz,
    extract(timezone from created_at_tz) as tz_offset
FROM timezone_test;

-- Clean up
DROP TABLE timezone_test;
```

## Application-Level Configuration

### Python (psycopg2/asyncpg)

```python
import psycopg2
from datetime import datetime, timezone

# Set timezone in connection string
conn = psycopg2.connect(
    host="localhost",
    database="your_db",
    user="your_user",
    password="your_password",
    options="-c timezone=UTC"
)

# Or set after connection
with conn.cursor() as cur:
    cur.execute("SET timezone = 'UTC'")
    conn.commit()

# Always use timezone-aware datetime objects
utc_now = datetime.now(timezone.utc)
```

### Node.js (pg)

```javascript
const { Pool } = require('pg');

const pool = new Pool({
  host: 'localhost',
  database: 'your_db',
  user: 'your_user',
  password: 'your_password',
  // Set timezone in connection
  options: '-c timezone=UTC'
});

// Or set after connection
pool.query('SET timezone = \'UTC\'');
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have proper permissions to edit postgresql.conf
2. **Config Not Loading**: Verify the config file path and restart PostgreSQL
3. **Timezone Not Found**: Use `SELECT * FROM pg_timezone_names;` to see available timezones
4. **Mixed Timezones**: Ensure all applications use the same timezone setting

### Verify Configuration is Active

```sql
-- Check if configuration changes are active
SELECT name, setting, source 
FROM pg_settings 
WHERE name IN ('timezone', 'log_timezone');

-- Check configuration file location
SHOW config_file;

-- Check if pending restart is needed
SELECT name, setting, pending_restart 
FROM pg_settings 
WHERE pending_restart = true;
```

## Best Practices

1. **Always use UTC in the database** - Store all timestamps in UTC
2. **Convert at application layer** - Convert to local time only for display
3. **Use TIMESTAMPTZ** - Always use timezone-aware timestamp columns
4. **Consistent configuration** - Ensure all database instances use the same timezone
5. **Test thoroughly** - Verify timezone behavior after changes
6. **Document changes** - Keep track of timezone configuration changes

## Migration Script for Existing Data

If you have existing data with incorrect timezone assumptions:

```sql
-- Example: Convert existing timestamps assuming they were in local time
-- WARNING: Test this thoroughly before running on production data

-- Create backup
CREATE TABLE your_table_backup AS SELECT * FROM your_table;

-- Update timestamps (example for EST to UTC conversion)
UPDATE your_table 
SET timestamp_column = timestamp_column AT TIME ZONE 'America/New_York' AT TIME ZONE 'UTC'
WHERE timestamp_column IS NOT NULL;

-- Verify the conversion
SELECT 
    original.timestamp_column as original,
    updated.timestamp_column as updated
FROM your_table_backup original
JOIN your_table updated ON original.id = updated.id
LIMIT 10;
```

This guide provides multiple approaches to ensure your PostgreSQL database permanently uses UTC timezone. Choose the method that best fits your deployment and requirements.
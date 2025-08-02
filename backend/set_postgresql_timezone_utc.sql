-- PostgreSQL UTC Timezone Configuration SQL Script
-- Method 2: SQL Commands - Using ALTER SYSTEM SET timezone = 'UTC' for immediate changes

-- Check current timezone setting
SELECT 'Current timezone setting:' as info, current_setting('timezone') as timezone;

-- Show current timestamp in different formats
SELECT 
    'Current timestamps:' as info,
    NOW() as local_time,
    NOW() AT TIME ZONE 'UTC' as utc_time,
    EXTRACT(timezone FROM NOW()) as timezone_offset_seconds;

-- Set timezone to UTC using ALTER SYSTEM (requires superuser privileges)
ALTER SYSTEM SET timezone = 'UTC';

-- Reload PostgreSQL configuration to apply changes immediately
SELECT pg_reload_conf() as config_reloaded;

-- Verify the timezone change
SELECT 'New timezone setting:' as info, current_setting('timezone') as timezone;

-- Show timestamps after timezone change
SELECT 
    'Updated timestamps:' as info,
    NOW() as local_time,
    NOW() AT TIME ZONE 'UTC' as utc_time,
    EXTRACT(timezone FROM NOW()) as timezone_offset_seconds;

-- Test timestamp storage and retrieval
CREATE TEMPORARY TABLE test_timezone_storage (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    created_at_utc TIMESTAMP DEFAULT (NOW() AT TIME ZONE 'UTC')
);

-- Insert test data
INSERT INTO test_timezone_storage DEFAULT VALUES;

-- Retrieve and display test data
SELECT 
    'Test data:' as info,
    created_at,
    created_at_utc,
    created_at AT TIME ZONE 'UTC' as created_at_converted_to_utc
FROM test_timezone_storage;

-- Show timezone-related settings
SELECT 
    name,
    setting,
    unit,
    context,
    source
FROM pg_settings 
WHERE name LIKE '%timezone%' OR name LIKE '%tz%'
ORDER BY name;

-- Final verification message
SELECT 
    CASE 
        WHEN current_setting('timezone') = 'UTC' THEN '✅ SUCCESS: PostgreSQL timezone is now set to UTC'
        ELSE '❌ WARNING: PostgreSQL timezone is not UTC, current setting: ' || current_setting('timezone')
    END as result;
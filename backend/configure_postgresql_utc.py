#!/usr/bin/env python3
"""
PostgreSQL UTC Configuration Script
Implements three methods to configure PostgreSQL to use UTC timezone:
1. Configuration File Method - Modifying postgresql.conf
2. SQL Commands - Using ALTER SYSTEM SET timezone = 'UTC'
3. Environment Variables - Setting PGTZ=UTC and TZ=UTC
"""

import os
import sys
import subprocess
import psycopg2
from pathlib import Path
import shutil
from datetime import datetime

class PostgreSQLUTCConfigurator:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'algo_trading',  # Connect to default database first
            'user': 'anthonyxiao',
            'password': os.getenv('DATABASE_PASSWORD', '')
        }
        
    def method_1_modify_postgresql_conf(self):
        """
        Method 1: Configuration File Method - Modifying postgresql.conf with timezone = 'UTC'
        """
        print("\n=== Method 1: Modifying postgresql.conf ===")
        
        # Find PostgreSQL configuration file
        possible_paths = [
            '/usr/local/var/postgres/postgresql.conf',  # Homebrew on macOS
            '/opt/homebrew/var/postgres/postgresql.conf',  # Homebrew on Apple Silicon
            '/usr/local/pgsql/data/postgresql.conf',  # Standard installation
            '/var/lib/postgresql/data/postgresql.conf',  # Linux
            '/etc/postgresql/*/main/postgresql.conf',  # Ubuntu/Debian
        ]
        
        postgresql_conf_path = None
        for path in possible_paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                if matches:
                    postgresql_conf_path = matches[0]
                    break
            elif os.path.exists(path):
                postgresql_conf_path = path
                break
        
        if not postgresql_conf_path:
            print("❌ Could not find postgresql.conf file.")
            print("   Please locate it manually and modify it with: timezone = 'UTC'")
            return False
            
        print(f"📁 Found postgresql.conf at: {postgresql_conf_path}")
        
        try:
            # Create backup
            backup_path = f"{postgresql_conf_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(postgresql_conf_path, backup_path)
            print(f"💾 Created backup: {backup_path}")
            
            # Read current configuration
            with open(postgresql_conf_path, 'r') as f:
                lines = f.readlines()
            
            # Check if timezone is already set
            timezone_found = False
            modified_lines = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('timezone') and not stripped.startswith('#'):
                    # Replace existing timezone setting
                    modified_lines.append("timezone = 'UTC'\n")
                    timezone_found = True
                    print(f"🔄 Replaced existing timezone setting: {stripped}")
                elif stripped.startswith('#timezone'):
                    # Uncomment and set to UTC
                    modified_lines.append("timezone = 'UTC'\n")
                    timezone_found = True
                    print(f"🔓 Uncommented and set timezone: {stripped}")
                else:
                    modified_lines.append(line)
            
            # If no timezone setting found, add it
            if not timezone_found:
                modified_lines.append("\n# Timezone configuration\n")
                modified_lines.append("timezone = 'UTC'\n")
                print("➕ Added new timezone = 'UTC' setting")
            
            # Write modified configuration
            with open(postgresql_conf_path, 'w') as f:
                f.writelines(modified_lines)
            
            print("✅ Successfully modified postgresql.conf")
            print("⚠️  PostgreSQL restart required for changes to take effect")
            return True
            
        except PermissionError:
            print("❌ Permission denied. Please run with sudo or as postgres user.")
            return False
        except Exception as e:
            print(f"❌ Error modifying postgresql.conf: {e}")
            return False
    
    def method_2_sql_commands(self):
        """
        Method 2: SQL Commands - Using ALTER SYSTEM SET timezone = 'UTC' for immediate changes
        """
        print("\n=== Method 2: SQL Commands (ALTER SYSTEM) ===")
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check current timezone
            cursor.execute("SHOW timezone;")
            current_tz = cursor.fetchone()[0]
            print(f"📊 Current timezone: {current_tz}")
            
            # Set timezone to UTC using ALTER SYSTEM
            cursor.execute("ALTER SYSTEM SET timezone = 'UTC';")
            print("🔧 Executed: ALTER SYSTEM SET timezone = 'UTC'")
            
            # Reload configuration
            cursor.execute("SELECT pg_reload_conf();")
            print("🔄 Reloaded PostgreSQL configuration")
            
            # Verify the change
            cursor.execute("SHOW timezone;")
            new_tz = cursor.fetchone()[0]
            print(f"✅ New timezone: {new_tz}")
            
            # Show current timestamp in UTC
            cursor.execute("SELECT NOW() AT TIME ZONE 'UTC' as utc_time, NOW() as local_time;")
            utc_time, local_time = cursor.fetchone()
            print(f"🕐 UTC time: {utc_time}")
            print(f"🕐 Local time: {local_time}")
            
            cursor.close()
            conn.close()
            
            print("✅ Successfully configured timezone via SQL commands")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Database error: {e}")
            return False
        except Exception as e:
            print(f"❌ Error executing SQL commands: {e}")
            return False
    
    def method_3_environment_variables(self):
        """
        Method 3: Environment Variables - Setting PGTZ=UTC and TZ=UTC
        """
        print("\n=== Method 3: Environment Variables ===")
        
        # Set environment variables for current session
        os.environ['PGTZ'] = 'UTC'
        os.environ['TZ'] = 'UTC'
        print("🌍 Set environment variables for current session:")
        print("   PGTZ=UTC")
        print("   TZ=UTC")
        
        # Create shell script for permanent environment variables
        env_script_content = '''#!/bin/bash
# PostgreSQL UTC Environment Variables
# Add these lines to your shell profile (~/.bashrc, ~/.zshrc, etc.)

export PGTZ=UTC
export TZ=UTC

echo "PostgreSQL timezone environment variables set to UTC"
echo "PGTZ=$PGTZ"
echo "TZ=$TZ"
'''
        
        script_path = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/set_postgresql_env.sh'
        with open(script_path, 'w') as f:
            f.write(env_script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        print(f"📝 Created environment script: {script_path}")
        
        # Create systemd service file for Linux systems
        systemd_content = '''[Unit]
Description=PostgreSQL UTC Environment
After=postgresql.service

[Service]
Type=oneshot
Environment=PGTZ=UTC
Environment=TZ=UTC
ExecStart=/bin/true
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
'''
        
        systemd_path = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/postgresql-utc.service'
        with open(systemd_path, 'w') as f:
            f.write(systemd_content)
        print(f"📝 Created systemd service file: {systemd_path}")
        
        # Create Docker environment file
        docker_env_content = '''# PostgreSQL UTC Environment Variables for Docker
PGTZ=UTC
TZ=UTC
POSTGRES_INITDB_ARGS=--timezone=UTC
'''
        
        docker_env_path = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/.env.postgresql'
        with open(docker_env_path, 'w') as f:
            f.write(docker_env_content)
        print(f"📝 Created Docker environment file: {docker_env_path}")
        
        # Instructions for permanent setup
        print("\n📋 To make environment variables permanent:")
        print("\n1. For current user (add to ~/.bashrc or ~/.zshrc):")
        print("   echo 'export PGTZ=UTC' >> ~/.zshrc")
        print("   echo 'export TZ=UTC' >> ~/.zshrc")
        print("   source ~/.zshrc")
        
        print("\n2. For system-wide (add to /etc/environment):")
        print("   echo 'PGTZ=UTC' | sudo tee -a /etc/environment")
        print("   echo 'TZ=UTC' | sudo tee -a /etc/environment")
        
        print("\n3. For Docker containers:")
        print(f"   Use the environment file: {docker_env_path}")
        print("   docker run --env-file .env.postgresql postgres")
        
        print("\n4. For systemd services (Linux):")
        print(f"   sudo cp {systemd_path} /etc/systemd/system/")
        print("   sudo systemctl enable postgresql-utc.service")
        print("   sudo systemctl start postgresql-utc.service")
        
        return True
    
    def verify_configuration(self):
        """
        Verify that PostgreSQL is using UTC timezone
        """
        print("\n=== Verification ===")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check timezone setting
            cursor.execute("SHOW timezone;")
            timezone = cursor.fetchone()[0]
            print(f"🔍 Current PostgreSQL timezone: {timezone}")
            
            # Check current timestamp
            cursor.execute("SELECT NOW(), NOW() AT TIME ZONE 'UTC';")
            local_time, utc_time = cursor.fetchone()
            print(f"🕐 PostgreSQL local time: {local_time}")
            print(f"🕐 PostgreSQL UTC time: {utc_time}")
            
            # Check environment variables
            print(f"🌍 PGTZ environment variable: {os.getenv('PGTZ', 'Not set')}")
            print(f"🌍 TZ environment variable: {os.getenv('TZ', 'Not set')}")
            
            # Test timestamp storage and retrieval
            cursor.execute("""
                CREATE TEMPORARY TABLE test_timestamps (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cursor.execute("INSERT INTO test_timestamps DEFAULT VALUES;")
            cursor.execute("SELECT created_at FROM test_timestamps;")
            test_timestamp = cursor.fetchone()[0]
            print(f"🧪 Test timestamp stored: {test_timestamp}")
            
            cursor.close()
            conn.close()
            
            if timezone.upper() == 'UTC':
                print("✅ PostgreSQL is correctly configured to use UTC timezone")
                return True
            else:
                print(f"⚠️  PostgreSQL timezone is '{timezone}', not UTC")
                return False
                
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False
    
    def run_all_methods(self):
        """
        Execute all three configuration methods
        """
        print("🚀 PostgreSQL UTC Configuration Tool")
        print("=====================================\n")
        
        results = {
            'method_1': False,
            'method_2': False,
            'method_3': False,
            'verification': False
        }
        
        # Method 1: Configuration File
        try:
            results['method_1'] = self.method_1_modify_postgresql_conf()
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: SQL Commands
        try:
            results['method_2'] = self.method_2_sql_commands()
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Environment Variables
        try:
            results['method_3'] = self.method_3_environment_variables()
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
        
        # Verification
        try:
            results['verification'] = self.verify_configuration()
        except Exception as e:
            print(f"❌ Verification failed: {e}")
        
        # Summary
        print("\n=== Summary ===")
        print(f"Method 1 (postgresql.conf): {'✅ Success' if results['method_1'] else '❌ Failed'}")
        print(f"Method 2 (SQL commands): {'✅ Success' if results['method_2'] else '❌ Failed'}")
        print(f"Method 3 (Environment vars): {'✅ Success' if results['method_3'] else '❌ Failed'}")
        print(f"Verification: {'✅ Success' if results['verification'] else '❌ Failed'}")
        
        if any(results.values()):
            print("\n🎉 At least one configuration method succeeded!")
            if not results['verification']:
                print("⚠️  You may need to restart PostgreSQL for changes to take effect.")
        else:
            print("\n❌ All configuration methods failed. Please check the errors above.")
        
        return results

if __name__ == "__main__":
    configurator = PostgreSQLUTCConfigurator()
    
    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "1":
            configurator.method_1_modify_postgresql_conf()
        elif method == "2":
            configurator.method_2_sql_commands()
        elif method == "3":
            configurator.method_3_environment_variables()
        elif method == "verify":
            configurator.verify_configuration()
        else:
            print("Usage: python configure_postgresql_utc.py [1|2|3|verify]")
            print("  1: Modify postgresql.conf")
            print("  2: Use SQL commands")
            print("  3: Set environment variables")
            print("  verify: Verify configuration")
    else:
        configurator.run_all_methods()
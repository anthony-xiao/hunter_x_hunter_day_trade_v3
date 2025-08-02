#!/usr/bin/env python3
"""
Timezone Fix Script

This script fixes the timezone issue in the PostgreSQL database and provides
options to correct existing data.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db_manager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class TimezoneFixer:
    def __init__(self):
        self.engine = db_manager.get_engine()
        self.Session = db_manager.SessionLocal
        
    def check_current_timezone(self):
        """
        Check current database timezone settings
        """
        print("=== CURRENT TIMEZONE SETTINGS ===")
        
        with self.Session() as session:
            # Check database timezone
            result = session.execute(text("SELECT current_setting('timezone') as db_timezone"))
            db_tz = result.fetchone()
            print(f"Database timezone: {db_tz.db_timezone}")
            
            # Check system timezone
            result = session.execute(text("SELECT now() as current_time"))
            current_time = result.fetchone()
            print(f"Database current time: {current_time.current_time}")
            
            # Check UTC time
            result = session.execute(text("SELECT now() AT TIME ZONE 'UTC' as utc_time"))
            utc_time = result.fetchone()
            print(f"Database UTC time: {utc_time.utc_time}")
    
    def fix_database_timezone(self):
        """
        Set database timezone to UTC
        """
        print("\n=== FIXING DATABASE TIMEZONE ===")
        
        try:
            with self.Session() as session:
                # Set timezone to UTC for current session
                session.execute(text("SET timezone = 'UTC'"))
                session.commit()
                print("✓ Session timezone set to UTC")
                
                # Set timezone to UTC globally (requires superuser privileges)
                try:
                    session.execute(text("ALTER DATABASE hunter_x_hunter_day_trade SET timezone = 'UTC'"))
                    session.commit()
                    print("✓ Database default timezone set to UTC")
                except Exception as e:
                    print(f"⚠ Could not set database default timezone (requires superuser): {e}")
                    print("  You may need to run: ALTER DATABASE hunter_x_hunter_day_trade SET timezone = 'UTC';")
                    print("  as a PostgreSQL superuser")
                
                # Verify the change
                result = session.execute(text("SELECT current_setting('timezone') as new_timezone"))
                new_tz = result.fetchone()
                print(f"✓ Current session timezone: {new_tz.new_timezone}")
                
        except Exception as e:
            print(f"✗ Error setting timezone: {e}")
    
    def analyze_data_impact(self):
        """
        Analyze how many records would be affected by timezone correction
        """
        print("\n=== DATA IMPACT ANALYSIS ===")
        
        with self.Session() as session:
            # Count market data records
            result = session.execute(text("SELECT COUNT(*) as count FROM market_data"))
            market_count = result.fetchone().count
            print(f"Market data records: {market_count:,}")
            
            # Count features records
            result = session.execute(text("SELECT COUNT(*) as count FROM features"))
            features_count = result.fetchone().count
            print(f"Features records: {features_count:,}")
            
            # Sample timestamp ranges
            result = session.execute(text("""
                SELECT 
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM market_data
            """))
            market_range = result.fetchone()
            print(f"Market data range: {market_range.earliest} to {market_range.latest}")
            
            if features_count > 0:
                result = session.execute(text("""
                    SELECT 
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM features
                """))
                features_range = result.fetchone()
                print(f"Features range: {features_range.earliest} to {features_range.latest}")
    
    def backup_data(self):
        """
        Create backup tables before making changes
        """
        print("\n=== CREATING BACKUP TABLES ===")
        
        try:
            with self.Session() as session:
                # Backup market_data
                session.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_data_backup AS 
                    SELECT * FROM market_data
                """))
                print("✓ Created market_data_backup table")
                
                # Backup features if it has data
                result = session.execute(text("SELECT COUNT(*) as count FROM features"))
                if result.fetchone().count > 0:
                    session.execute(text("""
                        CREATE TABLE IF NOT EXISTS features_backup AS 
                        SELECT * FROM features
                    """))
                    print("✓ Created features_backup table")
                
                session.commit()
                
        except Exception as e:
            print(f"✗ Error creating backups: {e}")
    
    def correct_timestamps(self, dry_run: bool = True):
        """
        Correct timestamps by converting from Shanghai time to UTC
        
        The issue: timestamps were stored as UTC but interpreted as Shanghai time
        Solution: subtract 8 hours to get the correct UTC time
        """
        print(f"\n=== {'DRY RUN: ' if dry_run else ''}CORRECTING TIMESTAMPS ===")
        
        try:
            with self.Session() as session:
                if dry_run:
                    # Show what would be changed
                    result = session.execute(text("""
                        SELECT 
                            timestamp as original,
                            timestamp - INTERVAL '8 hours' as corrected
                        FROM market_data 
                        ORDER BY timestamp 
                        LIMIT 5
                    """))
                    
                    print("Sample timestamp corrections (market_data):")
                    for row in result:
                        print(f"  {row.original} -> {row.corrected}")
                    
                    # Check features if they exist
                    result = session.execute(text("SELECT COUNT(*) as count FROM features"))
                    if result.fetchone().count > 0:
                        result = session.execute(text("""
                            SELECT 
                                timestamp as original,
                                timestamp - INTERVAL '8 hours' as corrected
                            FROM features 
                            ORDER BY timestamp 
                            LIMIT 5
                        """))
                        
                        print("\nSample timestamp corrections (features):")
                        for row in result:
                            print(f"  {row.original} -> {row.corrected}")
                else:
                    # Actually correct the timestamps
                    print("Correcting market_data timestamps...")
                    result = session.execute(text("""
                        UPDATE market_data 
                        SET timestamp = timestamp - INTERVAL '8 hours'
                    """))
                    print(f"✓ Updated {result.rowcount} market_data records")
                    
                    # Correct features if they exist
                    result = session.execute(text("SELECT COUNT(*) as count FROM features"))
                    if result.fetchone().count > 0:
                        print("Correcting features timestamps...")
                        result = session.execute(text("""
                            UPDATE features 
                            SET timestamp = timestamp - INTERVAL '8 hours'
                        """))
                        print(f"✓ Updated {result.rowcount} features records")
                    
                    session.commit()
                    print("✓ All timestamp corrections committed")
                    
        except Exception as e:
            print(f"✗ Error correcting timestamps: {e}")
            if not dry_run:
                print("Rolling back changes...")
    
    def verify_fix(self):
        """
        Verify that the timezone fix worked correctly
        """
        print("\n=== VERIFYING FIX ===")
        
        with self.Session() as session:
            # Check timezone setting
            result = session.execute(text("SELECT current_setting('timezone') as timezone"))
            tz = result.fetchone()
            print(f"Database timezone: {tz.timezone}")
            
            # Check sample timestamps
            result = session.execute(text("""
                SELECT 
                    DATE(timestamp) as date,
                    MIN(timestamp) as first,
                    MAX(timestamp) as last,
                    EXTRACT(HOUR FROM MIN(timestamp)) as first_hour,
                    EXTRACT(HOUR FROM MAX(timestamp)) as last_hour
                FROM market_data 
                WHERE timestamp >= NOW() - INTERVAL '3 days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 3
            """))
            
            print("\nSample corrected timestamps:")
            for row in result:
                print(f"Date: {row.date}")
                print(f"  First: {row.first} (Hour: {row.first_hour})")
                print(f"  Last:  {row.last} (Hour: {row.last_hour})")
            
            # Expected hours in UTC:
            # Pre-market: 09:00-14:30 UTC (4:00-9:30 AM ET)
            # Regular: 14:30-21:00 UTC (9:30 AM-4:00 PM ET)
            # After-hours: 21:00-01:00+1 UTC (4:00-8:00 PM ET)
            
            print("\nExpected trading hours in UTC:")
            print("  Pre-market: 09:00-14:30")
            print("  Regular: 14:30-21:00")
            print("  After-hours: 21:00-01:00+1")
    
    def run_complete_fix(self, backup: bool = True, dry_run: bool = True):
        """
        Run the complete timezone fix process
        """
        print("TIMEZONE FIX PROCESS")
        print("=" * 50)
        
        # Step 1: Check current state
        self.check_current_timezone()
        
        # Step 2: Analyze impact
        self.analyze_data_impact()
        
        # Step 3: Create backups
        if backup:
            self.backup_data()
        
        # Step 4: Fix database timezone
        self.fix_database_timezone()
        
        # Step 5: Correct timestamps
        self.correct_timestamps(dry_run=dry_run)
        
        # Step 6: Verify fix
        if not dry_run:
            self.verify_fix()
        
        print("\n=== NEXT STEPS ===")
        if dry_run:
            print("This was a dry run. To apply the fixes:")
            print("1. Run: python fix_timezone_issue.py --apply")
            print("2. Restart your application to pick up timezone changes")
            print("3. Re-run feature engineering to regenerate features with correct timestamps")
        else:
            print("✓ Timezone fix completed!")
            print("1. Restart your application to pick up timezone changes")
            print("2. Consider re-running feature engineering to ensure consistency")
            print("3. Monitor data quality to ensure everything looks correct")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix timezone issues in the database')
    parser.add_argument('--apply', action='store_true', help='Apply the fixes (default is dry run)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup tables')
    
    args = parser.parse_args()
    
    fixer = TimezoneFixer()
    
    # Run the complete fix process
    fixer.run_complete_fix(
        backup=not args.no_backup,
        dry_run=not args.apply
    )


if __name__ == "__main__":
    main()
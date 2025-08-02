#!/usr/bin/env python3
"""
Script to clear all data from market_data and features tables
Uses database credentials from .env file
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

def load_env_config():
    """Load database configuration from .env file"""
    # Load .env file from the backend directory
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    return {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': int(os.getenv('DATABASE_PORT', 5432)),
        'database': os.getenv('DATABASE_NAME', 'algo_trading'),
        'user': os.getenv('DATABASE_USER', 'anthonyxiao'),
        'password': os.getenv('DATABASE_PASSWORD', '')
    }

def get_table_counts(cursor):
    """Get current record counts for both tables"""
    counts = {}
    
    # Get market_data count
    cursor.execute("SELECT COUNT(*) FROM market_data;")
    counts['market_data'] = cursor.fetchone()[0]
    
    # Get features count
    cursor.execute("SELECT COUNT(*) FROM features;")
    counts['features'] = cursor.fetchone()[0]
    
    return counts

def clear_tables(db_config):
    """Clear all data from market_data and features tables"""
    try:
        # Connect to PostgreSQL
        print(f"Connecting to PostgreSQL database: {db_config['database']}")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Get initial counts
        print("\n=== Current Table Status ===")
        initial_counts = get_table_counts(cursor)
        print(f"market_data table: {initial_counts['market_data']:,} records")
        print(f"features table: {initial_counts['features']:,} records")
        
        if initial_counts['market_data'] == 0 and initial_counts['features'] == 0:
            print("\n✅ Both tables are already empty!")
            return
        
        # Confirm deletion
        total_records = initial_counts['market_data'] + initial_counts['features']
        print(f"\n⚠️  WARNING: About to delete {total_records:,} total records!")
        
        response = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Operation cancelled.")
            return
        
        print("\n=== Clearing Tables ===")
        
        # Clear features table first (may have foreign key references)
        print("Clearing features table...")
        cursor.execute("TRUNCATE TABLE features RESTART IDENTITY CASCADE;")
        features_deleted = initial_counts['features']
        print(f"✅ Deleted {features_deleted:,} records from features table")
        
        # Clear market_data table
        print("Clearing market_data table...")
        cursor.execute("TRUNCATE TABLE market_data RESTART IDENTITY CASCADE;")
        market_data_deleted = initial_counts['market_data']
        print(f"✅ Deleted {market_data_deleted:,} records from market_data table")
        
        # Commit the transaction
        conn.commit()
        
        # Verify tables are empty
        print("\n=== Verification ===")
        final_counts = get_table_counts(cursor)
        print(f"market_data table: {final_counts['market_data']:,} records")
        print(f"features table: {final_counts['features']:,} records")
        
        if final_counts['market_data'] == 0 and final_counts['features'] == 0:
            print("\n🎉 SUCCESS: All tables cleared successfully!")
            print(f"Total records deleted: {total_records:,}")
            print(f"- market_data: {market_data_deleted:,}")
            print(f"- features: {features_deleted:,}")
        else:
            print("\n❌ ERROR: Tables not completely cleared!")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"\n❌ PostgreSQL Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 60)
    print("PostgreSQL Table Cleanup Script")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load database configuration
    try:
        db_config = load_env_config()
        print(f"Database: {db_config['database']}@{db_config['host']}:{db_config['port']}")
        print(f"User: {db_config['user']}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Clear tables
    clear_tables(db_config)
    
    print("\n=== Script Complete ===")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to clear all data from market_data and features tables
Uses Supabase database connection
"""

import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, text
from config import settings

def get_table_counts(supabase):
    """Get current record counts for both tables"""
    counts = {}
    
    # Get market_data count
    result = supabase.table('market_data').select('id', count='exact').execute()
    counts['market_data'] = result.count
    
    # Get features count
    result = supabase.table('features').select('id', count='exact').execute()
    counts['features'] = result.count
    
    return counts

def clear_tables():
    """Clear all data from market_data and features tables"""
    try:
        # Connect to Supabase
        print(f"Connecting to Supabase database...")
        from database import db_manager
        supabase = db_manager.get_supabase_client()
        
        if not supabase:
            print("❌ Error: Supabase client not available")
            return
        
        # Get initial counts
        print("\n=== Current Table Status ===")
        initial_counts = get_table_counts(supabase)
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
        supabase.table('features').delete().neq('id', 0).execute()
        features_deleted = initial_counts['features']
        print(f"✅ Deleted {features_deleted:,} records from features table")
        
        # Clear market_data table
        print("Clearing market_data table...")
        supabase.table('market_data').delete().neq('id', 0).execute()
        market_data_deleted = initial_counts['market_data']
        print(f"✅ Deleted {market_data_deleted:,} records from market_data table")
        
        # Note: Supabase handles transactions automatically
        
        # Verify tables are empty
        print("\n=== Verification ===")
        final_counts = get_table_counts(supabase)
        print(f"market_data table: {final_counts['market_data']:,} records")
        print(f"features table: {final_counts['features']:,} records")
        
        if final_counts['market_data'] == 0 and final_counts['features'] == 0:
            print("\n🎉 SUCCESS: All tables cleared successfully!")
            print(f"Total records deleted: {total_records:,}")
            print(f"- market_data: {market_data_deleted:,}")
            print(f"- features: {features_deleted:,}")
        else:
            print("\n❌ ERROR: Tables not completely cleared!")
        
    except Exception as e:
        print(f"\n❌ Database Error: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("=" * 60)
    print("Supabase Table Cleanup Script")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Clear tables
    clear_tables()
    
    print("\n=== Script Complete ===")

if __name__ == "__main__":
    main()
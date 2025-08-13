#!/usr/bin/env python3
"""
Debug script to test Supabase connection and check market data existence.
This script helps identify why main.py reports no data despite data being visible in Supabase.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import settings
from supabase import create_client, Client
import pandas as pd

def test_supabase_connection():
    """Test basic Supabase connection and authentication."""
    print("=== Testing Supabase Connection ===")
    
    try:
        # Check if environment variables are loaded
        print(f"Supabase URL: {settings.supabase_url}")
        print(f"Service Role Key exists: {'Yes' if settings.supabase_service_role_key else 'No'}")
        print(f"Anon Key exists: {'Yes' if settings.supabase_anon_key else 'No'}")
        
        if not settings.supabase_url or not settings.supabase_service_role_key:
            print("❌ ERROR: Missing Supabase configuration!")
            return None
        
        # Create Supabase client
        supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
        
        print("✅ Supabase client created successfully")
        return supabase
        
    except Exception as e:
        print(f"❌ ERROR creating Supabase client: {e}")
        traceback.print_exc()
        return None

def check_table_exists(supabase: Client, table_name: str):
    """Check if a table exists and is accessible."""
    try:
        # Try to query the table with a limit to test accessibility
        response = supabase.table(table_name).select("*").limit(1).execute()
        print(f"✅ Table '{table_name}' exists and is accessible")
        return True
    except Exception as e:
        print(f"❌ ERROR accessing table '{table_name}': {e}")
        return False

def check_market_data(supabase: Client):
    """Check market_data table for existing data."""
    print("\n=== Checking Market Data Table ===")
    
    try:
        # Check if table exists
        if not check_table_exists(supabase, "market_data"):
            return
        
        # Count total records
        response = supabase.table("market_data").select("*", count="exact").limit(1).execute()
        total_count = response.count
        print(f"Total records in market_data: {total_count}")
        
        if total_count == 0:
            print("❌ No data found in market_data table")
            return
        
        # Get sample data
        print("\n--- Sample Data ---")
        response = supabase.table("market_data").select("*").limit(5).execute()
        
        if response.data:
            for i, record in enumerate(response.data[:3]):
                print(f"Record {i+1}:")
                print(f"  Symbol: {record.get('symbol', 'N/A')}")
                print(f"  Timestamp: {record.get('timestamp', 'N/A')}")
                print(f"  Open: {record.get('open', 'N/A')}")
                print(f"  Close: {record.get('close', 'N/A')}")
                print(f"  Volume: {record.get('volume', 'N/A')}")
                print()
        
        # Check for specific symbols
        print("--- Checking for AAPL data ---")
        response = supabase.table("market_data").select("*", count="exact").eq("symbol", "AAPL").limit(1).execute()
        aapl_count = response.count
        print(f"AAPL records: {aapl_count}")
        
        if aapl_count > 0:
            # Get date range for AAPL
            response = supabase.table("market_data").select("timestamp").eq("symbol", "AAPL").order("timestamp", desc=False).limit(1).execute()
            earliest = response.data[0]['timestamp'] if response.data else 'N/A'
            
            response = supabase.table("market_data").select("timestamp").eq("symbol", "AAPL").order("timestamp", desc=True).limit(1).execute()
            latest = response.data[0]['timestamp'] if response.data else 'N/A'
            
            print(f"AAPL date range: {earliest} to {latest}")
        
    except Exception as e:
        print(f"❌ ERROR checking market_data: {e}")
        traceback.print_exc()

async def test_data_pipeline_query():
    """Test the same query logic used by DataPipeline.load_market_data."""
    print("\n=== Testing DataPipeline Query Logic ===")
    
    try:
        from data.data_pipeline import DataPipeline
        
        # Create DataPipeline instance
        pipeline = DataPipeline()
        
        # Test loading AAPL data for the last 60 days (should trigger chunking)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        print(f"Testing load_market_data for AAPL from {start_date.date()} to {end_date.date()} (60 days - should chunk)")
        
        # This is the same method main.py uses - now properly awaited
        df = await pipeline.load_market_data("AAPL", start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"✅ DataPipeline.load_market_data returned {len(df)} records")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Columns: {list(df.columns)}")
        else:
            print("❌ DataPipeline.load_market_data returned empty or None")
            
    except Exception as e:
        print(f"❌ ERROR testing DataPipeline query: {e}")
        traceback.print_exc()

def check_database_manager():
    """Test DatabaseManager connection."""
    print("\n=== Testing DatabaseManager ===")
    
    try:
        from database import DatabaseManager
        
        db_manager = DatabaseManager()
        db_manager.initialize()
        
        # Test connection
        if db_manager.test_connection():
            print("✅ DatabaseManager connection test passed")
        else:
            print("❌ DatabaseManager connection test failed")
            
    except Exception as e:
        print(f"❌ ERROR testing DatabaseManager: {e}")
        traceback.print_exc()

async def main():
    """Main debug function."""
    print("Supabase Connection Debug Script")
    print("=" * 50)
    
    # Test basic connection
    supabase = test_supabase_connection()
    if not supabase:
        print("\n❌ Cannot proceed without valid Supabase connection")
        return
    
    # Check market data
    check_market_data(supabase)
    
    # Test DatabaseManager
    check_database_manager()
    
    # Test DataPipeline query logic
    await test_data_pipeline_query()
    
    print("\n=== Debug Complete ===")
    print("If market data exists in Supabase but DataPipeline returns empty,")
    print("check for:")
    print("1. Date range issues (timezone, format)")
    print("2. Symbol case sensitivity")
    print("3. Query filtering logic")
    print("4. Data type mismatches")
    print("5. Query timeout issues with large datasets")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
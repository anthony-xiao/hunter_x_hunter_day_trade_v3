#!/usr/bin/env python3
"""
Check the actual date range of AAPL data in the database and test loading with correct dates.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data.data_pipeline import DataPipeline

async def check_data_range_and_test():
    """Check the actual date range and test loading with correct dates"""
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    if not db_manager.test_connection():
        print("❌ Database connection failed")
        return
    
    print("✅ Database connection successful")
    
    # Get Supabase client
    supabase = db_manager.get_supabase_client()
    
    # Check the actual date range of AAPL data
    print("\n=== Checking actual AAPL data range ===")
    
    # Get min and max timestamps for AAPL
    min_response = supabase.table('market_data').select(
        'timestamp'
    ).eq('symbol', 'AAPL').order('timestamp').limit(1).execute()
    
    max_response = supabase.table('market_data').select(
        'timestamp'
    ).eq('symbol', 'AAPL').order('timestamp', desc=True).limit(1).execute()
    
    if min_response.data and max_response.data:
        min_date = min_response.data[0]['timestamp']
        max_date = max_response.data[0]['timestamp']
        print(f"AAPL data range: {min_date} to {max_date}")
        
        # Parse the dates
        start_date = datetime.fromisoformat(min_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(max_date.replace('Z', '+00:00'))
        
        print(f"Parsed range: {start_date} to {end_date}")
        
        # Test loading with the actual date range
        print("\n=== Testing load with actual date range ===")
        
        pipeline = DataPipeline()
        
        # Load data using the actual date range
        aapl_data = await pipeline.load_market_data('AAPL', start_date, end_date)
        
        if not aapl_data.empty:
            print(f"✅ Successfully loaded {len(aapl_data)} AAPL records")
            print(f"Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
            
            # Check if we got close to the expected 399.3K records
            if len(aapl_data) > 350000:  # Allow some tolerance
                print(f"🎉 SUCCESS: Loaded {len(aapl_data)} records (expected ~399.3K)")
            else:
                print(f"⚠️  Records loaded: {len(aapl_data)} (expected ~399.3K)")
        else:
            print("❌ No AAPL data loaded")
            
        # Also test a single chunk to see the pagination in action
        print("\n=== Testing single chunk with pagination ===")
        
        # Use first 10 days of data
        chunk_end = start_date + timedelta(days=10)
        chunk_data = await pipeline._load_market_data_chunk('AAPL', start_date, chunk_end)
        print(f"10-day chunk loaded: {len(chunk_data)} records")
        
    else:
        print("❌ Could not determine AAPL data range")
    
    # Close database connection
    db_manager.close()
    print("\n✅ Check completed")

if __name__ == "__main__":
    asyncio.run(check_data_range_and_test())
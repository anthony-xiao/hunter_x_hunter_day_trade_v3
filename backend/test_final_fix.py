#!/usr/bin/env python3
"""
Test script to verify the final timestamp-based pagination fix for loading market data.
This should now be able to load all 399.3K AAPL records.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data.data_pipeline import DataPipeline

async def test_full_data_loading():
    """Test loading the full AAPL dataset using the updated pagination method"""
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Test connection
    if not db_manager.test_connection():
        print("❌ Database connection failed")
        return
    
    print("✅ Database connection successful")
    
    # Initialize data pipeline
    pipeline = DataPipeline()
    
    # Test loading a smaller chunk first (5 days)
    print("\n=== Testing 5-day chunk ===") 
    start_date = datetime(2025, 6, 16)
    end_date = datetime(2025, 6, 20)
    
    chunk_data = await pipeline._load_market_data_chunk('AAPL', start_date, end_date)
    print(f"5-day chunk loaded: {len(chunk_data)} records")
    
    # Test loading a larger chunk (30 days)
    print("\n=== Testing 30-day chunk ===")
    start_date = datetime(2025, 6, 16)
    end_date = datetime(2025, 7, 15)
    
    chunk_data = await pipeline._load_market_data_chunk('AAPL', start_date, end_date)
    print(f"30-day chunk loaded: {len(chunk_data)} records")
    
    # Test the full load_market_data method for 760 days
    print("\n=== Testing full load_market_data method (760 days) ===")
    
    # Calculate 760 days back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=760)
    
    print(f"Loading AAPL data from {start_date.date()} to {end_date.date()}")
    
    full_data = await pipeline.load_market_data(['AAPL'], start_date, end_date)
    
    if 'AAPL' in full_data:
        aapl_data = full_data['AAPL']
        print(f"✅ Successfully loaded {len(aapl_data)} AAPL records")
        print(f"Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
        
        # Check if we got close to the expected 399.3K records
        if len(aapl_data) > 350000:  # Allow some tolerance
            print(f"🎉 SUCCESS: Loaded {len(aapl_data)} records (expected ~399.3K)")
        else:
            print(f"⚠️  WARNING: Only loaded {len(aapl_data)} records (expected ~399.3K)")
    else:
        print("❌ No AAPL data loaded")
    
    # Close database connection
    db_manager.close()
    print("\n✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_full_data_loading())
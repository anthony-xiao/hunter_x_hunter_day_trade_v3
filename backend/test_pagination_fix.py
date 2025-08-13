#!/usr/bin/env python3
"""
Test script to verify the pagination fix for loading market data.
This will test if we can now load all 399.3K AAPL records instead of just 26K.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from database import db_manager
from config import Settings

async def test_pagination_fix():
    """Test the pagination fix for loading market data"""
    print("=== Testing Pagination Fix ===")
    
    # Initialize components
    settings = Settings()
    
    # Test database connection
    if not db_manager.test_connection():
        print("❌ Database connection failed")
        return
    
    # Create data pipeline
    data_pipeline = DataPipeline()
    
    # Test loading a small date range first
    print("\n1. Testing small date range (1 day)...")
    start_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    end_date = datetime(2025, 6, 17, tzinfo=timezone.utc)
    
    small_data = await data_pipeline._load_market_data_chunk('AAPL', start_date, end_date)
    print(f"Small range result: {len(small_data)} records")
    
    # Test loading a medium date range
    print("\n2. Testing medium date range (7 days)...")
    start_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    end_date = datetime(2025, 6, 23, tzinfo=timezone.utc)
    
    medium_data = await data_pipeline._load_market_data_chunk('AAPL', start_date, end_date)
    print(f"Medium range result: {len(medium_data)} records")
    
    # Test loading a larger date range that should exceed 1000 records
    print("\n3. Testing large date range (30 days)...")
    start_date = datetime(2025, 6, 16, tzinfo=timezone.utc)
    end_date = datetime(2025, 7, 16, tzinfo=timezone.utc)
    
    large_data = await data_pipeline._load_market_data_chunk('AAPL', start_date, end_date)
    print(f"Large range result: {len(large_data)} records")
    
    if len(large_data) > 1000:
        print("✅ SUCCESS: Pagination is working! Got more than 1000 records.")
    else:
        print("❌ ISSUE: Still limited to 1000 or fewer records.")
    
    # Test the full load_market_data method with a reasonable range
    print("\n4. Testing full load_market_data method (60 days)...")
    start_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    
    full_data = await data_pipeline.load_market_data('AAPL', start_date, end_date)
    print(f"Full method result: {len(full_data)} records")
    
    # Calculate expected vs actual
    total_days = (end_date - start_date).days
    print(f"Date range: {total_days} days")
    
    if len(full_data) > 26000:
        print("✅ SUCCESS: Loading significantly more data than before!")
    else:
        print("❌ ISSUE: Still getting similar record count as before.")
    
    print("\n=== Test Complete ===")
    
    db_manager.close()

if __name__ == "__main__":
    asyncio.run(test_pagination_fix())
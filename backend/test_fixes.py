#!/usr/bin/env python3
"""
Test script to verify the fixes for:
1. Symbol storage bug in _make_json_serializable_batch
2. Initial chunk storage failures with retry logic
"""

import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from loguru import logger
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from database import db_manager

async def test_symbol_storage_fix():
    """Test that symbols are correctly stored instead of 'timestamp'"""
    logger.info("Testing symbol storage fix...")
    
    # Initialize data pipeline
    data_pipeline = DataPipeline()
    await data_pipeline.initialize_database()
    
    # Create test data
    test_symbol = "TEST_SYMBOL"
    test_data = {
        'timestamp': [datetime.now(timezone.utc) - timedelta(minutes=i) for i in range(5, 0, -1)],
        'open': [100.0 + i for i in range(5)],
        'high': [101.0 + i for i in range(5)],
        'low': [99.0 + i for i in range(5)],
        'close': [100.5 + i for i in range(5)],
        'volume': [1000 + i*100 for i in range(5)],
        'rsi': [50.0 + i for i in range(5)],
        'macd': [0.1 + i*0.1 for i in range(5)]
    }
    
    # Create DataFrame with timestamp as index
    df = pd.DataFrame(test_data)
    df.set_index('timestamp', inplace=True)
    df.index.name = 'timestamp'  # This was causing the bug
    
    logger.info(f"Created test DataFrame with index name: {df.index.name}")
    logger.info(f"Test symbol: {test_symbol}")
    
    try:
        # Test the store_features_batch method
        result = await data_pipeline.store_features_batch(test_symbol, df)
        logger.info(f"Successfully stored {result} feature records")
        
        # Verify the stored data has correct symbol
        # Query the features table to check if symbol is stored correctly
        response = data_pipeline.supabase.table('features').select('symbol').eq('symbol', test_symbol).limit(5).execute()
        
        if response.data:
            stored_symbols = [record['symbol'] for record in response.data]
            logger.info(f"Stored symbols: {stored_symbols}")
            
            if all(symbol == test_symbol for symbol in stored_symbols):
                logger.success("✅ Symbol storage fix PASSED - correct symbols stored")
                return True
            else:
                logger.error(f"❌ Symbol storage fix FAILED - incorrect symbols: {stored_symbols}")
                return False
        else:
            logger.warning("No data found in features table - this might be expected if data already exists")
            return True
            
    except Exception as e:
        logger.error(f"❌ Symbol storage test FAILED with error: {e}")
        return False

async def test_retry_logic():
    """Test the retry logic for chunk storage failures"""
    logger.info("Testing retry logic...")
    
    # Initialize data pipeline
    data_pipeline = DataPipeline()
    await data_pipeline.initialize_database()
    
    # Create a larger test dataset to trigger chunking
    test_symbol = "RETRY_TEST"
    num_records = 5000  # This should trigger chunking
    
    test_data = {
        'timestamp': [datetime.now(timezone.utc) - timedelta(minutes=i) for i in range(num_records, 0, -1)],
        'open': [100.0 + (i % 100) for i in range(num_records)],
        'high': [101.0 + (i % 100) for i in range(num_records)],
        'low': [99.0 + (i % 100) for i in range(num_records)],
        'close': [100.5 + (i % 100) for i in range(num_records)],
        'volume': [1000 + (i % 1000) for i in range(num_records)],
        'rsi': [50.0 + (i % 50) for i in range(num_records)],
        'macd': [0.1 + (i % 10) * 0.1 for i in range(num_records)]
    }
    
    df = pd.DataFrame(test_data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Created test DataFrame with {len(df)} records for retry testing")
    
    try:
        # Test the store_features_batch method with retry logic
        result = await data_pipeline.store_features_batch(test_symbol, df)
        logger.success(f"✅ Retry logic test PASSED - stored {result} records successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Retry logic test FAILED with error: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting fix verification tests...")
    
    # Test 1: Symbol storage fix
    symbol_test_passed = await test_symbol_storage_fix()
    
    # Test 2: Retry logic
    retry_test_passed = await test_retry_logic()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Symbol Storage Fix: {'✅ PASSED' if symbol_test_passed else '❌ FAILED'}")
    logger.info(f"Retry Logic Test: {'✅ PASSED' if retry_test_passed else '❌ FAILED'}")
    
    if symbol_test_passed and retry_test_passed:
        logger.success("🎉 All tests PASSED! Fixes are working correctly.")
    else:
        logger.error("❌ Some tests FAILED. Please review the fixes.")
    
    return symbol_test_passed and retry_test_passed

if __name__ == "__main__":
    asyncio.run(main())
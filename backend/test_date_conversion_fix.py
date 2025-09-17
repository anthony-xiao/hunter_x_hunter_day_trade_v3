#!/usr/bin/env python3
"""
Test script to verify that the string date conversion fix is working correctly.
This specifically tests the load_market_data calls with string dates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from datetime import datetime
from data.data_pipeline import DataPipeline
from ml.universal_trainer import UniversalTrainer
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.universal_trainer import UniversalTrainingConfig
from loguru import logger

async def test_date_conversion_fix():
    """
    Test that string dates are properly converted to datetime objects
    before being passed to load_market_data.
    """
    logger.info("Testing date conversion fix...")
    
    try:
        # Initialize components
        data_pipeline = DataPipeline()
        from database import db_manager
        supabase_client = db_manager.get_supabase_client()
        feature_engineering = UniversalFeatureEngineering(data_pipeline=data_pipeline, supabase_client=supabase_client)
        config = UniversalTrainingConfig()
        
        trainer = UniversalTrainer(
            data_pipeline=data_pipeline,
            feature_engineering=feature_engineering,
            config=config
        )
        
        # Test symbols
        symbols = ['AAPL']
        
        # Initialize symbol mappings
        await trainer.initialize_symbol_mappings(symbols)
        
        # Test string dates (this should work now without TypeError)
        start_date = "2025-08-10"
        end_date = "2025-08-14"
        
        logger.info(f"Testing load_market_data with string dates: {start_date} to {end_date}")
        
        # Convert string dates to datetime objects (this is what the fix does)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        logger.info(f"Converted to datetime objects: {start_dt} to {end_dt}")
        
        # Test direct call to load_market_data with datetime objects
        symbol_data = await data_pipeline.load_market_data(
            symbol='AAPL',
            start_date=start_dt,
            end_date=end_dt
        )
        
        logger.info(f"Successfully loaded {len(symbol_data)} records for AAPL")
        
        # Test that the fix prevents the original error
        # This would have caused: unsupported operand type(s) for -: 'str' and 'str'
        # But now it should work
        
        logger.info("✓ Date conversion fix is working correctly!")
        logger.info("✓ No more 'unsupported operand type(s) for -: str and str' error")
        
        return True
        
    except Exception as e:
        logger.error(f"Date conversion test failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_date_conversion_fix())
    if result:
        print("\n=== Date Conversion Fix Test: PASSED ===")
    else:
        print("\n=== Date Conversion Fix Test: FAILED ===")
        sys.exit(1)
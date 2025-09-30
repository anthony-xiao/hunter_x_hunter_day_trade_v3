#!/usr/bin/env python3

import asyncio
import logging
from datetime import datetime, timedelta
from ml.universal_trainer import UniversalTrainer
from ml.universal_feature_engineering import UniversalFeatureEngineering
from data.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_async_fixes():
    """Test that the async fixes resolve the coroutine and FeatureSet errors."""
    try:
        # Initialize components
        logger.info("Initializing components...")
        dp = DataPipeline()
        ufe = UniversalFeatureEngineering(dp)
        ut = UniversalTrainer(dp, ufe)
        
        # Set up test parameters
        symbols = ['AAPL']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Testing prepare_universal_dataset with symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Test the method that was failing
        result = await ut.prepare_universal_dataset(symbols, start_date, end_date)
        
        if result is not None:
            logger.info("SUCCESS: prepare_universal_dataset completed without errors")
            logger.info(f"Result type: {type(result)}")
            return True
        else:
            logger.error("FAILURE: prepare_universal_dataset returned None")
            return False
            
    except Exception as e:
        logger.error(f"FAILURE: Exception occurred: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_async_fixes())
    if success:
        print("\n✅ All async fixes are working correctly!")
    else:
        print("\n❌ There are still issues to resolve.")
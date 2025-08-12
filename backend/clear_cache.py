#!/usr/bin/env python3
"""
Script to clear the in-memory feature cache.
This clears only the cache, not the database.
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from loguru import logger

async def clear_cache():
    """Clear the in-memory feature cache"""
    try:
        # Initialize data pipeline
        pipeline = DataPipeline()
        
        # Clear the cache
        symbols_cleared = pipeline.clear_feature_cache()
        
        if symbols_cleared > 0:
            logger.success(f"Successfully cleared cache for {symbols_cleared} symbols")
            print(f"✅ Cache cleared successfully! {symbols_cleared} symbols had cached features removed.")
        else:
            logger.info("Cache was already empty")
            print("ℹ️  Cache was already empty - nothing to clear.")
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting cache cleanup...")
    asyncio.run(clear_cache())
    logger.info("Cache cleanup completed.")
    print("\n🔄 Cache has been cleared. The system will rebuild features as needed.")
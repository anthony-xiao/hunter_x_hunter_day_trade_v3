#!/usr/bin/env python3
"""
Debug script to test feature engineering and identify the 179 vs 178 feature count issue.
"""

import sys
import os
sys.path.append('backend')

from backend.ml.universal_trainer import UniversalTrainer
from backend.ml.universal_feature_engineering import UniversalFeatureEngineering
from backend.data.data_pipeline import DataPipeline
from backend.config import settings
from loguru import logger
import asyncio

# Configure logging to see all debug output
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="DEBUG", format="{time} | {level} | {message}")

async def debug_feature_engineering():
    """Debug the feature engineering process."""
    logger.info("Starting debug feature engineering test")
    
    # Initialize components
    data_pipeline = DataPipeline()
    feature_engineering = UniversalFeatureEngineering()
    trainer = UniversalTrainer(data_pipeline, feature_engineering)
    
    # Test with a single symbol to trigger fallback logic
    symbols = ['AAPL']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    try:
        # This should trigger the feature engineering and validation
        results = await trainer.phase1_train_base_models(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            model_types=['lstm']
        )
        
        logger.info(f"Training completed successfully: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_feature_engineering())
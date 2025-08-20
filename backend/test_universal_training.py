#!/usr/bin/env python3

import asyncio
import logging
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.model_trainer import ModelTrainer
from data.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_universal_training():
    """Test universal training initialization and dataset preparation"""
    try:
        # Initialize components
        dp = DataPipeline()
        mt = ModelTrainer(dp)
        symbols = ['AAPL', 'TSLA']
        
        print("=== Testing Universal Training Initialization ===")
        success = mt.initialize_universal_training(symbols)
        print(f"Universal training initialization: {success}")
        
        if success:
            print("\n=== Testing Universal Dataset Preparation ===")
            test_result = await mt.universal_trainer.prepare_universal_dataset(
                symbols, '2025-07-19', '2025-08-14'
            )
            print(f"Universal dataset preparation result:")
            if isinstance(test_result[0], list):
                print(f"  X type: list with {len(test_result[0])} elements")
                if len(test_result[0]) > 0:
                    print(f"  X[0] shape: {test_result[0][0].shape}")
                    print(f"  X[1] shape: {test_result[0][1].shape}")
            else:
                print(f"  X shape: {test_result[0].shape}")
            print(f"  y shape: {test_result[1].shape}")
            
            if (isinstance(test_result[0], list) and len(test_result[0]) > 0 and test_result[0][0].shape[0] > 0) or \
               (hasattr(test_result[0], 'shape') and test_result[0].shape[0] > 0):
                print("\n=== Success! Universal training data preparation is working ===")
            else:
                print("\n=== Issue: Empty dataset returned ===")
        else:
            print("\n=== Issue: Universal training initialization failed ===")
            
    except Exception as e:
        print(f"\n=== Error during testing: {e} ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_universal_training())
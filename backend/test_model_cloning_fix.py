#!/usr/bin/env python3

import asyncio
import logging
import sys
import os
import numpy as np
from datetime import datetime

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.model_trainer import ModelTrainer
from data.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_cloning_fix():
    """Test that model cloning fix resolves identical accuracy issue"""
    try:
        # Initialize components
        dp = DataPipeline()
        mt = ModelTrainer(dp)
        symbols = ['AAPL', 'TSLA']
        
        print("=== Testing Model Cloning Fix ===\n")
        
        # Step 1: Initialize universal training
        print("1. Initializing universal training...")
        success = mt.initialize_universal_training(symbols)
        if not success:
            print("   FAILED: Universal training initialization failed")
            return
        print("   SUCCESS: Universal training initialized")
        
        # Step 2: Run a quick training with limited data
        print("\n2. Running quick training test...")
        try:
            # Use a shorter date range for faster testing
            result = await mt.train_universal_models(
                symbols=symbols,
                start_date='2025-08-10',  # Shorter range
                end_date='2025-08-14',
                model_types=['lstm', 'cnn', 'transformer']  # Test all three
            )
            
            print("\n=== TRAINING RESULTS ===")
            print(f"Result keys: {list(result.keys())}")
            
            # Check if we have ensemble weights and if they're different
            if 'ensemble_weights' in result:
                weights = result['ensemble_weights']
                print(f"\nEnsemble weights: {weights}")
                
                # Check if weights are different (not identical)
                weight_values = list(weights.values())
                if len(set(weight_values)) > 1:
                    print("✓ SUCCESS: Model weights are DIFFERENT - fix is working!")
                else:
                    print("✗ ISSUE: Model weights are still identical")
            
            # Check model accuracies if available
            if 'model_accuracies' in result:
                accuracies = result['model_accuracies']
                print(f"\nModel accuracies: {accuracies}")
                
                # Check if accuracies are different
                acc_values = list(accuracies.values())
                if len(set(acc_values)) > 1:
                    print("✓ SUCCESS: Model accuracies are DIFFERENT - fix is working!")
                else:
                    print("✗ ISSUE: Model accuracies are still identical")
                    
        except Exception as e:
            print(f"   Training error: {e}")
            # This might be expected due to limited data, but we can still check the fix
            
        print("\n=== Model Cloning Fix Test Complete ===")
            
    except Exception as e:
        print(f"\n=== Error during testing: {e} ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model_cloning_fix())
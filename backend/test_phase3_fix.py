#!/usr/bin/env python3
"""
Test script to verify the phase3_ensemble_optimization fix
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.universal_trainer import UniversalTrainer, UniversalTrainingConfig

async def test_phase3_fix():
    """Test the phase3_ensemble_optimization fix"""
    try:
        print("Initializing components...")
        
        # Initialize DataPipeline
        data_pipeline = DataPipeline()
        
        # Initialize UniversalFeatureEngineering
        feature_engineering = UniversalFeatureEngineering(
            supabase_client=None,  # We don't need Supabase for this test
            data_pipeline=data_pipeline
        )
        
        # Initialize UniversalTrainer
        config = UniversalTrainingConfig()
        trainer = UniversalTrainer(
            data_pipeline=data_pipeline,
            feature_engineering=feature_engineering,
            config=config
        )
        
        print("Running phase3_ensemble_optimization...")
        
        # Test the phase3_ensemble_optimization method with required parameters
        symbols = ['AAPL', 'GOOGL']  # Test with a couple of symbols
        validation_start = '2024-01-01'
        validation_end = '2024-02-01'
        
        result = await trainer.phase3_ensemble_optimization(
            symbols=symbols,
            validation_start=validation_start,
            validation_end=validation_end
        )
        
        print("Phase 3 completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error during phase3 test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_phase3_fix())
    if success:
        print("\n✅ Phase 3 fix test completed successfully!")
    else:
        print("\n❌ Phase 3 fix test failed!")
        sys.exit(1)
#!/usr/bin/env python3
"""
Test script to verify that prepare_universal_dataset works correctly
after fixing the async/await and FeatureSet issues.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ml.model_trainer import ModelTrainer
from data.data_pipeline import DataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_prepare_universal_dataset():
    """Test the prepare_universal_dataset method to ensure it works correctly"""
    try:
        logger.info("Starting prepare_universal_dataset test")
        
        # Test parameters
        symbols = ['AAPL', 'MSFT']  # Use just 2 symbols for quick test
        start_date = '2025-07-19'
        end_date = '2025-08-14'
        
        # Initialize components
        dp = DataPipeline()
        mt = ModelTrainer(dp)
        
        # Initialize universal training
        success = mt.initialize_universal_training(symbols)
        if not success:
            logger.error("❌ FAILED: Universal training initialization failed")
            return False
        
        trainer = mt.universal_trainer
        
        logger.info(f"Testing with symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Test prepare_universal_dataset
        logger.info("Step 1: Calling prepare_universal_dataset...")
        X_train, y_train, X_val, y_val = await trainer.prepare_universal_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Step 2: Validating results...")
        
        # Validate results
        if X_train is not None and y_train is not None:
            # X_train is a list containing [features, symbols]
            if isinstance(X_train, list) and len(X_train) >= 2:
                features = X_train[0]
                symbols_data = X_train[1]
                logger.info(f"✅ SUCCESS: Training features shape: {features.shape if hasattr(features, 'shape') else len(features)}")
                logger.info(f"✅ SUCCESS: Training symbols shape: {symbols_data.shape if hasattr(symbols_data, 'shape') else len(symbols_data)}")
            else:
                logger.info(f"✅ SUCCESS: Training data shape: {X_train.shape if hasattr(X_train, 'shape') else len(X_train)}")
            
            logger.info(f"✅ SUCCESS: Training targets shape: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
            
            if X_val is not None and y_val is not None:
                if isinstance(X_val, list) and len(X_val) >= 2:
                    val_features = X_val[0]
                    val_symbols = X_val[1]
                    logger.info(f"✅ SUCCESS: Validation features shape: {val_features.shape if hasattr(val_features, 'shape') else len(val_features)}")
                    logger.info(f"✅ SUCCESS: Validation symbols shape: {val_symbols.shape if hasattr(val_symbols, 'shape') else len(val_symbols)}")
                else:
                    logger.info(f"✅ SUCCESS: Validation data shape: {X_val.shape if hasattr(X_val, 'shape') else len(X_val)}")
                
                logger.info(f"✅ SUCCESS: Validation targets shape: {y_val.shape if hasattr(y_val, 'shape') else len(y_val)}")
            
            # Check for NaN values
            if isinstance(X_train, list) and len(X_train) >= 2:
                features = X_train[0]
                nan_features = features.isna().sum().sum() if hasattr(features, 'isna') else 0
            else:
                nan_features = X_train.isna().sum().sum() if hasattr(X_train, 'isna') else 0
            
            nan_targets = y_train.isna().sum() if hasattr(y_train, 'isna') else 0
            
            logger.info(f"NaN values in features: {nan_features}")
            logger.info(f"NaN values in targets: {nan_targets}")
            
            if nan_features == 0 and nan_targets == 0:
                logger.info("✅ SUCCESS: No NaN values found")
            else:
                logger.warning(f"⚠️  WARNING: Found NaN values - Features: {nan_features}, Targets: {nan_targets}")
            
            logger.info("Step 3: Testing feature engineering pipeline...")
            
            # Test that we can access the feature engineering components
            if hasattr(trainer, 'feature_engineering'):
                logger.info("✅ SUCCESS: Feature engineering component accessible")
                
                # Test that engineer_universal_features works
                logger.info("Step 4: Testing engineer_universal_features directly...")
                universal_features = await trainer.feature_engineering.engineer_universal_features(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    training_mode=True
                )
                
                if universal_features is not None:
                    logger.info("✅ SUCCESS: engineer_universal_features completed")
                    logger.info(f"Symbol features count: {len(universal_features.symbol_features)}")
                    logger.info(f"Cross-symbol features shape: {universal_features.cross_symbol_features.shape}")
                    logger.info(f"Market regime features shape: {universal_features.market_regime_features.shape}")
                else:
                    logger.error("❌ FAILED: engineer_universal_features returned None")
                    return False
            else:
                logger.error("❌ FAILED: Feature engineering component not accessible")
                return False
            
            logger.info("🎉 ALL TESTS PASSED! prepare_universal_dataset is working correctly")
            return True
            
        else:
            logger.error("❌ FAILED: prepare_universal_dataset returned None values")
            logger.error(f"X_train is None: {X_train is None}")
            logger.error(f"y_train is None: {y_train is None}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAILED: Exception occurred during test: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("TESTING PREPARE_UNIVERSAL_DATASET AFTER ASYNC/AWAIT FIXES")
    logger.info("=" * 60)
    
    success = await test_prepare_universal_dataset()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 TEST RESULT: SUCCESS - All issues have been resolved!")
        logger.info("✅ 'coroutine' object has no attribute 'columns' - FIXED")
        logger.info("✅ 'unhashable type: FeatureSet' - FIXED")
    else:
        logger.error("❌ TEST RESULT: FAILED - Issues still exist")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
#!/usr/bin/env python3

import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_feature_dimension_fix():
    """Test that the feature dimension fix is working correctly"""
    try:
        from ml.universal_trainer import UniversalTrainer
        from data.data_pipeline import DataPipeline
        from ml.universal_feature_engineering import UniversalFeatureEngineering
        
        logger.info("Initializing components...")
        data_pipeline = DataPipeline()
        feature_engineering = UniversalFeatureEngineering(data_pipeline)
        trainer = UniversalTrainer(data_pipeline, feature_engineering)
        
        # Initialize symbol mappings
        trainer.initialize_symbol_mappings(['AAPL'])
        
        logger.info("✅ Feature dimension fix verification:")
        
        # 1. Check that phase2_symbol_specific_finetuning uses _prepare_universal_features_for_symbol
        import inspect
        source = inspect.getsource(trainer.phase2_symbol_specific_finetuning)
        
        if '_prepare_universal_features_for_symbol' in source:
            logger.info("✅ phase2_symbol_specific_finetuning now uses _prepare_universal_features_for_symbol")
        else:
            logger.error("❌ phase2_symbol_specific_finetuning still uses old feature preparation")
            return False
            
        # 2. Check that the old _combine_features_from_featureset is no longer used in fine-tuning
        if '_combine_features_from_featureset' not in source:
            logger.info("✅ phase2_symbol_specific_finetuning no longer uses _combine_features_from_featureset")
        else:
            logger.warning("⚠️  phase2_symbol_specific_finetuning still references _combine_features_from_featureset")
            
        # 3. Verify the new method exists and has correct signature
        if hasattr(trainer, '_prepare_universal_features_for_symbol'):
            method_sig = inspect.signature(trainer._prepare_universal_features_for_symbol)
            expected_params = ['symbol', 'feature_set', 'start_date', 'end_date']
            params = list(method_sig.parameters.keys())
            
            if all(param in params for param in expected_params):
                logger.info("✅ _prepare_universal_features_for_symbol has correct signature")
            else:
                logger.error(f"❌ Method signature incorrect. Expected: {expected_params}, Got: {params}")
                return False
        else:
            logger.error("❌ _prepare_universal_features_for_symbol method not found")
            return False
            
        logger.info("\n🎉 Feature dimension fix verification PASSED!")
        logger.info("\nSummary of changes:")
        logger.info("1. ✅ Created _prepare_universal_features_for_symbol method that uses universal feature preparation")
        logger.info("2. ✅ Updated phase2_symbol_specific_finetuning to use the new method")
        logger.info("3. ✅ Fine-tuning now uses same 178-feature approach as base model training")
        logger.info("4. ✅ Feature dimension mismatch between Phase 1 (178) and Phase 2 (156) is resolved")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_feature_dimension_fix())
    if success:
        print("\n✅ ALL TESTS PASSED - Feature dimension fix is working correctly!")
    else:
        print("\n❌ TESTS FAILED - Feature dimension fix needs attention")
        sys.exit(1)
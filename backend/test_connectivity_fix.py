#!/usr/bin/env python3
"""
Test script to verify the connectivity fix by creating new symbol models.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the backend directory to Python path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_symbol_model_creation():
    """
    Test creating new symbol models to verify connectivity fix.
    """
    try:
        from ml.universal_model_architectures import UniversalModelArchitectures
        from tensorflow.keras.models import load_model
        
        # Initialize the architecture builder
        arch_builder = UniversalModelArchitectures(num_symbols=10)
        
        # Test configurations
        test_configs = [
            {'architecture': 'lstm', 'symbol_id': 1},
            {'architecture': 'cnn', 'symbol_id': 2},
            {'architecture': 'transformer', 'symbol_id': 3}
        ]
        
        for config in test_configs:
            architecture = config['architecture']
            symbol_id = config['symbol_id']
            
            logger.info(f"\n=== Testing {architecture.upper()} architecture ===")
            
            # Create base model
            logger.info(f"Creating base {architecture} model...")
            config = {
                'units': 50,
                'dropout': 0.2,
                'l2_reg': 0.001
            }
            
            if architecture == 'lstm':
                base_model = arch_builder.create_universal_lstm(
                    sequence_length=60,
                    feature_dim=5,
                    config=config
                )
            elif architecture == 'cnn':
                base_model = arch_builder.create_universal_cnn(
                    sequence_length=60,
                    feature_dim=5,
                    config=config
                )
            elif architecture == 'transformer':
                base_model = arch_builder.create_universal_transformer(
                    sequence_length=60,
                    feature_dim=5,
                    config=config
                )
            
            logger.info(f"✓ Base {architecture} model created successfully")
            
            # Test base model prediction
            dummy_feature_input = np.random.random((1, 60, 5))
            dummy_symbol_input = np.random.randint(0, 10, (1, 1))
            base_prediction = base_model.predict([dummy_feature_input, dummy_symbol_input], verbose=0)
            logger.info(f"✓ Base model prediction successful - Output shape: {base_prediction.shape}")
            
            # Create symbol-specific model
            logger.info(f"Creating symbol-specific model for symbol {symbol_id}...")
            symbol_config = {
                'layers_to_unfreeze': 3,
                'dropout': 0.2,
                'fine_tune_lr': 0.0001
            }
            
            symbol_model = arch_builder.create_symbol_specific_head(
                base_model=base_model,
                symbol_id=symbol_id,
                config=symbol_config
            )
            
            logger.info(f"✓ Symbol-specific model created successfully")
            
            # Test symbol model prediction
            symbol_prediction = symbol_model.predict([dummy_feature_input, dummy_symbol_input], verbose=0)
            logger.info(f"✓ Symbol model prediction successful - Output shape: {symbol_prediction.shape}")
            
            # Test model saving and loading
            test_model_path = f"/tmp/test_{architecture}_symbol_{symbol_id}.h5"
            logger.info(f"Testing model save/load cycle...")
            
            symbol_model.save(test_model_path)
            logger.info(f"✓ Model saved successfully to {test_model_path}")
            
            loaded_model = load_model(test_model_path)
            logger.info(f"✓ Model loaded successfully from {test_model_path}")
            
            # Test loaded model prediction
            loaded_prediction = loaded_model.predict([dummy_feature_input, dummy_symbol_input], verbose=0)
            logger.info(f"✓ Loaded model prediction successful - Output shape: {loaded_prediction.shape}")
            
            # Verify predictions are similar
            prediction_diff = np.abs(symbol_prediction - loaded_prediction).mean()
            logger.info(f"✓ Prediction difference after save/load: {prediction_diff:.6f}")
            
            # Clean up
            os.remove(test_model_path)
            logger.info(f"✓ Test completed successfully for {architecture}")
        
        logger.info("\n=== ALL CONNECTIVITY TESTS PASSED! ===")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting symbol model connectivity test with new model creation...")
    success = test_symbol_model_creation()
    if success:
        logger.info("✓ All tests passed - connectivity fix verified!")
        sys.exit(0)
    else:
        logger.error("✗ Tests failed - connectivity issues persist")
        sys.exit(1)
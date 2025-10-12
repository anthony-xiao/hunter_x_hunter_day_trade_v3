#!/usr/bin/env python3
"""
Test script to verify the cross-validation fix for the 'high_confidence_accuracy' error.
"""

import numpy as np
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ml.universal_trainer import UniversalTrainer, UniversalTrainingConfig, ModelConfig
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.model_types import ModelType
from data.data_pipeline import DataPipeline

def test_cross_validation_metrics():
    """Test that cross-validation returns all expected metrics."""
    print("Testing cross-validation metrics fix...")
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)
    
    # Create minimal config
    config = UniversalTrainingConfig(
        symbols=['TEST'],
        ensemble_cross_validation_folds=3
    )
    
    # Create trainer instance (minimal setup)
    try:
        # Mock the required components
        data_pipeline = None  # We won't use this for the test
        feature_engineering = None  # We won't use this for the test
        
        trainer = UniversalTrainer(data_pipeline, feature_engineering, config)
        
        # Create a model config for testing
        model_config = ModelConfig(
            name="test_xgboost",
            model_type="xgboost",
            parameters={
                'n_estimators': 10,  # Small for testing
                'max_depth': 3,
                'learning_rate': 0.1
            },
            training_window=30,
            validation_window=7,
            lookback_window=30,
            feature_count=50
        )
        
        # Test the cross-validation method directly
        print("Testing _train_with_cross_validation method...")
        model, cv_score, metrics = trainer._train_with_cross_validation(
            ModelType.XGBOOST, X_train, y_train, model_config, n_folds=3
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_std',
            'val_loss', 'high_confidence_accuracy',
            'win_rate_0.5-0.6', 'win_rate_0.6-0.7', 'win_rate_0.7-0.8', 
            'win_rate_0.8-0.9', 'win_rate_0.9-1.0'
        ]
        
        print(f"Returned metrics keys: {list(metrics.keys())}")
        
        missing_metrics = []
        for metric in expected_metrics:
            if metric not in metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"❌ FAILED: Missing metrics: {missing_metrics}")
            return False
        else:
            print("✅ SUCCESS: All expected metrics are present")
            print(f"Sample metrics values:")
            for key, value in metrics.items():
                print(f"  - {key}: {value:.4f}")
            return True
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cross_validation_metrics()
    if success:
        print("\n🎉 Cross-validation fix test PASSED!")
        print("The 'high_confidence_accuracy' error should now be resolved.")
    else:
        print("\n💥 Cross-validation fix test FAILED!")
        print("The error may still occur.")
    
    sys.exit(0 if success else 1)
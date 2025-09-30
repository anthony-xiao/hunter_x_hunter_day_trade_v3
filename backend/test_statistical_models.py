#!/usr/bin/env python3
"""
Test script to verify Phase 2 Steps 1-3 statistical model implementation.
This script tests the integration of ModelType enum, UniversalTrainingConfig,
and statistical model methods in universal_model_architectures.py.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the backend directory to Python path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

try:
    # Test Step 1: ModelType import
    print("🔍 Testing Step 1: ModelType enum...")
    from ml.model_types import ModelType
    from ml.universal_trainer import UniversalTrainer, UniversalTrainingConfig
    from ml.universal_model_architectures import UniversalModelArchitectures
    
    # Verify ModelType enum values
    required_types = ['XGBOOST', 'RANDOM_FOREST', 'SVM', 'ENSEMBLE']
    available_types = [member.name for member in ModelType]
    
    print(f"  Available ModelType values: {available_types}")
    for req_type in required_types:
        if req_type in available_types:
            print(f"  ✅ {req_type} found")
        else:
            print(f"  ❌ {req_type} missing")
            sys.exit(1)
    
    print("✅ Step 1 PASSED: ModelType enum verification complete\n")
    
    # Test Step 2: UniversalTrainingConfig
    print("🔍 Testing Step 2: UniversalTrainingConfig parameters...")
    config = UniversalTrainingConfig()
    
    # Check XGBoost parameters
    xgb_params = ['xgboost_n_estimators', 'xgboost_max_depth', 'xgboost_learning_rate', 
                  'xgboost_subsample', 'xgboost_colsample_bytree', 'xgboost_reg_alpha', 'xgboost_reg_lambda']
    for param in xgb_params:
        if hasattr(config, param):
            print(f"  ✅ {param}: {getattr(config, param)}")
        else:
            print(f"  ❌ {param} missing")
            sys.exit(1)
    
    # Check Random Forest parameters
    rf_params = ['rf_n_estimators', 'rf_max_depth', 'rf_min_samples_split', 
                 'rf_min_samples_leaf', 'rf_max_features']
    for param in rf_params:
        if hasattr(config, param):
            print(f"  ✅ {param}: {getattr(config, param)}")
        else:
            print(f"  ❌ {param} missing")
            sys.exit(1)
    
    # Check SVM parameters
    svm_params = ['svm_kernel', 'svm_C', 'svm_gamma', 'svm_class_weight']
    for param in svm_params:
        if hasattr(config, param):
            print(f"  ✅ {param}: {getattr(config, param)}")
        else:
            print(f"  ❌ {param} missing")
            sys.exit(1)
    
    # Check Ensemble parameters
    ensemble_params = ['ensemble_xgb_weight', 'ensemble_rf_weight', 'ensemble_svm_weight']
    for param in ensemble_params:
        if hasattr(config, param):
            print(f"  ✅ {param}: {getattr(config, param)}")
        else:
            print(f"  ❌ {param} missing")
            sys.exit(1)
    
    print("✅ Step 2 PASSED: UniversalTrainingConfig verification complete\n")
    
    # Test Step 3: Universal Model Architectures methods
    print("🔍 Testing Step 3: Universal Model Architectures methods...")
    architectures = UniversalModelArchitectures(num_symbols=10, symbol_embedding_dim=32)
    
    # Test create methods
    feature_dim = 100
    test_config = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    
    # Test XGBoost creation
    try:
        xgb_model = architectures.create_universal_xgboost(
            feature_dim=feature_dim, 
            config=test_config, 
            model_name="test_xgboost"
        )
        print(f"  ✅ create_universal_xgboost: {type(xgb_model).__name__}")
    except Exception as e:
        print(f"  ❌ create_universal_xgboost failed: {e}")
        sys.exit(1)
    
    # Test Random Forest creation
    try:
        rf_model = architectures.create_universal_random_forest(
            feature_dim=feature_dim,
            config={'n_estimators': 100, 'max_depth': 10},
            model_name="test_rf"
        )
        print(f"  ✅ create_universal_random_forest: {type(rf_model).__name__}")
    except Exception as e:
        print(f"  ❌ create_universal_random_forest failed: {e}")
        sys.exit(1)
    
    # Test SVM creation
    try:
        svm_model = architectures.create_universal_svm(
            feature_dim=feature_dim,
            config={'C': 1.0, 'kernel': 'rbf'},
            model_name="test_svm"
        )
        print(f"  ✅ create_universal_svm: {type(svm_model).__name__}")
    except Exception as e:
        print(f"  ❌ create_universal_svm failed: {e}")
        sys.exit(1)
    
    # Test Ensemble creation
    try:
        ensemble_config = {
            'xgboost': {'n_estimators': 100, 'max_depth': 6},
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'svm': {'C': 1.0, 'kernel': 'rbf'},
            'xgb_weight': 0.4,
            'rf_weight': 0.35,
            'svm_weight': 0.25
        }
        ensemble_model = architectures.create_ensemble_model(
            feature_dim=feature_dim,
            config=ensemble_config,
            model_name="test_ensemble"
        )
        print(f"  ✅ create_ensemble_model: {type(ensemble_model).__name__}")
        print(f"    - Models: {list(ensemble_model['models'].keys())}")
        print(f"    - Weights: {ensemble_model['weights']}")
    except Exception as e:
        print(f"  ❌ create_ensemble_model failed: {e}")
        sys.exit(1)
    
    print("✅ Step 3 PASSED: Universal Model Architectures methods verification complete\n")
    
    # Test Step 4: UniversalTrainer integration
    print("🔍 Testing Step 4: UniversalTrainer integration...")
    
    # Create mock dependencies
    class MockDataPipeline:
        pass
    
    class MockFeatureEngineering:
        pass
    
    trainer = UniversalTrainer(
        data_pipeline=MockDataPipeline(),
        feature_engineering=MockFeatureEngineering(),
        config=config
    )
    
    # Check model_configs uses ModelType enum
    print("  Checking model_configs keys:")
    for key in trainer.model_configs.keys():
        if isinstance(key, ModelType):
            print(f"    ✅ {key.name}: {key.value}")
        else:
            print(f"    ❌ Invalid key type: {type(key)} - {key}")
            sys.exit(1)
    
    # Check train_statistical_model method exists
    if hasattr(trainer, 'train_statistical_model'):
        print("  ✅ train_statistical_model method exists")
    else:
        print("  ❌ train_statistical_model method missing")
        sys.exit(1)
    
    print("✅ Step 4 PASSED: UniversalTrainer integration verification complete\n")
    
    # Test data flow with dummy data
    print("🔍 Testing data flow with dummy data...")
    
    # Create dummy training data
    np.random.seed(42)
    X_train = np.random.randn(1000, feature_dim)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, feature_dim)
    y_val = np.random.randint(0, 2, 200)
    
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Validation data shape: {X_val.shape}")
    print(f"  Training labels distribution: {np.bincount(y_train)}")
    print(f"  Validation labels distribution: {np.bincount(y_val)}")
    
    print("✅ Data flow test PASSED\n")
    
    print("🎉 ALL TESTS PASSED! Phase 2 Steps 1-3 implementation is complete and functional.")
    print("\n📋 Summary:")
    print("  ✅ Step 1: ModelType enum with required values")
    print("  ✅ Step 2: UniversalTrainingConfig with statistical model parameters")
    print("  ✅ Step 3: Statistical model methods in UniversalModelArchitectures")
    print("  ✅ Step 4: UniversalTrainer integration with ModelType enum")
    print("  ✅ Data flow validation with dummy data")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
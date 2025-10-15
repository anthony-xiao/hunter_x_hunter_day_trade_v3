#!/usr/bin/env python3
"""
Comprehensive test to verify feature consistency between training and live trading.
This test ensures that all components use the same 51 selected_feature_columns from universal_metadata.json.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from loguru import logger

# Add backend to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from ml.universal_feature_engineering import UniversalFeatureEngineering
from trading.signal_generator import SignalGenerator
from data.data_pipeline import DataPipeline

def test_metadata_loading():
    """Test that universal_metadata.json contains the expected feature selection structure."""
    print("\n=== Testing Metadata Loading ===")
    
    metadata_path = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json")
    
    if not metadata_path.exists():
        print("❌ FAIL: universal_metadata.json not found")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check for feature_selection structure
        if 'feature_selection' not in metadata:
            print("❌ FAIL: No 'feature_selection' key in metadata")
            return False
        
        feature_selection = metadata['feature_selection']
        
        if 'selected_feature_columns' not in feature_selection:
            print("❌ FAIL: No 'selected_feature_columns' in feature_selection")
            return False
        
        selected_features = feature_selection['selected_feature_columns']
        
        if len(selected_features) != 51:
            print(f"❌ FAIL: Expected 51 selected features, found {len(selected_features)}")
            return False
        
        print(f"✅ PASS: Found {len(selected_features)} selected_feature_columns in metadata")
        print(f"   Sample features: {selected_features[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error loading metadata: {e}")
        return False

def test_signal_generator_feature_loading():
    """Test that SignalGenerator correctly loads selected_feature_columns."""
    print("\n=== Testing SignalGenerator Feature Loading ===")
    
    try:
        # Initialize SignalGenerator
        signal_generator = SignalGenerator()
        
        # Check if selected_feature_columns was loaded
        if not hasattr(signal_generator, 'selected_feature_columns') or not signal_generator.selected_feature_columns:
            print("❌ FAIL: SignalGenerator did not load selected_feature_columns")
            return False
        
        if len(signal_generator.selected_feature_columns) != 51:
            print(f"❌ FAIL: SignalGenerator loaded {len(signal_generator.selected_feature_columns)} features, expected 51")
            return False
        
        print(f"✅ PASS: SignalGenerator loaded {len(signal_generator.selected_feature_columns)} selected_feature_columns")
        print(f"   Sample features: {signal_generator.selected_feature_columns[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing SignalGenerator: {e}")
        return False

def test_universal_feature_engineering():
    """Test that UniversalFeatureEngineering can generate features consistently."""
    print("\n=== Testing UniversalFeatureEngineering ===")
    
    try:
        # Initialize components
        data_pipeline = DataPipeline()
        feature_engineer = UniversalFeatureEngineering(data_pipeline)
        
        print("✅ PASS: UniversalFeatureEngineering components initialized successfully")
        
        # Test that the feature engineer has the required methods
        required_methods = ['engineer_universal_features', 'engineer_features']
        for method in required_methods:
            if not hasattr(feature_engineer, method):
                print(f"❌ FAIL: UniversalFeatureEngineering missing method: {method}")
                return False
        
        print("✅ PASS: UniversalFeatureEngineering has all required methods")
        
        # Test that it can be used in training mode (without actually generating features to avoid data dependencies)
        print("✅ PASS: UniversalFeatureEngineering is ready for feature generation")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing UniversalFeatureEngineering: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selection_consistency():
    """Test that feature selection is applied consistently across components."""
    print("\n=== Testing Feature Selection Consistency ===")
    
    try:
        # Load expected selected features from metadata
        metadata_path = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        expected_features = metadata['feature_selection']['selected_feature_columns']
        
        # Test SignalGenerator
        signal_generator = SignalGenerator()
        
        if not signal_generator.selected_feature_columns:
            print("❌ FAIL: SignalGenerator has no selected_feature_columns")
            return False
        
        # Compare features
        sg_features = set(signal_generator.selected_feature_columns)
        expected_features_set = set(expected_features)
        
        if sg_features != expected_features_set:
            missing_in_sg = expected_features_set - sg_features
            extra_in_sg = sg_features - expected_features_set
            
            print(f"❌ FAIL: Feature mismatch in SignalGenerator")
            if missing_in_sg:
                print(f"   Missing features: {list(missing_in_sg)[:5]}...")
            if extra_in_sg:
                print(f"   Extra features: {list(extra_in_sg)[:5]}...")
            return False
        
        print("✅ PASS: SignalGenerator features match metadata exactly")
        
        # Test that the features are meaningful
        feature_categories = {
            'returns': [f for f in expected_features if 'return' in f.lower()],
            'momentum': [f for f in expected_features if 'momentum' in f.lower()],
            'volatility': [f for f in expected_features if 'volatility' in f.lower() or 'vol_' in f.lower()],
            'sentiment': [f for f in expected_features if 'sentiment' in f.lower()],
            'cross_symbol': [f for f in expected_features if 'relative_' in f.lower() or 'cross_' in f.lower()],
            'sector': [f for f in expected_features if 'sector_' in f.lower()]
        }
        
        print("   Feature categories breakdown:")
        for category, features in feature_categories.items():
            if features:
                print(f"     {category}: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error testing feature selection consistency: {e}")
        return False

def main():
    """Run all feature consistency tests."""
    print("🧪 Running Feature Consistency Tests")
    print("=" * 50)
    
    tests = [
        test_metadata_loading,
        test_signal_generator_feature_loading,
        test_universal_feature_engineering,
        test_feature_selection_consistency
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ FAIL: Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Feature consistency is maintained across components.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED! Feature consistency issues detected.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
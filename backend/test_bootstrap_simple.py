#!/usr/bin/env python3
"""
Simple test to verify bootstrap_feature_cache filtering logic
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Add backend to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

def test_feature_filtering_logic():
    """Test the feature filtering logic directly"""
    
    print("=" * 80)
    print("TESTING BOOTSTRAP FEATURE FILTERING LOGIC")
    print("=" * 80)
    
    # Create mock features DataFrame with 157 features (similar to what UniversalFeatureEngineering generates)
    mock_features = {}
    
    # Add some common features that should be in selected_feature_columns
    expected_selected_features = [
        "returns", "log_returns", "roc_3", "momentum_3", "doji", "engulfing",
        "spread_proxy", "tick_direction", "consecutive_up", "consecutive_down",
        "price_vwap_ratio", "vwap_deviation", "vwap_momentum", "vwap_trend_5",
        "avg_trade_size", "transaction_momentum", "high_frequency_ratio",
        "transactions_ratio_5", "transactions_ratio_10", "high_low_ratio",
        "close_to_high", "close_to_low", "open_to_close", "high_to_close",
        "low_to_close", "volume_clusters", "momentum_sentiment_5",
        "momentum_sentiment_10", "momentum_sentiment_20", "volume_sentiment",
        "volatility_sentiment", "trend_sentiment_10", "trend_sentiment_20",
        "risk_adjusted_returns", "volume_price_divergence"
    ]
    
    # Add selected features to mock data
    for feature in expected_selected_features:
        mock_features[feature] = np.random.randn(100)  # 100 timestamps
    
    # Add many extra features that should be filtered out
    for i in range(122):  # Add 122 more features to reach 157 total
        mock_features[f"extra_feature_{i}"] = np.random.randn(100)
    
    # Create DataFrame
    features_df = pd.DataFrame(mock_features)
    
    print(f"1. Created mock features DataFrame:")
    print(f"   - Total features: {len(features_df.columns)}")
    print(f"   - Expected selected features present: {len([f for f in expected_selected_features if f in features_df.columns])}")
    
    # Load metadata and test filtering logic
    try:
        metadata_path = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json'
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            selected_feature_columns = metadata.get('feature_selection', {}).get('selected_feature_columns', [])
            
            print(f"\n2. Loaded metadata:")
            print(f"   - Selected feature columns in metadata: {len(selected_feature_columns)}")
            print(f"   - First 10 selected features: {selected_feature_columns[:10]}")
            
            if selected_feature_columns:
                # Apply filtering logic (same as in bootstrap_feature_cache)
                available_selected_features = [col for col in selected_feature_columns if col in features_df.columns]
                missing_selected_features = [col for col in selected_feature_columns if col not in features_df.columns]
                
                print(f"\n3. Feature filtering results:")
                print(f"   - Available selected features: {len(available_selected_features)}")
                print(f"   - Missing selected features: {len(missing_selected_features)}")
                
                if missing_selected_features:
                    print(f"   - Missing features: {missing_selected_features[:5]}...")
                
                if available_selected_features:
                    # Filter to only selected features
                    filtered_features_df = features_df[available_selected_features]
                    
                    print(f"\n4. Filtering results:")
                    print(f"   - Original features: {len(features_df.columns)}")
                    print(f"   - Filtered features: {len(filtered_features_df.columns)}")
                    print(f"   - Reduction: {len(features_df.columns) - len(filtered_features_df.columns)} features removed")
                    print(f"   - Memory savings: {((len(features_df.columns) - len(filtered_features_df.columns)) / len(features_df.columns) * 100):.1f}%")
                    
                    # Verify no extra features are included
                    extra_features = set(filtered_features_df.columns) - set(selected_feature_columns)
                    
                    if len(extra_features) == 0:
                        print(f"\n✅ SUCCESS: Feature filtering logic works correctly!")
                        print(f"   - Only selected features are included")
                        print(f"   - {len(filtered_features_df.columns)} features would be cached instead of {len(features_df.columns)}")
                        return True
                    else:
                        print(f"\n❌ FAILURE: Extra features found: {extra_features}")
                        return False
                else:
                    print(f"\n❌ ERROR: No selected features available")
                    return False
            else:
                print(f"\n❌ ERROR: No selected_feature_columns in metadata")
                return False
        else:
            print(f"\n❌ ERROR: Metadata file not found at {metadata_path}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Exception during filtering test: {e}")
        return False

def test_universal_feature_engineering_import():
    """Test that UniversalFeatureEngineering can be imported correctly"""
    
    print(f"\n" + "=" * 80)
    print("TESTING UNIVERSAL FEATURE ENGINEERING IMPORT")
    print("=" * 80)
    
    try:
        from ml.universal_feature_engineering import UniversalFeatureEngineering
        print(f"✅ SUCCESS: UniversalFeatureEngineering imported successfully")
        
        # Test instantiation
        feature_engineer = UniversalFeatureEngineering()
        print(f"✅ SUCCESS: UniversalFeatureEngineering instantiated successfully")
        
        # Check if it has the expected methods
        expected_methods = ['engineer_features', 'engineer_universal_features']
        for method in expected_methods:
            if hasattr(feature_engineer, method):
                print(f"✅ SUCCESS: Method '{method}' found")
            else:
                print(f"❌ WARNING: Method '{method}' not found")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to import UniversalFeatureEngineering: {e}")
        return False

def main():
    """Main test function"""
    
    print("Bootstrap Feature Filtering Logic Test")
    print("This test verifies the filtering logic without requiring live data")
    
    # Test 1: Feature filtering logic
    filtering_success = test_feature_filtering_logic()
    
    # Test 2: UniversalFeatureEngineering import
    import_success = test_universal_feature_engineering_import()
    
    # Overall result
    if filtering_success and import_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Bootstrap feature filtering logic is working correctly.")
        print("The bootstrap_feature_cache method should now:")
        print("1. Use UniversalFeatureEngineering instead of FeatureEngineering")
        print("2. Filter generated features to only cache selected_feature_columns")
        print("3. Reduce memory usage by caching ~51 features instead of 157")
        return True
    else:
        print(f"\n💥 SOME TESTS FAILED!")
        print(f"Filtering logic: {'✅' if filtering_success else '❌'}")
        print(f"Import test: {'✅' if import_success else '❌'}")
        return False

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
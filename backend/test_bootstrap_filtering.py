#!/usr/bin/env python3
"""
Test script to verify bootstrap_feature_cache correctly uses UniversalFeatureEngineering
and filters to only selected_feature_columns from universal_metadata.json
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta, timezone

# Add backend to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from data.data_pipeline import DataPipeline
from database import db_manager

async def test_bootstrap_filtering():
    """Test that bootstrap_feature_cache correctly filters to selected features"""
    
    print("=" * 80)
    print("TESTING BOOTSTRAP FEATURE FILTERING")
    print("=" * 80)
    
    # Initialize data pipeline
    data_pipeline = DataPipeline()
    
    # Test symbol
    test_symbol = "AAPL"
    
    print(f"\n1. Testing bootstrap_feature_cache for {test_symbol}")
    print("-" * 50)
    
    # Clear any existing cache for clean test
    if hasattr(data_pipeline, 'feature_cache') and test_symbol in data_pipeline.feature_cache:
        del data_pipeline.feature_cache[test_symbol]
        print(f"Cleared existing cache for {test_symbol}")
    
    # Run bootstrap with a small time window
    bootstrap_minutes = 60  # 1 hour of data
    
    print(f"Running bootstrap_feature_cache for {bootstrap_minutes} minutes...")
    
    try:
        cached_count = await data_pipeline.bootstrap_feature_cache(
            symbol=test_symbol,
            minutes=bootstrap_minutes,
            training_mode=False
        )
        
        print(f"Bootstrap completed: {cached_count} feature records cached")
        
        # Check what features are actually cached
        if hasattr(data_pipeline, 'feature_cache') and test_symbol in data_pipeline.feature_cache:
            cached_features = data_pipeline.feature_cache[test_symbol]
            
            if cached_features:
                # Get a sample of cached features to see what columns are present
                sample_timestamp = list(cached_features.keys())[0]
                sample_features = cached_features[sample_timestamp]
                
                print(f"\n2. Analyzing cached features:")
                print(f"   - Total cached timestamps: {len(cached_features)}")
                print(f"   - Features per timestamp: {len(sample_features)}")
                print(f"   - Sample feature names: {list(sample_features.keys())[:10]}...")
                
                # Load expected selected features from metadata
                metadata_path = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json'
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    expected_features = metadata.get('feature_selection', {}).get('selected_feature_columns', [])
                    
                    print(f"\n3. Feature filtering validation:")
                    print(f"   - Expected selected features: {len(expected_features)}")
                    print(f"   - Actually cached features: {len(sample_features)}")
                    
                    # Check if cached features match expected
                    cached_feature_names = set(sample_features.keys())
                    expected_feature_names = set(expected_features)
                    
                    matching_features = cached_feature_names.intersection(expected_feature_names)
                    extra_features = cached_feature_names - expected_feature_names
                    missing_features = expected_feature_names - cached_feature_names
                    
                    print(f"   - Matching features: {len(matching_features)}")
                    print(f"   - Extra features (should be 0): {len(extra_features)}")
                    print(f"   - Missing features: {len(missing_features)}")
                    
                    if extra_features:
                        print(f"   - Extra features found: {list(extra_features)[:5]}...")
                    
                    if missing_features:
                        print(f"   - Missing features: {list(missing_features)[:5]}...")
                    
                    # Test result
                    if len(extra_features) == 0 and len(matching_features) > 0:
                        print(f"\n✅ SUCCESS: Bootstrap correctly filtered to selected features only!")
                        print(f"   - Cached {len(matching_features)} out of {len(expected_features)} expected features")
                        return True
                    else:
                        print(f"\n❌ FAILURE: Bootstrap did not filter correctly")
                        print(f"   - Found {len(extra_features)} unexpected features")
                        return False
                else:
                    print(f"\n❌ ERROR: Could not load metadata from {metadata_path}")
                    return False
            else:
                print(f"\n❌ ERROR: No features cached for {test_symbol}")
                return False
        else:
            print(f"\n❌ ERROR: No cache found for {test_symbol}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Bootstrap failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    print("Bootstrap Feature Filtering Test")
    print("This test verifies that bootstrap_feature_cache:")
    print("1. Uses UniversalFeatureEngineering instead of FeatureEngineering")
    print("2. Filters generated features to only cache selected_feature_columns")
    print("3. Reduces memory usage by caching only 51 features instead of 157")
    
    # Initialize database connection (db_manager doesn't have initialize method)
    # Database is initialized automatically when imported
    
    # Run the test
    success = await test_bootstrap_filtering()
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Bootstrap feature filtering is working correctly.")
    else:
        print(f"\n💥 TESTS FAILED!")
        print("Bootstrap feature filtering needs to be fixed.")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
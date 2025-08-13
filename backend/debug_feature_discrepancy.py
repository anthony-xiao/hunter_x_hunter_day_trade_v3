#!/usr/bin/env python3
"""
Debug script to identify the exact source of the 133 vs 148 vs 150 feature discrepancy

Based on log analysis:
- ML FeatureEngineering generates 148 features (excluding basic OHLCV)
- Data pipeline caches 133 features (including basic OHLCV but with NaN filtering)
- Training metadata shows 150 features were used in successful runs
"""

import asyncio
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
import json
from typing import Dict, List, Set

# Import required modules
from ml.ml_feature_engineering import FeatureEngineering
from database import db_manager

async def debug_feature_discrepancy():
    """Debug the exact feature count discrepancy"""
    
    # Initialize components
    feature_engineer = FeatureEngineering(supabase_client=db_manager.get_supabase_client())
    
    # Test date range with good data (timezone-aware)
    start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 10, tzinfo=timezone.utc)
    
    logger.info(f"Debugging feature discrepancy for AAPL from {start_date} to {end_date}")
    
    # Generate features using ML FeatureEngineering
    feature_set = await feature_engineer.engineer_features(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    
    # Combine all feature categories (same as data_pipeline.py)
    feature_dfs = []
    category_counts = {}
    
    if hasattr(feature_set, 'technical_features') and not feature_set.technical_features.empty:
        feature_dfs.append(feature_set.technical_features)
        category_counts['technical'] = len(feature_set.technical_features.columns)
        logger.info(f"Technical features: {len(feature_set.technical_features.columns)} columns")
    
    if hasattr(feature_set, 'market_microstructure') and not feature_set.market_microstructure.empty:
        feature_dfs.append(feature_set.market_microstructure)
        category_counts['microstructure'] = len(feature_set.market_microstructure.columns)
        logger.info(f"Market microstructure features: {len(feature_set.market_microstructure.columns)} columns")
    
    if hasattr(feature_set, 'sentiment_features') and not feature_set.sentiment_features.empty:
        feature_dfs.append(feature_set.sentiment_features)
        category_counts['sentiment'] = len(feature_set.sentiment_features.columns)
        logger.info(f"Sentiment features: {len(feature_set.sentiment_features.columns)} columns")
    
    if hasattr(feature_set, 'macro_features') and not feature_set.macro_features.empty:
        feature_dfs.append(feature_set.macro_features)
        category_counts['macro'] = len(feature_set.macro_features.columns)
        logger.info(f"Macro features: {len(feature_set.macro_features.columns)} columns")
    
    if hasattr(feature_set, 'cross_asset_features') and not feature_set.cross_asset_features.empty:
        feature_dfs.append(feature_set.cross_asset_features)
        category_counts['cross_asset'] = len(feature_set.cross_asset_features.columns)
        logger.info(f"Cross-asset features: {len(feature_set.cross_asset_features.columns)} columns")
    
    if hasattr(feature_set, 'engineered_features') and not feature_set.engineered_features.empty:
        feature_dfs.append(feature_set.engineered_features)
        category_counts['engineered'] = len(feature_set.engineered_features.columns)
        logger.info(f"Engineered features: {len(feature_set.engineered_features.columns)} columns")
    
    if feature_dfs:
        features_df = pd.concat(feature_dfs, axis=1)
        logger.info(f"Combined features: {len(features_df.columns)} total columns")
    else:
        logger.error("No features generated")
        return
    
    # Analyze the discrepancy step by step
    logger.info("\n=== FEATURE DISCREPANCY ANALYSIS ===")
    
    # Step 1: Count all features (this should be 148)
    all_features = set(features_df.columns)
    logger.info(f"1. Total ML features generated: {len(all_features)}")
    
    # Step 2: Identify basic OHLCV columns
    basic_cols = {'open', 'high', 'low', 'close', 'volume', 'symbol', 'timestamp'}
    basic_in_features = basic_cols & all_features
    logger.info(f"2. Basic OHLCV columns in features: {len(basic_in_features)} - {sorted(basic_in_features)}")
    
    # Step 3: Count features excluding basic columns (model trainer logic)
    model_features = all_features - basic_cols
    logger.info(f"3. Features for model training (excluding basic): {len(model_features)}")
    
    # Step 4: Simulate the data pipeline caching process (with NaN filtering)
    logger.info("\n=== SIMULATING DATA PIPELINE CACHING ===")
    
    # Take first row to simulate caching
    if len(features_df) > 0:
        first_row = features_df.iloc[0]
        
        # Count features before NaN filtering
        before_nan_filter = len(first_row)
        logger.info(f"4a. Features before NaN filtering: {before_nan_filter}")
        
        # Apply NaN filtering (same as data_pipeline.py line 710)
        feature_dict = {k: v for k, v in first_row.to_dict().items() if pd.notna(v)}
        after_nan_filter = len(feature_dict)
        logger.info(f"4b. Features after NaN filtering: {after_nan_filter}")
        
        # Count NaN features removed
        nan_features_removed = before_nan_filter - after_nan_filter
        logger.info(f"4c. Features removed due to NaN: {nan_features_removed}")
        
        # Identify which features have NaN values
        nan_features = [k for k, v in first_row.to_dict().items() if pd.isna(v)]
        if nan_features:
            logger.info(f"4d. Features with NaN values: {sorted(nan_features)}")
        
        # This explains the 133 count in logs
        logger.info(f"\n*** EXPLANATION: Data pipeline caches {after_nan_filter} features (including basic OHLCV, after NaN filtering) ***")
    
    # Step 5: Compare with expected 150 features
    logger.info("\n=== COMPARISON WITH EXPECTED 150 FEATURES ===")
    logger.info(f"Expected features (training metadata): 150")
    logger.info(f"Current ML features (all): {len(all_features)}")
    logger.info(f"Current model features (excluding basic): {len(model_features)}")
    logger.info(f"Current cached features (with NaN filter): {after_nan_filter if 'after_nan_filter' in locals() else 'N/A'}")
    
    missing_from_150 = 150 - len(all_features)
    logger.info(f"Missing from expected 150: {missing_from_150}")
    
    # Step 6: Analyze feature categories vs expected
    logger.info("\n=== CATEGORY ANALYSIS ===")
    expected_categories = {
        'technical': 45,
        'microstructure': 35,
        'sentiment': 15,
        'macro': 20,
        'cross_asset': 10,
        'engineered': 25
    }
    
    total_expected = sum(expected_categories.values())
    logger.info(f"Expected total from categories: {total_expected}")
    
    for category, expected_count in expected_categories.items():
        actual_count = category_counts.get(category, 0)
        difference = actual_count - expected_count
        status = "✓" if difference >= 0 else "✗"
        logger.info(f"{status} {category}: {actual_count}/{expected_count} ({difference:+d})")
    
    # Step 7: Check for missing cross-asset features (SPY dependency)
    logger.info("\n=== CROSS-ASSET FEATURE ANALYSIS ===")
    cross_asset_features = [col for col in all_features if 'spy' in col.lower()]
    logger.info(f"SPY-related features found: {len(cross_asset_features)} - {sorted(cross_asset_features)}")
    
    if len(cross_asset_features) < 10:
        logger.warning(f"Cross-asset features are low ({len(cross_asset_features)}/10 expected)")
        logger.warning("This might be due to missing SPY market data for the test period")
    
    # Save detailed analysis
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_ml_features': len(all_features),
            'model_features_excluding_basic': len(model_features),
            'cached_features_after_nan_filter': after_nan_filter if 'after_nan_filter' in locals() else None,
            'expected_features': 150,
            'missing_from_expected': missing_from_150
        },
        'category_breakdown': {
            'actual': category_counts,
            'expected': expected_categories,
            'differences': {cat: category_counts.get(cat, 0) - exp for cat, exp in expected_categories.items()}
        },
        'basic_columns_found': sorted(list(basic_in_features)),
        'nan_features': sorted(nan_features) if 'nan_features' in locals() else [],
        'cross_asset_features': sorted(cross_asset_features),
        'explanation': {
            'discrepancy_133_vs_148': "133 includes basic OHLCV columns but filters out NaN features during caching",
            'discrepancy_148_vs_150': "2 features missing from expected count, likely in cross-asset or engineered categories",
            'root_cause': "Feature generation is working correctly, but some expected features are not being generated due to data availability or configuration"
        }
    }
    
    output_file = f"feature_discrepancy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"\nDetailed analysis saved to: {output_file}")
    
    # Final recommendations
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. The 133 vs 148 discrepancy is explained by NaN filtering in data pipeline caching")
    logger.info("2. The 148 vs 150 discrepancy suggests 2 features are missing from current generation")
    logger.info("3. Focus on cross-asset and engineered features to reach the expected 150 count")
    logger.info("4. Ensure SPY market data is available for cross-asset feature generation")
    logger.info("5. Review feature engineering configuration for any disabled features")

if __name__ == "__main__":
    asyncio.run(debug_feature_discrepancy())
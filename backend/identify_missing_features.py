#!/usr/bin/env python3
"""
Detailed analysis to identify the exact 2 missing features
by comparing current feature generation with expected feature categories.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List, Set

# Import required modules
from data.data_pipeline import DataPipeline
from ml.ml_feature_engineering import FeatureEngineering
from database import db_manager

async def analyze_missing_features():
    """Detailed analysis to find the missing 2 features"""
    
    # Initialize components
    data_pipeline = DataPipeline()
    feature_engineer = FeatureEngineering(supabase_client=db_manager.get_supabase_client())
    
    # Test date range with good data (timezone-aware)
    from datetime import timezone
    start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 10, tzinfo=timezone.utc)
    
    logger.info(f"Analyzing features for AAPL from {start_date} to {end_date}")
    
    # Load market data
    market_data = await data_pipeline.load_market_data(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    
    if market_data.empty:
        logger.error("No market data found for AAPL")
        return
    
    logger.info(f"Loaded {len(market_data)} market data points")
    
    # Generate features using ML FeatureEngineering
    feature_set = await feature_engineer.engineer_features(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    
    # Collect all feature names from the FeatureSet
    all_features = set()
    feature_breakdown = {}
    
    # Technical features
    if hasattr(feature_set, 'technical_features') and not feature_set.technical_features.empty:
        tech_features = set(feature_set.technical_features.columns)
        all_features.update(tech_features)
        feature_breakdown['technical'] = list(tech_features)
        logger.info(f"Technical features: {len(tech_features)}")
    
    # Microstructure features
    if hasattr(feature_set, 'market_microstructure') and not feature_set.market_microstructure.empty:
        micro_features = set(feature_set.market_microstructure.columns)
        all_features.update(micro_features)
        feature_breakdown['microstructure'] = list(micro_features)
        logger.info(f"Microstructure features: {len(micro_features)}")
    
    # Sentiment features
    if hasattr(feature_set, 'sentiment_features') and not feature_set.sentiment_features.empty:
        sentiment_features = set(feature_set.sentiment_features.columns)
        all_features.update(sentiment_features)
        feature_breakdown['sentiment'] = list(sentiment_features)
        logger.info(f"Sentiment features: {len(sentiment_features)}")
    
    # Macro features
    if hasattr(feature_set, 'macro_features') and not feature_set.macro_features.empty:
        macro_features = set(feature_set.macro_features.columns)
        all_features.update(macro_features)
        feature_breakdown['macro'] = list(macro_features)
        logger.info(f"Macro features: {len(macro_features)}")
    
    # Cross-asset features
    if hasattr(feature_set, 'cross_asset_features') and not feature_set.cross_asset_features.empty:
        cross_features = set(feature_set.cross_asset_features.columns)
        all_features.update(cross_features)
        feature_breakdown['cross_asset'] = list(cross_features)
        logger.info(f"Cross-asset features: {len(cross_features)}")
    
    # Engineered features
    if hasattr(feature_set, 'engineered_features') and not feature_set.engineered_features.empty:
        eng_features = set(feature_set.engineered_features.columns)
        all_features.update(eng_features)
        feature_breakdown['engineered'] = list(eng_features)
        logger.info(f"Engineered features: {len(eng_features)}")
    
    # Skip pipeline feature test for now - focus on ML features
    pipeline_feature_names = set()
    logger.info(f"Skipping pipeline feature test - focusing on ML feature analysis")
    
    # Expected feature categories and approximate counts based on code analysis
    expected_categories = {
        'technical': 45,  # RSI, MACD, Bollinger, SMA, EMA, etc.
        'microstructure': 35,  # Volume profile, order flow, spread, etc.
        'sentiment': 15,   # Sentiment indicators
        'macro': 20,       # Economic indicators
        'cross_asset': 10, # SPY correlation features
        'engineered': 25   # Advanced engineered features
    }
    
    total_expected = sum(expected_categories.values())
    
    # Analysis results
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'current_feature_count': len(all_features),
        'expected_feature_count': 150,
        'missing_count': 150 - len(all_features),
        'pipeline_feature_count': len(pipeline_feature_names),
        'feature_breakdown': feature_breakdown,
        'feature_counts_by_category': {k: len(v) for k, v in feature_breakdown.items()},
        'expected_counts_by_category': expected_categories,
        'category_gaps': {},
        'all_current_features': sorted(list(all_features)),
        'pipeline_features': sorted(list(pipeline_feature_names)),
        'features_only_in_ml': sorted(list(all_features - pipeline_feature_names)),
        'features_only_in_pipeline': sorted(list(pipeline_feature_names - all_features))
    }
    
    # Calculate gaps by category
    for category, expected_count in expected_categories.items():
        actual_count = len(feature_breakdown.get(category, []))
        gap = expected_count - actual_count
        analysis['category_gaps'][category] = {
            'expected': expected_count,
            'actual': actual_count,
            'gap': gap
        }
    
    # Detailed logging
    logger.info(f"\n=== MISSING FEATURES ANALYSIS ===")
    logger.info(f"Current ML features: {len(all_features)}")
    logger.info(f"Current pipeline features: {len(pipeline_feature_names)}")
    logger.info(f"Expected features: 150")
    logger.info(f"Missing features: {150 - len(all_features)}")
    
    logger.info(f"\n=== CATEGORY BREAKDOWN ===")
    for category, gap_info in analysis['category_gaps'].items():
        logger.info(f"{category}: {gap_info['actual']}/{gap_info['expected']} (gap: {gap_info['gap']})")
    
    # Check for potential issues
    if len(analysis['features_only_in_pipeline']) > 0:
        logger.warning(f"Features only in pipeline ({len(analysis['features_only_in_pipeline'])}): {analysis['features_only_in_pipeline'][:5]}")
    
    if len(analysis['features_only_in_ml']) > 0:
        logger.info(f"Features only in ML ({len(analysis['features_only_in_ml'])}): {analysis['features_only_in_ml'][:5]}")
    
    # Save detailed analysis
    output_file = f"missing_features_detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"\nDetailed analysis saved to: {output_file}")
    
    # Specific recommendations
    logger.info(f"\n=== RECOMMENDATIONS ===")
    
    # Find categories with the biggest gaps
    biggest_gaps = sorted(analysis['category_gaps'].items(), key=lambda x: x[1]['gap'], reverse=True)
    
    for category, gap_info in biggest_gaps[:3]:
        if gap_info['gap'] > 0:
            logger.info(f"1. Check {category} feature generation - missing {gap_info['gap']} features")
    
    # Check if cross-asset features are working
    cross_asset_count = len(feature_breakdown.get('cross_asset', []))
    if cross_asset_count < 5:
        logger.warning(f"Cross-asset features very low ({cross_asset_count}) - check SPY data availability")
    
    return analysis

if __name__ == "__main__":
    asyncio.run(analyze_missing_features())
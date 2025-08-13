#!/usr/bin/env python3
"""
Test the actual data pipeline feature generation to understand
the discrepancy between ML features (148) and pipeline features (133)
"""

import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
from typing import Dict, List, Set

# Import required modules
from data.data_pipeline import DataPipeline
from ml.ml_feature_engineering import FeatureEngineering
from database import db_manager

async def test_pipeline_vs_ml_features():
    """Test both pipeline and ML feature generation to find discrepancies"""
    
    # Initialize components
    data_pipeline = DataPipeline()
    feature_engineer = FeatureEngineering(supabase_client=db_manager.get_supabase_client())
    
    # Test date range with good data (timezone-aware)
    start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 10, tzinfo=timezone.utc)
    
    logger.info(f"Testing feature generation for AAPL from {start_date} to {end_date}")
    
    # Test 1: ML FeatureEngineering (what we know works)
    logger.info("\n=== Testing ML FeatureEngineering ===")
    feature_set = await feature_engineer.engineer_features(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    
    # Count ML features
    ml_features = set()
    ml_breakdown = {}
    
    if hasattr(feature_set, 'technical_features') and not feature_set.technical_features.empty:
        tech_features = set(feature_set.technical_features.columns)
        ml_features.update(tech_features)
        ml_breakdown['technical'] = len(tech_features)
        logger.info(f"ML Technical features: {len(tech_features)}")
    
    if hasattr(feature_set, 'market_microstructure') and not feature_set.market_microstructure.empty:
        micro_features = set(feature_set.market_microstructure.columns)
        ml_features.update(micro_features)
        ml_breakdown['microstructure'] = len(micro_features)
        logger.info(f"ML Microstructure features: {len(micro_features)}")
    
    if hasattr(feature_set, 'sentiment_features') and not feature_set.sentiment_features.empty:
        sentiment_features = set(feature_set.sentiment_features.columns)
        ml_features.update(sentiment_features)
        ml_breakdown['sentiment'] = len(sentiment_features)
        logger.info(f"ML Sentiment features: {len(sentiment_features)}")
    
    if hasattr(feature_set, 'macro_features') and not feature_set.macro_features.empty:
        macro_features = set(feature_set.macro_features.columns)
        ml_features.update(macro_features)
        ml_breakdown['macro'] = len(macro_features)
        logger.info(f"ML Macro features: {len(macro_features)}")
    
    if hasattr(feature_set, 'cross_asset_features') and not feature_set.cross_asset_features.empty:
        cross_features = set(feature_set.cross_asset_features.columns)
        ml_features.update(cross_features)
        ml_breakdown['cross_asset'] = len(cross_features)
        logger.info(f"ML Cross-asset features: {len(cross_features)}")
    
    if hasattr(feature_set, 'engineered_features') and not feature_set.engineered_features.empty:
        eng_features = set(feature_set.engineered_features.columns)
        ml_features.update(eng_features)
        ml_breakdown['engineered'] = len(eng_features)
        logger.info(f"ML Engineered features: {len(eng_features)}")
    
    logger.info(f"Total ML features: {len(ml_features)}")
    
    # Test 2: Data Pipeline bootstrap process (simulating the actual pipeline)
    logger.info("\n=== Testing Data Pipeline Bootstrap Process ===")
    
    # Load market data first
    market_data = await data_pipeline.load_market_data(
        symbol="AAPL",
        start_date=start_date,
        end_date=end_date
    )
    
    if market_data.empty:
        logger.error("No market data found for pipeline test")
        return
    
    logger.info(f"Loaded {len(market_data)} market data points for pipeline test")
    
    # Simulate the exact pipeline process from data_pipeline.py lines 670-700
    try:
        # This is the exact code from data_pipeline.py
        feature_engineer_pipeline = FeatureEngineering(supabase_client=db_manager.get_supabase_client())
        
        # Engineer features from market data
        start_date_pipeline = market_data.index.min()
        end_date_pipeline = market_data.index.max()
        feature_set_pipeline = await feature_engineer_pipeline.engineer_features(
            symbol="AAPL",
            start_date=start_date_pipeline,
            end_date=end_date_pipeline
        )
        
        # Combine all feature categories from FeatureSet (exact pipeline logic)
        feature_dfs = []
        pipeline_breakdown = {}
        
        if hasattr(feature_set_pipeline, 'technical_features') and not feature_set_pipeline.technical_features.empty:
            feature_dfs.append(feature_set_pipeline.technical_features)
            pipeline_breakdown['technical'] = len(feature_set_pipeline.technical_features.columns)
            logger.info(f"Pipeline Technical features: {len(feature_set_pipeline.technical_features.columns)} columns")
        
        if hasattr(feature_set_pipeline, 'market_microstructure') and not feature_set_pipeline.market_microstructure.empty:
            feature_dfs.append(feature_set_pipeline.market_microstructure)
            pipeline_breakdown['microstructure'] = len(feature_set_pipeline.market_microstructure.columns)
            logger.info(f"Pipeline Market microstructure features: {len(feature_set_pipeline.market_microstructure.columns)} columns")
        
        if hasattr(feature_set_pipeline, 'sentiment_features') and not feature_set_pipeline.sentiment_features.empty:
            feature_dfs.append(feature_set_pipeline.sentiment_features)
            pipeline_breakdown['sentiment'] = len(feature_set_pipeline.sentiment_features.columns)
            logger.info(f"Pipeline Sentiment features: {len(feature_set_pipeline.sentiment_features.columns)} columns")
        
        if hasattr(feature_set_pipeline, 'macro_features') and not feature_set_pipeline.macro_features.empty:
            feature_dfs.append(feature_set_pipeline.macro_features)
            pipeline_breakdown['macro'] = len(feature_set_pipeline.macro_features.columns)
            logger.info(f"Pipeline Macro features: {len(feature_set_pipeline.macro_features.columns)} columns")
        
        if hasattr(feature_set_pipeline, 'cross_asset_features') and not feature_set_pipeline.cross_asset_features.empty:
            feature_dfs.append(feature_set_pipeline.cross_asset_features)
            pipeline_breakdown['cross_asset'] = len(feature_set_pipeline.cross_asset_features.columns)
            logger.info(f"Pipeline Cross-asset features: {len(feature_set_pipeline.cross_asset_features.columns)} columns")
        
        if hasattr(feature_set_pipeline, 'engineered_features') and not feature_set_pipeline.engineered_features.empty:
            feature_dfs.append(feature_set_pipeline.engineered_features)
            pipeline_breakdown['engineered'] = len(feature_set_pipeline.engineered_features.columns)
            logger.info(f"Pipeline Engineered features: {len(feature_set_pipeline.engineered_features.columns)} columns")
        
        if feature_dfs:
            features_df = pd.concat(feature_dfs, axis=1)
            logger.info(f"Pipeline Combined features: {len(features_df.columns)} total columns")
            
            # Get pipeline feature names (excluding basic OHLCV)
            basic_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timestamp']
            pipeline_features = set(col for col in features_df.columns if col not in basic_cols)
            
        else:
            logger.warning("No features generated from pipeline")
            pipeline_features = set()
            features_df = pd.DataFrame()
        
        logger.info(f"Total pipeline features: {len(pipeline_features)}")
        
        # Test 3: Check for NaN filtering effects
        logger.info("\n=== Testing NaN Filtering Effects ===")
        
        if not features_df.empty:
            # Count features before and after NaN filtering
            original_features = len(features_df.columns)
            
            # Simulate the NaN filtering from the pipeline
            for timestamp, row in features_df.iterrows():
                # Convert row to dictionary, excluding NaN values (this is what pipeline does)
                feature_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                
                # Log first row to see filtering effect
                if timestamp == features_df.index[0]:
                    filtered_features = len(feature_dict)
                    logger.info(f"NaN filtering effect: {original_features} -> {filtered_features} features (removed {original_features - filtered_features} NaN features)")
                    break
        
        # Analysis and comparison
        logger.info("\n=== COMPARISON ANALYSIS ===")
        logger.info(f"ML features: {len(ml_features)}")
        logger.info(f"Pipeline features: {len(pipeline_features)}")
        logger.info(f"Difference: {len(ml_features) - len(pipeline_features)}")
        
        # Find differences
        only_in_ml = ml_features - pipeline_features
        only_in_pipeline = pipeline_features - ml_features
        
        if only_in_ml:
            logger.info(f"Features only in ML ({len(only_in_ml)}): {sorted(list(only_in_ml))[:10]}")
        
        if only_in_pipeline:
            logger.info(f"Features only in Pipeline ({len(only_in_pipeline)}): {sorted(list(only_in_pipeline))[:10]}")
        
        # Save detailed comparison
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'ml_feature_count': len(ml_features),
            'pipeline_feature_count': len(pipeline_features),
            'difference': len(ml_features) - len(pipeline_features),
            'ml_breakdown': ml_breakdown,
            'pipeline_breakdown': pipeline_breakdown,
            'features_only_in_ml': sorted(list(only_in_ml)),
            'features_only_in_pipeline': sorted(list(only_in_pipeline)),
            'common_features': sorted(list(ml_features & pipeline_features))
        }
        
        output_file = f"pipeline_vs_ml_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"\nComparison saved to: {output_file}")
        
        # Final summary
        logger.info("\n=== SUMMARY ===")
        logger.info(f"Expected features (from training): 150")
        logger.info(f"Current ML features: {len(ml_features)}")
        logger.info(f"Current pipeline features: {len(pipeline_features)}")
        logger.info(f"Missing from expected: {150 - len(ml_features)}")
        
        if len(ml_features) != len(pipeline_features):
            logger.warning(f"DISCREPANCY FOUND: ML and Pipeline generate different feature counts!")
            logger.warning(f"This explains the 133 vs 148 feature difference mentioned by user")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline_vs_ml_features())
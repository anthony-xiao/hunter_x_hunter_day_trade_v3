#!/usr/bin/env python3
"""
Feature Analysis Script

This script analyzes the current feature engineering pipeline to:
1. Count features by category
2. Identify missing features (133 vs 150 expected)
3. Test cross-asset features with SPY data
4. Save detailed analysis for comparison
"""

import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from ml.ml_feature_engineering import FeatureEngineering
from database import db_manager
from data.data_pipeline import DataPipeline
from loguru import logger

async def load_market_data(data_pipeline, symbol: str, start_date: str, end_date: str):
    """Load market data for analysis"""
    try:
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        
        logger.info(f"Loading market data for {symbol} from {start_date} to {end_date}")
        
        # Load market data using data pipeline
        data = await data_pipeline.load_market_data(
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt
        )
        
        if data is None or data.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
            
        logger.info(f"Loaded {len(data)} records for {symbol}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading market data for {symbol}: {e}")
        return pd.DataFrame()

async def analyze_feature_categories(feature_engineer, symbol: str, data: pd.DataFrame):
    """Analyze features by category"""
    feature_analysis = {
        'total_features': 0,
        'categories': {},
        'feature_names': [],
        'missing_data_info': {}
    }
    
    if data.empty:
        logger.warning(f"No data available for {symbol} - cannot analyze features")
        return feature_analysis
    
    try:
        logger.info(f"Analyzing features for {symbol} with {len(data)} data points")
        
        # Engineer all features
        feature_set = await feature_engineer.engineer_features(
            symbol=symbol,
            start_date=data.index.min(),
            end_date=data.index.max()
        )
        
        if not feature_set or not hasattr(feature_set, 'technical_features'):
            logger.warning(f"No features generated for {symbol}")
            return feature_analysis
            
        # Count features by category from FeatureSet
        categories = {
            'technical': [],
            'microstructure': [],
            'sentiment': [],
            'macro': [],
            'cross_asset': [],
            'advanced': [],
            'other': []
        }
        
        # Collect features from each category in the FeatureSet
        if hasattr(feature_set, 'technical_features') and not feature_set.technical_features.empty:
            categories['technical'] = [f"technical_{col}" for col in feature_set.technical_features.columns]
        if hasattr(feature_set, 'market_microstructure') and not feature_set.market_microstructure.empty:
            categories['microstructure'] = [f"microstructure_{col}" for col in feature_set.market_microstructure.columns]
        if hasattr(feature_set, 'sentiment_features') and not feature_set.sentiment_features.empty:
            categories['sentiment'] = [f"sentiment_{col}" for col in feature_set.sentiment_features.columns]
        if hasattr(feature_set, 'macro_features') and not feature_set.macro_features.empty:
            categories['macro'] = [f"macro_{col}" for col in feature_set.macro_features.columns]
        if hasattr(feature_set, 'cross_asset_features') and not feature_set.cross_asset_features.empty:
            categories['cross_asset'] = [f"cross_asset_{col}" for col in feature_set.cross_asset_features.columns]
        if hasattr(feature_set, 'engineered_features') and not feature_set.engineered_features.empty:
            categories['advanced'] = [f"engineered_{col}" for col in feature_set.engineered_features.columns]
        
        # Collect all feature names
        all_features = []
        for feature_list in categories.values():
            all_features.extend(feature_list)
        
        feature_analysis['total_features'] = len(all_features)
        feature_analysis['feature_names'] = all_features
        
        logger.info(f"Total features generated: {len(all_features)}")
        
        # Store category counts
        for category, features in categories.items():
            feature_analysis['categories'][category] = {
                'count': len(features),
                'features': features
            }
            
        # Log category breakdown
        logger.info("Feature breakdown by category:")
        for category, info in feature_analysis['categories'].items():
            logger.info(f"  {category}: {info['count']} features")
            
        return feature_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing features: {e}")
        return feature_analysis

async def test_cross_asset_features(feature_engineer, data_pipeline):
    """Test cross-asset features specifically with SPY data"""
    logger.info("Testing cross-asset features with SPY data")
    
    try:
        # Load SPY data for cross-asset analysis
        spy_data = await load_market_data(data_pipeline, 'SPY', '2025-08-01', '2025-08-10')
        aapl_data = await load_market_data(data_pipeline, 'AAPL', '2025-08-01', '2025-08-10')
        
        cross_asset_analysis = {
            'spy_data_available': not spy_data.empty,
            'spy_records': len(spy_data) if not spy_data.empty else 0,
            'aapl_data_available': not aapl_data.empty,
            'aapl_records': len(aapl_data) if not aapl_data.empty else 0,
            'cross_asset_features': []
        }
        
        if not spy_data.empty and not aapl_data.empty:
            # Test cross-asset feature generation
            logger.info(f"Testing cross-asset features with SPY ({len(spy_data)} records) and AAPL ({len(aapl_data)} records)")
            
            # Generate features for AAPL (which should include SPY cross-asset features)
            feature_set = await feature_engineer.engineer_features(
                symbol='AAPL',
                start_date=aapl_data.index.min(),
                end_date=aapl_data.index.max()
            )
            
            if feature_set and hasattr(feature_set, 'cross_asset_features') and not feature_set.cross_asset_features.empty:
                # Get cross-asset feature names
                cross_asset_features = [f"cross_asset_{col}" for col in feature_set.cross_asset_features.columns]
                
                cross_asset_analysis['cross_asset_features'] = cross_asset_features
                logger.info(f"Found {len(cross_asset_features)} cross-asset features: {cross_asset_features}")
            else:
                logger.warning("No cross-asset features generated during test")
        else:
            logger.warning("Missing SPY or AAPL data for cross-asset testing")
            
        return cross_asset_analysis
        
    except Exception as e:
        logger.error(f"Error testing cross-asset features: {e}")
        return {'error': str(e)}

async def main():
    """Main analysis function"""
    logger.info("Starting feature analysis...")
    
    try:
        # Initialize data pipeline and feature engineer
        data_pipeline = DataPipeline()
        feature_engineer = FeatureEngineering(
            supabase_client=db_manager.get_supabase_client()
        )
        
        # Test parameters
        symbol = 'AAPL'
        start_date = '2025-08-01'
        end_date = '2025-08-10'
        
        # Load market data
        logger.info(f"Loading market data for {symbol}...")
        market_data = await load_market_data(data_pipeline, symbol, start_date, end_date)
        
        # Analyze features
        logger.info("Analyzing feature categories...")
        feature_analysis = await analyze_feature_categories(feature_engineer, symbol, market_data)
        
        # Test cross-asset features
        logger.info("Testing cross-asset features...")
        cross_asset_analysis = await test_cross_asset_features(feature_engineer, data_pipeline)
        
        # Compile full analysis
        full_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'test_parameters': {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'market_data_records': len(market_data) if not market_data.empty else 0
            },
            'feature_analysis': feature_analysis,
            'cross_asset_analysis': cross_asset_analysis,
            'comparison_with_training': {
                'expected_features': 150,
                'current_features': feature_analysis['total_features'],
                'missing_features': 150 - feature_analysis['total_features'],
                'feature_gap_percentage': ((150 - feature_analysis['total_features']) / 150) * 100
            }
        }
        
        # Save analysis to file
        output_file = '/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/feature_analysis_results.json'
        with open(output_file, 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
            
        logger.info(f"Analysis saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Market Data Records: {len(market_data) if not market_data.empty else 0}")
        print(f"\nFeature Count:")
        print(f"  Expected: 150")
        print(f"  Current: {feature_analysis['total_features']}")
        print(f"  Missing: {150 - feature_analysis['total_features']}")
        print(f"  Gap: {((150 - feature_analysis['total_features']) / 150) * 100:.1f}%")
        
        print(f"\nFeatures by Category:")
        for category, info in feature_analysis['categories'].items():
            print(f"  {category.capitalize()}: {info['count']} features")
            
        print(f"\nCross-Asset Features:")
        if 'cross_asset_features' in cross_asset_analysis:
            print(f"  Found: {len(cross_asset_analysis['cross_asset_features'])} features")
            for feature in cross_asset_analysis['cross_asset_features']:
                print(f"    - {feature}")
        else:
            print(f"  Error or no cross-asset features found")
            
        print(f"\nDetailed analysis saved to: {output_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f"Analysis failed: {e}")
    
    finally:
        if 'db_manager' in locals():
            await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
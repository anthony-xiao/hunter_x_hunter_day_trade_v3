#!/usr/bin/env python3
"""
Comprehensive Feature Analysis Script

This script analyzes the ML feature engineering pipeline to:
1. Test feature generation for AAPL from 2025-08-01 to 2025-08-10
2. Count features in each category
3. List exact feature names
4. Identify missing cross-asset and advanced features
5. Test SPY data availability
6. Calculate total feature count (ML + OHLCV)
7. Compare with training metadata expectations
8. Save detailed analysis to JSON
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ml.ml_feature_engineering import FeatureEngineering
from data.data_pipeline import DataPipeline
from database import supabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self):
        self.supabase = supabase
        self.data_pipeline = DataPipeline()
        self.feature_engineer = FeatureEngineering()
        self.analysis_results = {}
        
    async def analyze_features_comprehensive(self, symbol: str = "AAPL", 
                                           start_date: str = "2024-08-01", 
                                           end_date: str = "2024-08-10"):
        """
        Comprehensive feature analysis for the specified symbol and date range
        """
        logger.info(f"Starting comprehensive feature analysis for {symbol} from {start_date} to {end_date}")
        
        try:
            # Convert dates
            start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            
            # 1. Test market data availability
            await self._test_market_data_availability(symbol, start_dt, end_dt)
            
            # 2. Test SPY data availability for cross-asset features
            await self._test_spy_data_availability(start_dt, end_dt)
            
            # 3. Generate features using the ML feature engineering pipeline
            features = await self._generate_features(symbol, start_dt, end_dt)
            
            # 4. Analyze feature categories and counts
            await self._analyze_feature_categories(features)
            
            # 5. Identify missing features
            await self._identify_missing_features(features)
            
            # 6. Calculate total feature count including OHLCV
            await self._calculate_total_feature_count(features)
            
            # 7. Compare with training expectations
            await self._compare_with_training_expectations()
            
            # 8. Test consistency between training and live trading
            await self._test_training_live_consistency(symbol, start_dt, end_dt)
            
            # 9. Save analysis results
            await self._save_analysis_results(symbol, start_date, end_date)
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            self.analysis_results['error'] = str(e)
            return self.analysis_results
    
    async def _test_market_data_availability(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """Test market data availability for the symbol"""
        logger.info(f"Testing market data availability for {symbol}")
        
        try:
            # Query market data directly from Supabase
            response = self.supabase.table('market_data').select(
                'timestamp, open, high, low, close, volume, vwap, transactions'
            ).eq('symbol', symbol).gte(
                'timestamp', start_dt.isoformat()
            ).lte(
                'timestamp', end_dt.isoformat()
            ).order('timestamp').limit(1000).execute()
            
            data_count = len(response.data) if response.data else 0
            
            self.analysis_results['market_data_availability'] = {
                'symbol': symbol,
                'records_found': data_count,
                'date_range': f"{start_dt.date()} to {end_dt.date()}",
                'data_available': data_count > 0,
                'sample_record': response.data[0] if response.data else None
            }
            
            logger.info(f"Found {data_count} market data records for {symbol}")
            
        except Exception as e:
            logger.error(f"Market data availability test failed: {e}")
            self.analysis_results['market_data_availability'] = {
                'error': str(e),
                'data_available': False
            }
    
    async def _test_spy_data_availability(self, start_dt: datetime, end_dt: datetime):
        """Test SPY data availability for cross-asset features"""
        logger.info("Testing SPY data availability for cross-asset features")
        
        try:
            # Query SPY data directly from Supabase
            response = self.supabase.table('market_data').select(
                'timestamp, open, high, low, close, volume'
            ).eq('symbol', 'SPY').gte(
                'timestamp', start_dt.isoformat()
            ).lte(
                'timestamp', end_dt.isoformat()
            ).order('timestamp').limit(1000).execute()
            
            spy_count = len(response.data) if response.data else 0
            
            self.analysis_results['spy_data_availability'] = {
                'records_found': spy_count,
                'date_range': f"{start_dt.date()} to {end_dt.date()}",
                'data_available': spy_count > 0,
                'sample_record': response.data[0] if response.data else None
            }
            
            logger.info(f"Found {spy_count} SPY data records")
            
        except Exception as e:
            logger.error(f"SPY data availability test failed: {e}")
            self.analysis_results['spy_data_availability'] = {
                'error': str(e),
                'data_available': False
            }
    
    async def _generate_features(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """Generate features using the ML feature engineering pipeline"""
        logger.info(f"Generating features for {symbol}")
        
        try:
            # Use the feature engineering pipeline
            features = await self.feature_engineer.engineer_features(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt
            )
            
            self.analysis_results['feature_generation'] = {
                'success': True,
                'total_features': (
                    len(features.technical_features.columns) +
                    len(features.market_microstructure.columns) +
                    len(features.sentiment_features.columns) +
                    len(features.macro_features.columns) +
                    len(features.cross_asset_features.columns) +
                    len(features.engineered_features.columns)
                ),
                'total_observations': len(features.technical_features) if len(features.technical_features) > 0 else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            self.analysis_results['feature_generation'] = {
                'success': False,
                'error': str(e)
            }
            return None
    
    async def _analyze_feature_categories(self, features):
        """Analyze feature categories and counts"""
        logger.info("Analyzing feature categories")
        
        if features is None:
            self.analysis_results['feature_categories'] = {'error': 'No features generated'}
            return
        
        try:
            categories = {
                'technical_features': {
                    'count': len(features.technical_features.columns),
                    'feature_names': list(features.technical_features.columns),
                    'sample_values': {}
                },
                'market_microstructure': {
                    'count': len(features.market_microstructure.columns),
                    'feature_names': list(features.market_microstructure.columns),
                    'sample_values': {}
                },
                'sentiment_features': {
                    'count': len(features.sentiment_features.columns),
                    'feature_names': list(features.sentiment_features.columns),
                    'sample_values': {}
                },
                'macro_features': {
                    'count': len(features.macro_features.columns),
                    'feature_names': list(features.macro_features.columns),
                    'sample_values': {}
                },
                'cross_asset_features': {
                    'count': len(features.cross_asset_features.columns),
                    'feature_names': list(features.cross_asset_features.columns),
                    'sample_values': {}
                },
                'engineered_features': {
                    'count': len(features.engineered_features.columns),
                    'feature_names': list(features.engineered_features.columns),
                    'sample_values': {}
                }
            }
            
            # Add sample values for each category (first non-null value)
            for category_name, category_data in categories.items():
                if hasattr(features, category_name.replace('_features', '_features')):
                    df = getattr(features, category_name.replace('_features', '_features'))
                    if len(df) > 0:
                        for col in df.columns:
                            first_valid = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                            category_data['sample_values'][col] = float(first_valid) if first_valid is not None else None
            
            self.analysis_results['feature_categories'] = categories
            
            # Calculate total ML features
            total_ml_features = sum(cat['count'] for cat in categories.values())
            self.analysis_results['total_ml_features'] = total_ml_features
            
            logger.info(f"Total ML features generated: {total_ml_features}")
            
        except Exception as e:
            logger.error(f"Feature category analysis failed: {e}")
            self.analysis_results['feature_categories'] = {'error': str(e)}
    
    async def _identify_missing_features(self, features):
        """Identify missing cross-asset and advanced features"""
        logger.info("Identifying missing features")
        
        if features is None:
            self.analysis_results['missing_features'] = {'error': 'No features to analyze'}
            return
        
        try:
            # Expected cross-asset features based on implementation
            expected_cross_asset = [
                'spy_correlation_20', 'spy_correlation_50', 'spy_correlation_100',
                'beta_50', 'beta_100'
            ]
            
            # Expected advanced/engineered features based on implementation
            expected_advanced = [
                'composite_momentum', 'composite_volatility', 'composite_trend',
                'risk_adjusted_returns', 'volume_price_divergence', 'volatility_regime',
                'mean_reversion_20', 'mean_reversion_50', 'mean_reversion_100'
            ]
            
            # Check which features are missing
            actual_cross_asset = list(features.cross_asset_features.columns)
            actual_advanced = list(features.engineered_features.columns)
            
            missing_cross_asset = [f for f in expected_cross_asset if f not in actual_cross_asset]
            missing_advanced = [f for f in expected_advanced if f not in actual_advanced]
            
            self.analysis_results['missing_features'] = {
                'cross_asset': {
                    'expected': expected_cross_asset,
                    'actual': actual_cross_asset,
                    'missing': missing_cross_asset,
                    'missing_count': len(missing_cross_asset)
                },
                'advanced_engineered': {
                    'expected': expected_advanced,
                    'actual': actual_advanced,
                    'missing': missing_advanced,
                    'missing_count': len(missing_advanced)
                },
                'total_missing': len(missing_cross_asset) + len(missing_advanced)
            }
            
            logger.info(f"Missing cross-asset features: {missing_cross_asset}")
            logger.info(f"Missing advanced features: {missing_advanced}")
            
        except Exception as e:
            logger.error(f"Missing feature identification failed: {e}")
            self.analysis_results['missing_features'] = {'error': str(e)}
    
    async def _calculate_total_feature_count(self, features):
        """Calculate total feature count including OHLCV"""
        logger.info("Calculating total feature count including OHLCV")
        
        try:
            # OHLCV features (basic market data)
            ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
            ohlcv_count = len(ohlcv_features)
            
            # ML features count
            ml_features_count = self.analysis_results.get('total_ml_features', 0)
            
            # Total features
            total_features = ml_features_count + ohlcv_count
            
            self.analysis_results['total_feature_count'] = {
                'ml_features': ml_features_count,
                'ohlcv_features': ohlcv_count,
                'ohlcv_feature_names': ohlcv_features,
                'total_features': total_features,
                'expected_total': 153,  # 148 ML + 5 OHLCV
                'discrepancy': 153 - total_features
            }
            
            logger.info(f"Total features: {total_features} (ML: {ml_features_count} + OHLCV: {ohlcv_count})")
            logger.info(f"Expected: 153, Actual: {total_features}, Discrepancy: {153 - total_features}")
            
        except Exception as e:
            logger.error(f"Total feature count calculation failed: {e}")
            self.analysis_results['total_feature_count'] = {'error': str(e)}
    
    async def _compare_with_training_expectations(self):
        """Compare with training metadata expectations"""
        logger.info("Comparing with training expectations")
        
        try:
            # Expected from training metadata (based on requirements document)
            training_expectations = {
                'total_ml_features': 148,  # From previous analysis
                'target_total_with_ohlcv': 153,
                'minimum_features_for_training': 150
            }
            
            actual_ml = self.analysis_results.get('total_ml_features', 0)
            actual_total = self.analysis_results.get('total_feature_count', {}).get('total_features', 0)
            
            self.analysis_results['training_comparison'] = {
                'expectations': training_expectations,
                'actual_results': {
                    'ml_features': actual_ml,
                    'total_with_ohlcv': actual_total
                },
                'meets_minimum_requirement': actual_total >= training_expectations['minimum_features_for_training'],
                'feature_gap': training_expectations['total_ml_features'] - actual_ml
            }
            
        except Exception as e:
            logger.error(f"Training comparison failed: {e}")
            self.analysis_results['training_comparison'] = {'error': str(e)}
    
    async def _test_training_live_consistency(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """Test consistency between training and live trading feature generation"""
        logger.info("Testing training vs live trading consistency")
        
        try:
            # This would involve comparing the feature generation process
            # used in training vs live trading - for now, document the approach
            
            consistency_check = {
                'feature_engineering_pipeline': 'Same FeatureEngineering class used',
                'data_source': 'Same Supabase market_data table',
                'feature_categories': 'Same categories (technical, microstructure, etc.)',
                'potential_inconsistencies': [
                    'Real-time vs batch processing timing',
                    'Data availability differences',
                    'Caching vs fresh computation'
                ],
                'recommendations': [
                    'Ensure SPY data is consistently available',
                    'Verify cross-asset feature generation in live trading',
                    'Test advanced feature computation with sufficient data'
                ]
            }
            
            self.analysis_results['training_live_consistency'] = consistency_check
            
        except Exception as e:
            logger.error(f"Training/live consistency test failed: {e}")
            self.analysis_results['training_live_consistency'] = {'error': str(e)}
    
    async def _save_analysis_results(self, symbol: str, start_date: str, end_date: str):
        """Save analysis results to JSON file"""
        try:
            # Add metadata
            self.analysis_results['metadata'] = {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol_analyzed': symbol,
                'date_range': f"{start_date} to {end_date}",
                'script_version': '1.0',
                'purpose': 'Comprehensive feature analysis for ML trading system'
            }
            
            # Save to file
            output_file = Path(__file__).parent / f"feature_analysis_{symbol}_{start_date}_{end_date}.json"
            
            with open(output_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {output_file}")
            
            # Also save a summary
            summary = {
                'total_ml_features': self.analysis_results.get('total_ml_features', 0),
                'total_with_ohlcv': self.analysis_results.get('total_feature_count', {}).get('total_features', 0),
                'missing_features_count': self.analysis_results.get('missing_features', {}).get('total_missing', 0),
                'spy_data_available': self.analysis_results.get('spy_data_availability', {}).get('data_available', False),
                'feature_generation_success': self.analysis_results.get('feature_generation', {}).get('success', False)
            }
            
            summary_file = Path(__file__).parent / f"feature_analysis_summary_{symbol}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Analysis summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")

async def main():
    """Main function to run the comprehensive feature analysis"""
    analyzer = FeatureAnalyzer()
    
    # Run analysis for AAPL
    results = await analyzer.analyze_features_comprehensive(
        symbol="AAPL",
        start_date="2024-08-01",
        end_date="2024-08-10"
    )
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
    print("="*80)
    
    if 'total_ml_features' in results:
        print(f"Total ML Features Generated: {results['total_ml_features']}")
    
    if 'total_feature_count' in results:
        total_info = results['total_feature_count']
        print(f"Total Features (ML + OHLCV): {total_info.get('total_features', 'N/A')}")
        print(f"Expected Total: {total_info.get('expected_total', 'N/A')}")
        print(f"Discrepancy: {total_info.get('discrepancy', 'N/A')}")
    
    if 'missing_features' in results:
        missing_info = results['missing_features']
        print(f"Missing Features: {missing_info.get('total_missing', 'N/A')}")
        if 'cross_asset' in missing_info:
            print(f"  - Missing Cross-Asset: {missing_info['cross_asset'].get('missing_count', 0)}")
        if 'advanced_engineered' in missing_info:
            print(f"  - Missing Advanced: {missing_info['advanced_engineered'].get('missing_count', 0)}")
    
    if 'spy_data_availability' in results:
        spy_available = results['spy_data_availability'].get('data_available', False)
        print(f"SPY Data Available: {spy_available}")
    
    print("\nDetailed results saved to JSON files.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
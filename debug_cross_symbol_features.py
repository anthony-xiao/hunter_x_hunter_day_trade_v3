#!/usr/bin/env python3

import sys
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add backend to path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_cross_symbol_features():
    """Debug cross-symbol and market regime feature generation"""
    try:
        # Initialize components
        data_pipeline = DataPipeline()
        feature_engineer = UniversalFeatureEngineering()
        
        # Test with single symbol (should trigger fallback)
        symbols = ['AAPL']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        logger.info(f"Testing with symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Get individual symbol features first
        symbol_features = await feature_engineer._engineer_individual_symbol_features(
            symbols, start_date, end_date, include_cross_asset=True, training_mode=True
        )
        
        logger.info(f"Symbol features keys: {list(symbol_features.keys())}")
        
        for symbol, features in symbol_features.items():
            logger.info(f"Symbol {symbol}:")
            if hasattr(features, 'technical_features') and features.technical_features is not None:
                logger.info(f"  Technical features shape: {features.technical_features.shape}")
                logger.info(f"  Technical features columns: {list(features.technical_features.columns)}")
                logger.info(f"  Has 'close' column: {'close' in features.technical_features.columns}")
            else:
                logger.info(f"  No technical features or None")
        
        # Test cross-symbol features (should generate fallback)
        logger.info("\n=== Testing Cross-Symbol Features ===")
        cross_symbol_features = await feature_engineer._engineer_cross_symbol_features(
            symbol_features, symbols, start_date, end_date
        )
        
        logger.info(f"Cross-symbol features shape: {cross_symbol_features.shape}")
        logger.info(f"Cross-symbol features columns: {list(cross_symbol_features.columns)}")
        
        # Test market regime features
        logger.info("\n=== Testing Market Regime Features ===")
        market_regime_features = await feature_engineer._engineer_market_regime_features(
            symbol_features, symbols
        )
        
        logger.info(f"Market regime features shape: {market_regime_features.shape}")
        logger.info(f"Market regime features columns: {list(market_regime_features.columns)}")
        
        # Test with multiple symbols
        logger.info("\n=== Testing with Multiple Symbols ===")
        multi_symbols = ['AAPL', 'MSFT']
        
        multi_symbol_features = await feature_engineer._engineer_individual_symbol_features(
            multi_symbols, start_date, end_date, include_cross_asset=True, training_mode=True
        )
        
        logger.info(f"Multi-symbol features keys: {list(multi_symbol_features.keys())}")
        
        # Test cross-symbol features with multiple symbols
        multi_cross_features = await feature_engineer._engineer_cross_symbol_features(
            multi_symbol_features, multi_symbols, start_date, end_date
        )
        
        logger.info(f"Multi cross-symbol features shape: {multi_cross_features.shape}")
        logger.info(f"Multi cross-symbol features columns: {list(multi_cross_features.columns)}")
        
        # Test market regime with multiple symbols
        multi_regime_features = await feature_engineer._engineer_market_regime_features(
            multi_symbol_features, multi_symbols
        )
        
        logger.info(f"Multi market regime features shape: {multi_regime_features.shape}")
        logger.info(f"Multi market regime features columns: {list(multi_regime_features.columns)}")
        
        # Test feature validation
        logger.info("\n=== Testing Feature Validation ===")
        
        # Create a test DataFrame with known feature types
        test_features = pd.DataFrame({
            'rsi_14': [0.5] * 100,
            'macd_signal': [0.1] * 100,
            'symbol_id': [1] * 100,
            'corr_AAPL_MSFT_20': [0.7] * 100,
            'beta_AAPL_market_20': [1.2] * 100,
            'market_dispersion_10': [0.02] * 100,
            'market_volatility': [0.15] * 100,
            'vol_regime_low': [1] * 100,
            'sector_technology': [1] * 100
        })
        
        logger.info(f"Test features shape: {test_features.shape}")
        logger.info(f"Test features columns: {list(test_features.columns)}")
        
        # Run validation
        validation_result = feature_engineer._validate_feature_dimensions(
            test_features, "Debug Test", expected_total=9
        )
        
        logger.info(f"Validation result: {validation_result}")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    asyncio.run(debug_cross_symbol_features())
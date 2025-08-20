#!/usr/bin/env python3

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from ml.universal_feature_engineering import UniversalFeatureEngineering
from database import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_feature_generation():
    """Test cross-symbol and market regime feature generation"""
    try:
        logger.info("Starting feature generation test...")
        
        # Initialize components
        supabase_client = db_manager.get_supabase_client()
        if not supabase_client:
            raise Exception("Supabase client not available")
        feature_engineer = UniversalFeatureEngineering(supabase_client)
        
        # Test with a small set of symbols
        symbols = ['AAPL', 'MSFT']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        logger.info(f"Testing with symbols: {symbols}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Test individual symbol features first
        logger.info("\n=== Testing Individual Symbol Features ===")
        symbol_features = await feature_engineer._engineer_individual_symbol_features(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_cross_asset=False,
            training_mode=True
        )
        
        logger.info(f"Individual features generated for {len(symbol_features)} symbols")
        for symbol, features in symbol_features.items():
            if hasattr(features, 'technical_features'):
                logger.info(f"  {symbol}: {len(features.technical_features)} rows, {len(features.technical_features.columns)} columns")
                logger.info(f"    Columns: {list(features.technical_features.columns)[:10]}...")  # First 10 columns
            else:
                logger.info(f"  {symbol}: No technical features")
        
        # Test cross-symbol features
        logger.info("\n=== Testing Cross-Symbol Features ===")
        available_symbols = list(symbol_features.keys())
        cross_symbol_features = await feature_engineer._engineer_cross_symbol_features(
            symbol_features=symbol_features,
            symbols=available_symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Cross-symbol features shape: {cross_symbol_features.shape}")
        if not cross_symbol_features.empty:
            logger.info(f"Cross-symbol feature columns: {list(cross_symbol_features.columns)}")
        else:
            logger.warning("Cross-symbol features are empty!")
        
        # Test market regime features
        logger.info("\n=== Testing Market Regime Features ===")
        market_regime_features = await feature_engineer._engineer_market_regime_features(
            symbol_features=symbol_features,
            symbols=available_symbols
        )
        
        logger.info(f"Market regime features shape: {market_regime_features.shape}")
        if not market_regime_features.empty:
            logger.info(f"Market regime feature columns: {list(market_regime_features.columns)}")
        else:
            logger.warning("Market regime features are empty!")
        
        # Test full universal feature engineering
        logger.info("\n=== Testing Full Universal Feature Engineering ===")
        universal_features = await feature_engineer.engineer_universal_features(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_cross_asset=False,
            training_mode=True
        )
        
        logger.info(f"Universal features generated successfully")
        logger.info(f"Cross-symbol features shape: {universal_features.cross_symbol_features.shape}")
        logger.info(f"Market regime features shape: {universal_features.market_regime_features.shape}")
        
        # Test training data preparation
        logger.info("\n=== Testing Training Data Preparation ===")
        X, y = await feature_engineer.prepare_universal_training_data(
            universal_features=universal_features
        )
        
        logger.info(f"Training data prepared: X shape={X.shape}, y shape={y.shape}")
        
        logger.info("\n=== Test completed successfully ===")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_feature_generation())
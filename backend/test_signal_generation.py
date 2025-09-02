#!/usr/bin/env python3
"""
Test script to verify universal model prediction shape fixes
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Add the backend directory to Python path
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

from trading.signal_generator import SignalGenerator
from data.data_pipeline import DataPipeline
from data.pipeline_feature_engineering import FeatureEngineer
from ml.model_trainer import ModelTrainer
from ml.universal_trainer import UniversalTrainer
from config import settings

async def test_signal_generation():
    """Test signal generation with universal models"""
    try:
        print("Initializing components...")
        
        # Initialize config
        config = settings
        
        # Initialize components
        data_pipeline = DataPipeline()
        feature_engineer = FeatureEngineer(data_pipeline)
        model_trainer = ModelTrainer(feature_count=187, create_model_dir=False)
        
        # Initialize universal feature engineering
        from ml.universal_feature_engineering import UniversalFeatureEngineering
        universal_feature_engineering = UniversalFeatureEngineering()
        universal_trainer = UniversalTrainer(data_pipeline, universal_feature_engineering)
        
        # Initialize signal generator
        signal_generator = SignalGenerator(model_trainer=model_trainer)
        
        print("Components initialized successfully")
        
        # Test with a sample symbol
        symbol = "AAPL"
        print(f"Testing signal generation for {symbol}...")
        
        # Create some mock market data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=1)
        
        # Generate sample data
        timestamps = pd.date_range(start=start_date, end=end_date, freq='1min')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(150, 160, len(timestamps)),
            'high': np.random.uniform(155, 165, len(timestamps)),
            'low': np.random.uniform(145, 155, len(timestamps)),
            'close': np.random.uniform(150, 160, len(timestamps)),
            'volume': np.random.randint(1000, 10000, len(timestamps))
        }, index=timestamps)
        
        market_data = {symbol: sample_data}
        
        print(f"Generated sample market data with {len(sample_data)} rows")
        
        # Test signal generation
        print("Calling generate_signals...")
        signals = await signal_generator.generate_signals(market_data)
        
        print(f"Signal generation completed successfully!")
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"Signal: {signal.symbol} - {signal.action} (confidence: {signal.confidence:.3f})")
            
        return True
        
    except Exception as e:
        print(f"Error during signal generation test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Starting signal generation test...")
    success = asyncio.run(test_signal_generation())
    
    if success:
        print("\n✅ Signal generation test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Signal generation test failed!")
        sys.exit(1)
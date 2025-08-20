#!/usr/bin/env python3
"""
Test script to debug NaN values in sector features
"""

import sys
import os
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3')
sys.path.append('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from backend.ml.universal_feature_engineering import UniversalFeatureEngineering
from backend.data.data_pipeline import DataPipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_mock_price_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Create mock price data for testing"""
    # Create date range for the last 'days' days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    
    # Generate minute-level timestamps (390 minutes per trading day)
    timestamps = []
    current_date = start_date.replace(hour=9, minute=30, second=0, microsecond=0)  # Market open
    
    while current_date <= end_date:
        # Only add weekday timestamps (Monday=0, Sunday=6)
        if current_date.weekday() < 5:  # Monday to Friday
            for minute in range(390):  # 6.5 hours * 60 minutes
                timestamp = current_date + timedelta(minutes=minute)
                timestamps.append(timestamp)
        
        # Move to next day
        current_date += timedelta(days=1)
        current_date = current_date.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Generate realistic price data
    np.random.seed(42 if symbol == 'AAPL' else 123)  # Different seeds for different symbols
    base_price = 150.0 if symbol == 'AAPL' else 200.0
    
    prices = []
    current_price = base_price
    
    for i, timestamp in enumerate(timestamps):
        # Add some realistic price movement
        change_pct = np.random.normal(0, 0.001)  # 0.1% standard deviation
        current_price *= (1 + change_pct)
        
        # Ensure price stays positive
        current_price = max(current_price, base_price * 0.5)
        
        prices.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'close': current_price,
            'open': current_price * (1 + np.random.normal(0, 0.0005)),
            'high': current_price * (1 + abs(np.random.normal(0, 0.001))),
            'low': current_price * (1 - abs(np.random.normal(0, 0.001))),
            'volume': int(np.random.normal(1000000, 200000))
        })
    
    df = pd.DataFrame(prices)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    logger.info(f"Created mock data for {symbol}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df

async def test_sector_features_directly():
    """Test the _engineer_sector_features method directly with mock data"""
    print("\n=== Testing _engineer_sector_features directly ===")
    
    # Initialize UniversalFeatureEngineering
    feature_eng = UniversalFeatureEngineering()
    
    # Create mock symbol features in the expected format
    symbols = ['AAPL', 'TSLA']
    symbol_features = {}
    
    # Create mock price data for each symbol
    for symbol in symbols:
        mock_data = create_mock_price_data(symbol=symbol)
        
        # Create a FeatureSet-like object
        class MockFeatureSet:
            def __init__(self, technical_features):
                self.technical_features = technical_features
        
        symbol_features[symbol] = MockFeatureSet(mock_data)
    
    # Call _engineer_sector_features directly
    print("\nCalling _engineer_sector_features...")
    sector_features = await feature_eng._engineer_sector_features(
        symbol_features=symbol_features, 
        symbols=symbols
    )
    
    print(f"\nSector features shape: {sector_features.shape}")
    print(f"Sector features columns: {list(sector_features.columns)}")
    
    # Analyze NaN values
    print("\n=== NaN Analysis ===")
    for col in sector_features.columns:
        nan_count = sector_features[col].isnull().sum()
        print(f"{col}: {nan_count} NaN values out of {len(sector_features)} total")
        
        if nan_count > 0:
            # Show first few NaN positions
            nan_positions = sector_features[col].isnull()
            first_nans = nan_positions[nan_positions].head(5).index.tolist()
            print(f"  First 5 NaN positions: {first_nans}")
    
    return sector_features

if __name__ == "__main__":
    asyncio.run(test_sector_features_directly())
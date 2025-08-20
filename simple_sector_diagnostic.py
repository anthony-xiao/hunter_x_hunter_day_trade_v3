#!/usr/bin/env python3
"""
Simple Sector NaN Diagnostic Script
Focused analysis of sector feature NaN generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import db_manager
from data.data_pipeline import DataPipeline

class SimpleSectorDiagnostic:
    def __init__(self):
        self.supabase = db_manager.get_supabase_client()
        self.data_pipeline = DataPipeline()
        
    async def analyze_sector_calculation(self):
        """Analyze the sector feature calculation step by step"""
        print("\n" + "="*80)
        print("SIMPLE SECTOR NaN DIAGNOSTIC")
        print("="*80)
        
        # Define analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days of data
        
        symbols = ['AAPL', 'TSLA']
        
        print(f"\nAnalysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Symbols: {symbols}")
        
        # Step 1: Load raw market data
        print("\n[1] LOADING RAW MARKET DATA")
        print("-" * 40)
        
        symbol_data = {}
        for symbol in symbols:
            try:
                data = await self.data_pipeline.load_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                symbol_data[symbol] = data
                print(f"✅ {symbol}: {len(data)} records loaded")
                
                # Check data structure
                if hasattr(data, 'columns'):
                    print(f"   Columns: {list(data.columns)}")
                    if 'timestamp' in data.columns:
                        print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                    if 'close' in data.columns:
                        print(f"   Close price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                        print(f"   NaN count in close: {data['close'].isna().sum()}")
                else:
                    print(f"   Data type: {type(data)}")
                    print(f"   Data: {data}")
            except Exception as e:
                print(f"❌ {symbol}: Error loading data - {e}")
                symbol_data[symbol] = None
        
        # Step 2: Simulate sector feature calculation
        print("\n[2] SECTOR FEATURE CALCULATION SIMULATION")
        print("-" * 40)
        
        if all(data is not None for data in symbol_data.values()):
            self._simulate_sector_calculation(symbol_data)
        else:
            print("❌ Cannot proceed with sector calculation - missing data")
        
        # Step 3: Window size analysis
        print("\n[3] WINDOW SIZE ANALYSIS")
        print("-" * 40)
        self._analyze_window_requirements(symbol_data)
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
    
    def _simulate_sector_calculation(self, symbol_data):
        """Simulate the exact sector feature calculation"""
        print("\nSimulating sector feature calculations...")
        
        for symbol, data in symbol_data.items():
            if data is None or len(data) == 0:
                print(f"❌ {symbol}: No data available")
                continue
                
            print(f"\n--- {symbol} Sector Features ---")
            
            # Sort by timestamp
            data_sorted = data.sort_values('timestamp').copy()
            close_prices = data_sorted['close']
            
            print(f"Total records: {len(close_prices)}")
            print(f"Close prices - Min: ${close_prices.min():.2f}, Max: ${close_prices.max():.2f}")
            print(f"Initial NaN count: {close_prices.isna().sum()}")
            
            # Calculate 20-day momentum (same as in _engineer_sector_features)
            try:
                momentum = close_prices.pct_change(20)
                momentum_nans = momentum.isna().sum()
                print(f"20-day momentum NaNs: {momentum_nans} ({momentum_nans/len(momentum)*100:.1f}%)")
                
                # Show first 25 values to understand pattern
                print(f"First 25 momentum values:")
                for i in range(min(25, len(momentum))):
                    val = momentum.iloc[i]
                    if pd.isna(val):
                        print(f"  [{i:2d}]: NaN")
                    else:
                        print(f"  [{i:2d}]: {val:.6f}")
                        
            except Exception as e:
                print(f"❌ Error calculating momentum: {e}")
            
            # Calculate 20-day volatility
            try:
                volatility = close_prices.rolling(window=20).std()
                volatility_nans = volatility.isna().sum()
                print(f"\n20-day volatility NaNs: {volatility_nans} ({volatility_nans/len(volatility)*100:.1f}%)")
                
                # Show first 25 values
                print(f"First 25 volatility values:")
                for i in range(min(25, len(volatility))):
                    val = volatility.iloc[i]
                    if pd.isna(val):
                        print(f"  [{i:2d}]: NaN")
                    else:
                        print(f"  [{i:2d}]: {val:.6f}")
                        
            except Exception as e:
                print(f"❌ Error calculating volatility: {e}")
    
    def _analyze_window_requirements(self, symbol_data):
        """Analyze window size requirements vs available data"""
        print("\nAnalyzing window size requirements...")
        
        window_size = 20
        print(f"Required window size: {window_size} periods")
        
        for symbol, data in symbol_data.items():
            if data is None:
                continue
                
            print(f"\n{symbol}:")
            total_records = len(data)
            expected_nans_momentum = window_size  # pct_change(20) creates 20 NaNs at start
            expected_nans_volatility = window_size - 1  # rolling(20).std() creates 19 NaNs at start
            
            print(f"  Total records: {total_records}")
            print(f"  Expected NaNs for momentum: {expected_nans_momentum}")
            print(f"  Expected NaNs for volatility: {expected_nans_volatility}")
            print(f"  Expected valid momentum values: {total_records - expected_nans_momentum}")
            print(f"  Expected valid volatility values: {total_records - expected_nans_volatility}")
            
            if total_records < window_size:
                print(f"  ⚠️  WARNING: Insufficient data for window calculations!")
            else:
                print(f"  ✅ Sufficient data for window calculations")

if __name__ == "__main__":
    diagnostic = SimpleSectorDiagnostic()
    asyncio.run(diagnostic.analyze_sector_calculation())
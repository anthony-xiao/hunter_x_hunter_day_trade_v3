#!/usr/bin/env python3
"""
Sector NaN Diagnostic Script

This script traces the exact data flow in sector feature calculations to identify
where NaN values are being introduced and whether they are expected or indicate issues.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

# Import project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from database import db_manager
from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.ml_feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorNaNDiagnostic:
    def __init__(self):
        self.supabase = db_manager.get_supabase_client()
        self.data_pipeline = DataPipeline()
        self.universal_engineer = UniversalFeatureEngineering()
        self.ml_engineer = FeatureEngineering()
        
    async def run_comprehensive_diagnostic(self):
        """Run comprehensive diagnostic analysis"""
        print("\n" + "="*80)
        print("SECTOR NaN DIAGNOSTIC ANALYSIS")
        print("="*80)
        
        # Test parameters
        symbols = ['AAPL', 'TSLA']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # 60 days for analysis
        
        print(f"\nAnalysis Period: {start_date.date()} to {end_date.date()}")
        print(f"Symbols: {symbols}")
        
        # Step 1: Raw data analysis
        await self._analyze_raw_data(symbols, start_date, end_date)
        
        # Step 2: Technical features analysis
        await self._analyze_technical_features(symbols, start_date, end_date)
        
        # Step 3: Sector calculation step-by-step trace
        await self._trace_sector_calculations(symbols, start_date, end_date)
        
        # Step 4: Window size validation
        await self._validate_window_requirements(symbols, start_date, end_date)
        
        # Step 5: Data alignment analysis
        await self._analyze_data_alignment(symbols, start_date, end_date)
        
        # Step 6: Expected vs actual NaN comparison
        await self._compare_expected_vs_actual_nans(symbols, start_date, end_date)
        
        print("\n" + "="*80)
        print("DIAGNOSTIC ANALYSIS COMPLETE")
        print("="*80)
        
    async def _analyze_raw_data(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Analyze raw market data quality and coverage"""
        print("\n[1] RAW DATA ANALYSIS")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                # Load raw market data
                raw_data = await self.data_pipeline.load_market_data(symbol, start_date, end_date)
                
                if raw_data.empty:
                    print(f"❌ {symbol}: NO DATA FOUND")
                    continue
                    
                print(f"\n📊 {symbol} Raw Data:")
                print(f"   Records: {len(raw_data):,}")
                print(f"   Date Range: {raw_data.index.min()} to {raw_data.index.max()}")
                print(f"   Frequency: {pd.infer_freq(raw_data.index) or 'Irregular'}")
                
                # Check for NaNs in raw data
                nan_counts = raw_data.isnull().sum()
                print(f"   NaN Counts: {dict(nan_counts[nan_counts > 0])}")
                
                # Check for data gaps
                time_diffs = raw_data.index.to_series().diff().dropna()
                median_interval = time_diffs.median()
                large_gaps = time_diffs[time_diffs > median_interval * 5]
                
                if len(large_gaps) > 0:
                    print(f"   ⚠️  Large gaps detected: {len(large_gaps)} gaps > {median_interval * 5}")
                    print(f"   Largest gap: {large_gaps.max()}")
                else:
                    print(f"   ✅ No significant data gaps detected")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error loading raw data - {e}")
                
    async def _analyze_technical_features(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Analyze technical features generation"""
        print("\n[2] TECHNICAL FEATURES ANALYSIS")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                # Generate technical features using ML engineer
                features = await self.ml_engineer.engineer_features(symbol, start_date, end_date)
                
                if 'technical_features' not in features or features['technical_features'].empty:
                    print(f"❌ {symbol}: NO TECHNICAL FEATURES GENERATED")
                    continue
                    
                tech_features = features['technical_features']
                
                print(f"\n🔧 {symbol} Technical Features:")
                print(f"   Records: {len(tech_features):,}")
                print(f"   Columns: {list(tech_features.columns)}")
                
                # Focus on close prices (used in sector calculations)
                if 'close' in tech_features.columns:
                    close_prices = tech_features['close']
                    close_nans = close_prices.isnull().sum()
                    print(f"   Close Price NaNs: {close_nans} ({close_nans/len(close_prices)*100:.2f}%)")
                    
                    if close_nans > 0:
                        # Find NaN positions
                        nan_positions = close_prices[close_prices.isnull()].index
                        print(f"   First NaN: {nan_positions[0] if len(nan_positions) > 0 else 'None'}")
                        print(f"   Last NaN: {nan_positions[-1] if len(nan_positions) > 0 else 'None'}")
                else:
                    print(f"   ❌ No 'close' column found in technical features")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error generating technical features - {e}")
                
    async def _trace_sector_calculations(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Step-by-step trace of sector feature calculations"""
        print("\n[3] SECTOR CALCULATION TRACE")
        print("-" * 40)
        
        try:
            # Get technical features for both symbols
            symbol_features = {}
            for symbol in symbols:
                features = await self.ml_engineer.engineer_features(symbol, start_date, end_date)
                if 'technical_features' in features and not features['technical_features'].empty:
                    symbol_features[symbol] = features['technical_features']
                else:
                    print(f"❌ No technical features for {symbol}")
                    return
                    
            print(f"\n🔍 Tracing sector calculations for {symbols}...")
            
            # Extract close prices
            close_prices = {}
            for symbol in symbols:
                if 'close' in symbol_features[symbol].columns:
                    close_prices[symbol] = symbol_features[symbol]['close']
                    print(f"   {symbol} close prices: {len(close_prices[symbol])} records")
                    print(f"   {symbol} close NaNs: {close_prices[symbol].isnull().sum()}")
                else:
                    print(f"❌ No close prices for {symbol}")
                    return
                    
            # Step 1: Calculate 20-day momentum (pct_change(20))
            print("\n   Step 1: 20-day Momentum Calculation")
            momentum_results = {}
            for symbol in symbols:
                momentum = close_prices[symbol].pct_change(20)
                momentum_results[symbol] = momentum
                
                momentum_nans = momentum.isnull().sum()
                print(f"   {symbol} momentum NaNs: {momentum_nans} ({momentum_nans/len(momentum)*100:.2f}%)")
                
                # Analyze first 25 values (should be NaN due to 20-day window)
                first_25_nans = momentum.iloc[:25].isnull().sum()
                print(f"   {symbol} expected NaNs (first 20): {first_25_nans}/25")
                
                # Check for unexpected NaNs after position 20
                unexpected_nans = momentum.iloc[20:].isnull().sum()
                print(f"   {symbol} unexpected NaNs (after pos 20): {unexpected_nans}")
                
            # Step 2: Calculate 20-day volatility (rolling(20).std())
            print("\n   Step 2: 20-day Volatility Calculation")
            volatility_results = {}
            for symbol in symbols:
                volatility = close_prices[symbol].rolling(window=20).std()
                volatility_results[symbol] = volatility
                
                volatility_nans = volatility.isnull().sum()
                print(f"   {symbol} volatility NaNs: {volatility_nans} ({volatility_nans/len(volatility)*100:.2f}%)")
                
                # Analyze first 25 values (should be NaN due to 20-day window)
                first_25_nans = volatility.iloc[:25].isnull().sum()
                print(f"   {symbol} expected NaNs (first 20): {first_25_nans}/25")
                
                # Check for unexpected NaNs after position 20
                unexpected_nans = volatility.iloc[20:].isnull().sum()
                print(f"   {symbol} unexpected NaNs (after pos 20): {unexpected_nans}")
                
            # Step 3: Analyze final fillna(0) impact
            print("\n   Step 3: Final fillna(0) Impact")
            for symbol in symbols:
                momentum_filled = momentum_results[symbol].fillna(0)
                volatility_filled = volatility_results[symbol].fillna(0)
                
                momentum_zeros = (momentum_filled == 0).sum()
                volatility_zeros = (volatility_filled == 0).sum()
                
                print(f"   {symbol} momentum zeros after fillna: {momentum_zeros}")
                print(f"   {symbol} volatility zeros after fillna: {volatility_zeros}")
                
        except Exception as e:
            print(f"❌ Error in sector calculation trace: {e}")
            
    async def _validate_window_requirements(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Validate if we have sufficient data for 20-day windows"""
        print("\n[4] WINDOW SIZE VALIDATION")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                raw_data = await self.data_pipeline.load_market_data(symbol, start_date, end_date)
                
                if raw_data.empty:
                    print(f"❌ {symbol}: No data for validation")
                    continue
                    
                # Convert to daily data for window analysis
                daily_data = raw_data.resample('D').last().dropna()
                
                print(f"\n📏 {symbol} Window Validation:")
                print(f"   Total days: {len(daily_data)}")
                print(f"   Required for 20-day window: 20 days")
                print(f"   Available for calculations: {max(0, len(daily_data) - 20)} days")
                
                if len(daily_data) < 20:
                    print(f"   ❌ INSUFFICIENT DATA: Need 20 days, have {len(daily_data)}")
                else:
                    print(f"   ✅ Sufficient data for 20-day calculations")
                    
                # Check for consecutive data availability
                date_diffs = daily_data.index.to_series().diff().dropna()
                max_gap = date_diffs.max()
                print(f"   Maximum gap between days: {max_gap}")
                
                if max_gap > timedelta(days=7):
                    print(f"   ⚠️  Large gaps may affect rolling calculations")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error in window validation - {e}")
                
    async def _analyze_data_alignment(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Analyze data alignment between symbols"""
        print("\n[5] DATA ALIGNMENT ANALYSIS")
        print("-" * 40)
        
        try:
            # Load data for both symbols
            symbol_data = {}
            for symbol in symbols:
                data = await self.data_pipeline.load_market_data(symbol, start_date, end_date)
                if not data.empty:
                    symbol_data[symbol] = data
                    
            if len(symbol_data) < 2:
                print("❌ Insufficient data for alignment analysis")
                return
                
            # Compare timestamps
            aapl_timestamps = set(symbol_data['AAPL'].index)
            tsla_timestamps = set(symbol_data['TSLA'].index)
            
            common_timestamps = aapl_timestamps.intersection(tsla_timestamps)
            aapl_only = aapl_timestamps - tsla_timestamps
            tsla_only = tsla_timestamps - aapl_timestamps
            
            print(f"\n🔄 Timestamp Alignment:")
            print(f"   AAPL timestamps: {len(aapl_timestamps):,}")
            print(f"   TSLA timestamps: {len(tsla_timestamps):,}")
            print(f"   Common timestamps: {len(common_timestamps):,}")
            print(f"   AAPL-only timestamps: {len(aapl_only):,}")
            print(f"   TSLA-only timestamps: {len(tsla_only):,}")
            
            alignment_ratio = len(common_timestamps) / max(len(aapl_timestamps), len(tsla_timestamps))
            print(f"   Alignment ratio: {alignment_ratio:.2%}")
            
            if alignment_ratio < 0.8:
                print(f"   ⚠️  Poor alignment may cause NaNs in cross-symbol calculations")
            else:
                print(f"   ✅ Good timestamp alignment")
                
        except Exception as e:
            print(f"❌ Error in alignment analysis: {e}")
            
    async def _compare_expected_vs_actual_nans(self, symbols: List[str], start_date: datetime, end_date: datetime):
        """Compare expected vs actual NaN patterns"""
        print("\n[6] EXPECTED VS ACTUAL NaN COMPARISON")
        print("-" * 40)
        
        try:
            # Calculate expected NaNs based on window requirements
            total_days = (end_date - start_date).days
            expected_nan_days = min(20, total_days)  # First 20 days should be NaN
            expected_valid_days = max(0, total_days - 20)
            
            print(f"\n📊 Expected NaN Pattern:")
            print(f"   Analysis period: {total_days} days")
            print(f"   Expected NaN days (first 20): {expected_nan_days}")
            print(f"   Expected valid days: {expected_valid_days}")
            print(f"   Expected NaN rate: {expected_nan_days/total_days*100:.1f}%")
            
            # Get actual sector features
            print(f"\n📈 Actual Sector Features:")
            
            # Use universal feature engineer to get sector features
            symbol_features = {}
            for symbol in symbols:
                features = await self.ml_engineer.engineer_features(symbol, start_date, end_date)
                if 'technical_features' in features:
                    symbol_features[symbol] = features['technical_features']
                    
            if len(symbol_features) >= 2:
                # Calculate sector features using the same logic as universal engineer
                sector_features = await self._calculate_sector_features_debug(symbol_features)
                
                for feature_name, values in sector_features.items():
                    actual_nans = values.isnull().sum()
                    actual_nan_rate = actual_nans / len(values) * 100
                    
                    print(f"   {feature_name}:")
                    print(f"     Total values: {len(values)}")
                    print(f"     Actual NaNs: {actual_nans}")
                    print(f"     Actual NaN rate: {actual_nan_rate:.1f}%")
                    print(f"     Expected NaN rate: {expected_nan_days/total_days*100:.1f}%")
                    
                    if actual_nan_rate > (expected_nan_days/total_days*100 + 5):  # 5% tolerance
                        print(f"     ❌ EXCESSIVE NaNs detected!")
                    else:
                        print(f"     ✅ NaN rate within expected range")
            else:
                print("❌ Insufficient symbol features for sector calculation")
                
        except Exception as e:
            print(f"❌ Error in NaN comparison: {e}")
            
    async def _calculate_sector_features_debug(self, symbol_features: Dict) -> Dict:
        """Debug version of sector feature calculation"""
        sector_features = {}
        
        try:
            # Extract close prices
            close_prices = {}
            for symbol, features in symbol_features.items():
                if 'close' in features.columns:
                    close_prices[symbol] = features['close']
                    
            # Calculate momentum and volatility for each symbol
            for symbol in close_prices.keys():
                momentum = close_prices[symbol].pct_change(20)
                volatility = close_prices[symbol].rolling(window=20).std()
                
                # Apply fillna(0) as in the original code
                momentum_filled = momentum.fillna(0)
                volatility_filled = volatility.fillna(0)
                
                sector_features[f'sector_{symbol}_momentum'] = momentum_filled
                sector_features[f'sector_{symbol}_volatility'] = volatility_filled
                
        except Exception as e:
            print(f"Error in debug sector calculation: {e}")
            
        return sector_features

async def main():
    """Main diagnostic function"""
    diagnostic = SectorNaNDiagnostic()
    await diagnostic.run_comprehensive_diagnostic()

if __name__ == "__main__":
    asyncio.run(main())
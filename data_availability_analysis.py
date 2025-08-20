#!/usr/bin/env python3
"""
Data Availability Analysis for Sector Features NaN Investigation

This script analyzes the actual market data availability for AAPL and TSLA
to determine if the high NaN frequency in sector features is due to:
1. Insufficient historical data
2. Data gaps or missing trading days
3. Database connectivity issues
4. Data quality problems
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering
from database import db_manager

class DataAvailabilityAnalyzer:
    """Comprehensive data availability analysis for sector features"""
    
    def __init__(self):
        self.supabase = db_manager.get_supabase_client()
        self.data_pipeline = DataPipeline()
        self.symbols = ['AAPL', 'TSLA']
        
    async def analyze_data_availability(self) -> Dict:
        """Comprehensive analysis of data availability"""
        print("=" * 80)
        print("DATA AVAILABILITY ANALYSIS FOR SECTOR FEATURES")
        print("=" * 80)
        
        results = {
            'database_connectivity': await self._test_database_connectivity(),
            'raw_data_coverage': await self._analyze_raw_data_coverage(),
            'data_quality': await self._analyze_data_quality(),
            'temporal_gaps': await self._analyze_temporal_gaps(),
            'sector_calculation_simulation': await self._simulate_sector_calculations(),
            'expected_vs_actual_nans': await self._compare_expected_actual_nans()
        }
        
        await self._generate_comprehensive_report(results)
        return results
    
    async def _test_database_connectivity(self) -> Dict:
        """Test database connectivity and basic queries"""
        print("\n1. TESTING DATABASE CONNECTIVITY")
        print("-" * 50)
        
        connectivity_results = {}
        
        try:
            # Test basic connection
            response = self.supabase.table('market_data').select('count').execute()
            connectivity_results['connection_status'] = 'SUCCESS'
            print("✓ Database connection successful")
            
            # Test symbol-specific queries
            for symbol in self.symbols:
                try:
                    response = self.supabase.table('market_data').select(
                        'timestamp'
                    ).eq('symbol', symbol).limit(1).execute()
                    
                    if response.data:
                        connectivity_results[f'{symbol}_accessible'] = True
                        print(f"✓ {symbol} data accessible")
                    else:
                        connectivity_results[f'{symbol}_accessible'] = False
                        print(f"✗ {symbol} data not found")
                        
                except Exception as e:
                    connectivity_results[f'{symbol}_accessible'] = False
                    print(f"✗ {symbol} query failed: {e}")
                    
        except Exception as e:
            connectivity_results['connection_status'] = f'FAILED: {e}'
            print(f"✗ Database connection failed: {e}")
            
        return connectivity_results
    
    async def _analyze_raw_data_coverage(self) -> Dict:
        """Analyze raw data coverage for each symbol"""
        print("\n2. ANALYZING RAW DATA COVERAGE")
        print("-" * 50)
        
        coverage_results = {}
        
        for symbol in self.symbols:
            print(f"\nAnalyzing {symbol}:")
            
            try:
                # Get total record count
                count_response = self.supabase.table('market_data').select(
                    'timestamp', count='exact'
                ).eq('symbol', symbol).execute()
                
                total_records = count_response.count if hasattr(count_response, 'count') else 0
                print(f"  Total records: {total_records:,}")
                
                # Get date range
                range_response = self.supabase.table('market_data').select(
                    'timestamp'
                ).eq('symbol', symbol).order('timestamp').limit(1).execute()
                
                earliest_date = None
                if range_response.data:
                    earliest_date = pd.to_datetime(range_response.data[0]['timestamp'])
                    print(f"  Earliest date: {earliest_date}")
                
                range_response = self.supabase.table('market_data').select(
                    'timestamp'
                ).eq('symbol', symbol).order('timestamp', desc=True).limit(1).execute()
                
                latest_date = None
                if range_response.data:
                    latest_date = pd.to_datetime(range_response.data[0]['timestamp'])
                    print(f"  Latest date: {latest_date}")
                
                # Calculate expected vs actual records
                if earliest_date and latest_date:
                    total_days = (latest_date - earliest_date).days
                    # Assuming minute data during market hours (6.5 hours * 60 minutes = 390 minutes per day)
                    expected_records_per_day = 390
                    # Assuming ~252 trading days per year
                    trading_days_estimate = int(total_days * (252/365))
                    expected_total_records = trading_days_estimate * expected_records_per_day
                    
                    coverage_percentage = (total_records / expected_total_records) * 100 if expected_total_records > 0 else 0
                    
                    print(f"  Date range: {total_days} calendar days")
                    print(f"  Estimated trading days: {trading_days_estimate}")
                    print(f"  Expected records: {expected_total_records:,}")
                    print(f"  Coverage: {coverage_percentage:.2f}%")
                    
                    coverage_results[symbol] = {
                        'total_records': total_records,
                        'earliest_date': earliest_date.isoformat() if earliest_date else None,
                        'latest_date': latest_date.isoformat() if latest_date else None,
                        'total_days': total_days,
                        'estimated_trading_days': trading_days_estimate,
                        'expected_records': expected_total_records,
                        'coverage_percentage': coverage_percentage
                    }
                else:
                    coverage_results[symbol] = {'error': 'No date range data available'}
                    
            except Exception as e:
                print(f"  Error analyzing {symbol}: {e}")
                coverage_results[symbol] = {'error': str(e)}
                
        return coverage_results
    
    async def _analyze_data_quality(self) -> Dict:
        """Analyze data quality issues"""
        print("\n3. ANALYZING DATA QUALITY")
        print("-" * 50)
        
        quality_results = {}
        
        for symbol in self.symbols:
            print(f"\nAnalyzing data quality for {symbol}:")
            
            try:
                # Sample recent data for quality analysis
                sample_response = self.supabase.table('market_data').select(
                    'timestamp, open, high, low, close, volume'
                ).eq('symbol', symbol).order('timestamp', desc=True).limit(1000).execute()
                
                if not sample_response.data:
                    quality_results[symbol] = {'error': 'No sample data available'}
                    continue
                
                df = pd.DataFrame(sample_response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Convert numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Quality checks
                null_counts = df[numeric_cols].isnull().sum()
                zero_counts = (df[numeric_cols] == 0).sum()
                negative_counts = (df[numeric_cols] < 0).sum()
                
                # Check for price inconsistencies
                price_issues = {
                    'high_less_than_low': (df['high'] < df['low']).sum(),
                    'close_outside_range': ((df['close'] > df['high']) | (df['close'] < df['low'])).sum(),
                    'open_outside_range': ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
                }
                
                print(f"  Sample size: {len(df)} records")
                print(f"  Null values: {null_counts.to_dict()}")
                print(f"  Zero values: {zero_counts.to_dict()}")
                print(f"  Negative values: {negative_counts.to_dict()}")
                print(f"  Price inconsistencies: {price_issues}")
                
                quality_results[symbol] = {
                    'sample_size': len(df),
                    'null_counts': null_counts.to_dict(),
                    'zero_counts': zero_counts.to_dict(),
                    'negative_counts': negative_counts.to_dict(),
                    'price_issues': price_issues
                }
                
            except Exception as e:
                print(f"  Error analyzing data quality for {symbol}: {e}")
                quality_results[symbol] = {'error': str(e)}
                
        return quality_results
    
    async def _analyze_temporal_gaps(self) -> Dict:
        """Analyze temporal gaps in the data"""
        print("\n4. ANALYZING TEMPORAL GAPS")
        print("-" * 50)
        
        gap_results = {}
        
        for symbol in self.symbols:
            print(f"\nAnalyzing temporal gaps for {symbol}:")
            
            try:
                # Get timestamps for gap analysis
                timestamps_response = self.supabase.table('market_data').select(
                    'timestamp'
                ).eq('symbol', symbol).order('timestamp').limit(5000).execute()
                
                if not timestamps_response.data:
                    gap_results[symbol] = {'error': 'No timestamp data available'}
                    continue
                
                timestamps = pd.to_datetime([row['timestamp'] for row in timestamps_response.data])
                timestamps = timestamps.sort_values()
                
                # Calculate time differences
                time_diffs = timestamps.diff().dropna()
                
                # Analyze gaps
                minute_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=1)]
                hour_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
                day_gaps = time_diffs[time_diffs > pd.Timedelta(days=1)]
                
                print(f"  Total timestamps analyzed: {len(timestamps)}")
                print(f"  Gaps > 1 minute: {len(minute_gaps)}")
                print(f"  Gaps > 1 hour: {len(hour_gaps)}")
                print(f"  Gaps > 1 day: {len(day_gaps)}")
                
                if len(day_gaps) > 0:
                    print(f"  Largest gap: {time_diffs.max()}")
                    print(f"  Average gap: {time_diffs.mean()}")
                
                gap_results[symbol] = {
                    'total_timestamps': len(timestamps),
                    'minute_gaps': len(minute_gaps),
                    'hour_gaps': len(hour_gaps),
                    'day_gaps': len(day_gaps),
                    'largest_gap': str(time_diffs.max()),
                    'average_gap': str(time_diffs.mean())
                }
                
            except Exception as e:
                print(f"  Error analyzing temporal gaps for {symbol}: {e}")
                gap_results[symbol] = {'error': str(e)}
                
        return gap_results
    
    async def _simulate_sector_calculations(self) -> Dict:
        """Simulate sector feature calculations to identify NaN sources"""
        print("\n5. SIMULATING SECTOR CALCULATIONS")
        print("-" * 50)
        
        simulation_results = {}
        
        for symbol in self.symbols:
            print(f"\nSimulating sector calculations for {symbol}:")
            
            try:
                # Get recent data for simulation
                data_response = self.supabase.table('market_data').select(
                    'timestamp, close'
                ).eq('symbol', symbol).order('timestamp', desc=True).limit(1000).execute()
                
                if not data_response.data:
                    simulation_results[symbol] = {'error': 'No data for simulation'}
                    continue
                
                df = pd.DataFrame(data_response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Simulate momentum calculation (20-day pct_change)
                momentum = df['close'].pct_change(20)
                momentum_nans = momentum.isnull().sum()
                
                # Simulate volatility calculation (20-day rolling std of returns)
                returns = df['close'].pct_change()
                volatility = returns.rolling(20).std()
                volatility_nans = volatility.isnull().sum()
                
                print(f"  Data points: {len(df)}")
                print(f"  Close price NaNs: {df['close'].isnull().sum()}")
                print(f"  Momentum NaNs (before fillna): {momentum_nans}")
                print(f"  Volatility NaNs (before fillna): {volatility_nans}")
                print(f"  Expected momentum NaNs: 20 (first 20 values)")
                print(f"  Expected volatility NaNs: 20 (first 20 values)")
                
                # Check if NaN count exceeds expected
                excess_momentum_nans = max(0, momentum_nans - 20)
                excess_volatility_nans = max(0, volatility_nans - 20)
                
                print(f"  Excess momentum NaNs: {excess_momentum_nans}")
                print(f"  Excess volatility NaNs: {excess_volatility_nans}")
                
                simulation_results[symbol] = {
                    'data_points': len(df),
                    'close_nans': int(df['close'].isnull().sum()),
                    'momentum_nans': int(momentum_nans),
                    'volatility_nans': int(volatility_nans),
                    'expected_momentum_nans': 20,
                    'expected_volatility_nans': 20,
                    'excess_momentum_nans': int(excess_momentum_nans),
                    'excess_volatility_nans': int(excess_volatility_nans)
                }
                
            except Exception as e:
                print(f"  Error simulating calculations for {symbol}: {e}")
                simulation_results[symbol] = {'error': str(e)}
                
        return simulation_results
    
    async def _compare_expected_actual_nans(self) -> Dict:
        """Compare expected vs actual NaN patterns from the original analysis"""
        print("\n6. COMPARING EXPECTED VS ACTUAL NaN PATTERNS")
        print("-" * 50)
        
        # Original NaN counts from the user's observation
        observed_nans = {
            'sector_AAPL_momentum': 92412,
            'sector_AAPL_volatility': 92412,
            'sector_TSLA_momentum': 102666,
            'sector_TSLA_volatility': 102666
        }
        
        total_observed_nans = sum(observed_nans.values())
        print(f"Total observed NaNs: {total_observed_nans:,}")
        
        # Calculate expected NaNs based on data availability
        comparison_results = {'observed_nans': observed_nans}
        
        for symbol in self.symbols:
            try:
                # Get total record count for this symbol
                count_response = self.supabase.table('market_data').select(
                    'timestamp', count='exact'
                ).eq('symbol', symbol).execute()
                
                total_records = count_response.count if hasattr(count_response, 'count') else 0
                
                # Expected NaNs: 20 for momentum + 20 for volatility per symbol
                expected_nans_per_feature = 20
                expected_total_nans = expected_nans_per_feature * 2  # momentum + volatility
                
                actual_momentum_nans = observed_nans.get(f'sector_{symbol}_momentum', 0)
                actual_volatility_nans = observed_nans.get(f'sector_{symbol}_volatility', 0)
                actual_total_nans = actual_momentum_nans + actual_volatility_nans
                
                excess_nans = actual_total_nans - expected_total_nans
                nan_rate = (actual_total_nans / (total_records * 2)) * 100 if total_records > 0 else 0
                
                print(f"\n{symbol} Analysis:")
                print(f"  Total records: {total_records:,}")
                print(f"  Expected NaNs: {expected_total_nans}")
                print(f"  Actual NaNs: {actual_total_nans:,}")
                print(f"  Excess NaNs: {excess_nans:,}")
                print(f"  NaN rate: {nan_rate:.2f}%")
                
                comparison_results[symbol] = {
                    'total_records': total_records,
                    'expected_nans': expected_total_nans,
                    'actual_nans': actual_total_nans,
                    'excess_nans': excess_nans,
                    'nan_rate_percentage': nan_rate
                }
                
            except Exception as e:
                print(f"  Error analyzing {symbol}: {e}")
                comparison_results[symbol] = {'error': str(e)}
        
        return comparison_results
    
    async def _generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)
        
        # Summary of findings
        print("\nKEY FINDINGS:")
        print("-" * 20)
        
        # Database connectivity
        if results['database_connectivity']['connection_status'] == 'SUCCESS':
            print("✓ Database connectivity is working")
        else:
            print("✗ Database connectivity issues detected")
        
        # Data coverage analysis
        coverage_data = results['raw_data_coverage']
        for symbol in self.symbols:
            if symbol in coverage_data and 'coverage_percentage' in coverage_data[symbol]:
                coverage = coverage_data[symbol]['coverage_percentage']
                if coverage > 80:
                    print(f"✓ {symbol} has good data coverage ({coverage:.1f}%)")
                elif coverage > 50:
                    print(f"⚠ {symbol} has moderate data coverage ({coverage:.1f}%)")
                else:
                    print(f"✗ {symbol} has poor data coverage ({coverage:.1f}%)")
        
        # NaN analysis
        nan_comparison = results['expected_vs_actual_nans']
        for symbol in self.symbols:
            if symbol in nan_comparison and 'excess_nans' in nan_comparison[symbol]:
                excess = nan_comparison[symbol]['excess_nans']
                rate = nan_comparison[symbol]['nan_rate_percentage']
                if excess > 1000:
                    print(f"✗ {symbol} has excessive NaNs: {excess:,} excess ({rate:.2f}% rate)")
                elif excess > 100:
                    print(f"⚠ {symbol} has moderate excess NaNs: {excess:,} excess ({rate:.2f}% rate)")
                else:
                    print(f"✓ {symbol} NaN levels are within expected range")
        
        # Root cause determination
        print("\nROOT CAUSE ANALYSIS:")
        print("-" * 25)
        
        total_excess_nans = 0
        for symbol in self.symbols:
            if symbol in nan_comparison and 'excess_nans' in nan_comparison[symbol]:
                total_excess_nans += nan_comparison[symbol]['excess_nans']
        
        if total_excess_nans > 100000:
            print("🔴 CRITICAL ISSUE: Excessive NaN values indicate a serious data problem")
            print("   Likely causes:")
            print("   - Insufficient historical data in database")
            print("   - Data loading/processing pipeline issues")
            print("   - Database query pagination problems")
            print("   - Data alignment issues between symbols")
        elif total_excess_nans > 10000:
            print("🟡 MODERATE ISSUE: Higher than expected NaN values")
            print("   Likely causes:")
            print("   - Some data gaps or missing trading periods")
            print("   - Partial data loading issues")
        else:
            print("🟢 NORMAL: NaN levels are within expected mathematical bounds")
            print("   The observed NaNs are primarily due to:")
            print("   - Mathematical requirements (first 20 values for rolling calculations)")
            print("   - Normal market data gaps (weekends, holidays)")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 20)
        
        if total_excess_nans > 100000:
            print("1. Investigate data pipeline and ensure complete historical data loading")
            print("2. Check database query pagination and data retrieval logic")
            print("3. Verify data alignment and timestamp consistency")
            print("4. Consider data backfill for missing periods")
        elif total_excess_nans > 10000:
            print("1. Review data quality and identify specific gap periods")
            print("2. Implement more robust data validation")
            print("3. Consider forward-fill strategies for minor gaps")
        else:
            print("1. Current NaN handling with fillna(0) is appropriate")
            print("2. Continue monitoring for any changes in data patterns")
            print("3. No immediate action required")

async def main():
    """Main analysis function"""
    analyzer = DataAvailabilityAnalyzer()
    results = await analyzer.analyze_data_availability()
    return results

if __name__ == "__main__":
    asyncio.run(main())
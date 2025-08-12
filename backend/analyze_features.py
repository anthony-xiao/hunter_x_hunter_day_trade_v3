#!/usr/bin/env python3
"""
Feature Count Analysis Script
Analyzes the current feature engineering implementation to count total features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Optional

# Import required modules
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings

class FeatureCounter:
    """Standalone feature counter to avoid circular imports"""
    
    def __init__(self):
        # Initialize database connection
        from database import db_manager
        self.supabase = db_manager.get_supabase_client()
        self.engine = None
        self.Session = None
        
    def count_technical_features(self) -> int:
        """Count technical indicator features"""
        # Based on the implementation in ml_feature_engineering.py
        features = [
            # Price-based indicators (14)
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'rsi_14', 'rsi_21',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            
            # Volume indicators (8)
            'volume_sma_10', 'volume_sma_20',
            'volume_ratio_5', 'volume_ratio_10',
            'obv', 'ad_line', 'cmf', 'vwap_ratio',
            
            # Momentum indicators (12)
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'atr_14', 'atr_21',
            'adx', 'plus_di', 'minus_di',
            
            # Volatility indicators (6)
            'volatility_5', 'volatility_10', 'volatility_20',
            'garch_volatility', 'parkinson_volatility', 'garman_klass_volatility',
            
            # Support/Resistance (4)
            'support_level', 'resistance_level', 'pivot_point', 'fibonacci_retracement',
            
            # Price patterns (3)
            'doji', 'hammer', 'engulfing',
            
            # Returns and ratios (8)
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'log_returns_1', 'log_returns_5', 'log_returns_10', 'log_returns_20',
            
            # Additional technical features (40)
            'price_position', 'volume_position', 'rsi_position',
            'bb_position', 'macd_position', 'stoch_position',
            'williams_position', 'cci_position', 'adx_position',
            'atr_position', 'obv_position', 'ad_position',
            'cmf_position', 'vwap_position', 'volatility_position',
            'support_distance', 'resistance_distance', 'pivot_distance',
            'fibonacci_distance', 'trend_strength', 'momentum_strength',
            'volume_strength', 'volatility_regime', 'price_acceleration',
            'volume_acceleration', 'momentum_divergence', 'volume_divergence',
            'price_momentum_correlation', 'volume_price_correlation',
            'volatility_momentum_correlation', 'trend_consistency',
            'momentum_consistency', 'volume_consistency', 'volatility_consistency',
            'price_efficiency', 'market_efficiency', 'information_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'recovery_factor'
        ]
        return len(features)  # Should be 95
    
    def count_microstructure_features(self) -> int:
        """Count market microstructure features"""
        features = [
            # Basic microstructure (10)
            'spread_proxy', 'price_impact', 'tick_direction',
            'consecutive_up_ticks', 'consecutive_down_ticks',
            'order_flow_imbalance_proxy', 'vwap_deviation',
            'volume_weighted_price', 'transaction_intensity',
            'accumulated_volume_ratio',
            
            # Advanced microstructure (31)
            'market_efficiency_ratio', 'price_discovery_efficiency',
            'liquidity_proxy', 'market_impact_proxy', 'order_flow_toxicity',
            'market_depth_proxy', 'bid_ask_spread_estimate',
            'effective_spread_estimate', 'realized_spread_estimate',
            'price_improvement_estimate', 'market_fragmentation_proxy',
            'adverse_selection_proxy', 'inventory_risk_proxy',
            'information_asymmetry_proxy', 'trading_intensity',
            'volume_clustering', 'price_clustering', 'time_clustering',
            'microstructure_noise', 'jump_detection', 'regime_detection',
            'volatility_clustering', 'autocorrelation_1', 'autocorrelation_5',
            'autocorrelation_10', 'partial_autocorrelation_1',
            'partial_autocorrelation_5', 'partial_autocorrelation_10',
            'hurst_exponent', 'fractal_dimension', 'entropy'
        ]
        return len(features)  # Should be 41
    
    def count_sentiment_features(self) -> int:
        """Count sentiment-based features"""
        features = [
            'price_momentum_sentiment', 'volume_sentiment',
            'volatility_sentiment', 'trend_strength_sentiment',
            'support_resistance_sentiment', 'technical_sentiment',
            'market_regime_sentiment', 'risk_sentiment'
        ]
        return len(features)  # Should be 8
    
    def count_macro_features(self) -> int:
        """Count macro/time-based features"""
        features = [
            # Time features (7)
            'hour', 'minute', 'day_of_week', 'day_of_month',
            'week_of_year', 'month', 'quarter',
            
            # Market session features (3)
            'is_market_open', 'session_type', 'time_to_close',
            
            # Seasonal features (2)
            'seasonal_pattern', 'holiday_effect'
        ]
        return len(features)  # Should be 12
    
    def count_cross_asset_features(self) -> int:
        """Count cross-asset features"""
        features = [
            # SPY correlation features (4)
            'spy_correlation_5', 'spy_correlation_20',
            'spy_beta_5', 'spy_beta_20',
            
            # Additional cross-asset features (4)
            'sector_correlation', 'market_correlation',
            'volatility_correlation', 'volume_correlation'
        ]
        return len(features)  # Should be 8
    
    def count_advanced_features(self) -> int:
        """Count advanced composite features"""
        features = [
            'composite_momentum', 'composite_volatility',
            'composite_trend', 'risk_adjusted_returns',
            'volume_price_divergence', 'regime_detection_score',
            'mean_reversion_signal', 'momentum_persistence'
        ]
        return len(features)  # Should be 8
    
    def get_total_feature_count(self) -> dict:
        """Get total feature count breakdown"""
        tech_count = self.count_technical_features()
        micro_count = self.count_microstructure_features()
        sentiment_count = self.count_sentiment_features()
        macro_count = self.count_macro_features()
        cross_count = self.count_cross_asset_features()
        advanced_count = self.count_advanced_features()
        
        total_engineered = tech_count + micro_count + sentiment_count + macro_count + cross_count + advanced_count
        
        # Basic OHLCV columns that might be added
        basic_cols = ['open', 'high', 'low', 'close', 'volume']
        total_with_basic = total_engineered + len(basic_cols)
        
        return {
            'technical': tech_count,
            'microstructure': micro_count,
            'sentiment': sentiment_count,
            'macro': macro_count,
            'cross_asset': cross_count,
            'advanced': advanced_count,
            'total_engineered': total_engineered,
            'basic_columns': len(basic_cols),
            'total_with_basic': total_with_basic
        }
    
    def check_database_features(self, symbol: str = 'AAPL') -> Optional[int]:
        """Check actual feature count in database"""
        try:
            if not self.supabase:
                print("Supabase client not available")
                return None
                
            # Get recent feature data from database using Supabase
            response = self.supabase.table('features').select('features').eq('symbol', symbol).order('timestamp', desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                features_data = response.data[0]['features']
                if features_data:
                    feature_count = len(features_data)
                    print(f"Database feature count for {symbol}: {feature_count}")
                    print(f"Sample feature keys: {list(features_data.keys())[:20]}")
                    return feature_count
                else:
                    print(f"No features found in database for {symbol}")
                    return None
            else:
                print(f"No features found in database for {symbol}")
                return None
                    
        except Exception as e:
            print(f"Error checking database features: {e}")
            return None

def main():
    """Main analysis function"""
    print("=== Feature Count Analysis ===")
    
    counter = FeatureCounter()
    
    # Get theoretical feature count
    feature_breakdown = counter.get_total_feature_count()
    
    print("\nTheoretical Feature Count Breakdown:")
    print(f"Technical Features: {feature_breakdown['technical']}")
    print(f"Microstructure Features: {feature_breakdown['microstructure']}")
    print(f"Sentiment Features: {feature_breakdown['sentiment']}")
    print(f"Macro Features: {feature_breakdown['macro']}")
    print(f"Cross-Asset Features: {feature_breakdown['cross_asset']}")
    print(f"Advanced Features: {feature_breakdown['advanced']}")
    print(f"Total Engineered Features: {feature_breakdown['total_engineered']}")
    print(f"Basic OHLCV Columns: {feature_breakdown['basic_columns']}")
    print(f"Total with Basic Columns: {feature_breakdown['total_with_basic']}")
    
    # Check database
    print("\n=== Database Analysis ===")
    db_count = counter.check_database_features('AAPL')
    
    if db_count:
        print(f"\n=== Comparison ===")
        print(f"Expected total features: {feature_breakdown['total_with_basic']}")
        print(f"Database features: {db_count}")
        print(f"Difference: {db_count - feature_breakdown['total_with_basic']}")
        
        if db_count == 153:
            print("\n✅ Database count matches reported 153 features")
            extra_features = 153 - feature_breakdown['total_with_basic']
            print(f"There are {extra_features} additional features beyond the expected {feature_breakdown['total_with_basic']}")
        elif db_count == 148:
            print("\n⚠️  Database count matches previous analysis of 148 features")
        else:
            print(f"\n❓ Database count ({db_count}) doesn't match expected values")
    
    # No need to dispose engine since we're using Supabase client

if __name__ == "__main__":
    main()
import json
from sqlalchemy import create_engine, text
from config import settings

# Expected features based on the code analysis
expected_features = {
    "Price Features": [
        'price_range', 'price_change', 'price_change_pct', 'close_position', 
        'gap', 'gap_pct', 'hl_ratio', 'oc_ratio'
    ],
    "Technical Indicators": [
        # Moving averages
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio', 'price_sma_50_ratio',
        'price_ema_5_ratio', 'price_ema_10_ratio', 'price_ema_20_ratio', 'price_ema_50_ratio',
        # Bollinger Bands
        'bb_upper_10', 'bb_middle_10', 'bb_lower_10', 'bb_width_10', 'bb_position_10',
        'bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_width_20', 'bb_position_20',
        # RSI
        'rsi_7', 'rsi_14', 'rsi_21',
        # MACD
        'macd', 'macd_signal', 'macd_histogram',
        # Stochastic
        'stoch_k', 'stoch_d',
        # Others
        'williams_r', 'atr', 'atr_pct', 'cci', 'mfi', 'obv', 'ad'
    ],
    "Volume Features": [
        'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
        'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
        'volume_price', 'vwap', 'price_vwap_ratio',
        'volume_oscillator', 'pvt'
    ],
    "Microstructure Features": [
        'spread_proxy', 'price_impact', 'tick_direction',
        'consecutive_up', 'consecutive_down', 'flow_imbalance'
    ],
    "Time Features": [
        'hour', 'minute', 'day_of_week',
        'is_market_open', 'is_pre_market', 'is_after_hours',
        'minutes_since_open'
    ],
    "Volatility Features": [
        'volatility_5', 'volatility_10', 'volatility_20',
        'parkinson_vol', 'gk_vol',
        'vol_ratio_5_20', 'vol_ratio_10_20'
    ],
    "Momentum Features": [
        'roc_1', 'roc_5', 'roc_10', 'roc_20',
        'momentum_5', 'momentum_10', 'momentum_20',
        'acceleration_5', 'acceleration_10',
        'trend_strength_10', 'trend_strength_20'
    ],
    "Statistical Features": [
        'skewness_10', 'skewness_20',
        'kurtosis_10', 'kurtosis_20',
        'percentile_rank_10', 'percentile_rank_20',
        'zscore_10', 'zscore_20',
        'autocorr_lag_1', 'autocorr_lag_5'
    ]
}

try:
    # Connect to Supabase
    from database import db_manager
    supabase = db_manager.get_supabase_client()
    
    if not supabase:
        print("Error: Supabase client not available")
        exit(1)
    
    # Get the most recent features for AAPL
    result = supabase.table('features').select('features').eq('symbol', 'AAPL').order('timestamp', desc=True).limit(1).execute()
    
    if result.data:
        row = result.data[0]
        features_data = row['features']
    else:
        row = None
        features_data = None
    
    if row:
        stored_features = set(features_data.keys())
        
        print("FEATURE VERIFICATION REPORT")
        print("=" * 50)
        
        total_expected = 0
        total_found = 0
        total_missing = 0
        
        for category, expected_list in expected_features.items():
            print(f"\n{category}:")
            print("-" * 30)
            
            found_features = []
            missing_features = []
            
            for feature in expected_list:
                if feature in stored_features:
                    found_features.append(feature)
                else:
                    missing_features.append(feature)
            
            total_expected += len(expected_list)
            total_found += len(found_features)
            total_missing += len(missing_features)
            
            print(f"Expected: {len(expected_list)}")
            print(f"Found: {len(found_features)}")
            print(f"Missing: {len(missing_features)}")
            
            if missing_features:
                print(f"Missing features: {', '.join(missing_features)}")
            else:
                print("✅ All features found!")
        
        # Check for unexpected features
        all_expected = set()
        for feature_list in expected_features.values():
            all_expected.update(feature_list)
        
        unexpected_features = stored_features - all_expected
        
        print(f"\nSUMMARY:")
        print("=" * 30)
        print(f"Total expected features: {total_expected}")
        print(f"Total found features: {total_found}")
        print(f"Total missing features: {total_missing}")
        print(f"Total stored features: {len(stored_features)}")
        print(f"Unexpected features: {len(unexpected_features)}")
        
        if unexpected_features:
            print(f"\nUnexpected features found: {', '.join(sorted(unexpected_features))}")
        
        coverage_percentage = (total_found / total_expected) * 100
        print(f"\nFeature coverage: {coverage_percentage:.1f}%")
        
        if total_missing == 0:
            print("\n🎉 SUCCESS: All expected features are being stored correctly!")
        else:
            print(f"\n⚠️  WARNING: {total_missing} features are missing from storage.")
    
except Exception as e:
    print(f"Error: {e}")
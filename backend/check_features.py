import json
from sqlalchemy import create_engine, text
from config import settings

try:
    # Connect to Supabase
    from database import db_manager
    supabase = db_manager.get_supabase_client()
    
    if not supabase:
        print("Error: Supabase client not available")
        exit(1)
    
    # Get the most recent features for AAPL
    result = supabase.table('features').select('symbol, timestamp, features').eq('symbol', 'AAPL').order('timestamp', desc=True).limit(1).execute()
    
    row = result.data[0] if result.data else None
    
    if row:
        symbol = row['symbol']
        timestamp = row['timestamp']
        features = row['features']
        print(f"Symbol: {symbol}")
        print(f"Timestamp: {timestamp}")
        print(f"Number of features: {len(features)}")
        print("\nFeature names and values:")
        print("=" * 50)
        
        # Group features by category
        price_features = []
        technical_features = []
        volume_features = []
        microstructure_features = []
        time_features = []
        volatility_features = []
        momentum_features = []
        statistical_features = []
        other_features = []
        
        for key, value in sorted(features.items()):
            if any(x in key.lower() for x in ['price', 'gap', 'hl_ratio', 'oc_ratio', 'close_position']):
                price_features.append((key, value))
            elif any(x in key.lower() for x in ['sma', 'ema', 'bb_', 'rsi', 'macd', 'stoch', 'williams', 'atr', 'cci', 'mfi', 'obv', 'ad']):
                technical_features.append((key, value))
            elif any(x in key.lower() for x in ['volume', 'vwap', 'pvt']):
                volume_features.append((key, value))
            elif any(x in key.lower() for x in ['spread', 'impact', 'tick', 'consecutive', 'flow']):
                microstructure_features.append((key, value))
            elif any(x in key.lower() for x in ['hour', 'minute', 'day_of_week', 'market', 'minutes_since']):
                time_features.append((key, value))
            elif any(x in key.lower() for x in ['volatility', 'vol_', 'parkinson', 'gk_vol']):
                volatility_features.append((key, value))
            elif any(x in key.lower() for x in ['roc_', 'momentum', 'acceleration', 'trend_strength']):
                momentum_features.append((key, value))
            elif any(x in key.lower() for x in ['skewness', 'kurtosis', 'percentile', 'zscore', 'autocorr']):
                statistical_features.append((key, value))
            else:
                other_features.append((key, value))
        
        # Print categorized features
        categories = [
            ("Price Features", price_features),
            ("Technical Indicators", technical_features),
            ("Volume Features", volume_features),
            ("Microstructure Features", microstructure_features),
            ("Time Features", time_features),
            ("Volatility Features", volatility_features),
            ("Momentum Features", momentum_features),
            ("Statistical Features", statistical_features),
            ("Other Features", other_features)
        ]
        
        for category_name, category_features in categories:
            if category_features:
                print(f"\n{category_name} ({len(category_features)} features):")
                print("-" * 40)
                for i, (key, value) in enumerate(category_features, 1):
                    print(f"{i:2d}. {key}: {value}")
        
        print(f"\nTotal features stored: {len(features)}")
        
    else:
        print("No features found for AAPL")
    
    # Also check how many feature records exist for AAPL
    result = supabase.table('features').select('id', count='exact').eq('symbol', 'AAPL').execute()
    count = result.count
    print(f"\nTotal feature records for AAPL: {count}")
    
except Exception as e:
    print(f"Error: {e}")
import psycopg2
import json
from config import settings

try:
    # Connect to PostgreSQL
    database_url = f"postgresql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()
    
    # Get the most recent features for AAPL
    cur.execute("""
        SELECT symbol, timestamp, features 
        FROM features 
        WHERE symbol = %s 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, ('AAPL',))
    
    result = cur.fetchone()
    
    if result:
        symbol, timestamp, features = result
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
    cur.execute("SELECT COUNT(*) FROM features WHERE symbol = %s", ('AAPL',))
    count = cur.fetchone()[0]
    print(f"\nTotal feature records for AAPL: {count}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
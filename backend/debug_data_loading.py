import asyncio
from data.data_pipeline import DataPipeline
from ml.universal_trainer import UniversalTrainer
from ml.universal_feature_engineering import UniversalFeatureEngineering
from datetime import datetime as dt

async def test_data_loading():
    print("=== Testing Data Loading ===")
    
    # Initialize data pipeline
    dp = DataPipeline()
    
    # Test market data loading
    print("\n=== Testing Market Data Loading ===")
    start_date = dt(2025, 7, 19)
    end_date = dt(2025, 8, 18)
    symbols = ['AAPL', 'TSLA']
    
    universal_data = await dp.load_universal_data(symbols, start_date, end_date)
    print(f"Loaded data for {len(universal_data)} symbols:")
    for symbol, data in universal_data.items():
        print(f"  {symbol}: {len(data)} records")
        if len(data) > 0:
            print(f"    Date range: {data.index.min()} to {data.index.max()}")
            print(f"    Columns: {list(data.columns)}")
    
    # Test individual feature engineering first
    print("\n=== Testing Individual Feature Engineering ===")
    ufe = UniversalFeatureEngineering(dp.supabase)
    
    for symbol in symbols:
        print(f"\nTesting feature engineering for {symbol}...")
        try:
            features = await ufe.engineer_features(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                include_cross_asset=True,
                training_mode=True
            )
            print(f"  Technical features: {len(features.technical_features)} rows, {len(features.technical_features.columns) if len(features.technical_features) > 0 else 0} columns")
            print(f"  Microstructure features: {len(features.market_microstructure)} rows, {len(features.market_microstructure.columns) if len(features.market_microstructure) > 0 else 0} columns")
            print(f"  Feature importance: {len(features.feature_importance)} items")
            
            if len(features.technical_features) > 0:
                print(f"  Technical feature columns: {list(features.technical_features.columns)[:10]}...")  # Show first 10 columns
                print(f"  Date range: {features.technical_features.index.min()} to {features.technical_features.index.max()}")
            else:
                print(f"  WARNING: No technical features generated for {symbol}")
        except Exception as e:
            print(f"  ERROR in feature engineering for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test universal training dataset preparation
    print("\n=== Testing Universal Training Dataset ===")
    trainer = UniversalTrainer(dp, ufe)
    
    try:
        result = await trainer.prepare_universal_dataset(
            symbols=symbols,
            start_date='2025-07-19',
            end_date='2025-08-18'
        )
        
        print(f"Result: {result}")
        if len(result) == 4:
            X_train, y_train, X_val, y_val = result
            print(f"X_train: {type(X_train)}, length: {len(X_train)}")
            print(f"y_train: {type(y_train)}, length: {len(y_train)}")
            print(f"X_val: {type(X_val)}, length: {len(X_val)}")
            print(f"y_val: {type(y_val)}, length: {len(y_val)}")
        else:
            print(f"Unexpected result format: {type(result)}, length: {len(result)}")
            
    except Exception as e:
        print(f"Error in prepare_universal_dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test universal feature engineering
    print("\n=== Testing Universal Feature Engineering ===")
    ufe = UniversalFeatureEngineering(dp.supabase)
    
    # Test individual feature engineering first
    print("\n=== Testing Individual Feature Engineering ===")
    from datetime import datetime
    start_date = datetime(2025, 7, 19)
    end_date = datetime(2025, 8, 18)
    symbols = ['AAPL', 'TSLA']
    
    for symbol in symbols:
        print(f"\nTesting feature engineering for {symbol}...")
        features = await ufe.engineer_features(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            include_cross_asset=True,
            training_mode=True
        )
        print(f"  Technical features: {len(features.technical_features)} rows, {len(features.technical_features.columns) if len(features.technical_features) > 0 else 0} columns")
        print(f"  Microstructure features: {len(features.market_microstructure)} rows, {len(features.market_microstructure.columns) if len(features.market_microstructure) > 0 else 0} columns")
        print(f"  Feature importance: {len(features.feature_importance)} items")
        
        if len(features.technical_features) > 0:
            print(f"  Technical feature columns: {list(features.technical_features.columns)[:10]}...")  # Show first 10 columns
            print(f"  Date range: {features.technical_features.index.min()} to {features.technical_features.index.max()}")
        else:
            print(f"  WARNING: No technical features generated for {symbol}")
    
    # Test universal feature engineering
    print("\n=== Testing Universal Feature Engineering ===")
    universal_features = await ufe.engineer_universal_features(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_cross_asset=True,
        training_mode=True
    )
    
    print(f"Universal features generated: {len(universal_features.symbol_features)} symbols")
    for symbol, features in universal_features.symbol_features.items():
        print(f"  {symbol}: {len(features.technical_features)} technical features")
    
    print(f"Cross-symbol features: {len(universal_features.cross_symbol_features.columns)} columns")
    print(f"Market regime features: {len(universal_features.market_regime_features.columns)} columns")

if __name__ == "__main__":
    asyncio.run(test_data_loading())
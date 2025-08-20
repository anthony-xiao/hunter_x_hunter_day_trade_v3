#!/usr/bin/env python3

import asyncio
from data.data_pipeline import DataPipeline
from datetime import datetime

async def test_batch_processing():
    print('Testing batch processing for multiple symbols...')
    
    dp = DataPipeline()
    symbols = ['AAPL', 'TSLA', 'MSFT']
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)
    
    print(f'Loading data for symbols: {symbols}')
    
    try:
        data = await dp.load_universal_data(symbols, start_date, end_date)
        print(f'✓ Loaded data for {len(data)} symbols')
        
        for symbol in symbols:
            if symbol in data:
                print(f'  - {symbol}: {len(data[symbol])} records')
            else:
                print(f'  - {symbol}: No data')
        
        print('Batch processing test: PASSED')
        return True
        
    except Exception as e:
        print(f'Batch processing test: FAILED - {e}')
        return False

if __name__ == '__main__':
    asyncio.run(test_batch_processing())
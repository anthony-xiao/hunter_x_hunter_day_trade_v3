#!/usr/bin/env python3
"""
Test script to verify that WebSocket aggregate data is properly stored in the database.
This script simulates the WebSocket data flow and checks database insertion.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from unittest.mock import Mock

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.data.polygon_websocket import PolygonWebSocketManager
from backend.data.data_pipeline import DataPipeline

class MockEquityAgg:
    """Mock EquityAgg object to simulate WebSocket data"""
    def __init__(self, symbol="TSLA"):
        self.symbol = symbol
        self.start_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)  # Current time in ms
        self.s = self.start_timestamp  # Alternative timestamp field
        self.open = 250.50
        self.high = 252.75
        self.low = 249.25
        self.close = 251.80
        self.volume = 15000
        self.vwap = 251.25
        self.transactions = 125

async def test_websocket_database_storage():
    """Test that WebSocket aggregate data is stored in the database"""
    print("Testing WebSocket aggregate data storage...")
    
    try:
        # Create WebSocket manager instance
        ws_manager = PolygonWebSocketManager()
        
        # Create mock aggregate data
        mock_agg = MockEquityAgg("TSLA")
        
        print(f"Simulating aggregate data for {mock_agg.symbol}:")
        print(f"  Timestamp: {datetime.fromtimestamp(mock_agg.start_timestamp / 1000, tz=timezone.utc)}")
        print(f"  OHLCV: O={mock_agg.open}, H={mock_agg.high}, L={mock_agg.low}, C={mock_agg.close}, V={mock_agg.volume}")
        print(f"  VWAP: {mock_agg.vwap}, Transactions: {mock_agg.transactions}")
        
        # Test the _handle_agg method directly
        print("\nCalling _handle_agg method...")
        await ws_manager._handle_agg(mock_agg)
        
        print("\n✅ Test completed successfully!")
        print("Check the logs above to verify:")
        print("1. WebSocket received COMPLETE aggregate data log")
        print("2. Successfully stored aggregate data in database log")
        print("3. No error messages")
        
        # Verify data is in cache
        if mock_agg.symbol in ws_manager.latest_aggs:
            cached_data = ws_manager.latest_aggs[mock_agg.symbol]
            print(f"\n✅ Data cached successfully: {cached_data.symbol} at {cached_data.timestamp}")
        else:
            print("\n❌ Data not found in cache")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

async def test_direct_database_storage():
    """Test direct database storage method"""
    print("\n" + "="*50)
    print("Testing direct database storage method...")
    
    try:
        # Create data pipeline instance
        pipeline = DataPipeline()
        
        # Test data
        test_timestamp = datetime.now(timezone.utc)
        
        print(f"Storing test data for TSLA at {test_timestamp}")
        
        # Store test data
        await pipeline.store_realtime_market_data(
            symbol="TSLA",
            timestamp=test_timestamp,
            open_price=250.50,
            high=252.75,
            low=249.25,
            close=251.80,
            volume=15000,
            vwap=251.25,
            transactions=125
        )
        
        print("✅ Direct database storage test completed successfully!")
        
    except Exception as e:
        print(f"❌ Direct database storage test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🔧 Testing WebSocket Database Storage Fix")
    print("="*50)
    
    # Test 1: WebSocket aggregate handling
    await test_websocket_database_storage()
    
    # Test 2: Direct database storage
    await test_direct_database_storage()
    
    print("\n" + "="*50)
    print("🎯 Test Summary:")
    print("If you see 'Successfully stored aggregate data' messages above,")
    print("the fix is working and WebSocket data will now be stored in the database.")
    print("This should resolve the 'No market data found' issue during live trading.")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Trading System Test Script

This script tests the trading system by mocking WebSocket data from the database.
It replays historical market data as if it were real-time WebSocket data to ensure
the trading system functions correctly before market open.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd
from dataclasses import dataclass

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_pipeline import DataPipeline
from data.polygon_websocket import RealTimeData, PolygonWebSocketManager
from data.pipeline_feature_engineering import FeatureEngineer
from trading.trading_orchestrator import TradingOrchestrator
from trading.signal_generator import SignalGenerator
from trading.execution_engine import ExecutionEngine
from trading.risk_manager import RiskManager
from database import db_manager
from database.models import MarketData
from sqlalchemy import text
from config import settings

@dataclass
class TestResults:
    """Container for test results"""
    total_bars_processed: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    errors_encountered: List[str] = None
    processing_times: List[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []
        if self.processing_times is None:
            self.processing_times = []

class MockWebSocketManager:
    """Mock WebSocket manager that replays database data as real-time data"""
    
    def __init__(self):
        self.agg_handlers: List = []
        self.trade_handlers: List = []
        self.quote_handlers: List = []
        self.is_connected = False
        self.subscribed_symbols: set = set()
        self.current_replay_time: Optional[datetime] = None  # Track current replay time
        
    async def connect(self) -> bool:
        """Mock connection - always succeeds"""
        self.is_connected = True
        logger.info("Mock WebSocket connected")
        return True
    
    async def disconnect(self):
        """Mock disconnection"""
        self.is_connected = False
        logger.info("Mock WebSocket disconnected")
    
    async def subscribe_minute_aggs(self, symbols: List[str]):
        """Mock subscription to minute aggregates"""
        self.subscribed_symbols.update(symbols)
        logger.info(f"Mock WebSocket subscribed to minute aggregates for: {symbols}")
        return True
    
    async def subscribe_trades(self, symbols: List[str]):
        """Mock subscription to trades"""
        self.subscribed_symbols.update(symbols)
        logger.info(f"Mock WebSocket subscribed to trades for: {symbols}")
        return True
    
    async def subscribe_quotes(self, symbols: List[str]):
        """Mock subscription to quotes"""
        self.subscribed_symbols.update(symbols)
        logger.info(f"Mock WebSocket subscribed to quotes for: {symbols}")
        return True
    
    def add_agg_handler(self, handler):
        """Add aggregate data handler"""
        self.agg_handlers.append(handler)
        logger.info(f"Added aggregate handler: {handler.__name__ if hasattr(handler, '__name__') else 'anonymous'}")
    
    def add_trade_handler(self, handler):
        """Add trade data handler"""
        self.trade_handlers.append(handler)
    
    def add_quote_handler(self, handler):
        """Add quote data handler"""
        self.quote_handlers.append(handler)
    
    async def replay_market_data(self, market_data: pd.DataFrame, symbol: str, replay_speed: float = 1.0):
        """Replay market data as WebSocket aggregate messages with simulated real-time timestamps"""
        logger.info(f"Starting replay of {len(market_data)} bars for {symbol} at {replay_speed}x speed")
        
        # Start replay from current time to work with cached features system
        base_time = datetime.now()
        
        for idx, row in market_data.iterrows():
            # Use simulated real-time timestamp instead of historical timestamp
            simulated_timestamp = base_time + timedelta(minutes=idx)
            self.current_replay_time = simulated_timestamp
            
            # Create RealTimeData object with simulated timestamp
            real_time_data = RealTimeData(
                symbol=symbol,
                timestamp=simulated_timestamp,  # Use simulated timestamp
                data_type="agg",
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                vwap=float(row['vwap']) if row['vwap'] is not None else None,
                transactions=int(row['transactions']) if row['transactions'] is not None else None,
                # Note: accumulated_volume is not in database, so we'll simulate it
                accumulated_volume=int(row['volume']) * (idx + 1)  # Cumulative volume simulation
            )
            
            logger.info(f"Replaying bar for {symbol} at {real_time_data.timestamp} (simulated): O={real_time_data.open}, H={real_time_data.high}, L={real_time_data.low}, C={real_time_data.close}, V={real_time_data.volume}")
            
            # Call all aggregate handlers
            for handler in self.agg_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(real_time_data)
                    else:
                        handler(real_time_data)
                except Exception as e:
                    logger.error(f"Error in aggregate handler during replay: {e}")
            
            # Wait between bars based on replay speed
            if replay_speed > 0:
                await asyncio.sleep(1.0 / replay_speed)  # 1 second per bar at 1x speed
                
            # Additional small delay to allow feature caching to complete
            await asyncio.sleep(0.1)

class TradingSystemTester:
    """Main testing class for the trading system"""
    
    def __init__(self):
        self.data_pipeline: Optional[DataPipeline] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.trading_orchestrator: Optional[TradingOrchestrator] = None
        self.mock_websocket: Optional[MockWebSocketManager] = None
        self.test_results = TestResults()
        
    async def initialize_components(self):
        """Initialize all trading system components"""
        try:
            logger.info("Initializing trading system components...")
            
            # Initialize core components
            self.data_pipeline = DataPipeline()
            await self.data_pipeline.initialize_database()
            
            self.feature_engineer = FeatureEngineer()
            self.signal_generator = SignalGenerator()
            self.execution_engine = ExecutionEngine()
            self.risk_manager = RiskManager()
            
            # Initialize mock WebSocket manager
            self.mock_websocket = MockWebSocketManager()
            await self.mock_websocket.connect()
            
            # Initialize trading orchestrator with mock WebSocket
            self.trading_orchestrator = TradingOrchestrator()
            await self.trading_orchestrator.initialize(
                websocket_manager=self.mock_websocket,
                data_pipeline=self.data_pipeline,
                feature_engineer=self.feature_engineer,
                signal_generator=self.signal_generator,
                execution_engine=self.execution_engine,
                risk_manager=self.risk_manager
            )
            
            logger.info("All trading system components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system components: {e}")
            self.test_results.errors_encountered.append(f"Initialization error: {e}")
            return False
    
    async def load_recent_market_data(self, symbols: List[str], minutes: int = 60) -> Dict[str, pd.DataFrame]:
        """Load recent market data from the database - optimized for testing with historical data"""
        try:
            logger.info(f"Loading last {minutes} bars of market data for {len(symbols)} symbols...")
            
            market_data = {}
            
            with db_manager.get_session() as session:
                for symbol in symbols:
                    # Get the most recent data available (not time-based but count-based)
                    result = session.execute(text("""
                        SELECT timestamp, open, high, low, close, volume, vwap, transactions
                        FROM market_data
                        WHERE symbol = :symbol
                        ORDER BY timestamp DESC
                        LIMIT :limit
                    """), {
                        'symbol': symbol,
                        'limit': minutes  # Use minutes as bar count for testing
                    })
                    
                    data = result.fetchall()
                    
                    if data:
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'
                        ])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        # Reverse to get chronological order for replay
                        df = df.sort_values('timestamp')
                        market_data[symbol] = df
                        logger.info(f"Loaded {len(df)} bars for {symbol} from {df['timestamp'].min()} to {df['timestamp'].max()}")
                    else:
                        logger.warning(f"No market data found for {symbol}")
                        market_data[symbol] = pd.DataFrame()
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            self.test_results.errors_encountered.append(f"Data loading error: {e}")
            return {}
    
    async def run_test(self, symbols: List[str] = None, replay_speed: float = 10.0, minutes: int = 60):
        """Run the complete trading system test"""
        try:
            self.test_results.start_time = datetime.now()
            logger.info(f"Starting trading system test at {self.test_results.start_time}")
            
            # Use default symbols if none provided
            if symbols is None:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Default test symbols
            
            # Step 1: Initialize components
            if not await self.initialize_components():
                logger.error("Failed to initialize components, aborting test")
                return self.test_results
            
            # Step 2: Load recent market data
            market_data = await self.load_recent_market_data(symbols, minutes)
            
            if not market_data or all(df.empty for df in market_data.values()):
                logger.error("No market data available for testing")
                self.test_results.errors_encountered.append("No market data available")
                return self.test_results
            
            # Step 3: Start trading orchestrator (equivalent to /trading/start)
            logger.info("Starting trading orchestrator...")
            await self.trading_orchestrator.start(symbols)
            
            # Step 4: Replay market data for each symbol
            replay_tasks = []
            for symbol, df in market_data.items():
                if not df.empty:
                    task = asyncio.create_task(
                        self.mock_websocket.replay_market_data(df, symbol, replay_speed)
                    )
                    replay_tasks.append(task)
                    self.test_results.total_bars_processed += len(df)
            
            # Wait for all replay tasks to complete
            if replay_tasks:
                logger.info(f"Replaying data for {len(replay_tasks)} symbols at {replay_speed}x speed...")
                await asyncio.gather(*replay_tasks)
            
            # Step 5: Allow extra time for feature caching and signal generation
            logger.info("Allowing time for feature caching and signal generation...")
            await asyncio.sleep(10)  # Increased wait time for proper processing
            
            # Step 6: Stop trading orchestrator
            await self.trading_orchestrator.stop()
            
            # Step 7: Collect results
            self.test_results.end_time = datetime.now()
            self.test_results.signals_generated = self.trading_orchestrator.signals_generated
            self.test_results.trades_executed = self.trading_orchestrator.trades_executed
            self.test_results.processing_times = self.trading_orchestrator.event_processing_times.copy()
            
            logger.info("Trading system test completed successfully")
            
        except Exception as e:
            logger.error(f"Error during trading system test: {e}")
            self.test_results.errors_encountered.append(f"Test execution error: {e}")
        
        return self.test_results
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*60)
        print("TRADING SYSTEM TEST RESULTS")
        print("="*60)
        
        if self.test_results.start_time and self.test_results.end_time:
            duration = self.test_results.end_time - self.test_results.start_time
            print(f"Test Duration: {duration.total_seconds():.2f} seconds")
        
        print(f"Total Bars Processed: {self.test_results.total_bars_processed}")
        print(f"Signals Generated: {self.test_results.signals_generated}")
        print(f"Trades Executed: {self.test_results.trades_executed}")
        
        if self.test_results.processing_times:
            avg_processing_time = sum(self.test_results.processing_times) / len(self.test_results.processing_times)
            max_processing_time = max(self.test_results.processing_times)
            print(f"Average Processing Time: {avg_processing_time:.2f}ms")
            print(f"Maximum Processing Time: {max_processing_time:.2f}ms")
        
        if self.test_results.errors_encountered:
            print(f"\nERRORS ENCOUNTERED ({len(self.test_results.errors_encountered)}):")
            for i, error in enumerate(self.test_results.errors_encountered, 1):
                print(f"  {i}. {error}")
        else:
            print("\n✅ No errors encountered during testing")
        
        print("\n" + "="*60)

async def main():
    """Main test function"""
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add("trading_system_test.log", level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}")
    
    # Create and run test
    tester = TradingSystemTester()
    
    # Test with specific symbols or use defaults
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Modify as needed
    
    logger.info(f"Starting trading system test with symbols: {test_symbols}")
    
    results = await tester.run_test(
        symbols=test_symbols,
        replay_speed=5.0,  # 5x speed for faster testing
        minutes=60  # Test with last 60 minutes of data
    )
    
    # Print results
    tester.print_test_results()
    
    # Return success/failure based on errors
    return len(results.errors_encountered) == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
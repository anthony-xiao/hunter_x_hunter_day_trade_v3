import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from loguru import logger
import json
from dataclasses import asdict

# Import existing system components
from trading.trading_orchestrator import TradingOrchestrator
from trading.execution_engine import ExecutionEngine, TradeSignal, Position
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from data.data_pipeline import DataPipeline
from data.pipeline_feature_engineering import FeatureEngineer
from ml.model_trainer import ModelTrainer
from database import db_manager
from config import settings
from data.polygon_websocket import RealTimeData

class MockTradingSystem:
    """
    Mock trading system for testing trading logic during off-market hours.
    Replays the last 3 days of minute-level data from the market_data table.
    """
    
    def __init__(self, data_pipeline=None, feature_engineer=None, signal_generator=None, risk_manager=None):
        self.db_manager = db_manager
        self.is_running = False
        self.current_time = None
        self.replay_data = {}
        self.mock_positions = {}
        self.mock_portfolio_value = 100000.0  # Start with $100k
        self.mock_cash = 100000.0
        self.trade_logs = []
        
        # Use provided components or initialize new ones
        self.data_pipeline = data_pipeline
        self.feature_engineer = feature_engineer
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.execution_engine = None
        
        logger.info("Mock Trading System initialized")
    
    async def initialize_components(self):
        """Initialize all trading system components"""
        try:
            logger.info("Initializing mock trading system components...")
            
            # Initialize components if not provided
            if self.signal_generator is None:
                logger.info("Creating SignalGenerator instance...")
                # Initialize ModelTrainer for universal model loading
                model_trainer = ModelTrainer(feature_count=50, create_model_dir=False)
                
                # Initialize universal training components
                symbols = ['TSLA', 'AAPL']  # Trading symbols
                model_trainer.initialize_universal_training(symbols)
                
                # Load universal models from the universal directory
                from pathlib import Path
                universal_dir = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal")
                await model_trainer.load_universal_models(universal_dir)
                
                supabase_client = self.db_manager.get_supabase_client()
                self.signal_generator = SignalGenerator(model_trainer=model_trainer, supabase_client=supabase_client, data_pipeline=self.data_pipeline)
                
                # Initialize models for the trading symbols
                symbols = ['TSLA', 'AAPL']
                await self.signal_generator.initialize_models(symbols)
                
            if self.risk_manager is None:
                logger.info("Creating RiskManager instance...")
                self.risk_manager = RiskManager()
                
            if self.data_pipeline is None:
                logger.info("Creating DataPipeline instance...")
                self.data_pipeline = DataPipeline()
                
            if self.feature_engineer is None:
                logger.info("Creating FeatureEngineer instance...")
                self.feature_engineer = FeatureEngineer()
            
            logger.info("✅ All mock trading system components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize mock trading system components: {e}")
            return False
    
    async def fetch_historical_data(self, days: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Fetch the last N days of minute-level data from market_data table
        
        Args:
            days: Number of days to fetch (default: 3)
            
        Returns:
            Dict mapping symbol to DataFrame with minute-level data
        """
        try:
            logger.info(f"Fetching last {days} days of historical data...")
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get Supabase client
            supabase = self.db_manager.get_supabase_client()
            
            # Fetch data from market_data table
            response = supabase.table('market_data').select('*').gte(
                'timestamp', start_date.isoformat()
            ).lte(
                'timestamp', end_date.isoformat()
            ).order('timestamp', desc=False).execute()
            
            if not response.data:
                logger.warning("No historical data found in market_data table")
                return {}
            
            # Convert to DataFrame and group by symbol
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Group by symbol
            historical_data = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.drop('symbol', axis=1)
                historical_data[symbol] = symbol_data
                
                logger.info(f"Loaded {len(symbol_data)} data points for {symbol}")
            
            logger.info(f"✅ Fetched historical data for {len(historical_data)} symbols")
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return {}
    
    def prepare_replay_timeline(self, historical_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Create a timeline of all unique timestamps for replay
        
        Args:
            historical_data: Dict mapping symbol to DataFrame
            
        Returns:
            Sorted list of unique timestamps
        """
        try:
            all_timestamps = set()
            
            for symbol, data in historical_data.items():
                all_timestamps.update(data.index.tolist())
            
            timeline = sorted(list(all_timestamps))
            logger.info(f"Created replay timeline with {len(timeline)} timestamps")
            
            return timeline
            
        except Exception as e:
            logger.error(f"Failed to prepare replay timeline: {e}")
            return []
    
    def get_data_at_timestamp(self, timestamp: datetime, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Get market data for all symbols up to a specific timestamp
        Ensures at least 100 data points are provided for signal generation and feature engineering
        
        Args:
            timestamp: Current replay timestamp
            historical_data: Full historical data
            
        Returns:
            Dict mapping symbol to DataFrame with data up to timestamp
        """
        try:
            current_data = {}
            min_required_points = 101  # Minimum data points for feature engineering (100 + 1 for rolling calculations) (correlation windows up to 100)
            
            for symbol, data in historical_data.items():
                # Get data up to current timestamp
                symbol_data = data[data.index <= timestamp].copy()
                
                # Ensure we have at least min_required_points for feature engineering
                if len(symbol_data) < min_required_points:
                    # If we don't have enough data up to current timestamp,
                    # take the last min_required_points from all available data
                    symbol_data = data.tail(min_required_points).copy()
                    logger.debug(f"Insufficient data up to {timestamp} for {symbol} ({len(data[data.index <= timestamp])} points). Using last {min_required_points} points from historical data.")
                
                if len(symbol_data) > 0:
                    # Convert market_data format to match polygon websocket RealTimeData format
                    converted_data = self._convert_market_data_to_realtime_format(symbol_data, symbol)
                    current_data[symbol] = converted_data
            
            return current_data
            
        except Exception as e:
            logger.error(f"Failed to get data at timestamp {timestamp}: {e}")
            return {}
    
    def _convert_market_data_to_realtime_format(self, market_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Convert market_data table format to match polygon websocket RealTimeData format
        
        Market_data fields: symbol, timestamp, open, high, low, close, volume, vwap, transactions
        RealTimeData fields: symbol, timestamp, volume, bid, ask, data_type, open, high, low, close, vwap, 
                           accumulated_volume, opening_price, average_trade_size, transactions
        
        Args:
            market_data: DataFrame with market_data table format
            symbol: Symbol name
            
        Returns:
            DataFrame with RealTimeData compatible format
        """
        try:
            converted_data = market_data.copy()
            
            # Add missing fields that exist in RealTimeData but not in market_data
            # Use reasonable approximations based on available data
            
            # Estimate bid/ask from close price (typical spread approximation)
            spread_pct = 0.001  # 0.1% spread approximation
            converted_data['bid'] = converted_data['close'] * (1 - spread_pct)
            converted_data['ask'] = converted_data['close'] * (1 + spread_pct)
            
            # Set data_type to indicate this is aggregate data (like minute bars)
            converted_data['data_type'] = 'agg'
            
            # Map accumulated_volume to volume (same concept)
            converted_data['accumulated_volume'] = converted_data['volume']
            
            # Map opening_price to open (same concept)
            converted_data['opening_price'] = converted_data['open']
            
            # Estimate average_trade_size from volume and transactions
            converted_data['average_trade_size'] = (
                converted_data['volume'] / converted_data['transactions']
            ).fillna(100).astype(int)  # Default to 100 if no transactions data
            
            # Ensure all numeric fields are properly typed
            numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 
                            'bid', 'ask', 'accumulated_volume', 'opening_price', 'average_trade_size']
            
            for field in numeric_fields:
                if field in converted_data.columns:
                    converted_data[field] = pd.to_numeric(converted_data[field], errors='coerce')
            
            logger.debug(f"Converted market_data to RealTimeData format for {symbol}: {len(converted_data)} rows")
            return converted_data
            
        except Exception as e:
            logger.error(f"Failed to convert market_data to RealTimeData format for {symbol}: {e}")
            return market_data
    
    async def mock_execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        Mock trade execution - logs the payload that would be sent to Alpaca
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Dict containing the mock trade execution details
        """
        try:
            # Calculate position size (this would normally be done by execution engine)
            position_size = await self._calculate_mock_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {signal.symbol}: {position_size}")
                return None
            
            # Create mock Alpaca payload
            alpaca_payload = {
                "symbol": signal.symbol,
                "qty": position_size,
                "side": "buy" if signal.action in ["buy"] else "sell",
                "type": "market",
                "time_in_force": "day",
                "order_class": "simple"
            }
            
            # Add stop loss and take profit if available
            if signal.stop_loss:
                alpaca_payload["stop_loss"] = {
                    "stop_price": str(signal.stop_loss)
                }
            
            if signal.take_profit:
                alpaca_payload["take_profit"] = {
                    "limit_price": str(signal.take_profit)
                }
            
            # Create trade log entry
            trade_log = {
                "timestamp": self.current_time.isoformat(),
                "signal": asdict(signal),
                "alpaca_payload": alpaca_payload,
                "position_size": position_size,
                "mock_execution": True,
                "portfolio_value": self.mock_portfolio_value,
                "cash_available": self.mock_cash
            }
            
            # Log the trade
            self.trade_logs.append(trade_log)
            
            logger.info(f"🔄 MOCK TRADE: {signal.action.upper()} {position_size} shares of {signal.symbol}")
            logger.info(f"📋 Alpaca Payload: {json.dumps(alpaca_payload, indent=2)}")
            
            # Update mock positions
            await self._update_mock_positions(signal, position_size)
            
            return trade_log
            
        except Exception as e:
            logger.error(f"Failed to mock execute trade for {signal.symbol}: {e}")
            return None
    
    async def _calculate_mock_position_size(self, signal: TradeSignal) -> float:
        """
        Calculate position size for mock trading
        
        Args:
            signal: Trading signal
            
        Returns:
            Position size in shares
        """
        try:
            # Use a simple position sizing: 2% of portfolio per trade
            position_value = self.mock_portfolio_value * 0.02
            
            # Assume current price (in real system this would come from market data)
            estimated_price = signal.price if signal.price else 100.0  # Default price
            
            position_size = position_value / estimated_price
            
            # Round to whole shares
            return int(position_size)
            
        except Exception as e:
            logger.error(f"Failed to calculate mock position size: {e}")
            return 0
    
    async def _update_mock_positions(self, signal: TradeSignal, position_size: float):
        """
        Update mock positions based on executed trade
        
        Args:
            signal: Trading signal
            position_size: Size of position
        """
        try:
            symbol = signal.symbol
            
            if signal.action == "buy":
                if symbol in self.mock_positions:
                    self.mock_positions[symbol] += position_size
                else:
                    self.mock_positions[symbol] = position_size
                    
                # Reduce cash
                estimated_price = signal.price if signal.price else 100.0
                self.mock_cash -= position_size * estimated_price
                
            elif signal.action == "sell":
                if symbol in self.mock_positions:
                    self.mock_positions[symbol] -= position_size
                    if self.mock_positions[symbol] <= 0:
                        del self.mock_positions[symbol]
                        
                # Add cash
                estimated_price = signal.price if signal.price else 100.0
                self.mock_cash += position_size * estimated_price
            
            logger.debug(f"Updated mock positions: {self.mock_positions}")
            
        except Exception as e:
            logger.error(f"Failed to update mock positions: {e}")
    
    async def replay_trading_session(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Replay a complete trading session using historical data
        
        Args:
            historical_data: Historical market data
            
        Returns:
            Dict containing session results
        """
        try:
            logger.info("🚀 Starting mock trading session replay...")
            
            # Prepare timeline
            timeline = self.prepare_replay_timeline(historical_data)
            
            if not timeline:
                logger.error("No timeline data available for replay")
                return {"error": "No timeline data"}
            
            self.is_running = True
            session_start = datetime.now(timezone.utc)
            signals_generated = 0
            trades_executed = 0
            
            # Replay each minute
            for i, timestamp in enumerate(timeline):
                if not self.is_running:
                    break
                
                self.current_time = timestamp
                
                # Get data up to current timestamp
                current_data = self.get_data_at_timestamp(timestamp, historical_data)
                
                if not current_data:
                    continue
                
                logger.info(f"📅 Replaying {timestamp} ({i+1}/{len(timeline)})")
                
                # Log the mock websocket minute-level data format for verification
                # Mimicking the comprehensive logging format from polygon_websocket.py
                for symbol, data in current_data.items():
                    if len(data) > 0:
                        latest_row = data.iloc[-1]  # Get the most recent data point
                        
                        # Create a mock raw payload representation with clean Python types
                        def clean_value(val):
                            """Convert numpy/pandas types to clean Python types"""
                            if val is None:
                                return None
                            elif hasattr(val, 'item'):  # numpy scalar
                                return val.item()
                            elif hasattr(val, 'tolist'):  # numpy array
                                return val.tolist()
                            else:
                                return val
                        
                        all_fields = {
                            'symbol': symbol,
                            'timestamp': timestamp.isoformat(),
                            'open': clean_value(latest_row.get('open')),
                            'high': clean_value(latest_row.get('high')),
                            'low': clean_value(latest_row.get('low')),
                            'close': clean_value(latest_row.get('close')),
                            'volume': clean_value(latest_row.get('volume')),
                            'vwap': clean_value(latest_row.get('vwap')),
                            'accumulated_volume': clean_value(latest_row.get('accumulated_volume')),
                            'opening_price': clean_value(latest_row.get('opening_price')),
                            'average_trade_size': clean_value(latest_row.get('average_trade_size')),
                            'transactions': clean_value(latest_row.get('transactions')),
                            'bid': clean_value(latest_row.get('bid')),
                            'ask': clean_value(latest_row.get('ask')),
                            'data_type': clean_value(latest_row.get('data_type'))
                        }
                        
                        # Comprehensive logging with all available fields (matching polygon_websocket.py format)
                        logger.info(f"WebSocket received COMPLETE aggregate data for {symbol}:")
                        logger.info(f"  Basic OHLCV: O={latest_row.get('open')}, H={latest_row.get('high')}, L={latest_row.get('low')}, C={latest_row.get('close')}, V={latest_row.get('volume')}")
                        logger.info(f"  Extended fields: VWAP={latest_row.get('vwap')}, AccumVol={latest_row.get('accumulated_volume')}, OpenPrice={latest_row.get('opening_price')}")
                        logger.info(f"  Trade metrics: AvgTradeSize={latest_row.get('average_trade_size')}, Transactions={latest_row.get('transactions')}")
                        logger.info(f"  Timestamp: {timestamp}")
                        logger.info(f"  ALL RAW FIELDS from MockEquityAgg: {all_fields}")
                
                try:
                    # Generate signals using existing signal generator
                    signals = await self.signal_generator.generate_signals(current_data)
                    
                    for signal in signals:
                        signals_generated += 1
                        
                        # Apply risk management
                        risk_approved, risk_reason = await self.risk_manager.check_risk_limits(
                            signal=signal,
                            positions=list(self.mock_positions.keys()),
                            portfolio_value=self.mock_portfolio_value
                        )
                        
                        if risk_approved:
                            # Execute mock trade
                            trade_result = await self.mock_execute_trade(signal)
                            if trade_result:
                                trades_executed += 1
                        else:
                            logger.info(f"❌ Trade rejected by risk management: {risk_reason}")
                
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {e}")
                    continue
                
                # Add delay to slow down replay speed for better observation
                await asyncio.sleep(10)
            
            session_end = datetime.now(timezone.utc)
            session_duration = (session_end - session_start).total_seconds()
            
            # Compile session results
            session_results = {
                "session_start": session_start.isoformat(),
                "session_end": session_end.isoformat(),
                "session_duration_seconds": session_duration,
                "timeline_points": len(timeline),
                "signals_generated": signals_generated,
                "trades_executed": trades_executed,
                "final_portfolio_value": self.mock_portfolio_value,
                "final_cash": self.mock_cash,
                "final_positions": self.mock_positions,
                "trade_logs": self.trade_logs,
                "success": True
            }
            
            logger.info(f"✅ Mock trading session completed!")
            logger.info(f"📊 Generated {signals_generated} signals, executed {trades_executed} trades")
            logger.info(f"💰 Final portfolio value: ${self.mock_portfolio_value:,.2f}")
            
            return session_results
            
        except Exception as e:
            logger.error(f"Failed to replay trading session: {e}")
            return {"error": str(e), "success": False}
    
    async def start_mock_trading(self) -> Dict[str, Any]:
        """
        Main entry point to start mock trading system
        
        Returns:
            Dict containing the results of the mock trading session
        """
        try:
            logger.info("🎯 Starting Mock Trading System...")
            
            # Initialize components
            if not await self.initialize_components():
                return {"error": "Failed to initialize components", "success": False}
            
            # Fetch historical data
            historical_data = await self.fetch_historical_data(days=5)
            
            if not historical_data:
                return {"error": "No historical data available", "success": False}
            
            # Start replay session
            results = await self.replay_trading_session(historical_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Mock trading system failed: {e}")
            return {"error": str(e), "success": False}
    
    def stop_mock_trading(self):
        """Stop the mock trading system"""
        self.is_running = False
        logger.info("🛑 Mock trading system stopped")
    
    def get_trade_logs(self) -> List[Dict[str, Any]]:
        """Get all trade logs from the session"""
        return self.trade_logs
    
    def reset_mock_system(self):
        """Reset the mock system to initial state"""
        self.mock_positions = {}
        self.mock_portfolio_value = 100000.0
        self.mock_cash = 100000.0
        self.trade_logs = []
        self.current_time = None
        logger.info("🔄 Mock trading system reset")

# Global instance for easy access
mock_trading_system = MockTradingSystem()

# Convenience function for external use
async def start_mock_trading_session() -> Dict[str, Any]:
    """
    Convenience function to start a mock trading session
    
    Returns:
        Dict containing session results
    """
    return await mock_trading_system.start_mock_trading()

def stop_mock_trading_session():
    """
    Convenience function to stop mock trading session
    """
    mock_trading_system.stop_mock_trading()

def get_mock_trade_logs() -> List[Dict[str, Any]]:
    """
    Convenience function to get trade logs
    
    Returns:
        List of trade log dictionaries
    """
    return mock_trading_system.get_trade_logs()

def reset_mock_system():
    """
    Convenience function to reset mock system
    """
    mock_trading_system.reset_mock_system()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test the mock trading system"""
        try:
            logger.info("🚀 Testing Mock Trading System...")
            results = await start_mock_trading_session()
            
            if results.get("success"):
                logger.info("✅ Mock trading test completed successfully!")
                logger.info(f"📊 Results: {results}")
            else:
                logger.error(f"❌ Mock trading test failed: {results.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Run the test
    asyncio.run(main())
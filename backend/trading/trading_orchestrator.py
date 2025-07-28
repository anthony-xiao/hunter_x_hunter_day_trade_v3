import asyncio
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import pandas as pd

from data.polygon_websocket import RealTimeData, PolygonWebSocketManager
from data.data_pipeline import DataPipeline
from data.pipeline_feature_engineering import FeatureEngineer
from trading.signal_generator import SignalGenerator
from trading.execution_engine import ExecutionEngine, TradeSignal
from trading.risk_manager import RiskManager
from config import settings

@dataclass
class MinuteBarEvent:
    """Event triggered when a minute bar completes"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bar_completion_time: datetime

class TradingOrchestrator:
    """
    Event-driven trading orchestrator that listens for minute aggregate completions
    and triggers immediate feature updates, signal generation, and trade execution.
    """
    
    def __init__(self):
        # Core components
        self.websocket_manager: Optional[PolygonWebSocketManager] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Event tracking
        self.active_symbols: Set[str] = set()
        self.last_bar_timestamps: Dict[str, datetime] = {}
        self.processing_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance tracking
        self.event_processing_times: List[float] = []
        self.signals_generated: int = 0
        self.trades_executed: int = 0
        
        # Configuration
        self.max_processing_time_ms = 500  # Maximum time to process a bar event
        self.enable_event_driven = True
        self.enable_polling_backup = False 
        self.polling_interval = 30  # seconds
        
        # State management
        self.is_running = False
        self.orchestrator_task: Optional[asyncio.Task] = None
        self.polling_task: Optional[asyncio.Task] = None
        
    async def initialize(self, 
                        websocket_manager: PolygonWebSocketManager,
                        data_pipeline: DataPipeline,
                        feature_engineer: FeatureEngineer,
                        signal_generator: SignalGenerator,
                        execution_engine: ExecutionEngine,
                        risk_manager: RiskManager):
        """Initialize the orchestrator with trading components"""
        try:
            self.websocket_manager = websocket_manager
            self.data_pipeline = data_pipeline
            self.feature_engineer = feature_engineer
            self.signal_generator = signal_generator
            self.execution_engine = execution_engine
            self.risk_manager = risk_manager
            
            # Register minute aggregate handler
            self.websocket_manager.add_agg_handler(self._on_minute_aggregate)
            
            logger.info("Trading orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading orchestrator: {e}")
            return False
    
    async def start(self, trading_symbols: List[str]):
        """Start the event-driven trading orchestrator"""
        try:
            if self.is_running:
                logger.warning("Trading orchestrator is already running")
                return
            
            self.active_symbols = set(trading_symbols)
            self.is_running = True
            
            # Initialize processing locks for each symbol
            for symbol in trading_symbols:
                self.processing_locks[symbol] = asyncio.Lock()
            
            # Start WebSocket data streaming
            if self.websocket_manager:
                await self.websocket_manager.subscribe_minute_aggs(trading_symbols)
                logger.info(f"Subscribed to minute aggregates for {len(trading_symbols)} symbols")
            
            # Start polling backup if enabled
            if self.enable_polling_backup:
                self.polling_task = asyncio.create_task(self._polling_backup_loop())
                logger.info("Started polling backup system")
            
            logger.info(f"Event-driven trading orchestrator started for {len(trading_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to start trading orchestrator: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the trading orchestrator"""
        try:
            self.is_running = False
            
            # Cancel polling task
            if self.polling_task and not self.polling_task.done():
                self.polling_task.cancel()
                try:
                    await self.polling_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Trading orchestrator stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading orchestrator: {e}")
    
    async def _on_minute_aggregate(self, agg_data: RealTimeData):
        """Handle minute aggregate completion event"""
        logger.info(f"WebSocket aggregate handler called for {agg_data.symbol} at {agg_data.timestamp} (running: {self.is_running}, event_driven: {self.enable_event_driven})")
        
        if not self.is_running or not self.enable_event_driven:
            logger.warning(f"Skipping aggregate processing for {agg_data.symbol} - orchestrator not running or event-driven disabled")
            return
        
        symbol = agg_data.symbol
        
        # Check if this is a new minute bar
        current_minute = agg_data.timestamp.replace(second=0, microsecond=0)
        last_minute = self.last_bar_timestamps.get(symbol)
        
        if last_minute and current_minute <= last_minute:
            logger.debug(f"Skipping duplicate minute bar for {symbol}: {current_minute} <= {last_minute}")
            return  # Not a new bar, skip processing
        
        # Update last bar timestamp
        self.last_bar_timestamps[symbol] = current_minute
        logger.info(f"Processing new WebSocket minute bar for {symbol} at {current_minute}")
        
        # Process the bar event asynchronously to avoid blocking WebSocket
        asyncio.create_task(self._process_minute_bar_event(symbol, agg_data))
    
    async def _process_minute_bar_event(self, symbol: str, agg_data: RealTimeData):
        """Process a minute bar completion event"""
        start_time = time.time()
        
        # Use lock to prevent concurrent processing for the same symbol
        async with self.processing_locks.get(symbol, asyncio.Lock()):
            try:
                logger.debug(f"Processing minute bar event for {symbol} at {agg_data.timestamp}")
                
                # Step 1: Update features with new bar data
                await self._update_features_for_symbol(symbol, agg_data)
                
                # Step 2: Generate trading signal
                signal = await self._generate_signal_for_symbol(symbol)
                
                # Step 3: Execute trade if signal is valid
                if signal:
                    await self._execute_signal_with_risk_management(signal)
                
                # Track processing time
                processing_time_ms = (time.time() - start_time) * 1000
                self.event_processing_times.append(processing_time_ms)
                
                # Keep only last 1000 processing times for memory efficiency
                if len(self.event_processing_times) > 1000:
                    self.event_processing_times = self.event_processing_times[-1000:]
                
                # Log performance warning if processing is slow
                if processing_time_ms > self.max_processing_time_ms:
                    logger.warning(f"Slow event processing for {symbol}: {processing_time_ms:.1f}ms")
                else:
                    logger.debug(f"Processed {symbol} bar event in {processing_time_ms:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error processing minute bar event for {symbol}: {e}")
    
    async def _update_features_for_symbol(self, symbol: str, agg_data: RealTimeData):
        """Update features for a symbol with new minute bar data using incremental approach"""
        try:
            if not self.feature_engineer or not self.data_pipeline:
                return
            
            current_timestamp = agg_data.timestamp
            
            # Check if we have exact cached features for this timestamp
            cached_features = await self.data_pipeline.get_cached_features(symbol, current_timestamp)
            
            if cached_features:
                # Features already exist in cache, no need to recalculate
                logger.debug(f"[{symbol}] Using cached features for {current_timestamp}")
                return
            
            # Check if we have recent features to determine if we need full historical data
            recent_features = await self.data_pipeline.get_recent_cached_features(symbol, minutes=60)
            
            if recent_features and len(recent_features) >= 5:
                # We have recent features, create a minimal DataFrame with just the new bar
                # and calculate features incrementally
                logger.debug(f"[{symbol}] Performing incremental feature update with new bar data")
                
                # Create a minimal DataFrame with just the current bar using complete OHLCV data
                import pandas as pd
                current_bar_df = pd.DataFrame([{
                    'timestamp': current_timestamp,
                    'open': agg_data.open if agg_data.open is not None else agg_data.close,
                    'high': agg_data.high if agg_data.high is not None else agg_data.close,
                    'low': agg_data.low if agg_data.low is not None else agg_data.close,
                    'close': agg_data.close if agg_data.close is not None else agg_data.close,
                    'volume': agg_data.volume if agg_data.volume is not None else 1000,
                    'vwap': agg_data.vwap if agg_data.vwap is not None else agg_data.close,
                    'transactions': agg_data.transactions if agg_data.transactions is not None else 1
                }])
                current_bar_df.set_index('timestamp', inplace=True)
                
                # Calculate basic features for just this bar using complete OHLCV data
                features_dict = {
                    'open': float(agg_data.open if agg_data.open is not None else agg_data.close),
                    'high': float(agg_data.high if agg_data.high is not None else agg_data.close),
                    'low': float(agg_data.low if agg_data.low is not None else agg_data.close),
                    'close': float(agg_data.close if agg_data.close is not None else agg_data.close),
                    'volume': float(agg_data.volume if agg_data.volume is not None else 1000),
                    'vwap': float(agg_data.vwap if agg_data.vwap is not None else agg_data.close),
                    'transactions': float(agg_data.transactions if agg_data.transactions is not None else 1),
                    'timestamp_hour': current_timestamp.hour,
                    'timestamp_minute': current_timestamp.minute,
                    'timestamp_weekday': current_timestamp.weekday()
                }
                
                # Store only the current timestamp's features
                await self.data_pipeline.store_features(symbol, current_timestamp, features_dict)
                logger.debug(f"[{symbol}] Incremental features calculated and cached for {current_timestamp}")
                
            else:
                # No recent features, need to download minimal historical data for initial feature calculation
                logger.info(f"[{symbol}] No recent features found, downloading minimal historical data for initial calculation")
                end_date = current_timestamp
                start_date = end_date - timedelta(hours=4)  # Minimal 4-hour window for initial features
                
                historical_data = await self.data_pipeline.download_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if historical_data is not None and len(historical_data) >= 10:  # Reduced minimum requirement
                    # Calculate features with the minimal historical data
                    features = await self.feature_engineer.engineer_features(historical_data, symbol)
                    
                    # Only store features for the current timestamp to avoid bulk storage
                    if len(features) > 0:
                        # Find the row closest to current_timestamp
                        closest_idx = features.index.get_indexer([current_timestamp], method='nearest')[0]
                        if closest_idx >= 0 and closest_idx < len(features):
                            latest_features = features.iloc[closest_idx].to_dict()
                            await self.data_pipeline.store_features(symbol, current_timestamp, latest_features)
                            logger.debug(f"[{symbol}] Initial features calculated and cached for {current_timestamp}")
                
        except Exception as e:
            logger.error(f"Error updating features for {symbol}: {e}")
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trading signal for a symbol using cached features when possible"""
        try:
            if not self.signal_generator or not self.data_pipeline:
                return None
            
            # First try to use recent cached features for signal generation
            recent_features = await self.data_pipeline.get_recent_cached_features(symbol, minutes=30)
            
            market_data = None
            
            if recent_features and len(recent_features) >= 10:  # If we have enough recent features
                logger.debug(f"[{symbol}] Using cached features for signal generation")
                
                # Convert cached features to DataFrame format expected by signal generator
                try:
                    # Sort features by timestamp
                    sorted_timestamps = sorted(recent_features.keys())
                    
                    # Create DataFrame from cached features
                    data_rows = []
                    for timestamp in sorted_timestamps:
                        feature_dict = recent_features[timestamp].copy()
                        feature_dict['timestamp'] = timestamp
                        data_rows.append(feature_dict)
                    
                    if data_rows:
                        df = pd.DataFrame(data_rows)
                        df.set_index('timestamp', inplace=True)
                        
                        # Ensure required columns exist with fallback values
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in required_cols:
                            if col not in df.columns:
                                if col == 'volume':
                                    df[col] = 1000  # Default volume
                                else:
                                    df[col] = df.get('close', 100.0)  # Use close price as fallback
                        
                        market_data = df
                        logger.debug(f"[{symbol}] Converted {len(df)} cached features to DataFrame")
                    
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to convert cached features to DataFrame: {e}")
                    market_data = None
            
            if market_data is None or len(market_data) < 10:
                # Fallback: get minimal market data (reduced from 30 days to 5 days)
                logger.debug(f"[{symbol}] Insufficient cached features, downloading minimal market data")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)  # Reduced from 30 days
                
                market_data = await self.data_pipeline.download_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if market_data is not None and len(market_data) >= 10:  # Reduced minimum requirement
                # Generate signal for this symbol
                signals = await self.signal_generator.generate_signals({symbol: market_data})
                
                if signals and len(signals) > 0:
                    self.signals_generated += 1
                    logger.info(f"[{symbol}] Generated signal: {signals[0].action} with confidence {signals[0].confidence:.3f}")
                    return signals[0]  # Return the first (and likely only) signal
                else:
                    logger.debug(f"[{symbol}] No signals generated from market data")
            else:
                logger.warning(f"[{symbol}] Insufficient market data for signal generation")
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _execute_signal_with_risk_management(self, signal: TradeSignal):
        """Execute signal with risk management using cached data when possible"""
        try:
            if not self.execution_engine or not self.risk_manager:
                return
            
            # Try to use recent cached features for risk management calculations
            recent_features = await self.data_pipeline.get_recent_cached_features(signal.symbol, minutes=60)
            
            market_data = None
            if recent_features and len(recent_features) >= 5:  # If we have enough recent data
                logger.debug(f"[{signal.symbol}] Using cached data for risk management")
                # For now, still need market data for risk manager, but reduced timeframe
                market_data = await self.data_pipeline.download_historical_data(
                    symbol=signal.symbol,
                    start_date=datetime.now() - timedelta(days=3),  # Reduced from 30 days to 3 days
                    end_date=datetime.now()
                )
            else:
                # Fallback: get minimal market data for risk management
                logger.debug(f"[{signal.symbol}] Insufficient cached data, downloading minimal market data for risk management")
                market_data = await self.data_pipeline.download_historical_data(
                    symbol=signal.symbol,
                    start_date=datetime.now() - timedelta(days=7),  # Reduced from 30 days to 7 days
                    end_date=datetime.now()
                )
            
            if market_data is not None and len(market_data) >= 10:  # Reduced minimum requirement
                # Calculate position size with risk management
                position_size = await self.risk_manager.calculate_position_size(
                    signal=signal,
                    market_data=market_data
                )
                
                if position_size > 0:
                    # Execute the trade
                    success = await self.execution_engine.execute_signal(
                        signal=signal,
                        position_size=position_size
                    )
                    
                    if success:
                        self.trades_executed += 1
                        logger.info(f"Event-driven trade executed: {signal.symbol} {signal.action}")
                    else:
                        logger.warning(f"Failed to execute event-driven trade for {signal.symbol}")
                else:
                    logger.debug(f"Signal for {signal.symbol} rejected by risk management")
            else:
                logger.warning(f"Insufficient market data for risk management: {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _polling_backup_loop(self):
        """Backup polling system that runs alongside event-driven processing"""
        logger.info("Starting polling backup loop")
        
        while self.is_running:
            try:
                # Check if market is open
                now = datetime.now()
                
                if (now.weekday() < 5 and  # Monday = 0, Friday = 4
                    9 <= now.hour < 16 and
                    not (now.hour == 9 and now.minute < 30)):
                    
                    # Process symbols that haven't been updated recently via events
                    await self._process_stale_symbols()
                
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in polling backup loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_stale_symbols(self):
        """Process symbols that haven't received recent minute bar events"""
        try:
            current_time = datetime.now()
            stale_threshold = timedelta(minutes=2)  # Consider stale if no update in 2 minutes
            
            for symbol in self.active_symbols:
                last_update = self.last_bar_timestamps.get(symbol)
                
                if not last_update or (current_time - last_update) > stale_threshold:
                    logger.debug(f"Processing stale symbol via polling backup: {symbol}")
                    
                    # Create a synthetic aggregate event for polling backup
                    current_price = None
                    if self.websocket_manager:
                        current_price = self.websocket_manager.get_latest_price(symbol)
                    
                    if current_price:
                        synthetic_agg = RealTimeData(
                            symbol=symbol,
                            timestamp=current_time,
                            price=current_price,
                            data_type="agg"
                        )
                        
                        # Process as if it were a minute bar event
                        await self._process_minute_bar_event(symbol, synthetic_agg)
                        
        except Exception as e:
            logger.error(f"Error processing stale symbols: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the orchestrator"""
        avg_processing_time = 0
        if self.event_processing_times:
            avg_processing_time = sum(self.event_processing_times) / len(self.event_processing_times)
        
        return {
            "is_running": self.is_running,
            "active_symbols_count": len(self.active_symbols),
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "max_processing_time_ms": self.max_processing_time_ms,
            "event_driven_enabled": self.enable_event_driven,
            "polling_backup_enabled": self.enable_polling_backup,
            "recent_processing_times": self.event_processing_times[-10:] if self.event_processing_times else []
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.event_processing_times.clear()
        self.signals_generated = 0
        self.trades_executed = 0
        logger.info("Trading orchestrator statistics reset")

# Global orchestrator instance
orchestrator = TradingOrchestrator()

# Convenience functions
async def start_event_driven_trading(trading_symbols: List[str],
                                    websocket_manager: PolygonWebSocketManager,
                                    data_pipeline: DataPipeline,
                                    feature_engineer: FeatureEngineer,
                                    signal_generator: SignalGenerator,
                                    execution_engine: ExecutionEngine,
                                    risk_manager: RiskManager) -> bool:
    """Start event-driven trading with all components"""
    try:
        # Initialize orchestrator
        success = await orchestrator.initialize(
            websocket_manager=websocket_manager,
            data_pipeline=data_pipeline,
            feature_engineer=feature_engineer,
            signal_generator=signal_generator,
            execution_engine=execution_engine,
            risk_manager=risk_manager
        )
        
        if not success:
            return False
        
        # Start orchestrator
        await orchestrator.start(trading_symbols)
        
        logger.info("Event-driven trading system started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start event-driven trading: {e}")
        return False

async def stop_event_driven_trading():
    """Stop event-driven trading"""
    await orchestrator.stop()
    logger.info("Event-driven trading system stopped")

def get_orchestrator_stats() -> Dict:
    """Get orchestrator performance statistics"""
    return orchestrator.get_performance_stats()
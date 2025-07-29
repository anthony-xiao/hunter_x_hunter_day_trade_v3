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
        """Enhanced real-time feature updates using complete OHLCV + VWAP + transactions data (Priority 2)"""
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
            
            # Get recent cached features for rolling window calculations (Priority 2: Enhanced Real-time Feature Updates)
            recent_features = await self.data_pipeline.get_recent_cached_features(symbol, minutes=120)  # Extended lookback for better technical indicators
            
            if recent_features and len(recent_features) >= 20:  # Increased minimum for proper technical indicator calculation
                # Priority 2: Calculate ALL technical indicators from WebSocket data using rolling windows
                logger.debug(f"[{symbol}] Performing enhanced real-time feature update with complete OHLCV+VWAP+transactions data")
                
                # Convert recent cached features to DataFrame for rolling window calculations
                import pandas as pd
                
                # Sort features by timestamp and create DataFrame
                sorted_timestamps = sorted(recent_features.keys())
                data_rows = []
                
                for timestamp in sorted_timestamps:
                    feature_dict = recent_features[timestamp].copy()
                    feature_dict['timestamp'] = timestamp
                    data_rows.append(feature_dict)
                
                # Add current bar with complete Polygon WebSocket fields (Priority 3: Model Input Optimization)
                current_bar = {
                    'timestamp': current_timestamp,
                    'open': float(agg_data.open if agg_data.open is not None else agg_data.close),
                    'high': float(agg_data.high if agg_data.high is not None else agg_data.close),
                    'low': float(agg_data.low if agg_data.low is not None else agg_data.close),
                    'close': float(agg_data.close if agg_data.close is not None else agg_data.close),
                    'volume': float(agg_data.volume if agg_data.volume is not None else 1000),
                    'vwap': float(agg_data.vwap if agg_data.vwap is not None else agg_data.close),
                    'transactions': float(agg_data.transactions if agg_data.transactions is not None else 1),
                    # Add accumulated_volume if available from WebSocket
                    'accumulated_volume': float(getattr(agg_data, 'accumulated_volume', agg_data.volume) if agg_data.volume is not None else 1000)
                }
                data_rows.append(current_bar)
                
                # Create DataFrame for feature engineering
                rolling_df = pd.DataFrame(data_rows)
                rolling_df.set_index('timestamp', inplace=True)
                rolling_df.sort_index(inplace=True)
                
                # Ensure required OHLCV columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in rolling_df.columns:
                        if col == 'volume':
                            rolling_df[col] = 1000
                        else:
                            rolling_df[col] = rolling_df.get('close', 100.0)
                
                # Priority 2 & 3: Use engineer_features function to calculate ALL technical indicators
                # This leverages complete OHLCV + VWAP + transactions data with rolling window calculations
                logger.debug(f"[{symbol}] Calculating comprehensive features using engineer_features with {len(rolling_df)} data points")
                
                # Use the existing engineer_features function for comprehensive feature calculation
                engineered_features = await self.feature_engineer.engineer_features(rolling_df, symbol)
                
                if engineered_features is not None and len(engineered_features) > 0:
                    # Extract features for the current timestamp only
                    if current_timestamp in engineered_features.index:
                        current_features = engineered_features.loc[current_timestamp].to_dict()
                        
                        # Ensure all Polygon WebSocket fields are included (Priority 3: Model Input Optimization)
                        current_features.update({
                            'open': current_bar['open'],
                            'high': current_bar['high'],
                            'low': current_bar['low'],
                            'close': current_bar['close'],
                            'volume': current_bar['volume'],
                            'vwap': current_bar['vwap'],
                            'transactions': current_bar['transactions'],
                            'accumulated_volume': current_bar['accumulated_volume']
                        })
                        
                        # Store comprehensive features
                        await self.data_pipeline.store_features(symbol, current_timestamp, current_features)
                        logger.debug(f"[{symbol}] Enhanced real-time features calculated and cached for {current_timestamp} ({len(current_features)} features)")
                    else:
                        # Fallback: use the last available features
                        latest_features = engineered_features.iloc[-1].to_dict()
                        latest_features.update(current_bar)  # Ensure current bar data is included
                        await self.data_pipeline.store_features(symbol, current_timestamp, latest_features)
                        logger.debug(f"[{symbol}] Fallback features stored for {current_timestamp}")
                else:
                    logger.warning(f"[{symbol}] engineer_features returned no results, storing basic features")
                    await self.data_pipeline.store_features(symbol, current_timestamp, current_bar)
                
            else:
                # Cold start: insufficient recent features, use engineer_features with minimal historical data
                logger.info(f"[{symbol}] Cold start: insufficient recent features ({len(recent_features) if recent_features else 0}), using minimal historical data")
                
                # Download minimal historical data for initial comprehensive feature calculation
                end_date = current_timestamp
                start_date = end_date - timedelta(hours=6)  # Extended window for better technical indicators
                
                historical_data = await self.data_pipeline.download_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if historical_data is not None and len(historical_data) >= 20:
                    # Add current bar to historical data
                    import pandas as pd
                    current_bar_df = pd.DataFrame([{
                        'open': float(agg_data.open if agg_data.open is not None else agg_data.close),
                        'high': float(agg_data.high if agg_data.high is not None else agg_data.close),
                        'low': float(agg_data.low if agg_data.low is not None else agg_data.close),
                        'close': float(agg_data.close if agg_data.close is not None else agg_data.close),
                        'volume': float(agg_data.volume if agg_data.volume is not None else 1000),
                        'vwap': float(agg_data.vwap if agg_data.vwap is not None else agg_data.close),
                        'transactions': float(agg_data.transactions if agg_data.transactions is not None else 1),
                        'accumulated_volume': float(getattr(agg_data, 'accumulated_volume', agg_data.volume) if agg_data.volume is not None else 1000)
                    }], index=[current_timestamp])
                    
                    # Combine historical data with current bar
                    combined_data = pd.concat([historical_data, current_bar_df])
                    combined_data.sort_index(inplace=True)
                    
                    # Use engineer_features for comprehensive feature calculation
                    features = await self.feature_engineer.engineer_features(combined_data, symbol)
                    
                    if features is not None and len(features) > 0:
                        # Store only features for the current timestamp
                        if current_timestamp in features.index:
                            latest_features = features.loc[current_timestamp].to_dict()
                        else:
                            latest_features = features.iloc[-1].to_dict()
                        
                        await self.data_pipeline.store_features(symbol, current_timestamp, latest_features)
                        logger.debug(f"[{symbol}] Cold start comprehensive features calculated and cached for {current_timestamp}")
                else:
                    # Absolute fallback: store basic WebSocket data
                    basic_features = {
                        'open': float(agg_data.open if agg_data.open is not None else agg_data.close),
                        'high': float(agg_data.high if agg_data.high is not None else agg_data.close),
                        'low': float(agg_data.low if agg_data.low is not None else agg_data.close),
                        'close': float(agg_data.close if agg_data.close is not None else agg_data.close),
                        'volume': float(agg_data.volume if agg_data.volume is not None else 1000),
                        'vwap': float(agg_data.vwap if agg_data.vwap is not None else agg_data.close),
                        'transactions': float(agg_data.transactions if agg_data.transactions is not None else 1),
                        'accumulated_volume': float(getattr(agg_data, 'accumulated_volume', agg_data.volume) if agg_data.volume is not None else 1000),
                        'timestamp_hour': current_timestamp.hour,
                        'timestamp_minute': current_timestamp.minute,
                        'timestamp_weekday': current_timestamp.weekday()
                    }
                    await self.data_pipeline.store_features(symbol, current_timestamp, basic_features)
                    logger.debug(f"[{symbol}] Basic WebSocket features stored for {current_timestamp}")
                
        except Exception as e:
            logger.error(f"Error updating enhanced real-time features for {symbol}: {e}")
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trading signal for a symbol using cached features when possible"""
        try:
            if not self.signal_generator or not self.data_pipeline:
                return None
            
            # Try to get recent cached features first (Priority 1: Optimize Feature Engineering Pipeline)
            recent_features = await self.data_pipeline.get_recent_cached_features(symbol, minutes=60)
            
            # Check if we have sufficient cached features (at least 60 data points for proper lookback)
            if recent_features and len(recent_features) >= 60:
                logger.info(f"Using cached features for signal generation: {symbol} ({len(recent_features)} points)")
                
                # Generate signal directly from cached features (eliminates historical data download)
                signals = await self.signal_generator.generate_signals_from_features({symbol: recent_features})
                
                if signals and len(signals) > 0:
                    self.signals_generated += 1
                    logger.info(f"[{symbol}] Generated signal from cached features: {signals[0].action} with confidence {signals[0].confidence:.3f}")
                    return signals[0]
                else:
                    logger.warning(f"Failed to generate signal from cached features for {symbol}")
            
            # Handle cold start problem: insufficient cached features
            elif recent_features and len(recent_features) >= 10:
                logger.info(f"Limited cached features for {symbol} ({len(recent_features)} points), attempting signal generation")
                
                # Try with limited cached features
                signals = await self.signal_generator.generate_signals_from_features({symbol: recent_features})
                
                if signals and len(signals) > 0:
                    self.signals_generated += 1
                    logger.info(f"[{symbol}] Generated signal with limited cached features: {signals[0].action} with confidence {signals[0].confidence:.3f}")
                    return signals[0]
            
            # Cold start mitigation: Skip signal generation instead of downloading historical data
            logger.warning(f"Insufficient cached features for {symbol} ({len(recent_features) if recent_features else 0} points). Skipping signal generation to avoid historical data download.")
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _execute_signal_with_risk_management(self, signal: TradeSignal):
        """Execute signal with risk management using cached features when possible"""
        try:
            if not self.execution_engine or not self.risk_manager:
                return
            
            # Get recent cached features for risk management calculations
            recent_features = await self.data_pipeline.get_recent_cached_features(
                symbol=signal.symbol, 
                minutes=60
            )
            
            # Check if we have sufficient cached features for risk management
            if not recent_features or len(recent_features) < 30:
                logger.warning(f"Insufficient cached features for risk management: {signal.symbol} ({len(recent_features) if recent_features else 0} points). Skipping execution.")
                return
            
            # Convert cached features to DataFrame for risk calculations
            market_data = None
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
                    
                    # Ensure required columns exist for risk calculations
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'volume':
                                df[col] = 1000  # Default volume
                            else:
                                df[col] = df.get('close', 100.0)  # Use close price as fallback
                    
                    market_data = df
                    logger.debug(f"Converted {len(df)} cached features to DataFrame for risk management: {signal.symbol}")
                
            except Exception as e:
                logger.error(f"Failed to convert cached features for risk management: {signal.symbol}: {e}")
                return
            
            if market_data is not None and len(market_data) >= 10:
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
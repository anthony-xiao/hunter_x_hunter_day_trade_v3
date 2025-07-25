import asyncio
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

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
        self.enable_polling_backup = True
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
        if not self.is_running or not self.enable_event_driven:
            return
        
        symbol = agg_data.symbol
        
        # Check if this is a new minute bar
        current_minute = agg_data.timestamp.replace(second=0, microsecond=0)
        last_minute = self.last_bar_timestamps.get(symbol)
        
        if last_minute and current_minute <= last_minute:
            return  # Not a new bar, skip processing
        
        # Update last bar timestamp
        self.last_bar_timestamps[symbol] = current_minute
        
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
        """Update features for a symbol with new minute bar data"""
        try:
            if not self.feature_engineer or not self.data_pipeline:
                return
            
            # Get recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Get last 5 days for feature calculation
            
            historical_data = await self.data_pipeline.download_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if historical_data is not None and len(historical_data) >= 60:
                # Calculate features with the latest data
                features = await self.feature_engineer.calculate_features(historical_data)
                
                # Store features in cache for immediate use
                if hasattr(self.feature_engineer, 'feature_cache'):
                    self.feature_engineer.feature_cache[symbol] = {
                        'features': features,
                        'timestamp': datetime.now(),
                        'data_timestamp': agg_data.timestamp
                    }
                
        except Exception as e:
            logger.error(f"Error updating features for {symbol}: {e}")
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trading signal for a symbol"""
        try:
            if not self.signal_generator or not self.data_pipeline:
                return None
            
            # Get recent market data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            market_data = await self.data_pipeline.download_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data is not None and len(market_data) >= 60:
                # Generate signal for this symbol
                signals = await self.signal_generator.generate_signals({symbol: market_data})
                
                if signals and len(signals) > 0:
                    self.signals_generated += 1
                    return signals[0]  # Return the first (and likely only) signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _execute_signal_with_risk_management(self, signal: TradeSignal):
        """Execute signal with risk management"""
        try:
            if not self.execution_engine or not self.risk_manager:
                return
            
            # Get market data for risk management
            market_data = await self.data_pipeline.download_historical_data(
                symbol=signal.symbol,
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now()
            )
            
            if market_data is not None:
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
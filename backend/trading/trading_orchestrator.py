import asyncio
import time
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from loguru import logger
import pandas as pd
import pytz

from data.polygon_websocket import RealTimeData, PolygonWebSocketManager
from data.data_pipeline import DataPipeline
from ml.universal_feature_engineering import UniversalFeatureEngineering
from trading.signal_generator import SignalGenerator
from trading.execution_engine import ExecutionEngine, TradeSignal
from trading.risk_manager import RiskManager
from ml.model_types import ModelType
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
        self.feature_engineer: Optional[UniversalFeatureEngineering] = None
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
        self.eod_liquidation_task: Optional[asyncio.Task] = None
        
        # End-of-day liquidation settings
        self.eod_liquidation_enabled = True
        self.eod_liquidation_minutes_before_close = 10  # Close positions 10 minutes before market close
        
    async def initialize(self, 
                        websocket_manager: PolygonWebSocketManager,
                        data_pipeline: DataPipeline,
                        feature_engineer: UniversalFeatureEngineering,
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
        """Start the event-driven trading orchestrator with proactive data bootstrapping"""
        try:
            if self.is_running:
                logger.warning("Trading orchestrator is already running")
                return
            
            self.active_symbols = set(trading_symbols)
            self.is_running = True
            
            # Initialize processing locks for each symbol
            for symbol in trading_symbols:
                self.processing_locks[symbol] = asyncio.Lock()
            
            # Phase 1: Proactive Data Bootstrapping (Cold Start Solution)
            logger.info("Starting proactive data bootstrapping to solve cold start problem...")
            await self._bootstrap_historical_data(trading_symbols)
            
            # Start WebSocket data streaming
            if self.websocket_manager:
                await self.websocket_manager.subscribe_minute_aggs(trading_symbols)
                logger.info(f"Subscribed to minute aggregates for {len(trading_symbols)} symbols")
            
            # Start polling backup if enabled
            if self.enable_polling_backup:
                self.polling_task = asyncio.create_task(self._polling_backup_loop())
                logger.info("Started polling backup system")
            
            # Start end-of-day liquidation scheduler if enabled
            if self.eod_liquidation_enabled:
                self.eod_liquidation_task = asyncio.create_task(self._eod_liquidation_scheduler())
                logger.info("Started end-of-day liquidation scheduler")
            
            logger.info(f"Event-driven trading orchestrator started for {len(trading_symbols)} symbols with data bootstrap complete")
            
        except Exception as e:
            logger.error(f"Failed to start trading orchestrator: {e}")
            self.is_running = False
            raise
    
    async def _bootstrap_historical_data(self, trading_symbols: List[str]):
        """Bootstrap historical data for all symbols to solve cold start problem"""
        try:
            if not self.feature_engineer or not self.data_pipeline:
                logger.warning("Cannot bootstrap data: feature_engineer or data_pipeline not available")
                return
            
            # Calculate required lookback period from feature engineering requirements
            required_lookback_minutes = self.feature_engineer.calculate_required_lookback()
            logger.info(f"Calculated required lookback period: {required_lookback_minutes} minutes")
            
            # Calculate bootstrap time window
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=required_lookback_minutes)
            
            logger.info(f"Bootstrapping historical data from {start_time} to {end_time} for {len(trading_symbols)} symbols")
            
            # Bootstrap data for each symbol
            bootstrap_tasks = []
            for symbol in trading_symbols:
                task = asyncio.create_task(self._bootstrap_symbol_data(symbol, start_time, end_time))
                bootstrap_tasks.append(task)
            
            # Execute all bootstrap tasks concurrently
            results = await asyncio.gather(*bootstrap_tasks, return_exceptions=True)
            
            # Log bootstrap results
            successful_bootstraps = 0
            for i, result in enumerate(results):
                symbol = trading_symbols[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to bootstrap data for {symbol}: {result}")
                else:
                    successful_bootstraps += 1
                    logger.debug(f"Successfully bootstrapped data for {symbol}")
            
            logger.info(f"Data bootstrap complete: {successful_bootstraps}/{len(trading_symbols)} symbols successful")
            
        except Exception as e:
            logger.error(f"Error during data bootstrapping: {e}")
    
    async def _bootstrap_symbol_data(self, symbol: str, start_time: datetime, end_time: datetime):
        """Bootstrap historical data for a single symbol with intelligent cache loading"""
        try:
            # Step 1: Try to bootstrap existing features from database first
            logger.info(f"Attempting to bootstrap existing features for {symbol} from database")
            bootstrap_count = await self.data_pipeline.bootstrap_feature_cache(symbol, minutes=600)  # 10 hours lookback for sufficient data (100+ points for rolling calculations)
            
            if bootstrap_count >= 100:  # Sufficient features available for 100-point rolling windows
                logger.info(f"Successfully bootstrapped {bootstrap_count} existing features for {symbol} from database")
                return
            
            logger.info(f"Insufficient existing features ({bootstrap_count}) for {symbol}, downloading fresh data")
            
            # Step 2: Download historical data for the required lookback period
            historical_data = await self.data_pipeline.download_historical_data(
                symbol=symbol,
                start_date=start_time,
                end_date=end_time
            )
            
            if historical_data is None or len(historical_data) == 0:
                logger.warning(f"No historical data available for {symbol} in bootstrap period")
                return
            
            logger.debug(f"Downloaded {len(historical_data)} historical bars for {symbol}")
            
            # Step 3: Generate features for all historical data points using UniversalFeatureEngineering
            # Calculate start_date and end_date from historical_data
            start_date = historical_data.index.min()
            end_date = historical_data.index.max()
            
            # Use UniversalFeatureEngineering to generate universal features
            # CRITICAL FIX: Use complete ticker universe for cross-symbol features (same as training)
            all_symbols = self.data_pipeline.get_ticker_universe() if self.data_pipeline else [symbol]
            logger.info(f"[{symbol}] BOOTSTRAP: Using {len(all_symbols)} symbols for universal features (same as training): {all_symbols}")
            
            universal_features = await self.feature_engineer.engineer_universal_features(
                symbols=all_symbols,  # Use complete universe, not just single symbol
                start_date=start_date,
                end_date=end_date,
                training_mode=True
            )
            
            # Check if UniversalFeatureSet is valid
            if (universal_features is None or 
                universal_features.symbol_features.get(symbol) is None or
                universal_features.symbol_features[symbol].empty):
                logger.warning(f"No universal features generated for {symbol} during bootstrap")
                return
            
            # Step 4: Load selected_feature_columns from universal_metadata.json
            import json
            from pathlib import Path
            
            metadata_path = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json")
            selected_feature_columns = []
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        # Load from feature_selection.selected_feature_columns
                        if 'feature_selection' in metadata and 'selected_feature_columns' in metadata['feature_selection']:
                            selected_feature_columns = metadata['feature_selection']['selected_feature_columns']
                            logger.info(f"Loaded {len(selected_feature_columns)} selected feature columns from universal_metadata.json")
                        else:
                            logger.warning("No feature_selection.selected_feature_columns found in universal_metadata.json")
                            selected_feature_columns = []
                except Exception as e:
                    logger.error(f"Error loading universal_metadata.json: {e}")
                    selected_feature_columns = []
            else:
                logger.warning("universal_metadata.json not found, using all features")
            
            # Step 5: Extract and filter features for the symbol
            symbol_features = universal_features.symbol_features[symbol]
            
            # Apply selected_feature_columns filtering if available
            if selected_feature_columns:
                # Filter to only include selected features that exist in the data
                available_features = [col for col in selected_feature_columns if col in symbol_features.columns]
                if available_features:
                    symbol_features = symbol_features[available_features]
                    logger.info(f"Applied feature selection: {len(available_features)}/{len(selected_feature_columns)} selected features available for {symbol}")
                else:
                    logger.warning(f"No selected features found in generated features for {symbol}, using all features")
            
            # Step 6: Cache all filtered features for immediate availability
            cached_count = 0
            for timestamp, feature_row in symbol_features.iterrows():
                feature_dict = feature_row.to_dict()
                await self.data_pipeline.store_features(symbol, timestamp, feature_dict)
                cached_count += 1
            
            logger.info(f"Bootstrapped and cached {cached_count} feature sets with {len(symbol_features.columns)} selected features for {symbol}")
            
        except Exception as e:
            logger.error(f"Error bootstrapping data for {symbol}: {e}")
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
            
            # Cancel end-of-day liquidation task
            if self.eod_liquidation_task and not self.eod_liquidation_task.done():
                self.eod_liquidation_task.cancel()
                try:
                    await self.eod_liquidation_task
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
            
            if recent_features and len(recent_features.keys()) >= 20:  # Increased minimum for proper technical indicator calculation
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
                
                # Create a copy without timestamp for feature storage (timestamp is passed separately)
                current_bar_features = {k: v for k, v in current_bar.items() if k != 'timestamp'}
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
                
                # Priority 2 & 3: Use universal feature engineering to match training pipeline
                # This generates comprehensive features (157+) including cross-symbol, regime, and sector features
                logger.debug(f"[{symbol}] Calculating comprehensive universal features with {len(rolling_df)} data points")
                
                # Calculate start_date and end_date from rolling_df
                start_date = rolling_df.index.min()
                end_date = rolling_df.index.max()
                
                logger.info(f"[{symbol}] FEATURE_DEBUG: Calling universal feature engineering with start_date={start_date}, end_date={end_date}, data_points={len(rolling_df)}")
                
                try:
                    # Import universal feature engineering
                    from ml.universal_feature_engineering import UniversalFeatureEngineering
                    
                    # Create universal feature engineering instance
                    universal_feature_engineering = UniversalFeatureEngineering()
                    
                    # Get all trading symbols for cross-symbol and sector features (same as training)
                    all_symbols = list(self.symbol_features.keys()) if hasattr(self, 'symbol_features') else [symbol]
                    if len(all_symbols) < 2:
                        # Add common symbols for cross-symbol features if only one symbol is being traded
                        all_symbols = list(set(all_symbols + ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']))
                    
                    logger.info(f"[{symbol}] Using {len(all_symbols)} symbols for universal features: {all_symbols[:5]}...")
                    
                    # Engineer universal features with training_mode=True (same as training pipeline)
                    # CRITICAL FIX: Use complete ticker universe for cross-symbol features (same as training)
                    all_symbols = self.data_pipeline.get_ticker_universe() if self.data_pipeline else [symbol]
                    logger.info(f"[{symbol}] FEATURE_RETRY: Using {len(all_symbols)} symbols for universal features (same as training): {all_symbols}")
                    
                    universal_features = await universal_feature_engineering.engineer_universal_features(
                        symbols=all_symbols,  # Use complete universe, not just current symbol
                        start_date=start_date,
                        end_date=end_date,
                        training_mode=True  # Use training mode for comprehensive feature generation
                    )
                    
                    # Extract features for the current symbol
                    if symbol in universal_features.symbol_features:
                        engineered_features = universal_features.symbol_features[symbol]
                        
                        # Count total features across all DataFrames
                        feature_counts = {
                            'technical_features': len(engineered_features.technical_features.columns) if hasattr(engineered_features, 'technical_features') and not engineered_features.technical_features.empty else 0,
                            'market_microstructure': len(engineered_features.market_microstructure.columns) if hasattr(engineered_features, 'market_microstructure') and not engineered_features.market_microstructure.empty else 0,
                            'sentiment_features': len(engineered_features.sentiment_features.columns) if hasattr(engineered_features, 'sentiment_features') and not engineered_features.sentiment_features.empty else 0,
                            'macro_features': len(engineered_features.macro_features.columns) if hasattr(engineered_features, 'macro_features') and not engineered_features.macro_features.empty else 0,
                            'cross_asset_features': len(engineered_features.cross_asset_features.columns) if hasattr(engineered_features, 'cross_asset_features') and not engineered_features.cross_asset_features.empty else 0,
                            'engineered_features': len(engineered_features.engineered_features.columns) if hasattr(engineered_features, 'engineered_features') and not engineered_features.engineered_features.empty else 0
                        }
                        
                        # Add universal features counts
                        universal_feature_counts = {
                            'cross_symbol_features': len(universal_features.cross_symbol_features.columns) if hasattr(universal_features, 'cross_symbol_features') and not universal_features.cross_symbol_features.empty else 0,
                            'market_regime_features': len(universal_features.market_regime_features.columns) if hasattr(universal_features, 'market_regime_features') and not universal_features.market_regime_features.empty else 0,
                            'sector_features': len(universal_features.sector_features.columns) if hasattr(universal_features, 'sector_features') and not universal_features.sector_features.empty else 0
                        }
                        
                        individual_features = sum(feature_counts.values())
                        universal_feature_total = sum(universal_feature_counts.values())
                        total_features = individual_features + universal_feature_total
                        
                        logger.info(f"[{symbol}] FEATURE_DEBUG: Universal feature engineering SUCCESS")
                        logger.info(f"[{symbol}] Individual features: {individual_features} - {feature_counts}")
                        logger.info(f"[{symbol}] Universal features: {universal_feature_total} - {universal_feature_counts}")
                        logger.info(f"[{symbol}] Total comprehensive features: {total_features}")
                        
                        if total_features < 150:
                            logger.warning(f"[{symbol}] FEATURE_DEBUG: Low feature count ({total_features}) - Expected 157+ features")
                        else:
                            logger.info(f"[{symbol}] FEATURE_DEBUG: Excellent feature count ({total_features}) - Matches training pipeline")
                    else:
                        logger.error(f"[{symbol}] FEATURE_DEBUG: Symbol {symbol} not found in universal features")
                        engineered_features = None
                        
                except Exception as e:
                    logger.error(f"[{symbol}] FEATURE_DEBUG: Universal feature engineering FAILED with exception: {type(e).__name__}: {str(e)}")
                    logger.error(f"[{symbol}] FEATURE_DEBUG: Exception traceback:", exc_info=True)
                    engineered_features = None
                
                # Validate feature count and retry if needed
                max_retries = 2
                retry_count = 0
                valid_engineered_features = None
                
                while retry_count <= max_retries and valid_engineered_features is None:
                    if (engineered_features is not None and 
                        hasattr(engineered_features, 'technical_features') and 
                        not engineered_features.technical_features.empty):
                        
                        # Count total features to validate
                        feature_counts = {
                            'technical_features': len(engineered_features.technical_features.columns) if hasattr(engineered_features, 'technical_features') and not engineered_features.technical_features.empty else 0,
                            'market_microstructure': len(engineered_features.market_microstructure.columns) if hasattr(engineered_features, 'market_microstructure') and not engineered_features.market_microstructure.empty else 0,
                            'sentiment_features': len(engineered_features.sentiment_features.columns) if hasattr(engineered_features, 'sentiment_features') and not engineered_features.sentiment_features.empty else 0,
                            'macro_features': len(engineered_features.macro_features.columns) if hasattr(engineered_features, 'macro_features') and not engineered_features.macro_features.empty else 0,
                            'cross_asset_features': len(engineered_features.cross_asset_features.columns) if hasattr(engineered_features, 'cross_asset_features') and not engineered_features.cross_asset_features.empty else 0,
                            'engineered_features': len(engineered_features.engineered_features.columns) if hasattr(engineered_features, 'engineered_features') and not engineered_features.engineered_features.empty else 0
                        }
                        total_features = sum(feature_counts.values())
                        
                        if total_features >= 150:  # Accept if we have at least 150 features (matching training)
                            valid_engineered_features = engineered_features
                            logger.info(f"[{symbol}] FEATURE_VALIDATION: Regular update - Accepted {total_features} features")
                            break
                        else:
                            logger.warning(f"[{symbol}] FEATURE_VALIDATION: Regular update - Insufficient features ({total_features}), retry {retry_count + 1}/{max_retries}")
                    
                    # Retry universal feature engineering if validation failed
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.info(f"[{symbol}] FEATURE_RETRY: Regular update - Attempting universal feature engineering retry {retry_count}/{max_retries}")
                        
                        try:
                            # Retry with universal feature engineering
                            from ml.universal_feature_engineering import UniversalFeatureEngineering
                            universal_feature_engineering = UniversalFeatureEngineering()
                            
                            # CRITICAL FIX: Use complete ticker universe for cross-symbol features (same as training)
                            all_symbols = self.data_pipeline.get_ticker_universe() if self.data_pipeline else [symbol]
                            logger.info(f"[{symbol}] FEATURE_RETRY: Using {len(all_symbols)} symbols for universal features (same as training): {all_symbols}")
                            
                            universal_features = await universal_feature_engineering.engineer_universal_features(
                                symbols=all_symbols,  # Use complete universe for consistency
                                start_date=start_date,
                                end_date=end_date,
                                training_mode=True
                            )
                            
                            if symbol in universal_features.symbol_features:
                                engineered_features = universal_features.symbol_features[symbol]
                                logger.info(f"[{symbol}] FEATURE_RETRY: Universal feature engineering retry {retry_count} completed")
                            else:
                                engineered_features = None
                                logger.error(f"[{symbol}] FEATURE_RETRY: Symbol {symbol} not found in universal features retry")
                        except Exception as e:
                            logger.error(f"[{symbol}] FEATURE_RETRY: Regular update - Retry {retry_count} failed: {type(e).__name__}: {str(e)}")
                            engineered_features = None
                    else:
                        break
                
                # Use validated features
                engineered_features = valid_engineered_features
                    
                if (engineered_features is not None and 
                    hasattr(engineered_features, 'technical_features') and 
                    not engineered_features.technical_features.empty):
                    # Combine all feature DataFrames into one for extraction
                    combined_features = pd.concat([
                        engineered_features.technical_features,
                        engineered_features.market_microstructure,
                        engineered_features.sentiment_features,
                        engineered_features.macro_features,
                        engineered_features.cross_asset_features,
                        engineered_features.engineered_features
                    ], axis=1)
                    
                    # Load selected_feature_columns from universal_metadata.json for filtering
                    import json
                    from pathlib import Path
                    
                    metadata_path = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json")
                    selected_feature_columns = []
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                # Load from feature_selection.selected_feature_columns
                                if 'feature_selection' in metadata and 'selected_feature_columns' in metadata['feature_selection']:
                                    selected_feature_columns = metadata['feature_selection']['selected_feature_columns']
                                    logger.debug(f"[{symbol}] Loaded {len(selected_feature_columns)} selected feature columns for WebSocket filtering")
                                else:
                                    logger.warning(f"[{symbol}] No feature_selection.selected_feature_columns found in universal_metadata.json")
                                    selected_feature_columns = []
                        except Exception as e:
                            logger.error(f"[{symbol}] Error loading universal_metadata.json: {e}")
                            selected_feature_columns = []
                    else:
                        logger.warning(f"[{symbol}] universal_metadata.json not found, using all features")
                    
                    # Apply selected_feature_columns filtering if available
                    if selected_feature_columns:
                        # Filter to only include selected features that exist in the data
                        available_features = [col for col in selected_feature_columns if col in combined_features.columns]
                        if available_features:
                            combined_features = combined_features[available_features]
                            logger.debug(f"[{symbol}] Applied WebSocket feature selection: {len(available_features)}/{len(selected_feature_columns)} selected features available")
                        else:
                            logger.warning(f"[{symbol}] No selected features found in WebSocket features, using all features")
                    
                    # Extract features for the current timestamp only
                    if current_timestamp in combined_features.index:
                        current_features = combined_features.loc[current_timestamp].to_dict()
                        
                        # Ensure all Polygon WebSocket fields are included (Priority 3: Model Input Optimization)
                        current_features.update(current_bar_features)
                        
                        # Store comprehensive features
                        await self.data_pipeline.store_features(symbol, current_timestamp, current_features)
                        logger.debug(f"[{symbol}] Enhanced real-time features calculated and cached for {current_timestamp} ({len(current_features)} features)")
                    else:
                        # Fallback: use the last available features
                        latest_features = combined_features.iloc[-1].to_dict()
                        latest_features.update(current_bar_features)  # Ensure current bar data is included
                        await self.data_pipeline.store_features(symbol, current_timestamp, latest_features)
                        logger.debug(f"[{symbol}] Fallback features stored for {current_timestamp}")
                else:
                    if valid_engineered_features is None:
                        logger.error(f"[{symbol}] CRITICAL: Regular update feature engineering failed after {max_retries} retries - falling back to basic features")
                    else:
                        logger.warning(f"[{symbol}] engineer_features returned no results, storing basic features")
                    await self.data_pipeline.store_features(symbol, current_timestamp, current_bar_features)
                
            else:
                # Cold start: insufficient recent features, use engineer_features with minimal historical data
                logger.info(f"[{symbol}] Cold start: insufficient recent features ({len(recent_features.keys()) if recent_features else 0}), using minimal historical data")
                
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
                    # Calculate start_date and end_date from combined_data
                    start_date = combined_data.index.min()
                    end_date = combined_data.index.max()
                    
                    logger.info(f"[{symbol}] FEATURE_DEBUG: Cold start - Calling engineer_features with start_date={start_date}, end_date={end_date}, data_points={len(combined_data)}")
                    
                    try:
                        # Use universal feature engineering for cold start as well
                        from ml.universal_feature_engineering import UniversalFeatureEngineering
                        universal_feature_engineering = UniversalFeatureEngineering()
                        
                        # Get all trading symbols for cross-symbol and sector features
                        all_symbols = list(self.symbol_features.keys()) if hasattr(self, 'symbol_features') else [symbol]
                        if len(all_symbols) < 2:
                            all_symbols = list(set(all_symbols + ['SPY', 'QQQ', 'IWM', 'VIX', 'TLT', 'GLD']))
                        
                        logger.info(f"[{symbol}] Cold start using universal feature engineering with {len(all_symbols)} symbols")
                        
                        universal_features = await universal_feature_engineering.engineer_universal_features(
                            symbols=all_symbols,
                            start_date=start_date,
                            end_date=end_date,
                            training_mode=True
                        )
                        
                        if symbol in universal_features.symbol_features:
                            features = universal_features.symbol_features[symbol]
                            logger.info(f"[{symbol}] FEATURE_DEBUG: Cold start - Universal feature engineering SUCCESS")
                        else:
                            logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - Symbol {symbol} not found in universal features")
                            features = None
                        
                        if features is None:
                            logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - Universal feature engineering returned None")
                        elif not hasattr(features, 'technical_features'):
                            logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - Universal features returned object without technical_features: {type(features)}")
                        elif features.technical_features.empty:
                            logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - Universal features returned empty technical_features")
                        else:
                            # Count total features across all DataFrames
                            feature_counts = {
                                'technical_features': len(features.technical_features.columns) if hasattr(features, 'technical_features') and not features.technical_features.empty else 0,
                                'market_microstructure': len(features.market_microstructure.columns) if hasattr(features, 'market_microstructure') and not features.market_microstructure.empty else 0,
                                'sentiment_features': len(features.sentiment_features.columns) if hasattr(features, 'sentiment_features') and not features.sentiment_features.empty else 0,
                                'macro_features': len(features.macro_features.columns) if hasattr(features, 'macro_features') and not features.macro_features.empty else 0,
                                'cross_asset_features': len(features.cross_asset_features.columns) if hasattr(features, 'cross_asset_features') and not features.cross_asset_features.empty else 0,
                                'engineered_features': len(features.engineered_features.columns) if hasattr(features, 'engineered_features') and not features.engineered_features.empty else 0
                            }
                            total_features = sum(feature_counts.values())
                            logger.info(f"[{symbol}] FEATURE_DEBUG: Cold start - engineer_features SUCCESS - Generated {total_features} total features: {feature_counts}")
                            
                            if total_features < 100:
                                logger.warning(f"[{symbol}] FEATURE_DEBUG: Cold start - Low feature count ({total_features}) - Expected 150+ features")
                                
                    except Exception as e:
                        logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - engineer_features FAILED with exception: {type(e).__name__}: {str(e)}")
                        logger.error(f"[{symbol}] FEATURE_DEBUG: Cold start - Exception traceback:", exc_info=True)
                        features = None
                    
                    # Validate feature count and retry if needed
                    max_retries = 2
                    retry_count = 0
                    valid_features = None
                    
                    while retry_count <= max_retries and valid_features is None:
                        if (features is not None and 
                            hasattr(features, 'technical_features') and 
                            not features.technical_features.empty):
                            
                            # Count total features to validate
                            feature_counts = {
                                'technical_features': len(features.technical_features.columns) if hasattr(features, 'technical_features') and not features.technical_features.empty else 0,
                                'market_microstructure': len(features.market_microstructure.columns) if hasattr(features, 'market_microstructure') and not features.market_microstructure.empty else 0,
                                'sentiment_features': len(features.sentiment_features.columns) if hasattr(features, 'sentiment_features') and not features.sentiment_features.empty else 0,
                                'macro_features': len(features.macro_features.columns) if hasattr(features, 'macro_features') and not features.macro_features.empty else 0,
                                'cross_asset_features': len(features.cross_asset_features.columns) if hasattr(features, 'cross_asset_features') and not features.cross_asset_features.empty else 0,
                                'engineered_features': len(features.engineered_features.columns) if hasattr(features, 'engineered_features') and not features.engineered_features.empty else 0
                            }
                            total_features = sum(feature_counts.values())
                            
                            if total_features >= 150:  # Accept if we have at least 150 features (matching training)
                                valid_features = features
                                logger.info(f"[{symbol}] FEATURE_VALIDATION: Cold start - Accepted {total_features} features")
                                break
                            else:
                                logger.warning(f"[{symbol}] FEATURE_VALIDATION: Cold start - Insufficient features ({total_features}), retry {retry_count + 1}/{max_retries}")
                        
                        # Retry universal feature engineering if validation failed
                        if retry_count < max_retries:
                            retry_count += 1
                            logger.info(f"[{symbol}] FEATURE_RETRY: Cold start - Attempting retry {retry_count}/{max_retries}")
                            
                            try:
                                # Use universal feature engineering for retry
                                from ml.universal_feature_engineering import UniversalFeatureEngineering
                                universal_feature_engineering = UniversalFeatureEngineering()
                                
                                # CRITICAL FIX: Use complete ticker universe for cross-symbol features (same as training)
                                all_symbols = self.data_pipeline.get_ticker_universe() if self.data_pipeline else [symbol]
                                logger.info(f"[{symbol}] FEATURE_RETRY: Cold start - Using {len(all_symbols)} symbols for universal features (same as training): {all_symbols}")
                                
                                # Generate universal features for all symbols
                                universal_features = await universal_feature_engineering.engineer_universal_features(
                                    symbols=all_symbols,
                                    start_date=start_date,
                                    end_date=end_date,
                                    training_mode=True
                                )
                                
                                # Extract features for current symbol
                                if universal_features and symbol in universal_features:
                                    features = universal_features[symbol]
                                    logger.info(f"[{symbol}] FEATURE_RETRY: Cold start - Universal retry {retry_count} SUCCESS")
                                else:
                                    features = None
                                    logger.error(f"[{symbol}] FEATURE_RETRY: Cold start - Universal retry {retry_count} - Symbol not found in results")
                                    
                            except Exception as e:
                                logger.error(f"[{symbol}] FEATURE_RETRY: Cold start - Universal retry {retry_count} failed: {type(e).__name__}: {str(e)}")
                                features = None
                        else:
                            break
                    
                    if valid_features is not None:
                        # Combine all feature DataFrames into one for storage
                        combined_features = pd.concat([
                            valid_features.technical_features,
                            valid_features.market_microstructure,
                            valid_features.sentiment_features,
                            valid_features.macro_features,
                            valid_features.cross_asset_features,
                            valid_features.engineered_features
                        ], axis=1)
                        
                        # Load selected_feature_columns from universal_metadata.json for cold start filtering
                        import json
                        from pathlib import Path
                        
                        metadata_path = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json")
                        selected_feature_columns = []
                        
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                    # Load from feature_selection.selected_feature_columns
                                    if 'feature_selection' in metadata and 'selected_feature_columns' in metadata['feature_selection']:
                                        selected_feature_columns = metadata['feature_selection']['selected_feature_columns']
                                        logger.debug(f"[{symbol}] Loaded {len(selected_feature_columns)} selected feature columns for cold start filtering")
                                    else:
                                        logger.warning(f"[{symbol}] No feature_selection.selected_feature_columns found in universal_metadata.json")
                                        selected_feature_columns = []
                            except Exception as e:
                                logger.error(f"[{symbol}] Error loading universal_metadata.json: {e}")
                                selected_feature_columns = []
                        else:
                            logger.warning(f"[{symbol}] universal_metadata.json not found, using all features")
                        
                        # Apply selected_feature_columns filtering if available
                        if selected_feature_columns:
                            # Filter to only include selected features that exist in the data
                            available_features = [col for col in selected_feature_columns if col in combined_features.columns]
                            if available_features:
                                combined_features = combined_features[available_features]
                                logger.debug(f"[{symbol}] Applied cold start feature selection: {len(available_features)}/{len(selected_feature_columns)} selected features available")
                            else:
                                logger.warning(f"[{symbol}] No selected features found in cold start features, using all features")
                        
                        # Store only features for the current timestamp
                        if current_timestamp in combined_features.index:
                            latest_features = combined_features.loc[current_timestamp].to_dict()
                        else:
                            latest_features = combined_features.iloc[-1].to_dict()
                        
                        await self.data_pipeline.store_features(symbol, current_timestamp, latest_features)
                        logger.info(f"[{symbol}] Cold start comprehensive features calculated and cached for {current_timestamp} - {len(latest_features)} features stored")
                    else:
                        logger.error(f"[{symbol}] CRITICAL: Cold start feature engineering failed after {max_retries} retries - falling back to basic features")
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
                    
                    # CRITICAL DEBUG: Log when MSFT falls back to basic features
                    if symbol == "MSFT":
                        logger.error(f"[CRITICAL_DEBUG] MSFT: Falling back to basic features only - {len(basic_features)} features: {list(basic_features.keys())}")
                    
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
            
            # Use smart caching logic - let signal generator decide if features are sufficient
            if recent_features:
                feature_count = len(recent_features.keys())
                logger.info(f"Using cached features for signal generation: {symbol} ({feature_count} points)")
                
                # Generate signal directly from cached features (eliminates historical data download)
                # The signal generator now handles smart feature count validation internally
                signal = await self.signal_generator.generate_signals_from_features(symbol, recent_features)
                
                if signal:
                    self.signals_generated += 1
                    logger.info(f"[{symbol}] Generated signal from cached features: {signal.action} with confidence {signal.confidence:.3f}")
                    return signal
                else:
                    logger.debug(f"Signal generator declined to generate signal for {symbol} with {feature_count} cached features")
            
            # Cold start mitigation: Skip signal generation instead of downloading historical data
            logger.warning(f"No cached features available for {symbol}. Skipping signal generation to avoid historical data download.")
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
            
            # Apply smart caching logic for risk management
            if not recent_features:
                logger.warning(f"No cached features available for risk management: {signal.symbol}")
                # Still proceed with signal execution but with significantly reduced confidence
                risk_adjusted_signal = TradeSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=max(0.1, signal.confidence * 0.3),  # Significantly reduce confidence
                    price=signal.price,
                    quantity=min(signal.quantity, 25),  # Significantly reduce position size
                    timestamp=signal.timestamp,
                    strategy_name=signal.strategy_name,
                    metadata={**signal.metadata, "risk_adjustment": "no_cached_features"}
                )
            elif len(recent_features.keys()) < 10:
                logger.warning(f"Very limited cached features for risk management: {signal.symbol} ({len(recent_features.keys())}/10)")
                # Proceed with minimal risk management
                risk_adjusted_signal = TradeSignal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=max(0.1, signal.confidence * 0.6),  # Moderately reduce confidence
                    price=signal.price,
                    quantity=min(signal.quantity, 50),  # Moderately reduce position size
                    timestamp=signal.timestamp,
                    strategy_name=signal.strategy_name,
                    metadata={**signal.metadata, "risk_adjustment": "minimal_features"}
                )
            else:
                # Apply full risk management with available cached features
                feature_count = len(recent_features.keys())
                logger.debug(f"Applying risk management with {feature_count} cached features for {signal.symbol}")
                
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
                    # Fallback to reduced confidence signal
                    risk_adjusted_signal = TradeSignal(
                        symbol=signal.symbol,
                        action=signal.action,
                        confidence=max(0.1, signal.confidence * 0.5),
                        price=signal.price,
                        quantity=min(signal.quantity, 50),
                        timestamp=signal.timestamp,
                        strategy_name=signal.strategy_name,
                        metadata={**signal.metadata, "risk_adjustment": "feature_conversion_error"}
                    )
                    market_data = None
                
                if market_data is not None and len(market_data) >= 10:
                    # Check if signal contains statistical model predictions
                    has_statistical_models = False
                    if hasattr(signal, 'model_predictions') and signal.model_predictions:
                        statistical_model_types = {ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM, ModelType.ENSEMBLE}
                        signal_model_types = set()
                        
                        # Extract model types from model_predictions
                        for model_name in signal.model_predictions.keys():
                            if 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
                                signal_model_types.add(ModelType.XGBOOST)
                            elif 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
                                signal_model_types.add(ModelType.RANDOM_FOREST)
                            elif 'svm' in model_name.lower():
                                signal_model_types.add(ModelType.SVM)
                            elif 'ensemble' in model_name.lower():
                                signal_model_types.add(ModelType.ENSEMBLE)
                        
                        has_statistical_models = bool(signal_model_types.intersection(statistical_model_types))
                    
                    # Use appropriate position sizing method
                    if has_statistical_models:
                        position_size = await self.risk_manager.calculate_statistical_model_position_size(
                            signal=signal,
                            market_data=market_data
                        )
                        logger.debug(f"Using statistical model position sizing for {signal.symbol}: {position_size}")
                    else:
                        position_size = await self.risk_manager.calculate_position_size(
                            signal=signal,
                            market_data=market_data
                        )
                        logger.debug(f"Using standard position sizing for {signal.symbol}: {position_size}")
                    
                    if position_size > 0:
                        risk_adjusted_signal = signal  # Use original signal with calculated position size
                    else:
                        logger.debug(f"Signal for {signal.symbol} rejected by risk management")
                        return
                else:
                    logger.warning(f"Insufficient market data for risk management: {signal.symbol}")
                    # Use fallback signal with reduced parameters
                    risk_adjusted_signal = TradeSignal(
                        symbol=signal.symbol,
                        action=signal.action,
                        confidence=max(0.1, signal.confidence * 0.7),
                        price=signal.price,
                        quantity=min(signal.quantity, 75),
                        timestamp=signal.timestamp,
                        strategy_name=signal.strategy_name,
                        metadata={**signal.metadata, "risk_adjustment": "insufficient_market_data"}
                    )
                    position_size = risk_adjusted_signal.quantity
            
            # Execute the risk-adjusted signal
            if 'position_size' not in locals():
                position_size = risk_adjusted_signal.quantity
            
            success = await self.execution_engine.execute_signal(
                signal=risk_adjusted_signal
            )
            
            if success:
                self.trades_executed += 1
                logger.info(f"Event-driven trade executed: {signal.symbol} {signal.action} (risk-adjusted)")
            else:
                logger.warning(f"Failed to execute event-driven trade for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    async def _polling_backup_loop(self):
        """Backup polling system that runs alongside event-driven processing"""
        logger.info("Starting polling backup loop")
        
        while self.is_running:
            try:
                # Check if market is open - convert UTC to Eastern Time
                now_utc = datetime.now(timezone.utc)
                eastern = pytz.timezone('US/Eastern')
                now_et = now_utc.astimezone(eastern)
                
                logger.debug(f"Time check - UTC: {now_utc.strftime('%H:%M:%S')}, ET: {now_et.strftime('%H:%M:%S')}")
                
                if (now_et.weekday() < 5 and  # Monday = 0, Friday = 4
                    9 <= now_et.hour < 16 and
                    not (now_et.hour == 9 and now_et.minute < 30)):
                    
                    logger.debug(f"Market is open - processing stale symbols")
                    # Process symbols that haven't been updated recently via events
                    await self._process_stale_symbols()
                else:
                    logger.debug(f"Market is closed - ET time: {now_et.strftime('%H:%M:%S')} on {now_et.strftime('%A')}")
                
                await asyncio.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in polling backup loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_stale_symbols(self):
        """Process symbols that haven't received recent minute bar events"""
        try:
            current_time = datetime.now(timezone.utc)
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
    
    async def _eod_liquidation_scheduler(self):
        """End-of-day liquidation scheduler that runs continuously"""
        logger.info("End-of-day liquidation scheduler started")
        
        while self.is_running:
            try:
                # Check if we need to perform end-of-day liquidation
                if await self._should_perform_eod_liquidation():
                    logger.info("Triggering end-of-day liquidation")
                    
                    if self.execution_engine:
                        success = await self.execution_engine.close_all_positions_eod()
                        if success:
                            logger.info("End-of-day liquidation completed successfully")
                        else:
                            logger.error("End-of-day liquidation failed")
                    else:
                        logger.error("Cannot perform end-of-day liquidation: execution_engine not available")
                    
                    # Sleep for 5 minutes after liquidation to avoid repeated triggers
                    await asyncio.sleep(300)
                else:
                    # Check every minute when not near market close
                    await asyncio.sleep(60)
                    
            except Exception as e:
                logger.error(f"Error in end-of-day liquidation scheduler: {e}")
                await asyncio.sleep(60)
    
    async def _should_perform_eod_liquidation(self) -> bool:
        """Check if end-of-day liquidation should be performed"""
        try:
            if not self.execution_engine:
                return False
            
            # Use execution engine's market clock functionality
            return self.execution_engine.is_market_near_close(self.eod_liquidation_minutes_before_close)
            
        except Exception as e:
            logger.error(f"Error checking if EOD liquidation should be performed: {e}")
            return False

# Global orchestrator instance
orchestrator = TradingOrchestrator()

# Convenience functions
async def start_event_driven_trading(trading_symbols: List[str],
                                    websocket_manager: PolygonWebSocketManager,
                                    data_pipeline: DataPipeline,
                                    feature_engineer: UniversalFeatureEngineering,
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
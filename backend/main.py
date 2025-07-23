import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger
import json
from pathlib import Path

# Import our modules
from config import settings
from database.models import init_db
from data.data_pipeline import DataPipeline
from data.pipeline_feature_engineering import FeatureEngineer
from data.polygon_websocket import websocket_manager
from ml.model_trainer import ModelTrainer
from trading.signal_generator import SignalGenerator
from trading.execution_engine import ExecutionEngine, TradeSignal
from trading.risk_manager import RiskManager
from trading.trading_orchestrator import orchestrator, start_event_driven_trading, stop_event_driven_trading, get_orchestrator_stats

# Global instances
data_pipeline: Optional[DataPipeline] = None
feature_engineer: Optional[FeatureEngineer] = None
model_trainer: Optional[ModelTrainer] = None
signal_generator: Optional[SignalGenerator] = None
execution_engine: Optional[ExecutionEngine] = None
risk_manager: Optional[RiskManager] = None

# Trading state
trading_active = False
trading_task: Optional[asyncio.Task] = None
event_driven_active = False
last_model_training: Optional[datetime] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Algorithmic Trading System...")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()
        
        # Initialize components
        await initialize_trading_system()
        
        # Start background tasks
        asyncio.create_task(background_model_training())
        # Temporarily disable background data collection to prevent API blocking
        # asyncio.create_task(background_data_collection())
        
        logger.info("Trading system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down trading system...")
    
    global trading_task, event_driven_active
    
    # Stop event-driven trading
    if event_driven_active:
        await stop_event_driven_trading()
        event_driven_active = False
    
    # Stop polling trading task
    if trading_task and not trading_task.done():
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass
    
    # Stop trading and close positions if active
    if execution_engine and trading_active:
        await execution_engine.stop_trading()
        await execution_engine.emergency_stop()
    
    # Disconnect WebSocket
    if websocket_manager:
        await websocket_manager.disconnect()
    
    logger.info("Trading system shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Algorithmic Day Trading System",
    description="Advanced ML-based day trading system with ensemble models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_trading_system():
    """Initialize all trading system components"""
    global data_pipeline, feature_engineer, model_trainer, signal_generator, execution_engine, risk_manager
    
    try:
        # Initialize data pipeline
        logger.info("Initializing data pipeline...")
        data_pipeline = DataPipeline()
        # Database initialization will be done in background
        asyncio.create_task(data_pipeline.initialize_database())
        
        # Initialize feature engineer with data pipeline reference for hybrid storage
        logger.info("Initializing feature engineer...")
        feature_engineer = FeatureEngineer(data_pipeline=data_pipeline)
        
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        model_trainer = ModelTrainer()
        
        # Initialize signal generator
        logger.info("Initializing signal generator...")
        signal_generator = SignalGenerator()
        
        # Get trading universe (this is not actually async)
        trading_symbols = data_pipeline.get_ticker_universe()
        logger.info(f"Trading universe: {trading_symbols}")
        
        # Initialize execution engine
        logger.info("Initializing execution engine...")
        execution_engine = ExecutionEngine()
        
        # Initialize risk manager
        logger.info("Initializing risk manager...")
        risk_manager = RiskManager()
        
        logger.info("Trading system basic initialization complete")
        logger.info("Model initialization and data download will continue in background...")
        
        # Temporarily disable background initialization to test API responsiveness
        # asyncio.create_task(delayed_background_initialization(signal_generator, trading_symbols))
        
    except Exception as e:
        logger.error(f"Error initializing trading system: {e}")
        raise

async def delayed_background_initialization(signal_generator, trading_symbols):
    """Delayed background task for heavy initialization operations"""
    try:
        # Wait 5 seconds to allow FastAPI to complete startup
        await asyncio.sleep(5)
        await background_initialization(signal_generator, trading_symbols)
    except Exception as e:
        logger.error(f"Error in delayed background initialization: {e}")

async def background_initialization(signal_generator, trading_symbols):
    """Background task for heavy initialization operations"""
    try:
        logger.info("Starting background initialization...")
        
        # Initialize models for trading symbols
        if signal_generator:
            await signal_generator.initialize_models(trading_symbols)
        
        # Download initial historical data
        logger.info("Downloading initial historical data...")
        if data_pipeline:
            for symbol in trading_symbols:
                try:
                    await data_pipeline.download_historical_data(
                        symbol=symbol,
                        start_date=datetime.now() - timedelta(days=760),
                        end_date=datetime.now()
                    )
                    logger.info(f"Downloaded historical data for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to download data for {symbol}: {e}")
        
        logger.info("Background initialization complete")
        
    except Exception as e:
        logger.error(f"Error in background initialization: {e}")

async def background_model_training():
    """Background task for periodic model training"""
    global last_model_training
    
    while True:
        try:
            # Check if it's time to retrain models (daily at market close)
            now = datetime.now()
            
            # Train models daily at 4:30 PM ET (after market close)
            if (now.hour == 16 and now.minute >= 30 and 
                (last_model_training is None or 
                 (now - last_model_training).total_seconds() > 23 * 3600)):
                
                logger.info("Starting scheduled model training...")
                
                if model_trainer and data_pipeline:
                    trading_symbols = data_pipeline.get_ticker_universe()
                    
                    for symbol in trading_symbols:
                        try:
                            # Get historical data
                            historical_data = await data_pipeline.load_historical_data(
                                symbol=symbol,
                                start_date=datetime.now() - timedelta(days=760),
                                end_date=datetime.now()
                            )
                            
                            if historical_data is not None and len(historical_data) > 100:
                                # Train models
                                await model_trainer.train_ensemble_models(
                                    symbol=symbol,
                                    data=historical_data
                                )
                                
                                # Save trained models
                                if signal_generator:
                                    await signal_generator.save_models(symbol)
                                
                                logger.info(f"Completed model training for {symbol}")
                            
                        except Exception as e:
                            logger.error(f"Error training models for {symbol}: {e}")
                    
                    last_model_training = now
                    logger.info("Scheduled model training completed")
            
            # Sleep for 30 minutes before checking again
            await asyncio.sleep(1800)
            
        except Exception as e:
            logger.error(f"Error in background model training: {e}")
            await asyncio.sleep(3600)  # Sleep for 1 hour on error

async def background_data_collection():
    """Background task for continuous data collection"""
    # Wait for system to fully initialize before starting data collection
    await asyncio.sleep(30)
    
    while True:
        try:
            if data_pipeline:
                # Get real-time data for active symbols (limit to reduce API calls)
                trading_symbols = data_pipeline.get_ticker_universe()[:5]  # Limit to first 5 symbols
                
                for symbol in trading_symbols:
                    try:
                        # Get latest market data with timeout
                        latest_data = await asyncio.wait_for(
                            data_pipeline.get_real_time_data(symbol),
                            timeout=5.0
                        )
                        
                        if latest_data:
                            # Store in database
                            await data_pipeline.store_market_data([latest_data])
                        
                        # Small delay between symbols to prevent overwhelming the API
                        await asyncio.sleep(1)
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout getting data for {symbol}")
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
            
            # Sleep for 5 minutes between data collection cycles (reduced frequency)
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in background data collection: {e}")
            await asyncio.sleep(600)  # Sleep for 10 minutes on error

async def trading_loop():
    """Main trading loop"""
    global trading_active
    
    logger.info("Starting trading loop...")
    
    while trading_active:
        try:
            # Check if market is open
            now = datetime.now()
            
            # Trading hours: 9:30 AM - 4:00 PM ET (Monday-Friday)
            if (now.weekday() < 5 and  # Monday = 0, Friday = 4
                9 <= now.hour < 16 and
                not (now.hour == 9 and now.minute < 30)):
                
                # Get current market data
                trading_symbols = data_pipeline.get_ticker_universe()
                market_data = {}
                
                for symbol in trading_symbols:
                    try:
                        # Get recent historical data for analysis
                        data = await data_pipeline.load_historical_data(
                            symbol=symbol,
                            start_date=datetime.now() - timedelta(days=30),
                            end_date=datetime.now()
                        )
                        
                        if data is not None and len(data) >= 60:
                            market_data[symbol] = data
                            
                    except Exception as e:
                        logger.error(f"Error loading data for {symbol}: {e}")
                
                # Generate trading signals
                if signal_generator and market_data:
                    signals = await signal_generator.generate_signals(market_data)
                    
                    # Execute signals
                    if execution_engine and signals:
                        for signal in signals:
                            try:
                                # Apply risk management
                                if risk_manager:
                                    position_size = await risk_manager.calculate_position_size(
                                        signal=signal,
                                        market_data=market_data.get(signal.symbol)
                                    )
                                    
                                    if position_size > 0:
                                        # Execute the trade
                                        success = await execution_engine.execute_signal(
                                            signal=signal,
                                            position_size=position_size
                                        )
                                        
                                        if success:
                                            logger.info(f"Executed {signal.action} signal for {signal.symbol}")
                                        else:
                                            logger.warning(f"Failed to execute signal for {signal.symbol}")
                                    else:
                                        logger.info(f"Signal for {signal.symbol} rejected by risk management")
                                
                            except Exception as e:
                                logger.error(f"Error executing signal for {signal.symbol}: {e}")
                
                # Update model performance based on recent trades
                if execution_engine and signal_generator:
                    recent_trades = await execution_engine.get_recent_trades()
                    
                    for trade in recent_trades:
                        if hasattr(trade, 'model_predictions') and trade.model_predictions:
                            await signal_generator.update_model_performance(
                                symbol=trade.symbol,
                                actual_return=trade.realized_pnl / trade.entry_price if trade.entry_price > 0 else 0,
                                predicted_return=trade.predicted_return or 0,
                                model_predictions=trade.model_predictions
                            )
            
            # Sleep for 30 seconds before next iteration
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(60)  # Sleep longer on error
    
    logger.info("Trading loop stopped")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trading_active": trading_active,
        "components": {
            "data_pipeline": data_pipeline is not None,
            "feature_engineer": feature_engineer is not None,
            "model_trainer": model_trainer is not None,
            "signal_generator": signal_generator is not None,
            "execution_engine": execution_engine is not None,
            "risk_manager": risk_manager is not None
        }
    }

@app.get("/trading/status")
async def get_trading_status():
    """Get current trading status including event-driven orchestrator"""
    try:
        status = {
            "trading_active": trading_active,
            "event_driven_active": event_driven_active,
            "timestamp": datetime.now().isoformat(),
            "last_model_training": last_model_training.isoformat() if last_model_training else None
        }
        
        if execution_engine:
            trading_status = execution_engine.get_trading_status()
            status.update(trading_status)
        
        if data_pipeline:
            pipeline_status = await data_pipeline.get_pipeline_status()
            status["data_pipeline"] = pipeline_status
        
        if signal_generator:
            signal_stats = signal_generator.get_signal_statistics()
            status["signal_generator"] = signal_stats
        
        # Add orchestrator performance stats
        orchestrator_stats = get_orchestrator_stats()
        status["orchestrator"] = orchestrator_stats
        
        # Add WebSocket connection status
        if websocket_manager:
            status["websocket"] = {
                "connected": websocket_manager.is_connected,
                "subscribed_symbols_count": len(websocket_manager.subscribed_symbols),
                "reconnect_attempts": websocket_manager.reconnect_attempts
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the trading system with event-driven orchestrator"""
    global trading_active, trading_task, event_driven_active
    
    try:
        if trading_active or event_driven_active:
            return {"message": "Trading is already active", "status": "active"}
        
        if not execution_engine:
            raise HTTPException(status_code=500, detail="Execution engine not initialized")
        
        # Validate all components are initialized
        if not all([data_pipeline, feature_engineer, signal_generator, risk_manager]):
            raise HTTPException(status_code=500, detail="Trading system components not fully initialized")
        
        # Start execution engine
        await execution_engine.start_trading()
        
        # Get trading symbols
        trading_symbols = data_pipeline.get_ticker_universe()
        
        # Start event-driven trading system (primary)
        event_driven_success = await start_event_driven_trading(
            trading_symbols=trading_symbols,
            websocket_manager=websocket_manager,
            data_pipeline=data_pipeline,
            feature_engineer=feature_engineer,
            signal_generator=signal_generator,
            execution_engine=execution_engine,
            risk_manager=risk_manager
        )
        
        if event_driven_success:
            event_driven_active = True
            logger.info("Event-driven trading system started successfully")
        else:
            logger.warning("Event-driven system failed to start, falling back to polling only")
        
        # Start polling backup system
        trading_active = True
        trading_task = asyncio.create_task(trading_loop())
        
        logger.info("Trading system started with event-driven orchestrator")
        
        return {
            "message": "Trading started successfully",
            "status": "active",
            "event_driven_active": event_driven_active,
            "polling_backup_active": trading_active,
            "trading_symbols_count": len(trading_symbols),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/stop")
async def stop_trading():
    """Stop the trading system (both event-driven and polling)"""
    global trading_active, trading_task, event_driven_active
    
    try:
        if not trading_active and not event_driven_active:
            return {"message": "Trading is not active", "status": "inactive"}
        
        # Stop event-driven trading
        if event_driven_active:
            await stop_event_driven_trading()
            event_driven_active = False
            logger.info("Event-driven trading stopped")
        
        # Stop polling trading loop
        if trading_active:
            trading_active = False
            
            if trading_task and not trading_task.done():
                trading_task.cancel()
                try:
                    await trading_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Polling trading loop stopped")
        
        # Stop execution engine
        if execution_engine:
            await execution_engine.stop_trading()
        
        logger.info("Trading system stopped completely")
        
        return {
            "message": "Trading stopped successfully",
            "status": "inactive",
            "event_driven_stopped": True,
            "polling_stopped": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/walk-forward-test")
async def run_walk_forward_test(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """Run comprehensive walk-forward testing as specified in requirements"""
    try:
        from datetime import datetime
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        if (end_dt - start_dt).days < 30:
            raise HTTPException(status_code=400, detail="Test period must be at least 30 days")
        
        results = await execution_engine.run_walk_forward_test(start_dt, end_dt)
        
        return {
            "status": "success",
            "message": "Walk-forward test completed",
            "results": results
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Walk-forward test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/validate-readiness")
async def validate_live_readiness():
    """Comprehensive validation for live trading readiness"""
    try:
        validation_result = await execution_engine.validate_live_readiness()
        
        return {
            "status": "success",
            "message": "Live readiness validation completed",
            "validation_result": validation_result
        }
        
    except Exception as e:
        logger.error(f"Live readiness validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/performance-validation")
async def get_performance_validation(
    days: int = Query(30, description="Number of days to validate")
):
    """Get detailed performance validation results"""
    try:
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        validation_result = await execution_engine.performance_validator.validate_system_performance(
            start_date, end_date
        )
        
        return {
            "status": "success",
            "validation_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "validation_result": {
                "passed": validation_result.passed,
                "metrics": {
                    "annual_return": validation_result.metrics.annual_return,
                    "sharpe_ratio": validation_result.metrics.sharpe_ratio,
                    "max_drawdown": validation_result.metrics.max_drawdown,
                    "win_rate": validation_result.metrics.win_rate,
                    "trades_per_day": validation_result.metrics.trades_per_day,
                    "profit_factor": validation_result.metrics.profit_factor,
                    "volatility": validation_result.metrics.volatility
                },
                "recommendations": validation_result.recommendations,
                "risk_warnings": validation_result.risk_warnings
            }
        }
        
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/drift-detection")
async def get_drift_detection():
    """Get concept drift detection results for all models"""
    try:
        drift_results = await execution_engine.drift_detector.detect_all_models_drift()
        
        # Get recent drift alerts
        drift_alerts = await execution_engine.drift_detector.get_drift_alerts()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "drift_results": drift_results,
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "model_name": alert.model_name,
                    "drift_type": alert.drift_type,
                    "severity": alert.severity,
                    "drift_score": alert.drift_score,
                    "affected_metrics": alert.affected_metrics,
                    "recommended_action": alert.recommended_action
                }
                for alert in drift_alerts[:10]  # Last 10 alerts
            ]
        }
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/optimize-ensemble")
async def optimize_ensemble_weights():
    """Optimize ensemble weights using Bayesian optimization"""
    try:
        # Get recent validation data for optimization
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        optimization_result = await execution_engine.model_trainer.optimize_ensemble_weights(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "status": "success",
            "message": "Ensemble weights optimized",
            "optimization_result": optimization_result
        }
        
    except Exception as e:
        logger.error(f"Ensemble optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/emergency-stop")
async def emergency_stop():
    """Emergency stop - close all positions and cancel orders"""
    try:
        if execution_engine:
            await execution_engine.emergency_stop()
        
        # Also stop trading
        await stop_trading()
        
        logger.warning("Emergency stop executed")
        
        return {
            "message": "Emergency stop executed - all positions closed",
            "status": "emergency_stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status"""
    try:
        if not execution_engine:
            raise HTTPException(status_code=500, detail="Execution engine not initialized")
        
        portfolio = execution_engine.get_account_info()
        return portfolio
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    try:
        if not execution_engine:
            raise HTTPException(status_code=500, detail="Execution engine not initialized")
        
        positions = execution_engine.get_positions()
        # Convert Position objects to dictionaries for JSON serialization
        positions_data = []
        for pos in positions.values():
            positions_data.append({
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_price": pos.avg_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_percent": pos.pnl_percentage,
                "side": pos.side,
                "market_price": pos.market_price,
                "timestamp": pos.timestamp.isoformat()
            })
        return positions_data
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals")
async def get_recent_signals():
    """Get recent trading signals"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=500, detail="Signal generator not initialized")
        
        stats = signal_generator.get_signal_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/performance/{symbol}")
async def get_model_performance(symbol: str):
    """Get model performance for a specific symbol"""
    try:
        if not signal_generator:
            raise HTTPException(status_code=500, detail="Signal generator not initialized")
        
        performance = signal_generator.get_model_performance(symbol)
        
        if not performance:
            raise HTTPException(status_code=404, detail=f"No performance data found for {symbol}")
        
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/train/{symbol}")
async def train_models(symbol: str, background_tasks: BackgroundTasks):
    """Manually trigger model training for a symbol"""
    try:
        if not model_trainer or not data_pipeline:
            raise HTTPException(status_code=500, detail="Model trainer or data pipeline not initialized")
        
        # Add training task to background
        background_tasks.add_task(train_symbol_models, symbol)
        
        return {
            "message": f"Model training started for {symbol}",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting model training for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_symbol_models(symbol: str):
    """Background task to train models for a specific symbol"""
    try:
        logger.info(f"Starting model training for {symbol}")
        
        # Define date range for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=760)
        
        # First, try to load existing data from database
        historical_data = await data_pipeline.load_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # If no data or insufficient data, download historical data
        if historical_data is None or len(historical_data) < 100:
            logger.info(f"Insufficient data for {symbol}, downloading historical data...")
            historical_data = await data_pipeline.download_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        
        if historical_data is not None and len(historical_data) > 100:
            logger.info(f"Processing {len(historical_data)} data points for {symbol}")
            
            # Check for existing features before engineering
            if feature_engineer:
                logger.info(f"Checking existing features for {symbol}")
                try:
                    # Check which timestamps already have features
                    existing_timestamps = await data_pipeline.check_existing_features(
                        symbol=symbol,
                        start_time=start_date,
                        end_time=end_date
                    )
                    
                    total_timestamps = len(historical_data)
                    existing_count = len(existing_timestamps)
                    coverage_percentage = (existing_count / total_timestamps * 100) if total_timestamps > 0 else 0
                    
                    logger.info(f"Feature coverage for {symbol}: {existing_count}/{total_timestamps} ({coverage_percentage:.1f}%)")
                    
                    # Only engineer features if coverage is less than 95%
                    if coverage_percentage < 95.0:
                        logger.info(f"Engineering features for {symbol} (coverage: {coverage_percentage:.1f}%)")
                        # Add technical indicators and features
                        featured_data = await feature_engineer.engineer_features(historical_data, symbol)
                        if featured_data is not None and not featured_data.empty:
                            historical_data = featured_data
                            logger.info(f"Feature engineering completed for {symbol}")
                    else:
                        logger.info(f"Skipping feature engineering for {symbol} - sufficient coverage ({coverage_percentage:.1f}%)")
                        # Load existing features from database
                        featured_data = await data_pipeline.load_features_from_db(
                            symbol=symbol,
                            start_time=start_date,
                            end_time=end_date
                        )
                        if featured_data is not None and not featured_data.empty:
                            historical_data = featured_data
                            logger.info(f"Loaded existing features for {symbol}")
                        
                except Exception as fe_error:
                    logger.warning(f"Feature engineering/loading failed for {symbol}: {fe_error}")
                    # Continue with basic data if feature engineering fails
            
            # Train models
            logger.info(f"Training ensemble models for {symbol}")
            await model_trainer.train_ensemble_models(
                symbol=symbol,
                data=historical_data
            )
            
            # Save trained models
            if signal_generator:
                await signal_generator.save_models(symbol)
                logger.info(f"Models saved for {symbol}")
            
            logger.info(f"Completed model training for {symbol}")
        else:
            logger.warning(f"Insufficient data for training models for {symbol}. Need at least 100 data points, got {len(historical_data) if historical_data is not None else 0}")
            
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

@app.get("/data/universe")
async def get_trading_universe():
    """Get current trading universe"""
    try:
        if not data_pipeline:
            raise HTTPException(status_code=500, detail="Data pipeline not initialized")
        
        universe = data_pipeline.get_ticker_universe()
        return {"symbols": universe}
        
    except Exception as e:
        logger.error(f"Error getting trading universe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/status")
async def get_data_status():
    """Get data pipeline status"""
    try:
        if not data_pipeline:
            raise HTTPException(status_code=500, detail="Data pipeline not initialized")
        
        status = await data_pipeline.get_pipeline_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics"""
    try:
        if not risk_manager:
            raise HTTPException(status_code=500, detail="Risk manager not initialized")
        
        metrics = await risk_manager.get_portfolio_risk_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trading/execute-manual-trade")
async def execute_manual_trade(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL)"),
    action: str = Query(..., description="Trade action: buy, sell, or close"),
    quantity: float = Query(None, description="Quantity to trade (optional)"),
    confidence: float = Query(0.8, description="Signal confidence (0.0-1.0)"),
    price: float = Query(None, description="Limit price (optional, market order if not provided)")
):
    """Execute a manual trade for testing purposes (paper trading only)"""
    try:
        if not execution_engine:
            raise HTTPException(status_code=500, detail="Execution engine not initialized")
            
        # Verify we're in paper trading mode for safety
        if settings.trading_mode != "paper":
            raise HTTPException(status_code=403, detail="Manual trades only allowed in paper trading mode")
            
        # Create a trade signal
        signal = TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            predicted_return=0.01,  # Placeholder value
            risk_score=0.5,         # Placeholder value
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )
        
        # Execute the trade
        logger.info(f"Executing manual trade: {symbol} {action} {quantity if quantity else 'auto-sized'}")
        
        # Start trading if not already active
        if not execution_engine.is_trading:
            await execution_engine.start_trading()
            
        # Execute the signal
        order = await execution_engine.execute_signal(signal)
        
        if order:
            return {
                "status": "success",
                "message": f"Trade executed: {action} {symbol}",
                "order": {
                    "id": order.id,
                    "symbol": order.symbol,
                    "quantity": order.quantity,
                    "side": order.side,
                    "order_type": str(order.order_type),
                    "status": str(order.status),
                    "timestamp": order.timestamp.isoformat()
                }
            }
        else:
            return {
                "status": "error",
                "message": "Trade execution failed",
                "details": "No order was created. Check logs for details."
            }
            
    except Exception as e:
        logger.error(f"Error executing manual trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/trading_system.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        log_level="info"
    )
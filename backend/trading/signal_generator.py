import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
import json
import os
from pathlib import Path
import pickle
from collections import defaultdict, deque

# ML libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Technical analysis
import talib

from config import settings
from .execution_engine import TradeSignal

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class ModelType(Enum):
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class ConfidenceLevel(Enum):
    LOW = "low"        # 0.5-0.6
    MEDIUM = "medium"   # 0.6-0.75
    HIGH = "high"      # 0.75-0.9
    VERY_HIGH = "very_high"  # 0.9+

@dataclass
class ModelPrediction:
    model_type: ModelType
    symbol: str
    prediction: float
    confidence: float
    probability: float
    features_used: List[str]
    timestamp: datetime
    model_version: str
    feature_importance: Dict[str, float] = None
    
    def __post_init__(self):
        if self.feature_importance is None:
            self.feature_importance = {}

@dataclass
class EnsemblePrediction:
    symbol: str
    final_prediction: float
    confidence: float
    individual_predictions: List[ModelPrediction]
    ensemble_weights: Dict[str, float]
    risk_score: float
    signal_strength: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    volatility: float
    var_95: float
    max_drawdown_risk: float
    correlation_risk: float
    liquidity_risk: float
    market_regime_risk: float
    overall_risk_score: float
    risk_factors: Dict[str, float]

@dataclass
class MarketRegime:
    regime_type: str  # "trending", "ranging", "volatile", "calm"
    confidence: float
    volatility_level: float
    trend_strength: float
    market_stress: float
    timestamp: datetime

class SignalGenerator:
    def __init__(self):
        self.models: Dict[str, Dict[ModelType, Any]] = {}  # symbol -> model_type -> model
        self.scalers: Dict[str, StandardScaler] = {}  # symbol -> scaler
        
        # Initialize ensemble weights - will be loaded from optimization results
        self.ensemble_weights: Dict[str, Dict[ModelType, float]] = {}  # symbol -> model_type -> weight
        self.default_ensemble_weights = {
            ModelType.LSTM: 0.2,
            ModelType.CNN: 0.15,
            ModelType.TRANSFORMER: 0.2,
            ModelType.RANDOM_FOREST: 0.15,
            ModelType.XGBOOST: 0.15,
            ModelType.LIGHTGBM: 0.15
        }
        
        # Initialize ensemble configuration manager with absolute path
        from ensemble.ensemble_config import EnsembleConfigManager
        ensemble_config_dir = "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/ensemble"
        self.ensemble_config = EnsembleConfigManager(config_dir=ensemble_config_dir)
        
        # Load optimized weights on startup
        self._load_optimized_ensemble_weights()
        
        # Performance tracking (simplified - no longer used for weight optimization)
        self.model_performance: Dict[str, Dict[ModelType, Dict]] = defaultdict(lambda: defaultdict(dict))
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Risk management
        self.risk_filters = {
            'min_confidence': 0.6,
            'max_risk_score': 0.7,
            'min_liquidity': 1000000,
            'max_correlation': 0.8,
            'max_volatility': 0.5
        }
        
        # Market regime detection
        self.current_market_regime: Optional[MarketRegime] = None
        self.regime_history: deque = deque(maxlen=100)
        
        # Initialize additional attributes
        self._initialize_attributes()
    
    def _load_optimized_ensemble_weights(self) -> None:
        """Load optimized ensemble weights from shared configuration"""
        try:
            # Load optimized weights from ensemble configuration
            optimized_weights = self.ensemble_config.load_optimized_weights()
            
            # Convert string keys to ModelType enum for internal use
            converted_weights = {}
            for model_name, weight in optimized_weights.items():
                try:
                    model_type = ModelType(model_name)
                    converted_weights[model_type] = weight
                except ValueError:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
            
            # Set default weights for all symbols (will be used until symbol-specific weights are available)
            self.default_ensemble_weights = converted_weights
            
            logger.info(f"Loaded optimized ensemble weights: {optimized_weights}")
            
            # Get metadata about the optimization
            metadata = self.ensemble_config.get_ensemble_metadata()
            if metadata:
                logger.info(f"Ensemble optimization from: {metadata.get('optimization_timestamp', 'unknown')}")
                logger.info(f"Sharpe ratio: {metadata.get('sharpe_ratio', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error loading optimized ensemble weights: {e}")
            # Fallback to default equal weights
            self.default_ensemble_weights = {
                ModelType.LSTM: 0.2,
                ModelType.CNN: 0.2,
                ModelType.RANDOM_FOREST: 0.2,
                ModelType.XGBOOST: 0.2,
                ModelType.TRANSFORMER: 0.2
            }
            logger.info("Using default equal ensemble weights as fallback")
    
    def refresh_ensemble_weights(self) -> bool:
        """Refresh ensemble weights from latest optimization results"""
        try:
            self._load_optimized_ensemble_weights()
            
            # Update weights for all active symbols
            for symbol in self.ensemble_weights.keys():
                self.ensemble_weights[symbol] = self.default_ensemble_weights.copy()
            
            logger.info("Successfully refreshed ensemble weights from optimization results")
            return True
            
        except Exception as e:
            logger.error(f"Error refreshing ensemble weights: {e}")
            return False
    
    def _initialize_attributes(self):
        """Initialize class attributes - called from __init__"""
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.technical_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic',
            'williams_r', 'atr', 'cci', 'mfi', 'obv', 'ad_line'
        ]
        
        # Signal generation parameters
        self.signal_thresholds = {
            'buy_threshold': 0.6,
            'sell_threshold': -0.6,
            'strong_buy_threshold': 0.8,
            'strong_sell_threshold': -0.8
        }
        
        # Model update frequency (in hours)
        self.model_update_frequency = 24
        self.last_model_update: Dict[str, datetime] = {}
        
        # Create directories with absolute paths
        models_base_dir = Path('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models')
        models_base_dir.mkdir(exist_ok=True)
        (models_base_dir / 'lstm').mkdir(exist_ok=True)
        (models_base_dir / 'cnn').mkdir(exist_ok=True)
        (models_base_dir / 'transformer').mkdir(exist_ok=True)
        (models_base_dir / 'ensemble').mkdir(exist_ok=True)
        Path('logs/signals').mkdir(parents=True, exist_ok=True)
        Path('logs/predictions').mkdir(parents=True, exist_ok=True)
        
        logger.info("SignalGenerator initialized")
    
    async def initialize_models(self, symbols: List[str]) -> bool:
        """Initialize ML models for given symbols"""
        try:
            for symbol in symbols:
                logger.info(f"Initializing models for {symbol}")
                
                # Initialize ensemble weights (equal weights initially)
                self.ensemble_weights[symbol] = {
                    ModelType.LSTM: 0.25,
                    ModelType.CNN: 0.20,
                    ModelType.TRANSFORMER: 0.20,
                    ModelType.RANDOM_FOREST: 0.15,
                    ModelType.XGBOOST: 0.10,
                    ModelType.LIGHTGBM: 0.10
                }
                
                # Load or create models
                await self._load_or_create_models(symbol)
                
                # Initialize scaler
                self.scalers[symbol] = StandardScaler()
                
                # Initialize performance tracking
                for model_type in ModelType:
                    self.model_performance[symbol][model_type] = {
                        'accuracy': 0.5,
                        'sharpe_ratio': 0.0,
                        'total_predictions': 0,
                        'correct_predictions': 0,
                        'last_updated': datetime.now()
                    }
            
            logger.info(f"Models initialized for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    async def _load_or_create_models(self, symbol: str) -> None:
        """Load existing models or create new ones"""
        try:
            self.models[symbol] = {}
            
            # Find the latest timestamped model directory
            latest_model_dir = self._find_latest_model_directory()
            
            if latest_model_dir:
                logger.info(f"Loading models from {latest_model_dir}")
                
                # Try to load existing models from the latest directory
                for model_type in ModelType:
                    try:
                        if model_type == ModelType.LSTM:
                            model_path = os.path.join(latest_model_dir, "lstm_model.h5")
                            if os.path.exists(model_path):
                                self.models[symbol][model_type] = load_model(model_path)
                                logger.info(f"Loaded {model_type.value} model for {symbol}")
                            else:
                                self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                                
                        elif model_type == ModelType.CNN:
                            model_path = os.path.join(latest_model_dir, "cnn_model.h5")
                            if os.path.exists(model_path):
                                self.models[symbol][model_type] = load_model(model_path)
                                logger.info(f"Loaded {model_type.value} model for {symbol}")
                            else:
                                self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                                
                        elif model_type == ModelType.TRANSFORMER:
                            model_path = os.path.join(latest_model_dir, "transformer_model.h5")
                            if os.path.exists(model_path):
                                try:
                                    # Try loading with custom objects for compatibility
                                    custom_objects = {
                                        'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
                                        'LayerNormalization': tf.keras.layers.LayerNormalization,
                                        'GlobalAveragePooling1D': tf.keras.layers.GlobalAveragePooling1D
                                    }
                                    self.models[symbol][model_type] = tf.keras.models.load_model(
                                        model_path, 
                                        custom_objects=custom_objects
                                    )
                                    logger.info(f"Loaded {model_type.value} model for {symbol}")
                                except Exception as load_error:
                                    logger.warning(f"Failed to load transformer model for {symbol}: {load_error}")
                                    self.models[symbol][model_type] = self._create_transformer_model()
                            else:
                                self.models[symbol][model_type] = self._create_transformer_model()
                                
                        elif model_type == ModelType.RANDOM_FOREST:
                            model_path = os.path.join(latest_model_dir, "random_forest_model.pkl")
                            if os.path.exists(model_path):
                                with open(model_path, 'rb') as f:
                                    self.models[symbol][model_type] = pickle.load(f)
                                logger.info(f"Loaded {model_type.value} model for {symbol}")
                            else:
                                self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                                
                        elif model_type == ModelType.XGBOOST:
                            model_path = os.path.join(latest_model_dir, "xgboost_model.pkl")
                            if os.path.exists(model_path):
                                with open(model_path, 'rb') as f:
                                    self.models[symbol][model_type] = pickle.load(f)
                                logger.info(f"Loaded {model_type.value} model for {symbol}")
                            else:
                                self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                                
                        elif model_type == ModelType.LIGHTGBM:
                            # LightGBM models might not be in all directories, create new one
                            self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                            
                    except Exception as e:
                        logger.warning(f"Error loading {model_type.value} model for {symbol}: {e}")
                        # Create new model if loading fails
                        self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                
                # Load scalers if available
                scalers_path = os.path.join(latest_model_dir, "scalers.pkl")
                if os.path.exists(scalers_path):
                    try:
                        with open(scalers_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)
                        logger.info(f"Loaded scalers for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to load scalers for {symbol}: {e}")
                        # Create default scaler
                        from sklearn.preprocessing import StandardScaler
                        self.scalers[symbol] = StandardScaler()
                else:
                    # Create default scaler
                    from sklearn.preprocessing import StandardScaler
                    self.scalers[symbol] = StandardScaler()
            else:
                # No trained models found, create new ones
                logger.info(f"No trained models found, creating new models for {symbol}")
                for model_type in ModelType:
                    self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                
                # Create default scaler
                from sklearn.preprocessing import StandardScaler
                self.scalers[symbol] = StandardScaler()
                    
        except Exception as e:
            logger.error(f"Error loading/creating models for {symbol}: {e}")
    
    def _find_latest_model_directory(self) -> Optional[str]:
        """Find the latest timestamped model directory with all required models"""
        try:
            # Use absolute path to models directory
            models_dir = "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models"
            if not os.path.exists(models_dir):
                logger.error(f"Models directory not found: {models_dir}")
                return None
            
            # Required model files for a complete model set
            required_models = {
                "lstm_model.h5",
                "cnn_model.h5", 
                "transformer_model.h5",
                "random_forest_model.pkl",
                "xgboost_model.pkl",
                "scalers.pkl"
            }
            
            # Get all timestamped directories (format: YYYYMMDD_HHMMSS)
            timestamped_dirs = []
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path) and len(item) == 15 and item[8] == '_':
                    try:
                        # Validate timestamp format
                        datetime.strptime(item, "%Y%m%d_%H%M%S")
                        
                        # Check if this directory has all required models
                        dir_files = set(os.listdir(item_path))
                        if required_models.issubset(dir_files):
                            timestamped_dirs.append(item)
                            logger.info(f"Found complete model set in: {item}")
                        else:
                            missing = required_models - dir_files
                            logger.warning(f"Directory {item} missing models: {missing}")
                    except ValueError:
                        continue
            
            if not timestamped_dirs:
                logger.error("No complete model directories found")
                return None
            
            # Sort by timestamp (latest first)
            timestamped_dirs.sort(reverse=True)
            latest_dir = os.path.join(models_dir, timestamped_dirs[0])
            
            logger.info(f"Using latest complete model directory: {latest_dir}")
            return latest_dir
            
        except Exception as e:
            logger.error(f"Error finding latest model directory: {e}")
            return None
    
    async def _create_model(self, model_type: ModelType, symbol: str) -> Any:
        """Create a new model of specified type"""
        try:
            if model_type == ModelType.LSTM:
                return self._create_lstm_model()
            elif model_type == ModelType.CNN:
                return self._create_cnn_model()
            elif model_type == ModelType.TRANSFORMER:
                return self._create_transformer_model()
            elif model_type == ModelType.RANDOM_FOREST:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            elif model_type == ModelType.XGBOOST:
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == ModelType.LIGHTGBM:
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            
        except Exception as e:
            logger.error(f"Error creating {model_type.value} model: {e}")
            return None
    
    def _create_lstm_model(self) -> tf.keras.Model:
        """Create LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 50)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')  # Output between -1 and 1
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_cnn_model(self) -> tf.keras.Model:
        """Create CNN model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(60, 50)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(16, 3, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_transformer_model(self) -> tf.keras.Model:
        """Create Transformer model architecture using TensorFlow/Keras"""
        # Create the same architecture as in model_trainer.py for consistency
        input_layer = tf.keras.layers.Input(shape=(60, 50), name='input_layer')
        
        # Multi-head attention (2 heads, key_dim=32 to match model_trainer.py)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=2, 
            key_dim=32,
            name='multi_head_attention'
        )(input_layer, input_layer)
        
        # Layer normalization
        norm1 = tf.keras.layers.LayerNormalization(name='layer_normalization')(attention)
        
        # Global average pooling
        pooling = tf.keras.layers.GlobalAveragePooling1D(name='global_average_pooling1d')(norm1)
        
        # Dense layer (50 units to match model_trainer.py)
        dense = tf.keras.layers.Dense(50, activation='relu', name='dense')(pooling)
        
        # Dropout (0.2 to match model_trainer.py)
        dropout = tf.keras.layers.Dropout(0.2, name='dropout')(dense)
        
        # Output layer (tanh activation for signal generation)
        output = tf.keras.layers.Dense(1, activation='tanh', name='dense_1')(dropout)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output, name='transformer_model')
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mse',  # Use MSE for regression
            metrics=['mae']
        )
        
        return model
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """Generate trading signals for multiple symbols"""
        signals = []
        
        try:
            # Update market regime
            await self._update_market_regime(market_data)
            
            for symbol, data in market_data.items():
                if symbol not in self.models:
                    logger.warning(f"No models found for {symbol}")
                    continue
                
                # Generate ensemble prediction
                ensemble_pred = await self._generate_ensemble_prediction(symbol, data)
                
                if ensemble_pred:
                    # Convert prediction to signal
                    signal = await self._prediction_to_signal(ensemble_pred)
                    
                    if signal:
                        signals.append(signal)
                        
                        # Log signal
                        await self._log_signal(signal, ensemble_pred)
            
            logger.info(f"Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_ensemble_prediction(self, symbol: str, data: pd.DataFrame) -> Optional[EnsemblePrediction]:
        """Generate ensemble prediction for a symbol"""
        try:
            # Prepare features
            features = await self._prepare_features(symbol, data)
            if features is None or len(features) == 0:
                return None
            
            # Get individual model predictions
            individual_predictions = []
            
            for model_type, model in self.models[symbol].items():
                try:
                    prediction = await self._get_model_prediction(model_type, model, symbol, features)
                    if prediction:
                        individual_predictions.append(prediction)
                except Exception as e:
                    logger.warning(f"Error getting prediction from {model_type.value} for {symbol}: {e}")
            
            if not individual_predictions:
                return None
            
            # Calculate ensemble prediction using optimized weights
            # Use optimized weights if available, otherwise use default weights for this symbol
            if symbol not in self.ensemble_weights:
                self.ensemble_weights[symbol] = self.default_ensemble_weights.copy()
            
            weights = self.ensemble_weights[symbol]
            weighted_predictions = []
            total_weight = 0
            
            for pred in individual_predictions:
                weight = weights.get(pred.model_type, 0.0)  # Use 0.0 for unknown models
                if weight > 0:  # Only include models with positive weights
                    weighted_predictions.append(pred.prediction * weight)
                    total_weight += weight
            
            if total_weight == 0:
                # Fallback to equal weights if no optimized weights available
                logger.warning(f"No valid weights found for {symbol}, using equal weights")
                equal_weight = 1.0 / len(individual_predictions)
                for pred in individual_predictions:
                    weighted_predictions.append(pred.prediction * equal_weight)
                total_weight = 1.0
            
            final_prediction = sum(weighted_predictions) / total_weight
            
            # Calculate ensemble confidence
            confidences = [pred.confidence for pred in individual_predictions]
            ensemble_confidence = np.mean(confidences) * (1 - np.std(confidences))  # Penalize disagreement
            
            # Calculate risk score
            risk_metrics = await self._calculate_risk_metrics(symbol, data)
            
            # Calculate signal strength
            signal_strength = abs(final_prediction) * ensemble_confidence
            
            ensemble_pred = EnsemblePrediction(
                symbol=symbol,
                final_prediction=final_prediction,
                confidence=ensemble_confidence,
                individual_predictions=individual_predictions,
                ensemble_weights=weights.copy(),
                risk_score=risk_metrics.overall_risk_score,
                signal_strength=signal_strength,
                timestamp=datetime.now()
            )
            
            # Store prediction history
            self.prediction_history[symbol].append(ensemble_pred)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction for {symbol}: {e}")
            return None
    
    async def _get_model_prediction(self, model_type: ModelType, model: Any, symbol: str, features: np.ndarray) -> Optional[ModelPrediction]:
        """Get prediction from individual model"""
        try:
            if model_type in [ModelType.LSTM, ModelType.CNN]:
                # TensorFlow models
                prediction = model.predict(features.reshape(1, features.shape[0], features.shape[1]), verbose=0)[0][0]
                confidence = min(0.9, 0.5 + abs(prediction) * 0.4)  # Higher confidence for stronger predictions
                
            elif model_type == ModelType.TRANSFORMER:
                # TensorFlow model (now consistent with LSTM and CNN)
                prediction = model.predict(features.reshape(1, features.shape[0], features.shape[1]), verbose=0)[0][0]
                confidence = min(0.9, 0.5 + abs(prediction) * 0.4)
                
            elif model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                # Sklearn-style models
                features_flat = features.reshape(1, -1)
                prediction = model.predict(features_flat)[0]
                
                # Calculate confidence based on feature importance and prediction strength
                if hasattr(model, 'feature_importances_'):
                    importance_score = np.mean(model.feature_importances_)
                    confidence = min(0.9, 0.5 + abs(prediction) * 0.3 + importance_score * 0.1)
                else:
                    confidence = min(0.9, 0.5 + abs(prediction) * 0.4)
            
            else:
                return None
            
            # Calculate probability (sigmoid of prediction)
            probability = 1 / (1 + np.exp(-prediction * 5))  # Scale prediction for sigmoid
            
            return ModelPrediction(
                model_type=model_type,
                symbol=symbol,
                prediction=float(prediction),
                confidence=float(confidence),
                probability=float(probability),
                features_used=list(range(features.shape[-1])),  # Feature indices
                timestamp=datetime.now(),
                model_version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_type.value}: {e}")
            return None
    
    async def _prepare_features(self, symbol: str, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for model prediction"""
        try:
            if len(data) < 60:  # Need at least 60 periods
                return None
            
            # Use the last 60 periods
            recent_data = data.tail(60).copy()
            
            features = []
            
            # Price-based features
            features.extend([
                recent_data['close'].values,
                recent_data['high'].values,
                recent_data['low'].values,
                recent_data['open'].values,
                recent_data['volume'].values
            ])
            
            # Technical indicators
            close_prices = recent_data['close'].values
            high_prices = recent_data['high'].values
            low_prices = recent_data['low'].values
            volume = recent_data['volume'].values
            
            # Moving averages
            for period in [5, 10, 20]:
                if len(close_prices) >= period:
                    sma = talib.SMA(close_prices, timeperiod=period)
                    ema = talib.EMA(close_prices, timeperiod=period)
                    features.extend([sma, ema])
            
            # Momentum indicators
            if len(close_prices) >= 14:
                rsi = talib.RSI(close_prices, timeperiod=14)
                features.append(rsi)
            
            if len(close_prices) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close_prices)
                features.extend([macd, macd_signal, macd_hist])
            
            # Volatility indicators
            if len(close_prices) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20)
                atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                features.extend([bb_upper, bb_middle, bb_lower, atr])
            
            # Volume indicators
            if len(close_prices) >= 14:
                obv = talib.OBV(close_prices, volume)
                ad = talib.AD(high_prices, low_prices, close_prices, volume)
                features.extend([obv, ad])
            
            # Oscillators
            if len(close_prices) >= 14:
                stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
                williams_r = talib.WILLR(high_prices, low_prices, close_prices)
                cci = talib.CCI(high_prices, low_prices, close_prices)
                features.extend([stoch_k, stoch_d, williams_r, cci])
            
            # Convert to numpy array and handle NaN values
            features_array = np.array(features).T  # Transpose to get (time, features)
            
            # Fill NaN values with forward fill then backward fill
            df_features = pd.DataFrame(features_array)
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
            features_array = df_features.values
            
            # Normalize features
            if symbol in self.scalers:
                features_array = self.scalers[symbol].fit_transform(features_array)
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return None
    
    async def _calculate_risk_metrics(self, symbol: str, data: pd.DataFrame) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            close_prices = data['close'].tail(252).values  # Last year of data
            returns = np.diff(np.log(close_prices))
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5) * np.sqrt(252)
            
            # Maximum drawdown risk
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown_risk = abs(np.min(drawdown))
            
            # Liquidity risk (based on volume)
            volume_data = data['volume'].tail(20).values
            avg_volume = np.mean(volume_data)
            volume_std = np.std(volume_data)
            liquidity_risk = volume_std / avg_volume if avg_volume > 0 else 1.0
            
            # Market regime risk
            market_regime_risk = 0.5  # Placeholder - would use market regime analysis
            if self.current_market_regime:
                market_regime_risk = self.current_market_regime.market_stress
            
            # Correlation risk (simplified)
            correlation_risk = 0.3  # Placeholder - would calculate with market/sector
            
            # Overall risk score (weighted combination)
            risk_factors = {
                'volatility': volatility,
                'var_95': abs(var_95),
                'max_drawdown': max_drawdown_risk,
                'liquidity': liquidity_risk,
                'market_regime': market_regime_risk,
                'correlation': correlation_risk
            }
            
            # Normalize and weight risk factors
            weights = {
                'volatility': 0.25,
                'var_95': 0.20,
                'max_drawdown': 0.20,
                'liquidity': 0.15,
                'market_regime': 0.10,
                'correlation': 0.10
            }
            
            overall_risk_score = sum(
                min(1.0, risk_factors[factor] / 0.5) * weights[factor]
                for factor in weights
            )
            
            return RiskMetrics(
                volatility=volatility,
                var_95=var_95,
                max_drawdown_risk=max_drawdown_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                market_regime_risk=market_regime_risk,
                overall_risk_score=overall_risk_score,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics for {symbol}: {e}")
            return RiskMetrics(
                volatility=0.5, var_95=-0.05, max_drawdown_risk=0.2,
                correlation_risk=0.3, liquidity_risk=0.3, market_regime_risk=0.5,
                overall_risk_score=0.5, risk_factors={}
            )
    
    async def _update_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Update current market regime"""
        try:
            # Use SPY or a broad market index to determine regime
            spy_data = market_data.get('SPY')
            if spy_data is None or len(spy_data) < 50:
                return
            
            close_prices = spy_data['close'].tail(50).values
            returns = np.diff(np.log(close_prices))
            
            # Calculate regime indicators
            volatility = np.std(returns) * np.sqrt(252)
            trend_strength = abs(np.mean(returns)) * np.sqrt(252)
            
            # Determine regime type
            if volatility > 0.3:
                regime_type = "volatile"
            elif volatility < 0.15:
                regime_type = "calm"
            elif trend_strength > 0.1:
                regime_type = "trending"
            else:
                regime_type = "ranging"
            
            # Calculate market stress (VIX-like measure)
            market_stress = min(1.0, volatility / 0.4)
            
            # Calculate confidence in regime classification
            confidence = 1.0 - min(0.5, abs(volatility - 0.2) / 0.3)
            
            self.current_market_regime = MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                volatility_level=volatility,
                trend_strength=trend_strength,
                market_stress=market_stress,
                timestamp=datetime.now()
            )
            
            self.regime_history.append(self.current_market_regime)
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    async def _prediction_to_signal(self, ensemble_pred: EnsemblePrediction) -> Optional[TradeSignal]:
        """Convert ensemble prediction to trading signal"""
        try:
            # Apply risk filters
            if not await self._apply_risk_filters(ensemble_pred):
                return None
            
            prediction = ensemble_pred.final_prediction
            confidence = ensemble_pred.confidence
            
            # Determine action based on prediction and thresholds
            if prediction >= self.signal_thresholds['strong_buy_threshold']:
                action = SignalType.BUY.value
                signal_strength = "strong"
            elif prediction >= self.signal_thresholds['buy_threshold']:
                action = SignalType.BUY.value
                signal_strength = "moderate"
            elif prediction <= self.signal_thresholds['strong_sell_threshold']:
                action = SignalType.SELL.value
                signal_strength = "strong"
            elif prediction <= self.signal_thresholds['sell_threshold']:
                action = SignalType.SELL.value
                signal_strength = "moderate"
            else:
                action = SignalType.HOLD.value
                signal_strength = "weak"
            
            # Skip weak signals
            if action == SignalType.HOLD.value:
                return None
            
            # Calculate predicted return
            predicted_return = prediction * 0.05  # Scale to reasonable return expectation
            
            # Create signal
            signal = TradeSignal(
                symbol=ensemble_pred.symbol,
                action=action,
                confidence=confidence,
                predicted_return=predicted_return,
                risk_score=ensemble_pred.risk_score,
                timestamp=datetime.now(),
                model_predictions={
                    pred.model_type.value: pred.prediction 
                    for pred in ensemble_pred.individual_predictions
                }
            )
            
            # Store signal history
            self.signal_history[ensemble_pred.symbol].append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
            return None
    
    async def _apply_risk_filters(self, ensemble_pred: EnsemblePrediction) -> bool:
        """Apply risk filters to ensemble prediction"""
        try:
            # Minimum confidence filter
            if ensemble_pred.confidence < self.risk_filters['min_confidence']:
                logger.debug(f"Signal filtered: low confidence {ensemble_pred.confidence:.3f} for {ensemble_pred.symbol}")
                return False
            
            # Maximum risk score filter
            if ensemble_pred.risk_score > self.risk_filters['max_risk_score']:
                logger.debug(f"Signal filtered: high risk {ensemble_pred.risk_score:.3f} for {ensemble_pred.symbol}")
                return False
            
            # Market regime filter
            if self.current_market_regime and self.current_market_regime.market_stress > 0.8:
                logger.debug(f"Signal filtered: high market stress for {ensemble_pred.symbol}")
                return False
            
            # Model agreement filter (check if models agree)
            predictions = [pred.prediction for pred in ensemble_pred.individual_predictions]
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                if prediction_std > 0.5:  # High disagreement
                    logger.debug(f"Signal filtered: model disagreement for {ensemble_pred.symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying risk filters: {e}")
            return False
    
    async def update_model_performance(self, symbol: str, actual_return: float, predicted_return: float, model_predictions: Dict[str, float]) -> None:
        """Update model performance metrics"""
        try:
            for model_name, prediction in model_predictions.items():
                try:
                    model_type = ModelType(model_name)
                    perf = self.model_performance[symbol][model_type]
                    
                    # Update prediction counts
                    perf['total_predictions'] += 1
                    
                    # Check if prediction was correct (same direction)
                    if (prediction > 0 and actual_return > 0) or (prediction < 0 and actual_return < 0):
                        perf['correct_predictions'] += 1
                    
                    # Update accuracy
                    perf['accuracy'] = perf['correct_predictions'] / perf['total_predictions']
                    
                    # Update Sharpe ratio (simplified)
                    if 'returns' not in perf:
                        perf['returns'] = []
                    perf['returns'].append(actual_return if prediction > 0 else -actual_return)
                    
                    if len(perf['returns']) > 1:
                        returns_array = np.array(perf['returns'][-100:])  # Last 100 trades
                        perf['sharpe_ratio'] = np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) > 0 else 0
                    
                    perf['last_updated'] = datetime.now()
                    
                except ValueError:
                    continue  # Skip unknown model types
            
            # Update ensemble weights based on performance
            await self._update_ensemble_weights(symbol)
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    async def _update_ensemble_weights(self, symbol: str) -> None:
        """Legacy method - weights are now managed centrally via ensemble optimization"""
        # This method is kept for compatibility but no longer performs weight updates
        # Weights are now loaded from the centralized ensemble optimization results
        logger.debug(f"Ensemble weights for {symbol} are managed centrally - no local updates performed")
        
        # Optionally refresh weights from latest optimization if needed
        # This could be called periodically or triggered by external events
        pass
    
    async def _log_signal(self, signal: TradeSignal, ensemble_pred: EnsemblePrediction) -> None:
        """Log signal details"""
        try:
            signal_log = {
                'timestamp': datetime.now().isoformat(),
                'signal': asdict(signal),
                'ensemble_prediction': asdict(ensemble_pred),
                'market_regime': asdict(self.current_market_regime) if self.current_market_regime else None,
                'ensemble_weights': self.ensemble_weights.get(signal.symbol, {})
            }
            
            # Save to file
            filename = f"logs/signals/signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.symbol}.json"
            with open(filename, 'w') as f:
                json.dump(signal_log, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error logging signal: {e}")
    
    def get_model_performance(self, symbol: str) -> Dict:
        """Get model performance metrics for a symbol"""
        try:
            if symbol not in self.model_performance:
                return {}
            
            performance_data = {}
            for model_type, perf in self.model_performance[symbol].items():
                performance_data[model_type.value] = {
                    'accuracy': perf['accuracy'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'total_predictions': perf['total_predictions'],
                    'correct_predictions': perf['correct_predictions'],
                    'last_updated': perf['last_updated'].isoformat()
                }
            
            return {
                'symbol': symbol,
                'models': performance_data,
                'ensemble_weights': self.ensemble_weights.get(symbol, {}),
                'recent_signals': len(self.signal_history.get(symbol, [])),
                'recent_predictions': len(self.prediction_history.get(symbol, []))
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def get_signal_statistics(self) -> Dict:
        """Get overall signal generation statistics"""
        try:
            total_signals = sum(len(signals) for signals in self.signal_history.values())
            total_predictions = sum(len(preds) for preds in self.prediction_history.values())
            
            # Calculate signal distribution
            signal_distribution = {'buy': 0, 'sell': 0, 'hold': 0, 'close': 0}
            for signals in self.signal_history.values():
                for signal in signals:
                    signal_distribution[signal.action] += 1
            
            # Calculate average confidence
            all_confidences = []
            for signals in self.signal_history.values():
                all_confidences.extend([signal.confidence for signal in signals])
            
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            
            return {
                'total_signals_generated': total_signals,
                'total_predictions_made': total_predictions,
                'signal_distribution': signal_distribution,
                'average_confidence': avg_confidence,
                'active_symbols': len(self.models),
                'current_market_regime': asdict(self.current_market_regime) if self.current_market_regime else None,
                'risk_filters': self.risk_filters,
                'signal_thresholds': self.signal_thresholds
            }
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}
    
    async def save_models(self, symbol: str) -> bool:
        """Save trained models for a symbol"""
        try:
            if symbol not in self.models:
                return False
            
            for model_type, model in self.models[symbol].items():
                model_path = f"models/{model_type.value}/{symbol}_model"
                
                try:
                    if model_type in [ModelType.LSTM, ModelType.CNN, ModelType.TRANSFORMER]:
                        # Save all neural network models as TensorFlow .h5 files
                        model.save(f"{model_path}.h5")
                        logger.info(f"✓ Saved {model_type.value} model for {symbol} as .h5 file")
                    elif model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
                        with open(f"{model_path}.pkl", 'wb') as f:
                            pickle.dump(model, f)
                    
                    logger.info(f"Saved {model_type.value} model for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error saving {model_type.value} model for {symbol}: {e}")
            
            # Save ensemble weights
            weights_path = f"models/ensemble/{symbol}_weights.json"
            with open(weights_path, 'w') as f:
                json.dump({
                    'weights': {k.value: v for k, v in self.ensemble_weights[symbol].items()},
                    'performance': {k.value: v for k, v in self.model_performance[symbol].items()},
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models for {symbol}: {e}")
            return False
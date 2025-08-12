import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Technical analysis
import talib

from config import settings
from .execution_engine import TradeSignal
from ml.model_trainer import ModelTrainer

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
            ModelType.LSTM: 0.25,
            ModelType.CNN: 0.20,
            ModelType.TRANSFORMER: 0.25,
            ModelType.RANDOM_FOREST: 0.15,
            ModelType.XGBOOST: 0.15
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
        
        # Model configurations for feature count requirements
        self.model_configs: Dict[str, Dict] = {}
        
        # Initialize additional attributes
        self._initialize_attributes()
        
        # Load model configurations
        self._load_model_configurations()
    
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
        
        # Signal generation parameters - Enhanced with lower thresholds
        self.signal_thresholds = {
            'buy_threshold': 0.4,
            'sell_threshold': -0.4,
            'strong_buy_threshold': 0.6,
            'strong_sell_threshold': -0.6
        }
        
        # Market-based sell signal parameters
        self.market_sell_conditions = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'high_volatility_threshold': 0.25,  # 25% annualized volatility
            'market_stress_threshold': 0.7,     # Market stress level
            'volume_spike_threshold': 2.0       # 2x average volume
        }
        
        # Time-based sell signal parameters
        self.time_sell_conditions = {
            'max_holding_hours': 4,              # Maximum 4 hours for intraday
            'force_sell_minutes_before_close': 10,  # Force sell 10 min before close (aligned with EOD liquidation)
            'position_age_warning_hours': 3      # Warning at 3 hours
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
    
    def _load_model_configurations(self) -> None:
        """Load model configurations from the latest training metadata"""
        try:
            models_dir = Path('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models')
            latest_link = models_dir / 'latest'
            
            if latest_link.exists() and latest_link.is_symlink():
                metadata_file = latest_link / 'training_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_configs = metadata.get('model_configs', {})
                    logger.info(f"Loaded model configurations for {len(self.model_configs)} models")
                    
                    # Log feature counts for each model
                    for model_name, config in self.model_configs.items():
                        feature_count = config.get('feature_count', 'unknown')
                        logger.debug(f"{model_name} expects {feature_count} features")
                else:
                    logger.warning("No training metadata found, using default configurations")
                    self._set_default_model_configs()
            else:
                logger.warning("No latest model directory found, using default configurations")
                self._set_default_model_configs()
                
        except Exception as e:
            logger.error(f"Error loading model configurations: {e}")
            self._set_default_model_configs()
    
    def _set_default_model_configs(self) -> None:
        """Set default model configurations if loading fails"""
        # Use 150 features to match the trained models
        default_feature_count = 150  # Expected feature count from training metadata
        self.model_configs = {
            'lstm': {'feature_count': default_feature_count},
            'cnn': {'feature_count': default_feature_count},
            'random_forest': {'feature_count': default_feature_count},
            'xgboost': {'feature_count': default_feature_count},
            'transformer': {'feature_count': default_feature_count}
        }
        logger.info(f"Using default model configurations with {default_feature_count} features for all models")
    
    def _determine_feature_count_from_data(self, data: pd.DataFrame) -> int:
        """Determine the actual feature count from cached data"""
        try:
            # Exclude non-feature columns
            exclude_columns = {'timestamp'}
            feature_columns = [col for col in data.columns if col not in exclude_columns]
            feature_count = len(feature_columns)
            logger.debug(f"Determined feature count: {feature_count} from columns: {feature_columns[:10]}...")
            return feature_count
        except Exception as e:
            logger.error(f"Error determining feature count: {e}")
            return 50  # Default fallback
    
    async def initialize_models(self, symbols: List[str]) -> bool:
        """Initialize ML models for given symbols"""
        try:
            for symbol in symbols:
                logger.info(f"Initializing models for {symbol}")
                
                # Initialize ensemble weights (equal weights initially)
                self.ensemble_weights[symbol] = {
                    ModelType.LSTM: 0.25,
                    ModelType.CNN: 0.20,
                    ModelType.TRANSFORMER: 0.25,
                    ModelType.RANDOM_FOREST: 0.15,
                    ModelType.XGBOOST: 0.15
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
        """Load existing models or create new ones using ModelTrainer's load_models method"""
        try:
            self.models[symbol] = {}
            
            # Create a ModelTrainer instance to use its load_models method
            # Set create_model_dir=False to avoid creating new directories
            model_trainer = ModelTrainer(create_model_dir=False)
            
            try:
                # Use ModelTrainer's sophisticated load_models method
                await model_trainer.load_models(version="latest")
                
                # Map ModelTrainer's models to SignalGenerator's model structure
                # ModelTrainer uses string keys, SignalGenerator uses ModelType enum
                model_mapping = {
                    'lstm': ModelType.LSTM,
                    'cnn': ModelType.CNN,
                    'transformer': ModelType.TRANSFORMER,
                    'random_forest': ModelType.RANDOM_FOREST,
                    'xgboost': ModelType.XGBOOST
                }
                
                # Copy loaded models from ModelTrainer to SignalGenerator
                for trainer_key, signal_type in model_mapping.items():
                    if trainer_key in model_trainer.models:
                        self.models[symbol][signal_type] = model_trainer.models[trainer_key]
                        logger.info(f"✓ Loaded {signal_type.value} model for {symbol} using ModelTrainer")
                    else:
                        logger.warning(f"Model {trainer_key} not found in ModelTrainer, creating fallback")
                        self.models[symbol][signal_type] = await self._create_model(signal_type, symbol)
                
                # All models loaded from ModelTrainer or created as fallback
                
                # Copy scalers from ModelTrainer if available
                if hasattr(model_trainer, 'scalers') and model_trainer.scalers:
                    # ModelTrainer might have a single scaler or multiple scalers
                    if isinstance(model_trainer.scalers, dict):
                        # If multiple scalers, use the first one or a default key
                        scaler_key = list(model_trainer.scalers.keys())[0] if model_trainer.scalers else None
                        if scaler_key:
                            self.scalers[symbol] = model_trainer.scalers[scaler_key]
                        else:
                            self.scalers[symbol] = StandardScaler()
                    else:
                        # Single scaler
                        self.scalers[symbol] = model_trainer.scalers
                    logger.info(f"✓ Loaded scalers for {symbol} from ModelTrainer")
                else:
                    # Create default scaler if none available
                    self.scalers[symbol] = StandardScaler()
                    logger.info(f"Created default scaler for {symbol}")
                
                logger.info(f"Successfully loaded models for {symbol} using ModelTrainer's load_models method")
                
            except Exception as trainer_error:
                logger.error(f"Failed to load models using ModelTrainer: {trainer_error}")
                logger.info(f"Falling back to creating new models for {symbol}")
                
                # Fallback: create new models if ModelTrainer loading fails
                for model_type in ModelType:
                    self.models[symbol][model_type] = await self._create_model(model_type, symbol)
                
                # Create default scaler
                self.scalers[symbol] = StandardScaler()
                    
        except Exception as e:
            logger.error(f"Error loading/creating models for {symbol}: {e}")
    

    
    async def _create_model(self, model_type: ModelType, symbol: str, feature_count: int = None) -> Any:
        """Create a new model of specified type with dynamic feature count"""
        try:
            # Use provided feature_count or default to 150 to match trained models
            if feature_count is None:
                feature_count = 150
                
            if model_type == ModelType.LSTM:
                return self._create_lstm_model(feature_count)
            elif model_type == ModelType.CNN:
                return self._create_cnn_model(feature_count)
            elif model_type == ModelType.TRANSFORMER:
                return self._create_transformer_model(feature_count)
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
            
        except Exception as e:
            logger.error(f"Error creating {model_type.value} model: {e}")
            return None
    
    def _create_lstm_model(self, feature_count: int = None) -> tf.keras.Model:
        """Create LSTM model architecture with dynamic feature count"""
        # Use provided feature_count or default to 150 to match trained models
        if feature_count is None:
            feature_count = 150
            
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, feature_count)),
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
    
    def _create_cnn_model(self, feature_count: int = None) -> tf.keras.Model:
        """Create CNN model architecture with dynamic feature count"""
        # Use provided feature_count or default to 150 to match trained models
        if feature_count is None:
            feature_count = 150
            
        # Create 1D CNN architecture that works better with time series data
        # Input shape: (time_steps, features) = (60, 153)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(60, feature_count)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='tanh')  # Use tanh for signal generation
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='mse',  # Use MSE for regression in signal generation
            metrics=['mae']
        )
        
        return model
    
    def _create_transformer_model(self, feature_count: int = None) -> tf.keras.Model:
        """Create Transformer model architecture using TensorFlow/Keras with dynamic feature count"""
        # Use provided feature_count or default to 150 to match trained models
        if feature_count is None:
            feature_count = 150
            
        # Create the same architecture as in model_trainer.py for consistency
        input_layer = tf.keras.layers.Input(shape=(60, feature_count), name='input_layer')
        
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
                    # Convert prediction to signal with market data and positions
                    signal = await self._prediction_to_signal(ensemble_pred, data, None)
                    
                    if signal:
                        signals.append(signal)
                        
                        # Log signal
                        await self._log_signal(signal, ensemble_pred)
            
            logger.info(f"Generated {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def generate_signals_from_features(self, symbol: str, cached_features: Dict[datetime, Dict]) -> Optional[TradeSignal]:
        """Generate trading signal from cached features without historical data download
        
        Smart feature handling:
        - Optimal: 60+ features for full signal generation
        - Acceptable: 30+ features for reduced confidence signals
        - Minimum: 10+ features for emergency signals during market gaps
        """
        try:
            if symbol not in self.models:
                logger.warning(f"No models found for {symbol}")
                return None
            
            if not cached_features:
                logger.warning(f"No cached features available for {symbol}")
                return None
            
            feature_count = len(cached_features)
            
            # Smart feature count handling based on market conditions
            if feature_count < 10:
                logger.warning(f"Insufficient cached features for {symbol}: {feature_count}/10 (minimum required)")
                return None
            elif feature_count < 30:
                logger.info(f"Limited cached features for {symbol}: {feature_count}/60 (emergency mode - market gap detected)")
            elif feature_count < 60:
                logger.info(f"Reduced cached features for {symbol}: {feature_count}/60 (acceptable for signal generation)")
            else:
                logger.debug(f"Optimal cached features for {symbol}: {feature_count}/60")
            
            # Convert cached features to DataFrame format
            features_df = await self._convert_cached_features_to_dataframe(symbol, cached_features)
            if features_df is None:
                return None
            
            # Update market regime using cached features (simplified)
            await self._update_market_regime_from_features({symbol: features_df})
            
            # Generate ensemble prediction using cached features
            ensemble_pred = await self._generate_ensemble_prediction(symbol, features_df)
            
            if ensemble_pred:
                # Convert prediction to signal with features data and positions
                signal = await self._prediction_to_signal(ensemble_pred, features_df, None)
                
                if signal:
                    # Log signal
                    await self._log_signal(signal, ensemble_pred)
                    logger.info(f"Generated signal from cached features for {symbol}: {signal.action} (confidence: {signal.confidence:.3f})")
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal from cached features for {symbol}: {e}")
            return None
    
    async def _generate_ensemble_prediction(self, symbol: str, data: pd.DataFrame, feature_count: int = None) -> Optional[EnsemblePrediction]:
        """Generate ensemble prediction for a symbol"""
        try:
            # Get individual model predictions with model-specific feature filtering
            individual_predictions = []
            
            for model_type, model in self.models[symbol].items():
                try:
                    # Get model-specific feature count from configurations
                    model_feature_count = self.model_configs.get(model_type.value, {}).get('feature_count', feature_count)
                    
                    # Prepare features specifically for this model
                    features = await self._prepare_features(symbol, data, model_type, model_feature_count)
                    if features is None or len(features) == 0:
                        logger.warning(f"No features prepared for {model_type.value} model for {symbol}")
                        continue
                    
                    prediction = await self._get_model_prediction(model_type, model, symbol, features, model_feature_count)
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
            
            # Calculate ensemble confidence with detailed logging
            confidences = [pred.confidence for pred in individual_predictions]
            predictions = [pred.prediction for pred in individual_predictions]
            
            # Log individual model results for debugging
            logger.info(f"[CONFIDENCE_DEBUG] {symbol}: Individual model results:")
            for pred in individual_predictions:
                logger.info(f"[CONFIDENCE_DEBUG] {symbol}: {pred.model_type.value} - prediction: {pred.prediction:.4f}, confidence: {pred.confidence:.4f}")
            
            # Calculate confidence statistics
            mean_confidence = np.mean(confidences)
            std_confidence = np.std(confidences)
            prediction_variance = np.var(predictions)
            
            logger.info(f"[CONFIDENCE_DEBUG] {symbol}: Confidence stats - mean: {mean_confidence:.4f}, std: {std_confidence:.4f}, pred_var: {prediction_variance:.4f}")
            
            # Improved ensemble confidence calculation
            # Base confidence from mean, but add variability factors
            base_confidence = mean_confidence
            
            # Disagreement penalty (less harsh than before)
            disagreement_penalty = std_confidence * 0.5  # Reduced from full std
            
            # Prediction variance bonus (higher variance = more interesting signal)
            variance_bonus = min(0.1, prediction_variance * 0.2)
            
            # Final ensemble confidence
            ensemble_confidence = max(0.1, min(0.95, base_confidence - disagreement_penalty + variance_bonus))
            
            logger.info(f"[CONFIDENCE_DEBUG] {symbol}: Ensemble calculation - base: {base_confidence:.4f}, penalty: {disagreement_penalty:.4f}, bonus: {variance_bonus:.4f}, final: {ensemble_confidence:.4f}")
            
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
    
    async def _get_model_prediction(self, model_type: ModelType, model: Any, symbol: str, features: np.ndarray, feature_count: int = None) -> Optional[ModelPrediction]:
        """Get prediction from individual model with proper feature handling"""
        try:
            # Determine actual feature count from the features array
            actual_feature_count = features.shape[1] if len(features.shape) > 1 else features.shape[0]
            
            if model_type == ModelType.CNN:
                # CNN model - expects 4D input (batch_size, height, width, channels) for 2D CNN
                # The trained model architecture: Conv2D(32,(3,3)) + MaxPool(2,2) + Conv2D(64,(3,3)) + MaxPool(2,2) + Flatten()
                # Dense layer expects 14976 features = 64 * 39 * 6 (calculated from conv/pool operations)
                # Working backwards: final size after flatten should be 64 * 39 * 6 = 14976
                # This means input should be reshaped to (1, 80, 26, 1) to produce this output
                
                time_steps = features.shape[0]
                
                # CNN model now uses all available features (153) with dynamic input shape
                # CNN architecture: Conv2D(32,(3,3)) -> MaxPool(2,2) -> Conv2D(64,(3,3)) -> MaxPool(2,2) -> Flatten()
                # 
                # Input dimensions are dynamically determined from actual feature count
                # Input: (30, actual_feature_count, 1)
                # Dense layer input size is calculated based on actual feature dimensions
                # after convolution and pooling operations
                #
                # Use actual feature dimensions from data (no hardcoded filtering)
                required_height = 30   # Match actual training lookback_window
                required_width = actual_feature_count   # Use actual feature count from data
                
                if len(features.shape) == 2:
                    # Features are in (time_steps, features) format
                    # Pad or truncate to match required dimensions
                    if time_steps >= required_height and actual_feature_count >= required_width:
                        # Truncate to required size
                        resized_features = features[:required_height, :required_width]
                    else:
                        # Pad with zeros to required size
                        resized_features = np.zeros((required_height, required_width))
                        h_end = min(time_steps, required_height)
                        w_end = min(actual_feature_count, required_width)
                        resized_features[:h_end, :w_end] = features[:h_end, :w_end]
                    
                    model_input = resized_features.reshape(1, required_height, required_width, 1)
                else:
                    # Features might be flattened, reshape to required dimensions
                    total_features = required_height * required_width
                    if features.size >= total_features:
                        model_input = features.flatten()[:total_features].reshape(1, required_height, required_width, 1)
                    else:
                        # Pad with zeros if insufficient features
                        padded_features = np.zeros(total_features)
                        padded_features[:features.size] = features.flatten()
                        model_input = padded_features.reshape(1, required_height, required_width, 1)
                
                logger.debug(f"CNN model input shape (4D for Conv2D): {model_input.shape}")
                prediction = model.predict(model_input, verbose=0)[0][0]
                
                # Enhanced confidence calculation for CNN
                base_confidence = 0.5 + abs(prediction) * 0.4
                # Add model-specific variance based on prediction strength and input complexity
                model_variance = np.random.normal(0, 0.05)  # Small random component
                input_complexity = np.std(model_input.flatten()) * 0.1  # Input data complexity
                confidence = min(0.9, max(0.3, base_confidence + model_variance + input_complexity))
                
            elif model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
                # LSTM/Transformer models - expect 3D input (batch_size, time_steps, features)
                time_steps = features.shape[0]
                
                # Reshape features to match model expectations
                if len(features.shape) == 2:
                    # Features are already in (time_steps, features) format
                    model_input = features.reshape(1, time_steps, actual_feature_count)
                else:
                    # Features might be flattened, reshape appropriately
                    model_input = features.reshape(1, time_steps, -1)
                
                logger.debug(f"Model input shape for {model_type.value}: {model_input.shape}")
                prediction = model.predict(model_input, verbose=0)[0][0]
                
                # Enhanced confidence calculation for LSTM/Transformer
                base_confidence = 0.5 + abs(prediction) * 0.4
                # Add model-specific factors
                model_variance = np.random.normal(0, 0.03)  # Smaller variance for sequence models
                sequence_stability = 1.0 - np.std(model_input.flatten()) * 0.05  # Reward stable sequences
                model_type_bonus = 0.02 if model_type == ModelType.TRANSFORMER else 0.01  # Transformer slight bonus
                confidence = min(0.9, max(0.3, base_confidence + model_variance + sequence_stability + model_type_bonus))
                
            elif model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST]:
                # Sklearn-style models - use only latest features (no time sequence)
                # Random Forest and XGBoost were trained with lookback_window=1
                if len(features.shape) > 1:
                    # Use only the latest row of features
                    latest_features = features[-1:].reshape(1, -1)
                else:
                    # Already flattened
                    latest_features = features.reshape(1, -1)
                
                # All models now use all available features (no filtering)
                logger.debug(f"Features shape for {model_type.value}: {latest_features.shape}")
                prediction = model.predict(latest_features)[0]
                
                # Enhanced confidence calculation for tree-based models
                base_confidence = 0.5 + abs(prediction) * 0.3
                
                if hasattr(model, 'feature_importances_'):
                    importance_score = np.mean(model.feature_importances_)
                    # Add feature importance and model-specific factors
                    importance_bonus = importance_score * 0.15
                    model_variance = np.random.normal(0, 0.04)  # Medium variance for tree models
                    feature_diversity = np.std(latest_features.flatten()) * 0.05  # Reward diverse features
                    model_type_bonus = 0.03 if model_type == ModelType.XGBOOST else 0.02  # XGBoost slight bonus
                    confidence = min(0.9, max(0.3, base_confidence + importance_bonus + model_variance + feature_diversity + model_type_bonus))
                else:
                    model_variance = np.random.normal(0, 0.04)
                    confidence = min(0.9, max(0.3, base_confidence + model_variance))
            
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
    
    async def _convert_cached_features_to_dataframe(self, symbol: str, cached_features: Dict[datetime, Dict]) -> Optional[pd.DataFrame]:
        """Convert cached features dictionary to DataFrame format expected by models"""
        try:
            if not cached_features:
                return None

            # Sort by timestamp and convert to DataFrame
            sorted_timestamps = sorted(cached_features.keys())
            
            # DEBUG: Log total input
            logger.debug(f"[DEBUG] {symbol}: Processing {len(cached_features)} cached feature records")
            logger.debug(f"[DEBUG] {symbol}: Timestamp range: {sorted_timestamps[0]} to {sorted_timestamps[-1]}")

            # Extract OHLCV data and engineered features
            rows = []
            fallback_rows = []  # For records with only engineered features
            last_close_price = None  # Track last known close price for synthesis
            
            # DEBUG: Track processing stats
            complete_ohlcv_count = 0
            engineered_only_count = 0
            skipped_count = 0
            
            for i, timestamp in enumerate(sorted_timestamps):
                features = cached_features[timestamp]
                
                # DEBUG: Log first few records in detail
                if i < 3:
                    logger.debug(f"[DEBUG] {symbol}: Record {i+1} at {timestamp}:")
                    ohlcv_present = [key for key in ['open', 'high', 'low', 'close', 'volume'] if key in features]
                    logger.debug(f"[DEBUG] {symbol}: OHLCV present: {ohlcv_present}")

                # Primary path: Ensure we have basic OHLCV data
                if all(key in features for key in ['open', 'high', 'low', 'close', 'volume']):
                    complete_ohlcv_count += 1
                    
                    row = {
                        'timestamp': timestamp,
                        'open': features['open'],
                        'high': features['high'], 
                        'low': features['low'],
                        'close': features['close'],
                        'volume': features['volume']
                    }
                    last_close_price = features['close']  # Update last known close

                    # Add Polygon WebSocket fields if available
                    for field in ['vwap', 'transactions', 'accumulated_volume']:
                        if field in features:
                            row[field] = features[field]

                    # Add all other engineered features (only numeric values)
                    exclude_keys = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'accumulated_volume'}
                    engineered_added = 0
                    for key, value in features.items():
                        if key not in exclude_keys and not pd.isna(value):
                            # Only include numeric values, skip strings like symbol names
                            try:
                                float(value)  # Test if value can be converted to float
                                row[key] = value
                                engineered_added += 1
                            except (ValueError, TypeError):
                                # Skip non-numeric values like symbol names
                                continue
                    
                    # DEBUG: Log engineered features added
                    if i < 3:
                        logger.debug(f"[DEBUG] {symbol}: Added {engineered_added} engineered features to complete OHLCV record")

                    rows.append(row)
                    
                # Fallback path: Handle engineered-only features when OHLCV is missing
                else:
                    # Check if we have any engineered features
                    engineered_features = {}
                    exclude_keys = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions', 'accumulated_volume'}
                    
                    for key, value in features.items():
                        if key not in exclude_keys and not pd.isna(value):
                            try:
                                float(value)  # Test if value can be converted to float
                                engineered_features[key] = value
                            except (ValueError, TypeError):
                                continue
                    
                    # If we have engineered features, create a fallback row
                    if engineered_features:
                        engineered_only_count += 1
                        
                        # DEBUG: Log engineered-only record details
                        if i < 3:
                            logger.debug(f"[DEBUG] {symbol}: Engineered-only record with {len(engineered_features)} features: {list(engineered_features.keys())[:10]}...")
                        
                        # Synthesize basic OHLCV using last known close or reasonable defaults
                        synthetic_close = last_close_price if last_close_price is not None else 100.0
                        
                        fallback_row = {
                            'timestamp': timestamp,
                            'open': synthetic_close,
                            'high': synthetic_close,
                            'low': synthetic_close,
                            'close': synthetic_close,
                            'volume': 1000  # Minimal volume
                        }
                        
                        # Add the engineered features
                        fallback_row.update(engineered_features)
                        fallback_rows.append(fallback_row)
                    else:
                        skipped_count += 1
                        # DEBUG: Log why record was skipped
                        if i < 3:
                            logger.debug(f"[DEBUG] {symbol}: Skipped record - no OHLCV and no valid engineered features")

            # DEBUG: Log processing summary
            logger.debug(f"[DEBUG] {symbol}: Processing summary:")
            logger.debug(f"[DEBUG] {symbol}: - Complete OHLCV records: {complete_ohlcv_count}")
            logger.debug(f"[DEBUG] {symbol}: - Engineered-only records: {engineered_only_count}")
            logger.debug(f"[DEBUG] {symbol}: - Skipped records: {skipped_count}")
            logger.debug(f"[DEBUG] {symbol}: - Total processed: {complete_ohlcv_count + engineered_only_count}")

            # Combine primary and fallback rows
            all_rows = rows + fallback_rows
            
            if not all_rows:
                logger.warning(f"No valid feature rows found for {symbol}")
                return None

            # Log fallback usage
            if fallback_rows:
                logger.info(f"Using fallback mode for {symbol}: {len(rows)} complete OHLCV records + {len(fallback_rows)} engineered-only records")

            # Create DataFrame with timestamp index
            df = pd.DataFrame(all_rows)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # DEBUG: Log DataFrame creation details
            logger.debug(f"[DEBUG] {symbol}: Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
            logger.debug(f"[DEBUG] {symbol}: Column names: {list(df.columns)[:20]}...")

            # Ensure all columns are numeric (convert to float, coerce errors to NaN)
            numeric_conversion_failures = []
            for col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().all():
                    numeric_conversion_failures.append(col)

            # DEBUG: Log numeric conversion issues
            if numeric_conversion_failures:
                logger.debug(f"[DEBUG] {symbol}: Numeric conversion failed for columns: {numeric_conversion_failures}")

            # Drop columns that are all NaN (failed numeric conversion)
            columns_before_drop = len(df.columns)
            df = df.dropna(axis=1, how='all')
            columns_after_drop = len(df.columns)
            
            if columns_before_drop != columns_after_drop:
                logger.debug(f"[DEBUG] {symbol}: Dropped {columns_before_drop - columns_after_drop} all-NaN columns")

            logger.debug(f"Converted {len(df)} cached feature records to DataFrame for {symbol} with {len(df.columns)} numeric features")
            return df

        except Exception as e:
            logger.error(f"Error converting cached features to DataFrame for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _update_market_regime_from_features(self, features_data: Dict[str, pd.DataFrame]) -> None:
        """Update market regime using cached features data"""
        try:
            # Use SPY or first available symbol for market regime analysis
            spy_data = features_data.get('SPY')
            if spy_data is None and features_data:
                # Use first available symbol as proxy
                spy_data = list(features_data.values())[0]
            
            if spy_data is None or len(spy_data) < 20:
                logger.debug("Insufficient data for market regime update from cached features")
                return
            
            # Use cached features for regime analysis
            if 'close' in spy_data.columns:
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
                
                # Calculate market stress and confidence
                market_stress = min(1.0, volatility / 0.4)
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
                logger.debug(f"Updated market regime from cached features: {regime_type} (volatility: {volatility:.3f})")
            
        except Exception as e:
            logger.error(f"Error updating market regime from cached features: {e}")
    
    async def _prepare_features(self, symbol: str, data: pd.DataFrame, model_type: ModelType = None, feature_count: int = None) -> Optional[np.ndarray]:
        """Prepare features for model prediction using cached engineered features with model-specific filtering"""
        try:
            # Determine minimum required periods based on model type
            if model_type == ModelType.TRANSFORMER:
                # Transformer model requires exactly 60 timesteps
                required_periods = 60
                min_periods = 60
            elif model_type == ModelType.LSTM:
                # LSTM model requires exactly 60 timesteps
                required_periods = 60
                min_periods = 60
            elif model_type == ModelType.CNN:
                # CNN model requires exactly 30 timesteps
                required_periods = 30
                min_periods = 30
            else:
                # For other models (Random Forest, XGBoost), use adaptive minimum periods
                min_periods = min(60, max(10, len(data)))
                required_periods = min_periods
            
            # For sequence models with padding capability, allow smaller datasets
            if model_type not in [ModelType.TRANSFORMER, ModelType.LSTM, ModelType.CNN]:
                if len(data) < min_periods:
                    logger.warning(f"Insufficient data for {symbol} and {model_type}: {len(data)} < {min_periods}")
                    return None
            else:
                # For sequence models, we need at least 1 row to pad from
                if len(data) < 1:
                    logger.warning(f"No data available for {symbol} and {model_type}")
                    return None
            
            # Use the available data based on model requirements
            if model_type in [ModelType.TRANSFORMER, ModelType.LSTM, ModelType.CNN]:
                # For sequence models, ensure we have exactly the required number of periods
                if len(data) >= required_periods:
                    recent_data = data.tail(required_periods).copy()
                else:
                    # Pad with the first available row if we don't have enough data
                    recent_data = data.copy()
                    first_row = recent_data.iloc[0:1]
                    padding_needed = required_periods - len(recent_data)
                    padding_data = pd.concat([first_row] * padding_needed, ignore_index=True)
                    recent_data = pd.concat([padding_data, recent_data], ignore_index=True).tail(required_periods)
                    logger.debug(f"Padded {padding_needed} rows for {model_type} model for {symbol}")
            else:
                # For non-sequence models, use adaptive periods
                recent_data = data.tail(min_periods).copy()
            
            # Exclude non-feature columns and ensure all columns are numeric
            exclude_columns = {'timestamp'}
            
            # Get all feature columns (everything except excluded columns)
            feature_columns = [col for col in recent_data.columns if col not in exclude_columns]
            
            if not feature_columns:
                logger.error(f"No feature columns found for {symbol}")
                return None
            
            # Ensure all feature columns are numeric
            numeric_data = recent_data[feature_columns].copy()
            for col in feature_columns:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            # Drop columns that are all NaN (failed numeric conversion)
            numeric_data = numeric_data.dropna(axis=1, how='all')
            
            if numeric_data.empty or len(numeric_data.columns) == 0:
                logger.error(f"No valid numeric features found for {symbol}")
                return None
            
            # Extract features as numpy array
            features_array = numeric_data.values
            
            # Handle NaN values with forward fill then backward fill
            df_features = pd.DataFrame(features_array, columns=numeric_data.columns)
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN values, fill with 0
            df_features = df_features.fillna(0)
            features_array = df_features.values
            
            # Validate that features_array contains only numeric values
            if not np.issubdtype(features_array.dtype, np.number):
                logger.error(f"Features array contains non-numeric values for {symbol}")
                return None
            
            # Apply feature selection if feature_count is specified and less than available features
            if feature_count and feature_count < features_array.shape[1]:
                logger.debug(f"Selecting top {feature_count} features from {features_array.shape[1]} available for {symbol}")
                # Use the first feature_count features (most important ones should be first)
                # This assumes features are ordered by importance from the feature engineering pipeline
                features_array = features_array[:, :feature_count]
                # Update the column names accordingly
                numeric_data = numeric_data.iloc[:, :feature_count]
                logger.debug(f"Feature selection applied: {features_array.shape[1]} features selected")
            
            # Create model-specific scaler key
            scaler_key = f"{symbol}_{model_type if model_type else 'default'}"
            
            # Initialize scaler if not exists
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
            
            # Normalize features using the model-specific scaler
            if features_array.size > 0:
                current_feature_count = features_array.shape[1]
                scaler_fitted = hasattr(self.scalers[scaler_key], 'scale_') and self.scalers[scaler_key].scale_ is not None
                
                if not scaler_fitted:
                    # First time fitting the scaler
                    logger.debug(f"Fitting scaler for {scaler_key} with {current_feature_count} features")
                    self.scalers[scaler_key].fit(features_array)
                elif scaler_fitted and len(self.scalers[scaler_key].scale_) != current_feature_count:
                    # Feature count changed, need to refit the scaler
                    logger.warning(f"Feature count changed for {scaler_key}: {len(self.scalers[scaler_key].scale_)} -> {current_feature_count}. Refitting scaler.")
                    self.scalers[scaler_key] = StandardScaler()  # Create new scaler
                    self.scalers[scaler_key].fit(features_array)
                
                # Transform features using the fitted scaler
                features_array = self.scalers[scaler_key].transform(features_array)
            
            logger.debug(f"Prepared features for {symbol} ({model_type if model_type else 'default'}): shape {features_array.shape}")
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    async def _prediction_to_signal(self, ensemble_pred: EnsemblePrediction, market_data: pd.DataFrame = None, current_positions: Dict = None) -> Optional[TradeSignal]:
        """Convert ensemble prediction to trading signal with enhanced sell logic"""
        try:
            # Apply risk filters
            if not await self._apply_risk_filters(ensemble_pred):
                return None
            
            prediction = ensemble_pred.final_prediction
            confidence = ensemble_pred.confidence
            symbol = ensemble_pred.symbol
            
            # Check for forced sell conditions first (market-based and time-based)
            force_sell = False
            force_sell_reason = ""
            
            if market_data is not None:
                force_sell, force_sell_reason = await self._should_force_sell_signal(
                    symbol, market_data, current_positions
                )
            
            # If forced sell conditions are met, override prediction
            if force_sell:
                action = SignalType.SELL.value
                signal_strength = "forced"
                # Boost confidence for forced sells to ensure execution
                confidence = min(confidence * 1.2, 0.95)
                predicted_return = -0.02  # Expect small loss to avoid larger loss
                logger.info(f"Forced sell signal for {symbol}: {force_sell_reason}")
            else:
                # Normal prediction-based signal generation with enhanced thresholds
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
            
            # Enhanced signal with additional metadata
            model_predictions = {
                pred.model_type.value: pred.prediction 
                for pred in ensemble_pred.individual_predictions
            }
            
            # Add force sell information to model predictions for tracking
            if force_sell:
                model_predictions['force_sell_reason'] = force_sell_reason
            
            # Create signal
            signal = TradeSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                predicted_return=predicted_return,
                risk_score=ensemble_pred.risk_score,
                timestamp=datetime.now(),
                model_predictions=model_predictions
            )
            
            # Store signal history
            self.signal_history[symbol].append(signal)
            
            # Log enhanced signal information
            logger.info(f"Generated {signal_strength} {action} signal for {symbol} "
                       f"(confidence: {confidence:.3f}, prediction: {prediction:.3f})")
            
            if force_sell:
                logger.warning(f"FORCED SELL: {symbol} - {force_sell_reason}")
            
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
    
    async def _check_market_based_sell_conditions(self, symbol: str, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check market-based conditions that should trigger sell signals
        
        Returns:
            Tuple[bool, str]: (should_sell, reason)
        """
        try:
            if len(data) < 20:
                return False, "insufficient_data"
            
            # Calculate RSI for overbought/oversold conditions
            close_prices = data['close'].tail(14).values
            if len(close_prices) >= 14:
                gains = np.where(np.diff(close_prices) > 0, np.diff(close_prices), 0)
                losses = np.where(np.diff(close_prices) < 0, -np.diff(close_prices), 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # RSI overbought condition (sell signal)
                    if rsi > self.market_sell_conditions['rsi_overbought']:
                        return True, f"rsi_overbought_{rsi:.1f}"
                    
                    # RSI oversold condition (avoid new sells, but don't force buy)
                    if rsi < self.market_sell_conditions['rsi_oversold']:
                        return False, f"rsi_oversold_{rsi:.1f}"
            
            # Check volatility conditions
            returns = data['close'].pct_change().dropna().tail(20)
            if len(returns) > 5:
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                if volatility > self.market_sell_conditions['high_volatility_threshold']:
                    return True, f"high_volatility_{volatility:.3f}"
            
            # Check volume spike conditions
            if 'volume' in data.columns and len(data) >= 20:
                recent_volume = data['volume'].tail(5).mean()
                avg_volume = data['volume'].tail(20).mean()
                if avg_volume > 0 and recent_volume / avg_volume > self.market_sell_conditions['volume_spike_threshold']:
                    return True, f"volume_spike_{recent_volume/avg_volume:.1f}x"
            
            # Check market regime stress
            if (self.current_market_regime and 
                self.current_market_regime.market_stress > self.market_sell_conditions['market_stress_threshold']):
                return True, f"market_stress_{self.current_market_regime.market_stress:.2f}"
            
            return False, "no_market_sell_conditions"
            
        except Exception as e:
            logger.error(f"Error checking market-based sell conditions for {symbol}: {e}")
            return False, "error_checking_conditions"
    
    async def _check_time_based_sell_conditions(self, symbol: str, current_positions: Dict = None) -> Tuple[bool, str]:
        """Check time-based conditions that should trigger sell signals
        
        Args:
            symbol: Symbol to check
            current_positions: Current positions from execution engine
            
        Returns:
            Tuple[bool, str]: (should_sell, reason)
        """
        try:
            if not current_positions or symbol not in current_positions:
                return False, "no_position"
            
            position = current_positions[symbol]
            current_time = datetime.now(timezone.utc)
            
            # Check if position has entry_time
            if not hasattr(position, 'entry_time') or not position.entry_time:
                return False, "no_entry_time"
            
            # Calculate holding time
            holding_time = current_time - position.entry_time
            holding_hours = holding_time.total_seconds() / 3600
            
            # Force sell if held too long (max 4 hours for intraday)
            if holding_hours > self.time_sell_conditions['max_holding_hours']:
                return True, f"max_holding_time_{holding_hours:.1f}h"
            
            # Check if we're near market close (force sell 30 minutes before)
            try:
                # Import here to avoid circular imports
                from execution_engine import ExecutionEngine
                
                # Check if we're near market close
                if hasattr(ExecutionEngine, 'is_market_near_close'):
                    # This would need to be called on an instance, but for now we'll use a simple time check
                    # In a real implementation, you'd pass the execution engine instance
                    pass
                
                # Simple time-based check for market close (4 PM ET = 9 PM UTC)
                market_close_time = current_time.replace(hour=21, minute=0, second=0, microsecond=0)
                force_sell_time = market_close_time - timedelta(minutes=self.time_sell_conditions['force_sell_minutes_before_close'])
                
                if current_time >= force_sell_time:
                    return True, "approaching_market_close"
                    
            except Exception as time_check_error:
                logger.debug(f"Could not check market close time: {time_check_error}")
            
            # Warning for positions approaching max holding time
            if holding_hours > self.time_sell_conditions['position_age_warning_hours']:
                logger.info(f"Position {symbol} held for {holding_hours:.1f} hours - approaching max holding time")
            
            return False, f"holding_time_ok_{holding_hours:.1f}h"
            
        except Exception as e:
            logger.error(f"Error checking time-based sell conditions for {symbol}: {e}")
            return False, "error_checking_time_conditions"
    
    async def _should_force_sell_signal(self, symbol: str, data: pd.DataFrame, current_positions: Dict = None) -> Tuple[bool, str]:
        """Determine if a sell signal should be forced based on market or time conditions
        
        Returns:
            Tuple[bool, str]: (should_force_sell, reason)
        """
        try:
            # Check market-based sell conditions
            market_sell, market_reason = await self._check_market_based_sell_conditions(symbol, data)
            if market_sell:
                return True, f"market_condition_{market_reason}"
            
            # Check time-based sell conditions
            time_sell, time_reason = await self._check_time_based_sell_conditions(symbol, current_positions)
            if time_sell:
                return True, f"time_condition_{time_reason}"
            
            return False, "no_force_sell_conditions"
            
        except Exception as e:
            logger.error(f"Error checking force sell conditions for {symbol}: {e}")
            return False, "error_checking_force_sell"
    
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
            # Convert ModelType enum keys to string values for JSON serialization
            ensemble_weights = self.ensemble_weights.get(signal.symbol, {})
            serializable_weights = {model_type.value: weight for model_type, weight in ensemble_weights.items()}
            
            signal_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': asdict(signal),
                'ensemble_prediction': asdict(ensemble_pred),
                'market_regime': asdict(self.current_market_regime) if self.current_market_regime else None,
                'ensemble_weights': serializable_weights
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
                    elif model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST]:
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
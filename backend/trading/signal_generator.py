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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Technical analysis
import talib

from config import settings
from .execution_engine import TradeSignal
from ml.universal_trainer import UniversalTrainer
from ml.universal_model_architectures import UniversalModelArchitectures
from ml.universal_feature_engineering import UniversalFeatureEngineering
from ml.ml_feature_engineering import FeatureEngineering
from ml.feature_selector import UniversalFeatureSelector, FeatureSelectionConfig

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

# Import ModelType from ml module to avoid circular imports
from ml.model_types import ModelType

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

@dataclass
class DirectionalConfidence:
    """Separate confidence metrics for buy vs. sell predictions"""
    buy_confidence: float
    sell_confidence: float
    direction_clarity: float
    prediction_strength: float
    model_uncertainty: float
    
    @classmethod
    def calculate(cls, prediction: float, base_confidence: float, model_variance: float = 0.0) -> 'DirectionalConfidence':
        """Calculate directional confidence metrics"""
        # Direction clarity - how clear the directional signal is
        direction_clarity = abs(prediction)
        
        # Prediction strength without absolute value bias
        prediction_strength = prediction * prediction
        
        # Model uncertainty from variance
        model_uncertainty = min(0.5, model_variance * 2.0)
        
        # Calculate buy vs sell confidence based on prediction direction
        if prediction > 0:  # Buy signal
            buy_confidence = min(0.95, base_confidence + (direction_clarity * 0.2))
            sell_confidence = max(0.05, base_confidence - (direction_clarity * 0.3))
        else:  # Sell signal
            sell_confidence = min(0.95, base_confidence + (direction_clarity * 0.2))
            buy_confidence = max(0.05, base_confidence - (direction_clarity * 0.3))
        
        # Adjust for model uncertainty
        uncertainty_penalty = model_uncertainty * 0.1
        buy_confidence = max(0.05, buy_confidence - uncertainty_penalty)
        sell_confidence = max(0.05, sell_confidence - uncertainty_penalty)
        
        return cls(
            buy_confidence=buy_confidence,
            sell_confidence=sell_confidence,
            direction_clarity=direction_clarity,
            prediction_strength=prediction_strength,
            model_uncertainty=model_uncertainty
        )

class SignalGenerator:
    def __init__(self, model_trainer: Optional[UniversalTrainer] = None, supabase_client=None, data_pipeline=None):
        self.models: Dict[str, Dict[ModelType, Any]] = {}  # symbol -> model_type -> model
        self.scalers: Dict[str, StandardScaler] = {}  # symbol -> scaler
        
        # Store UniversalTrainer instance for universal model loading
        self.model_trainer = model_trainer
        
        # Store Supabase client for database operations
        if supabase_client:
            self.supabase_client = supabase_client
        else:
            from database import db_manager
            self.supabase_client = db_manager.get_supabase_client()
        
        # Store data pipeline for feature engineering
        if data_pipeline:
            self.data_pipeline = data_pipeline
        elif model_trainer and hasattr(model_trainer, 'data_pipeline'):
            self.data_pipeline = model_trainer.data_pipeline
        else:
            self.data_pipeline = None
        
        # Initialize ensemble weights - will be loaded from optimization results
        self.ensemble_weights: Dict[str, Dict[ModelType, float]] = {}  # symbol -> model_type -> weight
        self.default_ensemble_weights = {
            ModelType.XGBOOST: 0.35,
            ModelType.RANDOM_FOREST: 0.30,
            ModelType.SVM: 0.35
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
            'min_confidence': 0.4,
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
        
        # Universal model components
        self.universal_trainer: Optional[UniversalTrainer] = None
        self.universal_architectures: Optional[UniversalModelArchitectures] = None
        self.universal_feature_engineering: Optional[UniversalFeatureEngineering] = None
        self.is_universal_mode: bool = False
        self.universal_models: Dict[str, Any] = {}  # Store loaded universal models
        
        # Feature selection attributes - Initialize to None, will be loaded by load_feature_selection_results
        self.selected_features: Optional[List[str]] = None
        self.selected_feature_columns: Optional[List[str]] = None
        self.feature_selection_metadata: Optional[Dict] = None
        self.universal_symbol_models: Dict[str, Dict[str, Any]] = {}  # Store symbol-specific universal models
        self.universal_metadata: Dict[str, Any] = {}  # Store universal metadata
        
        # Feature selection components
        self.feature_selector: Optional[UniversalFeatureSelector] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_selection_metadata: Dict[str, Any] = {}
        
        # Initialize additional attributes
        self._initialize_attributes()
        
        # Load model configurations
        self._load_model_configurations()
        
        # Load feature selection results during initialization
        import asyncio
        try:
            # Run the async method in a synchronous context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                asyncio.create_task(self.load_feature_selection_results())
            else:
                # If not in async context, run it directly
                loop.run_until_complete(self.load_feature_selection_results())
        except RuntimeError:
            # If no event loop exists, create one
            asyncio.run(self.load_feature_selection_results())
    
    def _load_optimized_ensemble_weights(self) -> None:
        """Load optimized ensemble weights from universal training metadata or shared configuration"""
        try:
            # First try to load weights from universal training metadata
            universal_metadata_path = "/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json"
            optimized_weights = None
            weights_source = None
            training_timestamp = None
            
            logger.info("Loading ensemble weights...")
            
            if os.path.exists(universal_metadata_path):
                try:
                    logger.info("Attempting to load from universal training metadata...")
                    with open(universal_metadata_path, 'r') as f:
                        universal_data = json.load(f)
                    
                    if 'ensemble_weights' in universal_data:
                        optimized_weights = universal_data['ensemble_weights']
                        weights_source = "universal training metadata"
                        training_timestamp = universal_data.get('training_timestamp', 'unknown')
                        
                        # Log detailed universal weights information
                        logger.info("✓ Successfully loaded ensemble weights from universal training metadata")
                        logger.info("📊 Universal ensemble weights:")
                        for model_name, weight in optimized_weights.items():
                            percentage = weight * 100
                            logger.info(f"  - {model_name.upper()}: {percentage:.2f}% ({weight:.4f})")
                        logger.info(f"🕐 Universal weights generated on: {training_timestamp}")
                        
                except Exception as e:
                    logger.warning(f"Error reading universal metadata: {e}")
            
            # Fallback to ensemble configuration if universal weights not available
            if optimized_weights is None:
                logger.info("Universal weights not available, loading from ensemble configuration...")
                optimized_weights = self.ensemble_config.load_optimized_weights()
                weights_source = "ensemble configuration"
                
                # Log detailed ensemble config weights information
                logger.info("✓ Successfully loaded ensemble weights from ensemble configuration")
                logger.info("📊 Ensemble config weights:")
                for model_name, weight in optimized_weights.items():
                    percentage = weight * 100
                    logger.info(f"  - {model_name.upper()}: {percentage:.2f}% ({weight:.4f})")
                
                # Get metadata about the optimization
                metadata = self.ensemble_config.get_ensemble_metadata()
                if metadata:
                    optimization_timestamp = metadata.get('optimization_timestamp', 'unknown')
                    sharpe_ratio = metadata.get('sharpe_ratio', 'unknown')
                    logger.info(f"🕐 Ensemble optimization from: {optimization_timestamp}")
                    logger.info(f"📈 Sharpe ratio: {sharpe_ratio}")
            
            # Convert string keys to ModelType enum for internal use
            # Filter out unsupported model types (legacy neural network models)
            supported_models = {'xgboost', 'random_forest', 'svm', 'ensemble'}
            converted_weights = {}
            filtered_models = []
            
            for model_name, weight in optimized_weights.items():
                if model_name.lower() not in supported_models:
                    filtered_models.append(model_name)
                    continue  # Skip unsupported models silently
                try:
                    model_type = ModelType(model_name)
                    converted_weights[model_type] = weight
                except ValueError:
                    logger.warning(f"Unknown model type: {model_name}")
                    continue
            
            if filtered_models:
                logger.info(f"ℹ️  Filtered out unsupported models: {', '.join(filtered_models)}")
            
            # Set default weights for all symbols (will be used until symbol-specific weights are available)
            self.default_ensemble_weights = converted_weights
            
            logger.info(f"✅ Ensemble weights loaded successfully from {weights_source}")
                
        except Exception as e:
            logger.error(f"❌ Error loading optimized ensemble weights: {e}")
            # Fallback to default equal weights
            self.default_ensemble_weights = {
                ModelType.XGBOOST: 0.33,
                ModelType.RANDOM_FOREST: 0.33,
                ModelType.SVM: 0.34
            }
            logger.info("⚠️  Using default equal ensemble weights as fallback:")
            logger.info("📊 Default ensemble weights:")
            for model_type, weight in self.default_ensemble_weights.items():
                percentage = weight * 100
                logger.info(f"  - {model_type.value.upper()}: {percentage:.2f}% ({weight:.4f})")
    
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
            'buy_threshold': 0.55,         # Reduce from 0.6
            'sell_threshold': 0.45,        # Keep same
            'strong_buy_threshold': 0.65,  # Reduce from 0.75
            'strong_sell_threshold': 0.35 # Increase from 0.25
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
        """Load model configurations from universal metadata or latest training metadata"""
        try:
            # First, try to load from universal metadata
            universal_metadata_path = Path('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json')
            
            if universal_metadata_path.exists():
                with open(universal_metadata_path, 'r') as f:
                    universal_metadata = json.load(f)
                
                universal_model_configs = universal_metadata.get('model_configs', {})
                if universal_model_configs:
                    self.model_configs = universal_model_configs
                    logger.info(f"Loaded universal model configurations for {len(self.model_configs)} models")
                    
                    # Log configurations for each model
                    for model_name, config in self.model_configs.items():
                        logger.debug(f"{model_name} config: {config}")
                    return
                else:
                    logger.warning("Universal metadata exists but no model_configs found")
            
            # Fall back to existing logic - load from latest training metadata
            models_dir = Path('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models')
            latest_link = models_dir / 'latest'
            
            if latest_link.exists() and latest_link.is_symlink():
                metadata_file = latest_link / 'training_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_configs = metadata.get('model_configs', {})
                    logger.info(f"Loaded model configurations for {len(self.model_configs)} models from latest training metadata")
                    
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
        # Use 2D aggregated features for statistical models
        default_feature_count = 150  # Expected feature count from training metadata
        self.model_configs = {
            'xgboost': {'feature_count': default_feature_count},
            'random_forest': {'feature_count': default_feature_count},
            'svm': {'feature_count': default_feature_count},
            'ensemble': {'feature_count': default_feature_count}
        }
        logger.info(f"Using default model configurations with {default_feature_count} features for statistical models")
    
    async def load_feature_selection_results(self) -> bool:
        """Load feature selection results from universal metadata"""
        try:
            # Load directly from universal metadata file (no longer using latest symlink)
            universal_metadata_file = Path('/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal/universal_metadata.json')
            
            if universal_metadata_file.exists():
                logger.info("Loading feature selection from universal metadata...")
                with open(universal_metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load selected features from feature_selection section
                if 'feature_selection' in metadata:
                    feature_selection = metadata['feature_selection']
                    
                    # Load both selected_features and selected_feature_columns
                    if 'selected_features' in feature_selection:
                        self.selected_features = feature_selection['selected_features']
                        self.feature_selection_metadata = feature_selection
                        logger.info(f"✓ Loaded {len(self.selected_features)} selected features from universal metadata")
                        logger.info(f"✓ Selected features: {self.selected_features[:5]}... (showing first 5)")
                        
                        # Also load selected_feature_columns if available
                        if 'selected_feature_columns' in feature_selection:
                            self.selected_feature_columns = feature_selection['selected_feature_columns']
                            logger.info(f"✓ FEATURE_SELECTION_LOADED: {len(self.selected_feature_columns)} selected feature columns from universal_metadata.json")
                            logger.info(f"✓ FEATURE_COLUMNS_PREVIEW: {self.selected_feature_columns[:10]}... (showing first 10 of {len(self.selected_feature_columns)})")
                            logger.info(f"✓ FEATURE_COLUMNS_VALIDATION: Expected 51 features, loaded {len(self.selected_feature_columns)} features")
                        
                        # Load selected_feature_indices for generic feature filtering
                        if 'selected_feature_indices' in feature_selection:
                            self.selected_feature_indices = feature_selection['selected_feature_indices']
                            logger.info(f"✓ Also loaded {len(self.selected_feature_indices)} selected feature indices")
                        
                        return True
                    else:
                        logger.warning("No 'selected_features' found in feature_selection section")
                else:
                    logger.warning("No 'feature_selection' section found in universal metadata")
            else:
                logger.warning(f"Universal metadata file not found: {universal_metadata_file}")
            
            # Fallback: Try ensemble config if universal metadata doesn't have feature selection
            ensemble_config_file = Path(os.path.dirname(os.path.dirname(__file__))) / 'models' / 'universal' / 'base_models' / 'ensemble_base_ensemble' / 'ensemble_config.json'
            
            if ensemble_config_file.exists():
                logger.info("Trying ensemble config as fallback...")
                with open(ensemble_config_file, 'r') as f:
                    config = json.load(f)
                
                if 'selected_feature_columns' in config:
                    self.selected_features = config['selected_feature_columns']
                    self.selected_feature_columns = config['selected_feature_columns']
                    self.feature_selection_metadata = {
                        'selected_features': config['selected_feature_columns'],
                        'selected_feature_columns': config['selected_feature_columns'],
                        'selected_feature_count': config.get('selected_feature_count', len(config['selected_feature_columns']))
                    }
                    logger.info(f"✓ Loaded {len(self.selected_features)} selected features from ensemble config")
                    logger.info(f"✓ Selected features: {self.selected_features[:5]}... (showing first 5)")
                    return True
                else:
                    logger.warning("No 'selected_feature_columns' found in ensemble config")
            else:
                logger.warning(f"Ensemble config file not found: {ensemble_config_file}")
            
            logger.warning("No feature selection results found, using all features")
            return False
            
        except Exception as e:
            logger.error(f"Error loading feature selection results: {e}")
            return False
    
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
            # Try to initialize universal models first
            universal_success = False
            try:
                universal_success = await self.initialize_universal_models(symbols)
                if universal_success:
                    logger.info("Universal models initialized successfully")
                    self.is_universal_mode = True
                else:
                    logger.warning("Universal model initialization returned False, falling back to symbol-specific models")
                    self.is_universal_mode = False
            except Exception as universal_error:
                logger.error(f"Error during universal model initialization: {universal_error}")
                logger.info("Falling back to symbol-specific models due to universal model error")
                self.is_universal_mode = False
                universal_success = False
            
            # Load feature selection results after model initialization
            try:
                await self.load_feature_selection_results()
                if self.selected_features:
                    logger.info(f"Loaded {len(self.selected_features)} selected features for signal generation")
                else:
                    logger.info("No feature selection applied - using all available features")
            except Exception as fs_error:
                logger.warning(f"Failed to load feature selection results: {fs_error}")
                logger.info("Continuing without feature selection - using all available features")
            
            for symbol in symbols:
                logger.info(f"Initializing models for {symbol}")
                
                # Initialize ensemble weights (equal weights initially)
                self.ensemble_weights[symbol] = {
                    ModelType.XGBOOST: 0.33,
                    ModelType.RANDOM_FOREST: 0.33,
                    ModelType.SVM: 0.34
                }
                
                # Load or create symbol-specific models (as fallback or primary)
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
                        'last_updated': datetime.now(timezone.utc)
                    }
            
            logger.info(f"Models initialized for {len(symbols)} symbols (Universal mode: {self.is_universal_mode})")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    async def _load_or_create_models(self, symbol: str) -> None:
        """Load existing statistical models using joblib"""
        try:
            self.models[symbol] = {}
            
            # Load statistical models from joblib files
            from pathlib import Path
            import joblib
            
            try:
                # Load universal statistical models from ensemble directory
                universal_models_dir = Path(os.path.dirname(os.path.dirname(__file__))) / 'models' / 'universal' / 'base_models' / 'ensemble_base_ensemble'
                
                # Map statistical model types to file names
                model_mapping = {
                    ModelType.XGBOOST: 'xgboost.joblib',
                    ModelType.RANDOM_FOREST: 'random_forest.joblib', 
                    ModelType.SVM: 'svm.joblib',
                    ModelType.ENSEMBLE: 'ensemble_config.json'  # Ensemble uses config file, not joblib
                }
                
                # Load each statistical model
                for model_type, filename in model_mapping.items():
                    model_path = universal_models_dir / filename
                    if model_path.exists():
                        try:
                            if model_type == ModelType.ENSEMBLE:
                                # Load ensemble config as JSON
                                with open(model_path, 'r') as f:
                                    model = json.load(f)
                            else:
                                # Load statistical models with joblib
                                model = joblib.load(model_path)
                            self.models[symbol][model_type] = model
                            logger.info(f"✓ Loaded {model_type.value} model for {symbol} from {model_path}")
                        except Exception as load_error:
                            logger.error(f"Failed to load {model_type.value} model: {load_error}")
                            self.models[symbol][model_type] = None
                    else:
                        logger.warning(f"Statistical model file not found: {model_path}")
                        self.models[symbol][model_type] = None
                
                # Load scalers from joblib file if available
                scaler_path = universal_models_dir / 'scalers.joblib'
                if scaler_path.exists():
                    try:
                        scalers = joblib.load(scaler_path)
                        if isinstance(scalers, dict):
                            # If multiple scalers, use the first one or a default key
                            scaler_key = list(scalers.keys())[0] if scalers else None
                            if scaler_key:
                                self.scalers[symbol] = scalers[scaler_key]
                            else:
                                self.scalers[symbol] = StandardScaler()
                        else:
                            # Single scaler
                            self.scalers[symbol] = scalers
                        logger.info(f"✓ Loaded scalers for {symbol} from {scaler_path}")
                    except Exception as scaler_error:
                        logger.error(f"Failed to load scalers: {scaler_error}")
                        self.scalers[symbol] = StandardScaler()
                        logger.info(f"Created default scaler for {symbol}")
                else:
                    # Create default scaler if none available
                    self.scalers[symbol] = StandardScaler()
                    logger.info(f"Created default scaler for {symbol}")
                
                logger.info(f"Successfully configured statistical models for {symbol}")
                
            except Exception as model_error:
                logger.error(f"Failed to load statistical models: {model_error}")
                logger.info(f"Initializing with None models for {symbol}")
                
                # Fallback: set models to None if loading fails
                for model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM, ModelType.ENSEMBLE]:
                    self.models[symbol][model_type] = None
                
                # Create default scaler
                self.scalers[symbol] = StandardScaler()
                    
        except Exception as e:
            logger.error(f"Error loading/creating models for {symbol}: {e}")
    

    
    async def _create_model(self, model_type: ModelType, symbol: str, feature_count: int = None) -> Any:
        """Load statistical model from joblib file"""
        try:
            # Statistical models are loaded from .joblib files, not created dynamically
            # This method now serves as a placeholder for model loading
            logger.warning(f"Statistical models should be loaded from .joblib files, not created dynamically for {model_type.value}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating {model_type.value} model: {e}")
            return None
    
    # Statistical model creation methods removed - models are now loaded from .joblib files
    
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
            
            # FEATURE COUNT DEBUG: Log DataFrame shape after conversion
            logger.info(f"[FEATURE_DEBUG] {symbol}: generate_signals_from_features - DataFrame after conversion: shape {features_df.shape}, columns: {len(features_df.columns)}")
            logger.info(f"[FEATURE_DEBUG] {symbol}: DataFrame columns sample: {list(features_df.columns)[:10]}...")  # Show first 10 columns
            
            # FEATURE SELECTION DEBUG: Check if we have selected_feature_columns loaded
            if hasattr(self, 'selected_feature_columns') and self.selected_feature_columns:
                logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ selected_feature_columns loaded: {len(self.selected_feature_columns)} features")
                logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Selected features sample: {self.selected_feature_columns[:5]}...")
            else:
                logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  NO selected_feature_columns loaded - will use all features!")
            
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
        """Generate ensemble prediction for a symbol using universal models when available"""
        try:
            # Try universal models first if available
            if self.is_universal_mode and self.universal_models:
                logger.debug(f"Using universal models for {symbol}")
                
                # Use market data directly for universal models
                universal_prediction = await self._generate_universal_prediction(symbol, data)
                if universal_prediction is not None:
                    logger.info(f"✓ Generated universal prediction for {symbol}: {universal_prediction.final_prediction:.4f} (confidence: {universal_prediction.confidence:.4f})")
                    return universal_prediction
                
                logger.warning(f"Universal prediction failed for {symbol}, falling back to symbol-specific models")
            
            # Fallback to symbol-specific models
            if symbol not in self.models or not self.models[symbol]:
                logger.warning(f"No models available for {symbol}")
                return None
            
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
                timestamp=datetime.now(timezone.utc)
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
            
            # FEATURE COUNT DEBUG: Log feature count being passed to model
            logger.info(f"[FEATURE_DEBUG] {symbol}: _get_model_prediction - {model_type.value} model receiving features with shape {features.shape}, actual_feature_count: {actual_feature_count}, requested_feature_count: {feature_count}")
            
            # FEATURE SELECTION DEBUG: Log if feature selection was applied
            if hasattr(self, 'selected_feature_columns') and self.selected_feature_columns:
                logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Model prediction using {actual_feature_count} features (selected from {len(self.selected_feature_columns)} selected_feature_columns)")
            else:
                logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  Model prediction using {actual_feature_count} features (NO FEATURE SELECTION APPLIED)")
            
            # Statistical models expect 2D input (n_samples, n_features)
            # All statistical models are universal by design and work with 2D aggregated features
            logger.info(f"[STATISTICAL_DEBUG] Processing {model_type.value} statistical model with 2D features")
            
            # Ensure features are in 2D format for statistical models
            if len(features.shape) > 2:
                # If features are 3D, take the last time step (most recent)
                features_2d = features[-1] if len(features.shape) == 3 else features.reshape(-1)
            elif len(features.shape) == 1:
                # If features are 1D, reshape to 2D
                features_2d = features.reshape(1, -1)
            else:
                # Features are already 2D, use as is
                features_2d = features
            
            logger.info(f"[STATISTICAL_DEBUG] {model_type.value} input features shape: {features_2d.shape}")
            
            # Statistical models handle predictions differently based on type
            if model_type == ModelType.XGBOOST:
                # XGBoost expects 2D input (n_samples, n_features)
                prediction = model.predict(features_2d)[0]
                
                # Get raw probability prediction
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(features_2d)[0]
                        raw_confidence = proba[1]  # Probability of positive class
                        
                        # Apply calibration if available
                        if hasattr(model, 'confidence_calibrator'):
                            calibrated_confidence = model.confidence_calibrator.predict_proba(
                                np.array([[raw_confidence]])
                            )[0, 1]
                            confidence = calibrated_confidence
                            logger.debug(f"[CALIBRATION] {model_type.value} raw: {raw_confidence:.4f}, calibrated: {confidence:.4f}")
                        else:
                            confidence = raw_confidence
                    except Exception as e:
                        logger.warning(f"Error getting XGBoost probability: {e}")
                        confidence = 0.6 + abs(prediction) * 0.3
                else:
                    confidence = 0.6 + abs(prediction) * 0.3
                
            elif model_type == ModelType.RANDOM_FOREST:
                # Random Forest expects 2D input (n_samples, n_features)
                prediction = model.predict(features_2d)[0]
                
                # Get raw probability prediction
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(features_2d)[0]
                        raw_confidence = proba[1]  # Probability of positive class
                        
                        # Apply calibration if available
                        if hasattr(model, 'confidence_calibrator'):
                            calibrated_confidence = model.confidence_calibrator.predict_proba(
                                np.array([[raw_confidence]])
                            )[0, 1]
                            confidence = calibrated_confidence
                            logger.debug(f"[CALIBRATION] {model_type.value} raw: {raw_confidence:.4f}, calibrated: {confidence:.4f}")
                        else:
                            confidence = raw_confidence
                    except Exception as e:
                        logger.warning(f"Error getting Random Forest probability: {e}")
                        confidence = 0.55 + abs(prediction) * 0.35
                else:
                    confidence = 0.55 + abs(prediction) * 0.35
                
            elif model_type == ModelType.SVM:
                # SVM expects 2D input (n_samples, n_features)
                prediction = model.predict(features_2d)[0]
                
                # Get raw probability prediction
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(features_2d)[0]
                        raw_confidence = proba[1]  # Probability of positive class
                        
                        # Apply calibration if available
                        if hasattr(model, 'confidence_calibrator'):
                            calibrated_confidence = model.confidence_calibrator.predict_proba(
                                np.array([[raw_confidence]])
                            )[0, 1]
                            confidence = calibrated_confidence
                            logger.debug(f"[CALIBRATION] {model_type.value} raw: {raw_confidence:.4f}, calibrated: {confidence:.4f}")
                        else:
                            confidence = raw_confidence
                    except Exception as e:
                        logger.warning(f"Error getting SVM probability: {e}")
                        # Fallback to decision function
                        if hasattr(model, 'decision_function'):
                            try:
                                decision_score = model.decision_function(features_2d)[0]
                                confidence = min(0.8, 0.5 + abs(decision_score) * 0.1)
                            except:
                                confidence = 0.5 + abs(prediction) * 0.3
                        else:
                            confidence = 0.5 + abs(prediction) * 0.3
                else:
                    confidence = 0.5 + abs(prediction) * 0.3
                
            elif model_type == ModelType.ENSEMBLE:
                # Ensemble model combines multiple statistical models
                if isinstance(model, dict) and 'models' in model:
                    # Handle ensemble dictionary format
                    models = model['models']
                    weights = model['weights']
                    
                    # Get predictions from individual models
                    xgb_pred = models['xgboost'].predict_proba(features_2d)[0, 1] if 'xgboost' in models else 0.5
                    rf_pred = models['random_forest'].predict_proba(features_2d)[0, 1] if 'random_forest' in models else 0.5
                    svm_pred = models['svm'].predict_proba(features_2d)[0, 1] if 'svm' in models else 0.5
                    
                    # Calculate weighted ensemble prediction
                    prediction = (weights.get('xgboost', 0.33) * xgb_pred + 
                                weights.get('random_forest', 0.33) * rf_pred + 
                                weights.get('svm', 0.34) * svm_pred)
                    
                    # Apply ensemble calibration if available
                    if 'confidence_calibrator' in model:
                        try:
                            calibrated_confidence = model['confidence_calibrator'].predict(np.array([prediction]))[0]
                            confidence = calibrated_confidence
                            logger.debug(f"[CALIBRATION] {model_type.value} raw: {prediction:.4f}, calibrated: {confidence:.4f}")
                        except Exception as e:
                            logger.warning(f"Error applying ensemble calibration: {e}")
                            confidence = prediction
                    else:
                        confidence = prediction
                else:
                    # Fallback for other ensemble formats
                    prediction = model.predict(features_2d)[0]
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(features_2d)[0]
                            confidence = max(proba) * 0.95
                        except:
                            confidence = 0.65 + abs(prediction) * 0.25
                    else:
                        confidence = 0.65 + abs(prediction) * 0.25
                
            else:
                # Fallback for unknown model types
                logger.warning(f"Unknown statistical model type: {model_type.value}")
                prediction = 0.0
                confidence = 0.3
                
            # Ensure confidence is within valid range
            confidence = max(0.2, min(0.9, confidence))
            
            logger.info(f"[STATISTICAL_DEBUG] {model_type.value} prediction: {prediction:.4f}, confidence: {confidence:.4f}")
            
            # Calculate probability (sigmoid of prediction)
            probability = 1 / (1 + np.exp(-prediction * 5))  # Scale prediction for sigmoid
            
            return ModelPrediction(
                model_type=model_type,
                symbol=symbol,
                prediction=float(prediction),
                confidence=float(confidence),
                probability=float(probability),
                features_used=list(range(features.shape[-1])),  # Feature indices
                timestamp=datetime.now(timezone.utc),
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
            
            # CRITICAL DEBUG: Log exact column names for MSFT to identify the 7 columns
            if symbol == "MSFT":
                logger.error(f"[CRITICAL_DEBUG] MSFT: Exact column names ({len(df.columns)} total): {list(df.columns)}")

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

            # FEATURE COUNT DEBUG: Log final feature count after DataFrame processing
            logger.info(f"[FEATURE_DEBUG] {symbol}: _convert_cached_features_to_dataframe - Final DataFrame has {len(df.columns)} features")
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
                    timestamp=datetime.now(timezone.utc)
                )
                
                self.regime_history.append(self.current_market_regime)
                logger.debug(f"Updated market regime from cached features: {regime_type} (volatility: {volatility:.3f})")
            
        except Exception as e:
            logger.error(f"Error updating market regime from cached features: {e}")
    
    async def _prepare_features(self, symbol: str, data: pd.DataFrame, model_type: ModelType = None, feature_count: int = None) -> Optional[np.ndarray]:
        """Prepare 2D aggregated features for statistical model prediction"""
        try:
            # Statistical models use 2D aggregated features, not sequential data
            # Use the most recent data point for feature extraction
            required_periods = 1
            min_periods = 1
            
            # Statistical models only need the most recent data point
            if len(data) < min_periods:
                logger.warning(f"Insufficient data for {symbol} and {model_type}: {len(data)} < {min_periods}")
                return None
            
            # Use the most recent data point for statistical models
            recent_data = data.tail(required_periods).copy()
            
            # Exclude non-feature columns and ensure all columns are numeric
            exclude_columns = {'timestamp'}
            
            # Get all feature columns (everything except excluded columns)
            all_feature_columns = [col for col in recent_data.columns if col not in exclude_columns]
            
            # Apply feature selection if selected features are available
            if self.selected_feature_columns:
                # Check if we're dealing with generic feature names (feature_0, feature_1, etc.)
                has_generic_features = any(col.startswith('feature_') and col.split('_')[1].isdigit() 
                                         for col in all_feature_columns if '_' in col)
                
                if has_generic_features and hasattr(self, 'selected_feature_indices') and self.selected_feature_indices:
                    # Use feature indices for generic feature names
                    logger.info(f"[FEATURE_DEBUG] {symbol}: Detected generic feature names, using selected_feature_indices for filtering")
                    feature_columns = []
                    for idx in self.selected_feature_indices:
                        if idx < len(all_feature_columns):
                            feature_columns.append(all_feature_columns[idx])
                        else:
                            logger.warning(f"Feature index {idx} exceeds available features ({len(all_feature_columns)})")
                    
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ FEATURE_SELECTION_BY_INDICES - Using {len(feature_columns)} selected features out of {len(all_feature_columns)} available")
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Selected feature indices: {len(self.selected_feature_indices)} total")
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Feature filtering: {len(all_feature_columns)} -> {len(feature_columns)} features")
                else:
                    # Use feature column names for human-readable feature names
                    feature_columns = [col for col in self.selected_feature_columns if col in all_feature_columns]
                    if len(feature_columns) < len(self.selected_feature_columns):
                        missing_features = set(self.selected_feature_columns) - set(all_feature_columns)
                        logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  MISSING_SELECTED_FEATURES: {len(missing_features)} features not found in generated data")
                        logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  Missing features: {list(missing_features)[:10]}... (showing first 10)")
                        logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Available features that match selection: {len(feature_columns)}/{len(self.selected_feature_columns)}")
                    else:
                        logger.info(f"[FEATURE_DEBUG] {symbol}: ✅ ALL_SELECTED_FEATURES_FOUND: {len(feature_columns)}/{len(self.selected_feature_columns)} selected features available")
                    
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ FEATURE_SELECTION_BY_NAMES - Using {len(feature_columns)} selected features out of {len(all_feature_columns)} available")
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Selected feature columns loaded: {len(self.selected_feature_columns)} total")
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Feature filtering: {len(all_feature_columns)} -> {len(feature_columns)} features")
            else:
                # Use all available features if no feature selection is applied
                feature_columns = all_feature_columns
                logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  NO_FEATURE_SELECTION - selected_feature_columns is None/empty, using all {len(feature_columns)} features")
                logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  This may cause model prediction failures due to feature dimension mismatch!")
            
            # FEATURE COUNT DEBUG: Log initial feature count
            logger.info(f"[FEATURE_DEBUG] {symbol}: _prepare_features - Input data has {len(recent_data.columns)} total columns, {len(feature_columns)} feature columns")
            
            if not feature_columns:
                logger.error(f"No feature columns found for {symbol}")
                return None
            
            # Ensure all feature columns are numeric
            numeric_data = recent_data[feature_columns].copy()
            for col in feature_columns:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            # Drop columns that are all NaN (failed numeric conversion)
            columns_before_nan_drop = len(numeric_data.columns)
            numeric_data = numeric_data.dropna(axis=1, how='all')
            
            # FEATURE COUNT DEBUG: Log after NaN column removal
            logger.info(f"[FEATURE_DEBUG] {symbol}: _prepare_features - After NaN removal: {len(numeric_data.columns)} features (dropped {columns_before_nan_drop - len(numeric_data.columns)} NaN columns)")
            
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
            
            # FEATURE COUNT DEBUG: Log before feature selection
            logger.info(f"[FEATURE_DEBUG] {symbol}: _prepare_features - Before feature selection: {features_array.shape[1]} features available, requested: {feature_count}")
            
            # Apply feature selection if feature_count is specified and less than available features
            if feature_count and feature_count < features_array.shape[1]:
                logger.debug(f"Selecting top {feature_count} features from {features_array.shape[1]} available for {symbol}")
                # Use the first feature_count features (most important ones should be first)
                # This assumes features are ordered by importance from the feature engineering pipeline
                features_array = features_array[:, :feature_count]
                # Update the column names accordingly
                numeric_data = numeric_data.iloc[:, :feature_count]
                logger.debug(f"Feature selection applied: {features_array.shape[1]} features selected")
                
                # FEATURE COUNT DEBUG: Log after feature selection
                logger.info(f"[FEATURE_DEBUG] {symbol}: _prepare_features - After feature selection: {features_array.shape[1]} features")
            else:
                # FEATURE COUNT DEBUG: Log when no feature selection is applied
                logger.info(f"[FEATURE_DEBUG] {symbol}: _prepare_features - No feature selection applied, using all {features_array.shape[1]} features")
            
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
                timestamp=datetime.now(timezone.utc)
            )
            
            self.regime_history.append(self.current_market_regime)
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    async def _prediction_to_signal(self, ensemble_pred: EnsemblePrediction, market_data: pd.DataFrame = None, current_positions: Dict = None) -> Optional[TradeSignal]:
        """Convert ensemble prediction to trading signal - sell signals only for closing existing long positions"""
        try:
            # Apply risk filters
            if not await self._apply_risk_filters(ensemble_pred):
                return None
            
            prediction = ensemble_pred.final_prediction
            confidence = ensemble_pred.confidence
            symbol = ensemble_pred.symbol
            
            # Check if we have an existing long position for this symbol
            has_long_position = False
            if current_positions and symbol in current_positions:
                position = current_positions[symbol]
                has_long_position = hasattr(position, 'quantity') and position.quantity > 0
            
            # Check for forced sell conditions first (market-based and time-based)
            # Only check if we have a long position to close
            force_sell = False
            force_sell_reason = ""
            
            if has_long_position and market_data is not None:
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
                logger.info(f"Threshold checking for {symbol}: prediction={prediction:.4f}, "
                           f"buy_threshold={self.signal_thresholds['buy_threshold']}, "
                           f"sell_threshold={self.signal_thresholds['sell_threshold']}, "
                           f"strong_buy_threshold={self.signal_thresholds['strong_buy_threshold']}, "
                           f"strong_sell_threshold={self.signal_thresholds['strong_sell_threshold']}")
                
                if prediction >= self.signal_thresholds['strong_buy_threshold']:
                    action = SignalType.BUY.value
                    signal_strength = "strong"
                    logger.info(f"Signal decision for {symbol}: STRONG BUY (prediction {prediction:.4f} >= {self.signal_thresholds['strong_buy_threshold']})")
                elif prediction >= self.signal_thresholds['buy_threshold']:
                    action = SignalType.BUY.value
                    signal_strength = "moderate"
                    logger.info(f"Signal decision for {symbol}: MODERATE BUY (prediction {prediction:.4f} >= {self.signal_thresholds['buy_threshold']})")
                elif prediction <= self.signal_thresholds['strong_sell_threshold'] and has_long_position:
                    action = SignalType.SELL.value
                    signal_strength = "strong"
                    logger.info(f"Signal decision for {symbol}: STRONG SELL (prediction {prediction:.4f} <= {self.signal_thresholds['strong_sell_threshold']}) - closing long position")
                elif prediction <= self.signal_thresholds['sell_threshold'] and has_long_position:
                    action = SignalType.SELL.value
                    signal_strength = "moderate"
                    logger.info(f"Signal decision for {symbol}: MODERATE SELL (prediction {prediction:.4f} <= {self.signal_thresholds['sell_threshold']}) - closing long position")
                elif (prediction <= self.signal_thresholds['strong_sell_threshold'] or prediction <= self.signal_thresholds['sell_threshold']) and not has_long_position:
                    # Would be a sell signal but no position to close
                    logger.info(f"Skipping SELL signal for {symbol} - no existing long position to close (prediction {prediction:.4f})")
                    return None
                else:
                    action = SignalType.HOLD.value
                    signal_strength = "weak"
                    logger.info(f"Signal decision for {symbol}: HOLD (prediction {prediction:.4f} between thresholds)")
                
                # Skip weak signals
                if action == SignalType.HOLD.value:
                    logger.info(f"Skipping HOLD signal for {symbol} (weak signal filtered out)")
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
                timestamp=datetime.now(timezone.utc),
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
            logger.info(f"Applying risk filters for {ensemble_pred.symbol}: "
                       f"confidence={ensemble_pred.confidence:.3f}, "
                       f"risk_score={ensemble_pred.risk_score:.3f}, "
                       f"prediction={ensemble_pred.final_prediction:.4f}")
            
            # Minimum confidence filter
            if ensemble_pred.confidence < self.risk_filters['min_confidence']:
                logger.info(f"❌ Signal filtered: low confidence {ensemble_pred.confidence:.3f} < {self.risk_filters['min_confidence']} for {ensemble_pred.symbol}")
                return False
            
            # Maximum risk score filter
            if ensemble_pred.risk_score > self.risk_filters['max_risk_score']:
                logger.info(f"❌ Signal filtered: high risk {ensemble_pred.risk_score:.3f} > {self.risk_filters['max_risk_score']} for {ensemble_pred.symbol}")
                return False
            
            # Market regime filter
            if self.current_market_regime and self.current_market_regime.market_stress > 0.8:
                logger.info(f"❌ Signal filtered: high market stress {self.current_market_regime.market_stress:.2f} for {ensemble_pred.symbol}")
                return False
            
            # Model agreement filter (check if models agree)
            predictions = [pred.prediction for pred in ensemble_pred.individual_predictions]
            if len(predictions) > 1:
                prediction_std = np.std(predictions)
                if prediction_std > 0.5:  # High disagreement
                    logger.info(f"❌ Signal filtered: model disagreement (std={prediction_std:.3f}) for {ensemble_pred.symbol}")
                    return False
            
            logger.info(f"✅ Risk filters passed for {ensemble_pred.symbol}")
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
                    
                    perf['last_updated'] = datetime.now(timezone.utc)
                    
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
    
    def _serialize_for_json(self, obj) -> dict:
        """Custom serialization function that handles ModelType enums and other complex objects"""
        if hasattr(obj, '__dict__'):
            # Handle dataclass objects
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, ModelType):
                    result[key] = value.value
                elif isinstance(value, datetime):
                    result[key] = value.isoformat()
                elif isinstance(value, list):
                    result[key] = [self._serialize_for_json(item) if hasattr(item, '__dict__') else item for item in value]
                elif isinstance(value, dict):
                    # Convert both keys and values if they are ModelType enums
                    # Also handle nested dictionaries recursively
                    result[key] = self._serialize_dict(value)
                elif hasattr(value, '__dict__'):
                    result[key] = self._serialize_for_json(value)
                else:
                    result[key] = value
            return result
        else:
            return obj
    
    def _serialize_dict(self, d: dict) -> dict:
        """Recursively serialize dictionary with ModelType enum handling"""
        result = {}
        for k, v in d.items():
            # Convert ModelType keys to strings
            key = k.value if isinstance(k, ModelType) else k
            
            # Handle different value types
            if isinstance(v, ModelType):
                result[key] = v.value
            elif isinstance(v, dict):
                result[key] = self._serialize_dict(v)
            elif isinstance(v, list):
                result[key] = [self._serialize_for_json(item) if hasattr(item, '__dict__') else item for item in v]
            elif hasattr(v, '__dict__'):
                result[key] = self._serialize_for_json(v)
            else:
                result[key] = v
        return result
    
    async def _log_signal(self, signal: TradeSignal, ensemble_pred: EnsemblePrediction) -> None:
        """Log signal details"""
        try:
            # Convert ModelType enum keys to string values for JSON serialization
            ensemble_weights = self.ensemble_weights.get(signal.symbol, {})
            serializable_weights = {model_type.value: weight for model_type, weight in ensemble_weights.items()}
            
            signal_log = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal': self._serialize_for_json(signal),
                'ensemble_prediction': self._serialize_for_json(ensemble_pred),
                'market_regime': self._serialize_for_json(self.current_market_regime) if self.current_market_regime else None,
                'ensemble_weights': serializable_weights
            }
            
            # Save to file
            filename = f"logs/signals/signal_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{signal.symbol}.json"
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
            
            # Convert ModelType keys to strings for JSON serialization
            ensemble_weights = self.ensemble_weights.get(symbol, {})
            serializable_weights = {model_type.value: weight for model_type, weight in ensemble_weights.items()}
            
            return {
                'symbol': symbol,
                'models': performance_data,
                'ensemble_weights': serializable_weights,
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
        """Save trained statistical models for a symbol"""
        try:
            if symbol not in self.models:
                return False
            
            import joblib
            from pathlib import Path
            
            for model_type, model in self.models[symbol].items():
                if model is None:
                    continue
                    
                # Create model directory if it doesn't exist
                model_dir = Path(f"models/{model_type.value}")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = model_dir / f"{symbol}_model.joblib"
                
                try:
                    # Save all statistical models as .joblib files
                    joblib.dump(model, model_path)
                    logger.info(f"✓ Saved {model_type.value} model for {symbol} as .joblib file")
                except Exception as e:
                    logger.error(f"Error saving {model_type.value} model for {symbol}: {e}")
            
            # Save scaler using joblib
            if symbol in self.scalers and self.scalers[symbol] is not None:
                scaler_dir = Path("models/scalers")
                scaler_dir.mkdir(parents=True, exist_ok=True)
                scaler_path = scaler_dir / f"{symbol}_scaler.joblib"
                
                try:
                    joblib.dump(self.scalers[symbol], scaler_path)
                    logger.info(f"✓ Saved scaler for {symbol} as .joblib file")
                except Exception as e:
                    logger.error(f"Error saving scaler for {symbol}: {e}")
            
            # Save model metadata
            metadata_dir = Path("models/metadata")
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"{symbol}_metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'ensemble_weights': {k.value: v for k, v in self.ensemble_weights.get(symbol, {}).items()},
                    'performance': {k.value: v for k, v in self.model_performance.get(symbol, {}).items()},
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models for {symbol}: {e}")
            return False
    
    async def initialize_universal_models(self, symbols: List[str] = None) -> bool:
        """Initialize universal models for cross-symbol prediction"""
        try:
            logger.info("Initializing universal models...")
            
            # Check if UniversalTrainer is available
            if not self.model_trainer:
                logger.warning("UniversalTrainer not available, cannot initialize universal models")
                return False
            
            # Initialize universal components - ensure consistent data_pipeline usage
            # Use the same data_pipeline reference as the rest of the SignalGenerator
            data_pipeline = self.data_pipeline or (self.model_trainer.data_pipeline if self.model_trainer else None)
            
            if not data_pipeline:
                logger.error("No data pipeline available for universal feature engineering")
                return False
                
            logger.info(f"Using data_pipeline for universal features: {type(data_pipeline).__name__}")
            feature_engineering = UniversalFeatureEngineering(self.supabase_client, data_pipeline)
            
            # Set symbol mappings for embedding lookup
            if symbols:
                symbol_mappings = {symbol: idx for idx, symbol in enumerate(symbols)}
                feature_engineering.set_symbol_mappings(symbol_mappings)
                logger.info(f"Set symbol mappings for {len(symbols)} symbols: {list(symbols)}")
            
            self.universal_trainer = UniversalTrainer(
                data_pipeline=data_pipeline,
                feature_engineering=feature_engineering
            )
                
            # Initialize UniversalModelArchitectures with number of symbols
            num_symbols = len(symbols) if symbols else 10  # Default to 10 if no symbols provided
            self.universal_architectures = UniversalModelArchitectures(num_symbols=num_symbols)
            self.universal_feature_engineering = feature_engineering
            
            # Load universal models using UniversalTrainer's method
            universal_models_loaded = await self._load_universal_models()
            
            if universal_models_loaded:
                self.is_universal_mode = True
                logger.info("✓ Universal models initialized and loaded successfully")
                return True
            else:
                logger.warning("Universal models not found - falling back to symbol-specific models")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing universal models: {e}")
            return False
    
    async def _load_universal_models(self) -> bool:
        """Load trained universal models using UniversalTrainer's universal loading method"""
        try:
            if not self.model_trainer:
                logger.error("UniversalTrainer not available for loading universal models")
                return False
            
            # Use UniversalTrainer's load_universal_models method
            from pathlib import Path
            universal_dir = Path("/Users/anthonyxiao/Dev/hunter_x_hunter_day_trade_v3/backend/models/universal")
            
            # Check if universal directory exists
            if not universal_dir.exists():
                logger.warning(f"Universal models directory not found: {universal_dir}")
                return False
            
            # Load universal models using UniversalTrainer
            success = await self.model_trainer.load_universal_models(universal_dir)
            
            if success:
                # Get the loaded models from UniversalTrainer directly
                if self.model_trainer:
                    universal_trainer = self.model_trainer
                    
                    # Copy base models to our universal_models dict with proper ModelType enum keys
                    if hasattr(universal_trainer, 'base_models'):
                        for model_type_str, model in universal_trainer.base_models.items():
                            try:
                                # Convert string key to ModelType enum
                                model_type_enum = ModelType(model_type_str)
                                self.universal_models[model_type_enum] = model
                                logger.info(f"✓ Loaded universal base {model_type_str} model (type: {type(model).__name__})")
                                
                                # Validate that the model has predict method
                                if hasattr(model, 'predict'):
                                    logger.debug(f"✓ Model {model_type_str} has predict method")
                                else:
                                    logger.warning(f"⚠ Model {model_type_str} missing predict method")
                                    
                            except ValueError as e:
                                logger.error(f"✗ Failed to convert model type '{model_type_str}' to ModelType enum: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"✗ Error loading model {model_type_str}: {e}")
                                continue
                    
                    # Store symbol-specific models if available
                    if hasattr(universal_trainer, 'symbol_models'):
                        self.universal_symbol_models = universal_trainer.symbol_models
                        logger.info(f"✓ Loaded symbol-specific models for {len(self.universal_symbol_models)} model types")
                    
                    # Load universal metadata
                    metadata_path = universal_dir / 'universal_metadata.json'
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            self.universal_metadata = json.load(f)
                        logger.info("✓ Loaded universal metadata")
                        
                        # Update ensemble weights from universal metadata
                        if 'ensemble_weights' in self.universal_metadata:
                            universal_weights = self.universal_metadata['ensemble_weights']
                            # Convert to ModelType enum keys
                            converted_weights = {}
                            for model_name, weight in universal_weights.items():
                                try:
                                    model_type = ModelType(model_name.lower())
                                    converted_weights[model_type] = weight
                                except ValueError:
                                    logger.warning(f"Unknown model type in universal weights: {model_name}")
                            
                            if converted_weights:
                                self.default_ensemble_weights = converted_weights
                                logger.info("✓ Updated ensemble weights from universal metadata")
                        
                        # Load selected features from universal metadata
                        if not await self.load_feature_selection_results():
                            logger.warning("⚠️  Failed to load feature selection results during universal model loading")
                        else:
                            logger.info(f"✓ Feature selection loaded: {len(self.selected_features) if self.selected_features else 0} features")
                    
                    logger.info(f"Successfully loaded {len(self.universal_models)} universal models")
                    return True
                else:
                    logger.error("UniversalTrainer not available after loading")
                    return False
            else:
                logger.warning("Failed to load universal models using UniversalTrainer")
                return False
                
        except Exception as e:
            logger.error(f"Error loading universal models: {e}")
            return False
    
    async def _prepare_universal_features(self, symbol: str, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare universal features for a symbol using the same process as training
        
        This method follows the same universal feature engineering process as
        _prepare_universal_features_for_symbol in universal_trainer.py to ensure
        consistent feature dimensions (446 features for 9 symbols) between training and live trading.
        """
        try:
            logger.info(f"[UNIVERSAL_FEATURES] Preparing universal features for {symbol} in live trading")
            
            # Get all trading symbols for universal feature engineering
            if self.data_pipeline:
                all_symbols = self.data_pipeline.get_ticker_universe()
            elif hasattr(self.universal_feature_engineering, '_symbol_mappings'):
                all_symbols = list(self.universal_feature_engineering._symbol_mappings.keys())
            else:
                # Fallback to current symbol only (this will cause feature mismatch)
                all_symbols = [symbol]
                logger.warning(f"[UNIVERSAL_FEATURES] Using fallback single symbol {symbol} - this may cause feature dimension mismatch")
            
            logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Using {len(all_symbols)} symbols for universal features: {all_symbols}")
            
            # Engineer universal features for all symbols (same as training)
            # Use training_mode=True to ensure sufficient historical data for all features
            logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Calling engineer_universal_features with training_mode=True")
            universal_features = await self.universal_feature_engineering.engineer_universal_features(
                symbols=all_symbols,
                start_date=market_data.index[0].to_pydatetime(),
                end_date=market_data.index[-1].to_pydatetime(),
                training_mode=True  # Use training mode to ensure full feature generation
            )
            
            # Get the individual symbol's features from universal features
            if symbol not in universal_features.symbol_features:
                logger.error(f"[UNIVERSAL_FEATURES] Symbol {symbol} not found in universal features")
                return None
            
            # Get individual symbol features
            symbol_feature_set = universal_features.symbol_features[symbol]
            
            # Combine individual symbol features (same as training)
            individual_feature_dfs = []
            individual_feature_counts = {}
            
            # Add technical features
            if hasattr(symbol_feature_set, 'technical_features') and symbol_feature_set.technical_features is not None and not symbol_feature_set.technical_features.empty:
                individual_feature_dfs.append(symbol_feature_set.technical_features)
                individual_feature_counts['technical'] = len(symbol_feature_set.technical_features.columns)
            
            # Add market microstructure features
            if hasattr(symbol_feature_set, 'market_microstructure') and symbol_feature_set.market_microstructure is not None and not symbol_feature_set.market_microstructure.empty:
                individual_feature_dfs.append(symbol_feature_set.market_microstructure)
                individual_feature_counts['market_microstructure'] = len(symbol_feature_set.market_microstructure.columns)
            
            # Add sentiment features
            if hasattr(symbol_feature_set, 'sentiment_features') and symbol_feature_set.sentiment_features is not None and not symbol_feature_set.sentiment_features.empty:
                individual_feature_dfs.append(symbol_feature_set.sentiment_features)
                individual_feature_counts['sentiment'] = len(symbol_feature_set.sentiment_features.columns)
            
            # Add macro features
            if hasattr(symbol_feature_set, 'macro_features') and symbol_feature_set.macro_features is not None and not symbol_feature_set.macro_features.empty:
                individual_feature_dfs.append(symbol_feature_set.macro_features)
                individual_feature_counts['macro'] = len(symbol_feature_set.macro_features.columns)
            
            # Add cross-asset features
            if hasattr(symbol_feature_set, 'cross_asset_features') and symbol_feature_set.cross_asset_features is not None and not symbol_feature_set.cross_asset_features.empty:
                individual_feature_dfs.append(symbol_feature_set.cross_asset_features)
                individual_feature_counts['cross_asset'] = len(symbol_feature_set.cross_asset_features.columns)
            
            # Add engineered features
            if hasattr(symbol_feature_set, 'engineered_features') and symbol_feature_set.engineered_features is not None and not symbol_feature_set.engineered_features.empty:
                individual_feature_dfs.append(symbol_feature_set.engineered_features)
                individual_feature_counts['engineered'] = len(symbol_feature_set.engineered_features.columns)
            
            if not individual_feature_dfs:
                logger.error(f"[UNIVERSAL_FEATURES] No valid individual feature DataFrames found for symbol {symbol}")
                return None
            
            # Combine individual features
            individual_df = pd.concat(individual_feature_dfs, axis=1)
            total_individual_features = sum(individual_feature_counts.values())
            logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Individual features: {total_individual_features} columns")
            logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Feature breakdown: {individual_feature_counts}")
            
            # Add cross-symbol features
            cross_symbol_df = universal_features.cross_symbol_features
            if not cross_symbol_df.empty:
                # Align cross-symbol features with individual features by index
                aligned_cross_symbol = cross_symbol_df.reindex(individual_df.index)
                individual_df = pd.concat([individual_df, aligned_cross_symbol], axis=1)
                logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Added {len(cross_symbol_df.columns)} cross-symbol features")
                
                # Log some cross-symbol feature names to verify they're being generated
                cross_symbol_feature_names = list(cross_symbol_df.columns)
                logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Cross-symbol features: {cross_symbol_feature_names[:10]}... (showing first 10)")
            else:
                logger.warning(f"[UNIVERSAL_FEATURES] {symbol}: ⚠️  NO CROSS-SYMBOL FEATURES GENERATED - this is the root cause of missing features!")
            
            # Add market regime features
            regime_df = universal_features.market_regime_features
            if not regime_df.empty:
                # Align regime features with individual features by index
                aligned_regime = regime_df.reindex(individual_df.index)
                individual_df = pd.concat([individual_df, aligned_regime], axis=1)
                logger.info(f"[UNIVERSAL_FEATURES] {symbol}: Added {len(regime_df.columns)} market regime features")
            
            # Add sector features
            sector_df = universal_features.sector_features
            if not sector_df.empty:
                # Align sector features with individual features by index
                aligned_sector = sector_df.reindex(individual_df.index)
                individual_df = pd.concat([individual_df, aligned_sector], axis=1)
                logger.info(f"[{symbol}] Added {len(sector_df.columns)} sector features")
            
            # Add universal embeddings
            embeddings_df = universal_features.universal_embeddings
            if not embeddings_df.empty:
                # Align embeddings with individual features by index
                aligned_embeddings = embeddings_df.reindex(individual_df.index)
                individual_df = pd.concat([individual_df, aligned_embeddings], axis=1)
                logger.info(f"[{symbol}] Added {len(embeddings_df.columns)} universal embedding features")
                
                # Add symbol_id for the current symbol
                symbol_mappings = universal_features.symbol_mappings
                if symbol in symbol_mappings:
                    symbol_id = symbol_mappings[symbol]
                    individual_df['symbol_id'] = symbol_id
                    logger.info(f"[{symbol}] Added symbol_id={symbol_id}")
                
                # Remove symbol embedding columns (same as training)
                symbol_embedding_cols = [col for col in individual_df.columns if (
                    col == 'symbol_id' or (
                        col.startswith('symbol_') and not any([
                            col.startswith('corr_'),
                            col.startswith('beta_'),
                            col.startswith('relative_strength_'),
                            col.startswith('market_dispersion_'),
                            col.startswith('market_volatility'),
                            col.startswith('vol_regime_'),
                            col.startswith('vol_trend'),
                            col.startswith('vol_correlation')
                        ])
                    )
                )]
                
                # Keep all features except symbol embedding columns and target
                all_feature_columns = [col for col in individual_df.columns if col not in symbol_embedding_cols and col != 'target']
                
                # Apply feature selection if selected feature columns are available
                if self.selected_feature_columns:
                    # Use only selected feature columns that are available in the data
                    feature_columns = [col for col in self.selected_feature_columns if col in all_feature_columns]
                    if len(feature_columns) < len(self.selected_feature_columns):
                        missing_features = set(self.selected_feature_columns) - set(all_feature_columns)
                        logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  MISSING_UNIVERSAL_FEATURES: {len(missing_features)} selected features not found in universal data")
                        logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  Missing features: {list(missing_features)[:10]}... (showing first 10)")
                        logger.warning(f"[FEATURE_DEBUG] {symbol}: ⚠️  Available: {len(all_feature_columns)}, Selected: {len(self.selected_feature_columns)}, Found: {len(feature_columns)}")
                    else:
                        logger.info(f"[FEATURE_DEBUG] {symbol}: ✅ ALL_UNIVERSAL_FEATURES_FOUND: {len(feature_columns)}/{len(self.selected_feature_columns)} selected features available")
                    
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ UNIVERSAL_FEATURE_SELECTION: Using {len(feature_columns)} selected feature columns out of {len(all_feature_columns)} available")
                    logger.info(f"[FEATURE_DEBUG] {symbol}: ✓ Selected feature column names: {feature_columns[:10]}... (showing first 10)")
                else:
                    # Use all available features if no feature selection is applied
                    feature_columns = all_feature_columns
                    logger.warning(f"[{symbol}] ⚠️  NO FEATURE SELECTION: Using all {len(feature_columns)} features - this may cause dimension mismatch!")
                    logger.warning(f"[{symbol}] ⚠️  Expected features from ensemble_config: 45, Got: {len(feature_columns)}")
                
                symbol_df = individual_df[feature_columns]
                
                logger.info(f"[{symbol}] Excluded symbol embedding columns ({len(symbol_embedding_cols)}): {symbol_embedding_cols}")
                logger.info(f"[{symbol}] Final universal feature columns: {len(feature_columns)}")
                
            # except Exception as e:
            #     logger.warning(f"Failed to get universal features for {symbol}: {e}")
            #     logger.warning(f"Proceeding with individual features only for {symbol}")
                # Fall back to the original individual features approach
                pass
            
            # Step 4: Handle NaN and infinite values (universal features already filtered above)
            # If we're using individual features (fallback), apply the same filtering as training
            if 'symbol_id' in symbol_df.columns:
                # This means we're using individual features, apply same filtering as training
                symbol_embedding_cols = [col for col in symbol_df.columns if (
                    col == 'symbol_id' or (
                        col.startswith('symbol_') and not any([
                            col.startswith('corr_'),
                            col.startswith('beta_'),
                            col.startswith('relative_strength_'),
                            col.startswith('market_dispersion_'),
                            col.startswith('market_volatility'),
                            col.startswith('vol_regime_'),
                            col.startswith('vol_trend'),
                            col.startswith('vol_correlation')
                        ])
                    )
                )]
                
                # Keep all features except symbol embedding columns and target
                all_feature_columns = [col for col in symbol_df.columns if col not in symbol_embedding_cols and col != 'target']
                
                # Apply feature selection if selected feature columns are available
                if self.selected_feature_columns:
                    # Use only selected feature columns that are available in the data
                    feature_columns = [col for col in self.selected_feature_columns if col in all_feature_columns]
                    if len(feature_columns) < len(self.selected_feature_columns):
                        missing_features = set(self.selected_feature_columns) - set(all_feature_columns)
                        logger.warning(f"[{symbol}] Fallback: ⚠️  Some selected feature columns not found: {missing_features}")
                        logger.warning(f"[{symbol}] Fallback: ⚠️  Available features: {len(all_feature_columns)}, Selected feature columns: {len(self.selected_feature_columns)}, Found: {len(feature_columns)}")
                    logger.info(f"[{symbol}] Fallback: ✓ FEATURE_SELECTION: Using {len(feature_columns)} selected feature columns out of {len(all_feature_columns)} available")
                    logger.info(f"[{symbol}] Fallback: ✓ Selected feature column names: {feature_columns[:10]}... (showing first 10)")
                else:
                    # Use all available features if no feature selection is applied
                    feature_columns = all_feature_columns
                    logger.warning(f"[{symbol}] Fallback: ⚠️  NO FEATURE SELECTION: Using all {len(feature_columns)} features - this may cause dimension mismatch!")
                    logger.warning(f"[{symbol}] Fallback: ⚠️  Expected features from ensemble_config: 45, Got: {len(feature_columns)}")
                
                symbol_df = symbol_df[feature_columns]
                
                logger.info(f"[{symbol}] Fallback: Excluded symbol embedding columns ({len(symbol_embedding_cols)}): {symbol_embedding_cols}")
                logger.info(f"[{symbol}] Fallback: Final feature columns kept: {len(feature_columns)}")
            
            # Step 5: Handle NaN and infinite values
            symbol_df = symbol_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            symbol_df = symbol_df.replace([np.inf, -np.inf], [1e6, -1e6])
            
            logger.info(f"[{symbol}] Final feature shape: {symbol_df.shape}")
            
            # Return only the latest row of features as 2D array for statistical models
            if len(symbol_df) > 0:
                # Get the latest row and reshape to (1, features) for statistical models
                latest_features = symbol_df.iloc[-1:].values  # Shape: (1, features)
                logger.info(f"[{symbol}] Returning 2D features for statistical models: {latest_features.shape}")
                return latest_features
            else:
                logger.error(f"No feature data available for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing universal features for {symbol}: {e}")
            return None
    
    async def _generate_universal_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[EnsemblePrediction]:
        """Generate prediction using universal statistical models (XGBoost, Random Forest, SVM, Ensemble)"""
        try:
            if not self.is_universal_mode or not self.universal_models:
                return None

            # Prepare universal features using the same process as training
            universal_features = await self._prepare_universal_features(symbol, market_data)
            if universal_features is None:
                logger.error(f"Failed to prepare universal features for {symbol}")
                return None
            
            # universal_features is now a 2D array with shape (1, features) for statistical models
            logger.info(f"[{symbol}] Using 2D features for statistical models with shape: {universal_features.shape}")
            
            model_predictions = {}
            model_confidences = {}
            
            # Generate predictions from each statistical model
            for model_type, model in self.universal_models.items():
                try:
                    # model_type is already a ModelType enum from the fixed loading process
                    if not isinstance(model_type, ModelType):
                        logger.error(f"Invalid model type: {model_type} (type: {type(model_type)})")
                        continue
                    
                    # Only handle statistical models (XGBoost, Random Forest, SVM, Ensemble)
                    if model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM, ModelType.ENSEMBLE]:
                        if model_type == ModelType.ENSEMBLE:
                            # Handle ensemble model (dictionary with individual models)
                            if isinstance(model, dict) and 'models' in model:
                                models = model['models']
                                weights = model['weights']
                                
                                # Get predictions from each model in ensemble
                                xgb_pred = models['xgboost'].predict_proba(universal_features)[0, 1] if 'xgboost' in models else 0.5
                                rf_pred = models['random_forest'].predict_proba(universal_features)[0, 1] if 'random_forest' in models else 0.5
                                svm_pred = models['svm'].predict_proba(universal_features)[0, 1] if 'svm' in models else 0.5
                                
                                # Calculate weighted ensemble prediction
                                pred_value = (weights.get('xgboost', 0.33) * xgb_pred + 
                                            weights.get('random_forest', 0.33) * rf_pred + 
                                            weights.get('svm', 0.34) * svm_pred)
                                
                                # Use calibrated confidence if available
                                if hasattr(model, 'confidence_calibrator') and model['confidence_calibrator'] is not None:
                                    try:
                                        # Apply ensemble calibration to the prediction
                                        confidence = float(model['confidence_calibrator'].predict_proba([[pred_value]])[0, 1])
                                        logger.debug(f"[{symbol}] ENSEMBLE: Applied calibrated confidence: {confidence:.4f}")
                                    except Exception as e:
                                        logger.warning(f"[{symbol}] ENSEMBLE: Calibration failed, using fallback: {e}")
                                        confidence = pred_value  # Use prediction as confidence fallback
                                else:
                                    # Fallback: use prediction as confidence
                                    confidence = pred_value
                            else:
                                logger.warning(f"Ensemble model format not recognized for {symbol}")
                                continue
                        else:
                            # Individual statistical models
                            if hasattr(model, 'predict_proba'):
                                # Get probability prediction for binary classification
                                proba = model.predict_proba(universal_features)[0]
                                pred_value = float(proba[1])  # Probability of positive class
                                
                                # Use calibrated confidence if available
                                if hasattr(model, 'confidence_calibrator') and model.confidence_calibrator is not None:
                                    try:
                                        # Apply calibration to the raw confidence (probability of predicted class)
                                        confidence = float(model.confidence_calibrator.predict_proba([[pred_value]])[0, 1])
                                        logger.debug(f"[{symbol}] {model_type.value}: Applied calibrated confidence: {confidence:.4f}")
                                    except Exception as e:
                                        logger.warning(f"[{symbol}] {model_type.value}: Calibration failed, using fallback: {e}")
                                        confidence = pred_value  # Use prediction as confidence fallback
                                else:
                                    # Fallback: use prediction as confidence (corrected from max(proba))
                                    confidence = pred_value
                            else:
                                # Fallback for models without predict_proba
                                prediction = model.predict(universal_features)[0]
                                pred_value = float(prediction)
                                confidence = 0.5
                        
                        model_predictions[model_type] = pred_value
                        model_confidences[model_type] = confidence
                        
                        logger.debug(f"[{symbol}] {model_type.value} prediction: {pred_value:.4f}, confidence: {confidence:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error generating prediction with statistical model {model_type.value}: {e}")
                    logger.error(f"Model type: {type(model)}, Model object: {model}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            if not model_predictions:
                return None
            
            # Calculate ensemble prediction using universal weights
            ensemble_weights = self.ensemble_weights.get(symbol, self.default_ensemble_weights)
            
            weighted_prediction = 0.0
            total_weight = 0.0
            weighted_confidence = 0.0
            
            for model_type, prediction in model_predictions.items():
                weight = ensemble_weights.get(model_type, 0.2)  # Default weight
                weighted_prediction += prediction * weight
                weighted_confidence += model_confidences[model_type] * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction = weighted_prediction / total_weight
                ensemble_confidence = weighted_confidence / total_weight
            else:
                ensemble_prediction = np.mean(list(model_predictions.values()))
                ensemble_confidence = np.mean(list(model_confidences.values()))
            
            # Calculate prediction variance for additional confidence measure
            prediction_values = list(model_predictions.values())
            prediction_variance = np.var(prediction_values) if len(prediction_values) > 1 else 0.0
            
            # Adjust confidence based on model agreement
            disagreement_penalty = min(prediction_variance * 0.5, 0.3)
            final_confidence = max(0.1, ensemble_confidence - disagreement_penalty)
            
            # Create individual predictions list
            individual_predictions = []
            for model_type, pred_value in model_predictions.items():
                individual_predictions.append(ModelPrediction(
                    model_type=model_type,
                    symbol=symbol,
                    prediction=pred_value,
                    confidence=model_confidences[model_type],
                    probability=model_confidences[model_type],  # Use confidence as probability
                    features_used=[f"universal_feature_{i}" for i in range(universal_features.shape[1])],
                    timestamp=datetime.now(),
                    model_version="universal_v1.0"
                ))
            
            return EnsemblePrediction(
                symbol=symbol,
                final_prediction=ensemble_prediction,
                confidence=final_confidence,
                individual_predictions=individual_predictions,
                ensemble_weights=ensemble_weights,
                risk_score=self._calculate_risk_score(ensemble_prediction, prediction_variance),
                signal_strength=self._calculate_signal_strength(ensemble_prediction, final_confidence),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating universal prediction for {symbol}: {e}")
            return None
    
    def _calculate_risk_score(self, prediction: float, prediction_variance: float) -> float:
        """Calculate risk score based on prediction and variance - directional approach"""
        try:
            # Base risk from prediction strength (squared to avoid abs bias)
            prediction_strength = prediction * prediction
            prediction_risk = min(1.0, prediction_strength * 0.5)
            
            # Variance risk (higher variance = higher risk)
            variance_risk = min(1.0, prediction_variance * 2.0)
            
            # Direction clarity bonus - clearer direction = lower risk
            direction_clarity = abs(prediction)
            clarity_bonus = 0.1 if direction_clarity > 0.6 else 0.05 if direction_clarity > 0.3 else 0.0
            
            # Combined risk score (weighted average with clarity adjustment)
            risk_score = (prediction_risk * 0.7) + (variance_risk * 0.3) - clarity_bonus
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default moderate risk
    
    def _calculate_signal_strength(self, prediction: float, confidence: float) -> float:
        """Calculate signal strength based on prediction and confidence - directional approach"""
        try:
            # Signal strength using directional approach - avoid abs bias
            prediction_magnitude = prediction * prediction  # Squared for strength without sign bias
            direction_clarity = abs(prediction)  # How clear the direction is
            
            # Base strength from prediction magnitude
            base_strength = prediction_magnitude * 0.8
            
            # Directional bonus for clear signals
            directional_bonus = 0.2 if direction_clarity > 0.7 else 0.1 if direction_clarity > 0.4 else 0.0
            
            # Combine with confidence
            signal_strength = (base_strength + directional_bonus) * confidence
            
            return min(1.0, max(0.0, signal_strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5  # Default moderate strength
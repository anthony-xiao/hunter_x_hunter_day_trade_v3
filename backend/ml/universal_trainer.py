import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime, timezone, timedelta
from loguru import logger
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model
from tqdm import tqdm

class NaNDetectionCallback(Callback):
    """Custom callback to detect NaN values during training and stop if found."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Check for NaN in loss values
        for metric_name, metric_value in logs.items():
            if np.isnan(metric_value) or np.isinf(metric_value):
                logger.error(f"NaN/Inf detected in {metric_name}: {metric_value} at epoch {epoch}")
                logger.error("Stopping training due to numerical instability")
                self.model.stop_training = True
                return
        
        # Check model weights for NaN/Inf
        for layer in self.model.layers:
            if hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                for i, weight in enumerate(weights):
                    if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
                        logger.error(f"NaN/Inf detected in layer {layer.name} weights[{i}] at epoch {epoch}")
                        logger.error("Stopping training due to numerical instability in weights")
                        self.model.stop_training = True
                        return

from data.data_pipeline import DataPipeline
from .universal_feature_engineering import UniversalFeatureEngineering
from .universal_model_architectures import UniversalModelArchitectures
from .universal_feature_engineering import UniversalFeatureSet
from .class_imbalance_mitigation import ClassImbalanceMitigator, ImbalanceConfig, ImbalanceMetrics
from .feature_selector import UniversalFeatureSelector, FeatureSelectionConfig
from .temporal_aggregator import TemporalAggregator, AggregationConfig
from .model_types import ModelType

@dataclass
class ModelConfig:
    name: str
    model_type: str
    parameters: Dict
    training_window: int
    validation_window: int
    lookback_window: int
    feature_count: int
    train_test_split: float = 0.8
    learning_rate: float = 0.001
    prediction_threshold: float = 0.5

@dataclass
class ModelPerformance:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    returns: List[float]
    timestamp: datetime
    validation_score: float
    overfitting_score: float
    profit_factor: float = 0.0
    last_updated: datetime = None

@dataclass
class UniversalTrainingConfig:
    """Configuration for universal statistical model training phases"""
    # Phase 1: Universal Base Model Training
    base_training_window: int = 12  # months of training data
    base_validation_window: int = 3  # months of validation data
    base_lookback_window: int = 30  # minutes of lookback for features
    base_validation_split: float = 0.2
    
    # Symbols for training
    symbols: List[str] = None
    
    # Date range for training
    start_date: str = None
    end_date: str = None
    
    # Data pipeline configuration
    force_2d_for_statistical: bool = False  # Enforce 2D data pipeline for statistical models
    
    # Data augmentation settings
    enable_smote: bool = False
    
    # Phase 2: Symbol-Specific Fine-tuning
    finetune_training_window: int = 6  # months for fine-tuning
    finetune_validation_window: int = 2  # months for fine-tuning validation
    finetune_sample_weight_adjustment: float = 1.2  # boost recent samples
    
    # Phase 3: Ensemble Optimization
    ensemble_validation_periods: int = 10
    ensemble_rebalance_frequency: int = 5
    ensemble_cross_validation_folds: int = 5
    
    # XGBoost Configuration
    xgboost_n_estimators: int = 1000
    xgboost_max_depth: int = 7
    xgboost_learning_rate: float = 0.15
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8
    xgboost_reg_alpha: float = 0.1
    xgboost_reg_lambda: float = 0.1
    
    # Random Forest Configuration
    rf_n_estimators: int = 500
    rf_max_depth: int = 12
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5
    rf_max_features: str = 'sqrt'
    
    # SVM Configuration
    svm_kernel: str = 'rbf'
    svm_C: float = 1.0
    svm_gamma: str = 'scale'
    svm_class_weight: str = 'balanced'
    
    # Ensemble Configuration
    ensemble_xgb_weight: float = 0.45
    ensemble_rf_weight: float = 0.35
    ensemble_svm_weight: float = 0.20
    
    # Dual Exit Target Configuration
    prediction_window: int = 15  # Maximum prediction window in minutes (periods)
    take_profit_pct: float = 0.003
    stop_loss_pct: float = 0.001
    
    # Class Imbalance Mitigation Configuration
    enable_imbalance_mitigation: bool = True
    imbalance_config: ImbalanceConfig = None
    
    # Temporal Aggregation Configuration
    enable_temporal_aggregation: bool = True
    temporal_aggregation_config: AggregationConfig = None
    
    # General settings
    prediction_threshold: float = 0.55
    min_samples_per_symbol: int = 1000
    max_symbols_per_batch: int = 50
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        # Validate ensemble weights sum to 1.0
        total_weight = self.ensemble_xgb_weight + self.ensemble_rf_weight + self.ensemble_svm_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Ensemble weights sum to {total_weight}, normalizing to 1.0")
            self.ensemble_xgb_weight /= total_weight
            self.ensemble_rf_weight /= total_weight
            self.ensemble_svm_weight /= total_weight

@dataclass
class UniversalTrainingResult:
    """Results from universal training process"""
    phase: str
    model_name: str
    symbols_trained: List[str]
    base_model_performance: Dict[str, float]
    symbol_performances: Dict[str, Dict[str, float]]
    ensemble_weights: Dict[str, float]
    training_time: float
    total_samples: int
    validation_accuracy: float
    metadata: Dict[str, Any]

class UniversalTrainer:
    """
    Universal training system implementing 3-phase training strategy:
    1. Universal base model training on all symbols
    2. Symbol-specific fine-tuning
    3. Ensemble optimization
    """
    
    def __init__(
        self,
        data_pipeline: DataPipeline,
        feature_engineering: UniversalFeatureEngineering,
        config: UniversalTrainingConfig = None
    ):
        self.data_pipeline = data_pipeline
        self.feature_engineering = feature_engineering
        self.config = config or UniversalTrainingConfig()
        
        # Initialize class imbalance mitigation
        if self.config.imbalance_config is None:
            self.config.imbalance_config = ImbalanceConfig()
        
        self.imbalance_mitigator = ClassImbalanceMitigator(self.config.imbalance_config) if self.config.enable_imbalance_mitigation else None
        
        # Initialize temporal aggregator
        if self.config.temporal_aggregation_config is None:
            self.config.temporal_aggregation_config = AggregationConfig()
        
        self.temporal_aggregator = TemporalAggregator(self.config.temporal_aggregation_config) if self.config.enable_temporal_aggregation else None
        
        # Initialize feature selector
        self.feature_selector = None
        self.selected_features = None
        self.selected_feature_indices = None
        
        # Initialize components
        self.symbol_to_id = {}
        self.id_to_symbol = {}
        self.universal_architectures = None
        
        # Training state
        self.base_models = {}
        self.symbol_models = {}
        self.ensemble_weights = {}
        self.training_history = []
        self.imbalance_metrics = []
        
        self.model_configs = {
            ModelType.XGBOOST: ModelConfig(
                name='xgboost',
                model_type='statistical',
                parameters={
                    'n_estimators': self.config.xgboost_n_estimators,
                    'max_depth': self.config.xgboost_max_depth,
                    'learning_rate': self.config.xgboost_learning_rate,
                    'subsample': self.config.xgboost_subsample,
                    'colsample_bytree': self.config.xgboost_colsample_bytree,
                    'reg_alpha': self.config.xgboost_reg_alpha,
                    'reg_lambda': self.config.xgboost_reg_lambda,
                    'random_state': self.config.random_state,
                    'n_jobs': self.config.n_jobs,
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss'
                },
                training_window=self.config.base_training_window,
                validation_window=self.config.base_validation_window,
                lookback_window=self.config.base_lookback_window,
                feature_count=None,
                learning_rate=self.config.xgboost_learning_rate,
                prediction_threshold=self.config.prediction_threshold
            ),
            ModelType.RANDOM_FOREST: ModelConfig(
                name='random_forest',
                model_type='statistical',
                parameters={
                    'n_estimators': self.config.rf_n_estimators,
                    'max_depth': self.config.rf_max_depth,
                    'min_samples_split': self.config.rf_min_samples_split,
                    'min_samples_leaf': self.config.rf_min_samples_leaf,
                    'max_features': self.config.rf_max_features,
                    'bootstrap': True,
                    'random_state': self.config.random_state,
                    'n_jobs': self.config.n_jobs,
                    'class_weight': 'balanced'
                },
                training_window=self.config.base_training_window,
                validation_window=self.config.base_validation_window,
                lookback_window=self.config.base_lookback_window,
                feature_count=None,
                learning_rate=None,
                prediction_threshold=self.config.prediction_threshold
            ),
            ModelType.SVM: ModelConfig(
                name='svm',
                model_type='statistical',
                parameters={
                    'C': self.config.svm_C,
                    'kernel': self.config.svm_kernel,
                    'gamma': self.config.svm_gamma,
                    'probability': True,
                    'random_state': self.config.random_state,
                    'class_weight': self.config.svm_class_weight,
                    'cache_size': 1000
                },
                training_window=self.config.base_training_window,
                validation_window=self.config.base_validation_window,
                lookback_window=self.config.base_lookback_window,
                feature_count=None,
                learning_rate=None,
                prediction_threshold=self.config.prediction_threshold
            ),
            ModelType.ENSEMBLE: ModelConfig(
                name='ensemble',
                model_type='ensemble',
                parameters={
                    'base_models': [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM],
                    'voting': 'soft',
                    'weights': [self.config.ensemble_xgb_weight, self.config.ensemble_rf_weight, self.config.ensemble_svm_weight],
                    'n_jobs': self.config.n_jobs
                },
                training_window=self.config.base_training_window,
                validation_window=self.config.base_validation_window,
                lookback_window=self.config.base_lookback_window,
                feature_count=None,
                learning_rate=None,
                prediction_threshold=self.config.prediction_threshold
            )
        }


        logger.info("Initialized UniversalTrainer with 3-phase training strategy")
    
    async def initialize_symbol_mappings(self, symbols: List[str]) -> None:
        """
        Initialize symbol-to-ID mappings for embeddings.
        
        Args:
            symbols: List of trading symbols
        """
        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.id_to_symbol = {idx: symbol for idx, symbol in enumerate(symbols)}
        
        # Initialize universal architectures
        self.universal_architectures = UniversalModelArchitectures(
            num_symbols=len(symbols),
            symbol_embedding_dim=self.config.symbol_embedding_dim
        )
        
        logger.info(f"Initialized symbol mappings for {len(symbols)} symbols")
    
    def _combine_features_from_featureset(self, feature_set) -> pd.DataFrame:
        """
        Combine all feature components from a FeatureSet into a single features DataFrame.
        
        Args:
            feature_set: FeatureSet object containing various feature categories
            
        Returns:
            Combined features as pandas DataFrame (preserving column names)
        """
        feature_dfs = []
        
        # Collect all available feature DataFrames
        if hasattr(feature_set, 'technical_features') and feature_set.technical_features is not None:
            feature_dfs.append(feature_set.technical_features)
        if hasattr(feature_set, 'market_microstructure') and feature_set.market_microstructure is not None:
            feature_dfs.append(feature_set.market_microstructure)
        if hasattr(feature_set, 'sentiment_features') and feature_set.sentiment_features is not None:
            feature_dfs.append(feature_set.sentiment_features)
        if hasattr(feature_set, 'macro_features') and feature_set.macro_features is not None:
            feature_dfs.append(feature_set.macro_features)
        if hasattr(feature_set, 'cross_asset_features') and feature_set.cross_asset_features is not None:
            feature_dfs.append(feature_set.cross_asset_features)
        if hasattr(feature_set, 'engineered_features') and feature_set.engineered_features is not None:
            feature_dfs.append(feature_set.engineered_features)
        
        if not feature_dfs:
            raise ValueError("No valid feature DataFrames found in FeatureSet")
        
        # Combine all features horizontally
        combined_df = pd.concat(feature_dfs, axis=1)
        
        # Handle NaN values
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return combined_df
    
    def _validate_feature_dimensions(self, features_df: pd.DataFrame, context: str, expected_total: int = 262) -> bool:
        """
        Comprehensive feature count validation to ensure consistent dimensions across all training phases.
        
        Args:
            features_df: DataFrame containing features to validate
            context: Description of the validation context (e.g., "Phase 1 training", "Phase 2 fine-tuning")
            expected_total: Expected total feature count (default 178)
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"🔍 Feature Validation - {context}")
        
        # Analyze feature types based on actual naming patterns
        # Cross-symbol features: corr_, beta_, relative_strength_, market_dispersion_
        cross_cols = [col for col in features_df.columns if any([
            col.startswith('corr_'),
            col.startswith('beta_'),
            col.startswith('relative_strength_'),
            col.startswith('market_dispersion_')
        ])]
        
        # Market regime features: market_volatility, vol_regime_, vol_trend, vol_correlation
        regime_cols = [col for col in features_df.columns if any([
            col.startswith('market_volatility'),
            col.startswith('vol_regime_'),
            col.startswith('vol_trend'),
            col.startswith('vol_correlation')
        ])]
        
        # Sector features: sector_
        sector_cols = [col for col in features_df.columns if col.startswith('sector_')]
        
        # Symbol embedding columns: symbol_id and symbol_SYMBOL_NAME
        symbol_cols = [col for col in features_df.columns if col == 'symbol_id' or (
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
        )]
        
        # Target feature should not be classified as technical
        target_cols = [col for col in features_df.columns if col == 'target']
        
        # Technical features: everything else (excluding target)
        special_feature_cols = set(cross_cols + regime_cols + sector_cols + symbol_cols + target_cols)
        technical_cols = [col for col in features_df.columns if col not in special_feature_cols]
        
        total_features = len(features_df.columns)
        
        # Log detailed breakdown
        logger.info(f"  📊 Feature Breakdown:")
        logger.info(f"    • Technical features: {len(technical_cols)}")
        logger.info(f"    • Symbol embeddings: {len(symbol_cols)}")
        logger.info(f"    • Cross-symbol features: {len(cross_cols)}")
        logger.info(f"    • Market regime features: {len(regime_cols)}")
        logger.info(f"    • Sector features: {len(sector_cols)}")
        logger.info(f"    • Target features: {len(target_cols)}")
        logger.info(f"    • TOTAL: {total_features}")
        
        # Log detailed feature column names for debugging
        logger.info(f"  🔍 Detailed Feature Analysis:")
        logger.info(f"    Technical columns ({len(technical_cols)}): {technical_cols[:10]}{'...' if len(technical_cols) > 10 else ''}")
        logger.info(f"    Symbol columns ({len(symbol_cols)}): {symbol_cols}")
        logger.info(f"    Cross-symbol columns ({len(cross_cols)}): {cross_cols}")
        logger.info(f"    Market regime columns ({len(regime_cols)}): {regime_cols}")
        logger.info(f"    Sector columns ({len(sector_cols)}): {sector_cols}")
        logger.info(f"    Target columns ({len(target_cols)}): {target_cols}")
        
        # Check for any unclassified columns
        all_classified = technical_cols + symbol_cols + cross_cols + regime_cols + sector_cols + target_cols
        unclassified = [col for col in features_df.columns if col not in all_classified]
        if unclassified:
            logger.warning(f"    ⚠️  Unclassified columns ({len(unclassified)}): {unclassified}")
        
        # Check for duplicate columns
        duplicate_cols = features_df.columns[features_df.columns.duplicated()].tolist()
        if duplicate_cols:
            logger.warning(f"    ⚠️  Duplicate columns found: {duplicate_cols}")
        
        # Validation checks
        validation_passed = True
        
        # Check total feature count
        if total_features != expected_total:
            logger.error(f"  ❌ FEATURE COUNT MISMATCH: Got {total_features}, expected {expected_total}")
            logger.error(f"     Difference: {total_features - expected_total} features")
            
            # Enhanced feature comparison logging
            logger.error(f"  🔍 DETAILED FEATURE ANALYSIS:")
            logger.error(f"     Expected vs Received by Category:")
            
            # Define expected counts based on typical training configuration
            expected_technical = 120  # Typical technical feature count
            expected_symbol = 0       # Symbol embeddings excluded during training
            expected_cross = 15       # Cross-symbol features
            expected_regime = 49      # Market regime features
            expected_sector = 0       # Sector features (if any)
            
            logger.error(f"       • Technical: {len(technical_cols)} (expected ~{expected_technical})")
            logger.error(f"       • Symbol: {len(symbol_cols)} (expected {expected_symbol})")
            logger.error(f"       • Cross-symbol: {len(cross_cols)} (expected ~{expected_cross})")
            logger.error(f"       • Market regime: {len(regime_cols)} (expected ~{expected_regime})")
            logger.error(f"       • Sector: {len(sector_cols)} (expected {expected_sector})")
            
            # Log sample column names for each category
            logger.error(f"     Sample Column Names by Category:")
            if technical_cols:
                logger.error(f"       • Technical samples: {technical_cols[:5]}")
            if symbol_cols:
                logger.error(f"       • Symbol samples: {symbol_cols[:5]}")
            if cross_cols:
                logger.error(f"       • Cross-symbol samples: {cross_cols[:5]}")
            if regime_cols:
                logger.error(f"       • Market regime samples: {regime_cols[:5]}")
            if sector_cols:
                logger.error(f"       • Sector samples: {sector_cols[:5]}")
            if unclassified:
                logger.error(f"       • Unclassified samples: {unclassified[:5]}")
            
            # Identify potential discrepancies
            if len(symbol_cols) > 0:
                logger.error(f"     ⚠️  POTENTIAL ISSUE: Symbol embeddings found but should be excluded during training")
            
            if len(cross_cols) == 0:
                logger.error(f"     ⚠️  POTENTIAL ISSUE: No cross-symbol features found - check universal feature engineering")
            
            if len(regime_cols) == 0:
                logger.error(f"     ⚠️  POTENTIAL ISSUE: No market regime features found - check universal feature engineering")
            
            # Log all column names for complete debugging
            all_columns = list(features_df.columns)
            logger.error(f"     Complete Column List ({len(all_columns)} total):")
            for i in range(0, len(all_columns), 10):
                batch = all_columns[i:i+10]
                logger.error(f"       [{i+1}-{min(i+10, len(all_columns))}]: {batch}")
            
            validation_passed = False
        else:
            logger.info(f"  ✅ FEATURE COUNT VALIDATED: {total_features} features match expected {expected_total}")
        
        # Check for missing critical feature types
        if len(technical_cols) == 0:
            logger.error(f"  ❌ MISSING TECHNICAL FEATURES: No technical features found")
            validation_passed = False
        
        if len(symbol_cols) == 0:
            logger.error(f"  ❌ MISSING SYMBOL EMBEDDINGS: No symbol embeddings found")
            validation_passed = False
        
        # Enhanced validation for cross-symbol and market regime features
        if len(cross_cols) == 0:
            logger.warning(f"  ⚠️  NO CROSS-SYMBOL FEATURES: Expected features like corr_, beta_, relative_strength_, market_dispersion_")
            logger.warning(f"     This may indicate feature filtering issues or missing universal feature engineering")
        else:
            logger.info(f"  ✅ CROSS-SYMBOL FEATURES FOUND: {len(cross_cols)} features detected")
        
        if len(regime_cols) == 0:
            logger.warning(f"  ⚠️  NO MARKET REGIME FEATURES: Expected features like market_volatility, vol_regime_, vol_trend, vol_correlation")
            logger.warning(f"     This may indicate feature filtering issues or missing universal feature engineering")
        else:
            logger.info(f"  ✅ MARKET REGIME FEATURES FOUND: {len(regime_cols)} features detected")
        
        # Log specific feature examples for debugging
        if cross_cols:
            logger.info(f"     Cross-symbol examples: {cross_cols[:3]}{'...' if len(cross_cols) > 3 else ''}")
        if regime_cols:
            logger.info(f"     Market regime examples: {regime_cols[:3]}{'...' if len(regime_cols) > 3 else ''}")
        
        # Check for NaN or infinite values
        nan_count = features_df.isnull().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        
        if nan_count > 0:
            logger.warning(f"  ⚠️  Found {nan_count} NaN values in features")
        
        if inf_count > 0:
            logger.warning(f"  ⚠️  Found {inf_count} infinite values in features")
        
        # Log feature distribution percentages
        if total_features > 0:
            logger.info(f"  📈 Feature Distribution:")
            logger.info(f"    • Technical: {len(technical_cols)/total_features*100:.1f}%")
            logger.info(f"    • Symbol: {len(symbol_cols)/total_features*100:.1f}%")
            logger.info(f"    • Cross-symbol: {len(cross_cols)/total_features*100:.1f}%")
            logger.info(f"    • Market regime: {len(regime_cols)/total_features*100:.1f}%")
            logger.info(f"    • Sector: {len(sector_cols)/total_features*100:.1f}%")
            logger.info(f"    • Target: {len(target_cols)/total_features*100:.1f}%")
        
        return validation_passed

    async def _prepare_universal_features_for_symbol(self, symbol: str, feature_set, start_date, end_date) -> np.ndarray:
        """
        Prepare universal features for a single symbol using the same approach as Phase 1 training.
        This ensures consistent feature dimensions (178 features) between base training and fine-tuning.
        
        Args:
            symbol: Trading symbol
            feature_set: FeatureSet object containing individual symbol features
            start_date: Start date for universal features
            end_date: End date for universal features
            
        Returns:
            Universal features array with consistent 178-feature dimension
        """
        logger.info(f"Preparing universal features for symbol {symbol} fine-tuning")
        
        # Step 1: Combine individual symbol features (same as _combine_features_from_featureset)
        feature_dfs = []
        feature_counts = {}
        
        # Add technical features
        if hasattr(feature_set, 'technical_features') and feature_set.technical_features is not None and not feature_set.technical_features.empty:
            feature_dfs.append(feature_set.technical_features)
            feature_counts['technical'] = len(feature_set.technical_features.columns)
        
        # Add market microstructure features
        if hasattr(feature_set, 'market_microstructure') and feature_set.market_microstructure is not None and not feature_set.market_microstructure.empty:
            feature_dfs.append(feature_set.market_microstructure)
            feature_counts['market_microstructure'] = len(feature_set.market_microstructure.columns)
        
        # Add sentiment features
        if hasattr(feature_set, 'sentiment_features') and feature_set.sentiment_features is not None and not feature_set.sentiment_features.empty:
            feature_dfs.append(feature_set.sentiment_features)
            feature_counts['sentiment'] = len(feature_set.sentiment_features.columns)
        
        # Add macro features
        if hasattr(feature_set, 'macro_features') and feature_set.macro_features is not None and not feature_set.macro_features.empty:
            feature_dfs.append(feature_set.macro_features)
            feature_counts['macro'] = len(feature_set.macro_features.columns)
        
        # Add cross-asset features
        if hasattr(feature_set, 'cross_asset_features') and feature_set.cross_asset_features is not None and not feature_set.cross_asset_features.empty:
            feature_dfs.append(feature_set.cross_asset_features)
            feature_counts['cross_asset'] = len(feature_set.cross_asset_features.columns)
        
        # Add engineered features
        if hasattr(feature_set, 'engineered_features') and feature_set.engineered_features is not None and not feature_set.engineered_features.empty:
            feature_dfs.append(feature_set.engineered_features)
            feature_counts['engineered'] = len(feature_set.engineered_features.columns)
        
        if not feature_dfs:
            raise ValueError(f"No valid feature DataFrames found for symbol {symbol}")
        
        # Combine individual features
        symbol_df = pd.concat(feature_dfs, axis=1)
        total_individual_features = sum(feature_counts.values())
        logger.info(f"[{symbol}] Combined individual features: {total_individual_features} columns")
        
        # Step 2: Add symbol embeddings (same as prepare_universal_training_data)
        symbol_id = self.symbol_to_id[symbol]
        symbol_df['symbol_id'] = symbol_id
        
        # Add one-hot encoding for all symbols
        for other_symbol in self.symbol_to_id.keys():
            symbol_df[f'symbol_{other_symbol}'] = 1 if other_symbol == symbol else 0
        
        # Step 3: Add universal features (cross-symbol, regime, sector)
        # We need to engineer these universal features for the specific time period
        try:
            # Get universal features for this time period
            # Use training_mode=True to ensure sufficient historical data for all features
            # including sma_100, sma_200, and other long-period indicators
            # Use all symbols for cross-symbol and sector features (same as training)
            universal_features = await self.feature_engineering.engineer_universal_features(
                symbols=list(self.symbol_to_id.keys()),  # Use all symbols for consistent cross-symbol/sector features
                start_date=start_date,
                end_date=end_date,
                training_mode=True  # Use training mode to ensure full feature generation
            )
            
            # Add cross-symbol features if available
            cross_symbol_count = 0
            if hasattr(universal_features, 'cross_symbol_features') and not universal_features.cross_symbol_features.empty:
                aligned_cross = universal_features.cross_symbol_features.reindex(symbol_df.index)
                symbol_df = pd.concat([symbol_df, aligned_cross], axis=1)
                cross_symbol_count = len(universal_features.cross_symbol_features.columns)
                logger.info(f"[{symbol}] Added {cross_symbol_count} cross-symbol features")
            
            # Add regime features if available
            regime_count = 0
            if hasattr(universal_features, 'market_regime_features') and not universal_features.market_regime_features.empty:
                aligned_regime = universal_features.market_regime_features.reindex(symbol_df.index)
                symbol_df = pd.concat([symbol_df, aligned_regime], axis=1)
                regime_count = len(universal_features.market_regime_features.columns)
                logger.info(f"[{symbol}] Added {regime_count} market regime features")
            
            # Add sector features if available
            sector_count = 0
            if hasattr(universal_features, 'sector_features') and not universal_features.sector_features.empty:
                aligned_sector = universal_features.sector_features.reindex(symbol_df.index)
                symbol_df = pd.concat([symbol_df, aligned_sector], axis=1)
                sector_count = len(universal_features.sector_features.columns)
                logger.info(f"[{symbol}] Added {sector_count} sector features")
            
            # Calculate total features
            symbol_embedding_count = 1  # Only symbol_id (matching training phase)
            total_features = total_individual_features + symbol_embedding_count + cross_symbol_count + regime_count + sector_count
            logger.info(f"[{symbol}] Total universal features: {total_features} (individual: {total_individual_features}, embeddings: {symbol_embedding_count}, cross: {cross_symbol_count}, regime: {regime_count}, sector: {sector_count})")
        
        except Exception as e:
            logger.warning(f"Failed to add universal features for {symbol}: {e}")
            logger.warning(f"Proceeding with individual features only for {symbol}")
        
        # Apply same symbol embedding exclusion logic as prepare_universal_dataset
        # Define actual symbol embedding columns to exclude (not cross-symbol or market regime features)
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
        
        # Keep all features except actual symbol embedding columns and target
        feature_columns = [col for col in symbol_df.columns if col not in symbol_embedding_cols and col != 'target']
        
        # Log which columns are being excluded vs included for debugging
        excluded_cols = [col for col in symbol_df.columns if col in symbol_embedding_cols]
        cross_symbol_cols = [col for col in symbol_df.columns if any([
            col.startswith('corr_'),
            col.startswith('beta_'),
            col.startswith('relative_strength_'),
            col.startswith('market_dispersion_')
        ])]
        market_regime_cols = [col for col in symbol_df.columns if any([
            col.startswith('market_volatility'),
            col.startswith('vol_regime_'),
            col.startswith('vol_trend'),
            col.startswith('vol_correlation')
        ])]
        
        # Check if target column was excluded
        target_excluded = 'target' in symbol_df.columns and 'target' not in feature_columns
        
        logger.info(f"[{symbol}] Feature filtering results:")
        logger.info(f"  - Excluded symbol embedding columns ({len(excluded_cols)}): {excluded_cols}")
        logger.info(f"  - Excluded target column: {target_excluded}")
        logger.info(f"  - Included cross-symbol features ({len(cross_symbol_cols)}): {cross_symbol_cols}")
        logger.info(f"  - Included market regime features ({len(market_regime_cols)}): {market_regime_cols}")
        logger.info(f"  - Total feature columns kept: {len(feature_columns)}")
        
        # Apply feature selection if available
        if self.selected_features is not None and len(self.selected_features) > 0:
            # Filter feature columns to only include selected features
            selected_feature_columns = [col for col in feature_columns if col in self.selected_features]
            logger.info(f"[{symbol}] Applying feature selection: {len(feature_columns)} -> {len(selected_feature_columns)} features")
            feature_columns = selected_feature_columns
        
        # Filter to keep only non-symbol-embedding features
        symbol_df = symbol_df[feature_columns]
        
        # Handle NaN values
        symbol_df = symbol_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        symbol_df = symbol_df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Add detailed feature consistency logging for debugging
        logger.info(f"[{symbol}] === FEATURE CONSISTENCY DEBUG ===")
        logger.info(f"[{symbol}] Total feature columns: {len(symbol_df.columns)}")
        logger.info(f"[{symbol}] Feature column names: {sorted(list(symbol_df.columns))}")
        
        # Categorize features for consistency checking
        technical_features = [col for col in symbol_df.columns if any([
            'rsi' in col.lower(), 'macd' in col.lower(), 'bb' in col.lower(), 
            'sma' in col.lower(), 'ema' in col.lower(), 'stoch' in col.lower(),
            'atr' in col.lower(), 'volume' in col.lower(), 'price' in col.lower(),
            'return' in col.lower(), 'trend' in col.lower()
        ])]
        cross_symbol_features = [col for col in symbol_df.columns if any([
            col.startswith('corr_'), col.startswith('beta_'), 
            col.startswith('relative_strength_'), col.startswith('market_dispersion_')
        ])]
        market_regime_features = [col for col in symbol_df.columns if any([
            col.startswith('market_volatility'), col.startswith('vol_regime_'),
            col.startswith('vol_trend'), col.startswith('vol_correlation')
        ])]
        sector_features = [col for col in symbol_df.columns if col.startswith('sector_')]
        
        logger.info(f"[{symbol}] Technical features ({len(technical_features)}): {technical_features[:5]}..." if len(technical_features) > 5 else f"[{symbol}] Technical features ({len(technical_features)}): {technical_features}")
        logger.info(f"[{symbol}] Cross-symbol features ({len(cross_symbol_features)}): {cross_symbol_features}")
        logger.info(f"[{symbol}] Market regime features ({len(market_regime_features)}): {market_regime_features}")
        logger.info(f"[{symbol}] Sector features ({len(sector_features)}): {sector_features}")
        logger.info(f"[{symbol}] === END FEATURE CONSISTENCY DEBUG ===")
        
        # Validate feature dimensions before returning (updated to match consistent feature count)
        validation_passed = self._validate_feature_dimensions(
            symbol_df, 
            f"Phase 2 Fine-tuning - {symbol}", 
            expected_total=187  # Updated to match actual feature count from training
        )
        
        if not validation_passed:
            logger.error(f"[{symbol}] Feature validation failed - proceeding with caution")
        
        logger.info(f"[{symbol}] Final feature shape after filtering: {symbol_df.shape}")
        return symbol_df.values
    
    def _create_dual_exit_targets(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Create binary targets using dual exit conditions within a prediction window.
        
        This method implements a more sophisticated target generation approach that:
        1. Uses a maximum prediction window (default 15 minutes)
        2. Applies dual exit conditions:
           - Take profit: Trigger when price increases by take_profit_pct within the horizon
           - Stop loss: Trigger when price decreases by stop_loss_pct within the same period
        3. Returns 1 for take profit hit, 0 for stop loss hit or no exit
        
        Args:
            market_data: DataFrame with market data including 'close' prices
            
        Returns:
            Binary targets as numpy array (1 for take profit, 0 for stop loss or no exit)
        """
        if 'close' not in market_data.columns:
            raise ValueError("Market data must contain 'close' column for target extraction")
        
        close_prices = market_data['close'].values
        targets = np.zeros(len(close_prices) - self.config.prediction_window, dtype=int)
        
        logger.info(f"Creating dual exit targets with {self.config.prediction_window}-period window")
        logger.info(f"Take profit threshold: {self.config.take_profit_pct*100:.2f}%")
        logger.info(f"Stop loss threshold: {self.config.stop_loss_pct*100:.2f}%")
        
        # Iterate through each possible starting point
        for i in range(len(targets)):
            current_price = close_prices[i]
            take_profit_price = current_price * (1 + self.config.take_profit_pct)
            stop_loss_price = current_price * (1 - self.config.stop_loss_pct)
            
            # Look ahead within the prediction window
            window_end = min(i + self.config.prediction_window + 1, len(close_prices))
            future_prices = close_prices[i+1:window_end]
            
            # Check for exit conditions
            target_hit = False
            for future_price in future_prices:
                if future_price >= take_profit_price:
                    targets[i] = 1  # Take profit hit
                    target_hit = True
                    break
                elif future_price <= stop_loss_price:
                    targets[i] = 0  # Stop loss hit
                    target_hit = True
                    break
            
            # If no exit condition is met within the window, default to 0
            if not target_hit:
                targets[i] = 0
        
        # Log target distribution for analysis
        take_profit_count = np.sum(targets == 1)
        stop_loss_count = np.sum(targets == 0)
        total_targets = len(targets)
        
        logger.info(f"Target distribution:")
        logger.info(f"  Take profit hits: {take_profit_count} ({take_profit_count/total_targets*100:.2f}%)")
        logger.info(f"  Stop loss/no exit: {stop_loss_count} ({stop_loss_count/total_targets*100:.2f}%)")
        logger.info(f"  Total targets: {total_targets}")
        
        return targets
    
    def _extract_targets_from_market_data(self, market_data: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
        """
        Extract binary targets from market data using dual exit conditions.
        
        This method now uses the dual exit approach instead of simple next-period returns.
        
        Args:
            market_data: DataFrame with market data including 'close' prices
            threshold: Deprecated parameter (kept for compatibility)
            
        Returns:
            Binary targets as numpy array (1 for take profit, 0 for stop loss or no exit)
        """
        return self._create_dual_exit_targets(market_data)
    
    def _calculate_expected_unique_features(self) -> int:
        """
        Calculate the expected number of unique base features from selected features.
        This accounts for temporal aggregation where multiple aggregated features
        (e.g., feature_5_min, feature_5_max) map to the same base feature.
        
        Returns:
            Number of unique base features expected
        """
        try:
            # Load selected features
            feature_selector = UniversalFeatureSelector()
            selected_features = feature_selector.load_selected_features()
            
            if not selected_features:
                logger.warning("No selected features found, defaulting to 47")
                return 47
            
            # Extract unique base feature indices from temporal aggregation names
            unique_indices = set()
            for feature_name in selected_features:
                # Extract feature index from names like 'feature_5_min', 'feature_6_max', etc.
                if feature_name.startswith('feature_'):
                    parts = feature_name.split('_')
                    if len(parts) >= 2 and parts[1].isdigit():
                        unique_indices.add(int(parts[1]))
            
            return len(unique_indices)
            
        except Exception as e:
            logger.warning(f"Error calculating expected unique features: {e}, defaulting to 47")
            return 47
    
    async def prepare_universal_dataset(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: UniversalTrainingConfig = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare universal training dataset with symbol embeddings.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) with symbol embeddings
        """
        logger.info(f"Preparing universal dataset for {len(symbols)} symbols")
        
        # Convert dates to timezone-aware UTC datetime objects
        # Handle both string and datetime inputs
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        elif isinstance(start_date, datetime):
            # Ensure timezone-aware UTC
            start_dt = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date.astimezone(timezone.utc)
        else:
            raise TypeError(f"start_date must be str or datetime, got {type(start_date)}")
            
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        elif isinstance(end_date, datetime):
            # Ensure timezone-aware UTC
            end_dt = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date.astimezone(timezone.utc)
        else:
            raise TypeError(f"end_date must be str or datetime, got {type(end_date)}")
        
        # Load universal data
        logger.info("Step 1: Loading universal data...")
        universal_data = await self.data_pipeline.load_universal_data(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt
        )
        logger.info(f"Step 1 completed: universal_data type={type(universal_data)}, length={len(universal_data) if hasattr(universal_data, '__len__') else 'N/A'}")

        # Engineer universal features
        logger.info("Step 2: Engineering universal features...")
        
        universal_features = await self.feature_engineering.engineer_universal_features(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            training_mode=True
        )
        logger.info(f"Step 2 completed: universal_features type={type(universal_features)}")
        
        # Log detailed feature breakdown from universal features
        if universal_features:
            logger.info("Universal features breakdown:")
            logger.info(f"  - Symbol features count: {len(universal_features.symbol_features)}")
            logger.info(f"  - Cross-symbol features shape: {universal_features.cross_symbol_features.shape}")
            logger.info(f"  - Market regime features shape: {universal_features.market_regime_features.shape}")
            logger.info(f"  - Sector features shape: {universal_features.sector_features.shape}")
            logger.info(f"  - Universal embeddings shape: {universal_features.universal_embeddings.shape}")
            
            # Count total individual features across all symbols
            total_individual_features = 0
            for symbol, features in universal_features.symbol_features.items():
                if hasattr(features, 'technical_features'):
                    symbol_feature_count = len(features.technical_features.columns) if not features.technical_features.empty else 0
                    logger.info(f"  - {symbol} individual features: {symbol_feature_count}")
                    total_individual_features += symbol_feature_count
            
            logger.info(f"  - Total individual features across all symbols: {total_individual_features}")
            logger.info(f"  - Expected total features (153 validated from Supabase): Should match or exceed this number")
        
        # Get universal training data
        logger.info("Step 3: Preparing universal training data...")
        X, y = await self.feature_engineering.prepare_universal_training_data(
            universal_features=universal_features,
            target_column='target'
        )
        logger.info(f"Step 3 completed: X type={type(X)}, y type={type(y)}")
        
        logger.info(f"Raw data shapes: X={X.shape if not X.empty else 'empty'}, y={y.shape if not y.empty else 'empty'}")
        
        if not X.empty:
            logger.info(f"Step 3 - Feature analysis:")
            logger.info(f"  - Total columns in X: {len(X.columns)}")
            logger.info(f"  - Total rows in X: {len(X)}")
            logger.info(f"  - Column names: {list(X.columns)}")
            
            # Analyze feature types with correct naming patterns
            # Cross-symbol features: corr_, beta_, relative_strength_, market_dispersion_
            cross_cols = [col for col in X.columns if any([
                col.startswith('corr_'),
                col.startswith('beta_'),
                col.startswith('relative_strength_'),
                col.startswith('market_dispersion_')
            ])]
            
            # Market regime features: market_volatility, vol_regime_, vol_trend, vol_correlation
            regime_cols = [col for col in X.columns if any([
                col.startswith('market_volatility'),
                col.startswith('vol_regime_'),
                col.startswith('vol_trend'),
                col.startswith('vol_correlation')
            ])]
            
            # Symbol and sector features
            symbol_cols = [col for col in X.columns if col.startswith('symbol_') or col == 'symbol_id']
            sector_cols = [col for col in X.columns if col.startswith('sector_')]
            
            # Technical features: everything else except the above categories
            excluded_prefixes = set()
            for col in cross_cols + regime_cols + symbol_cols + sector_cols:
                excluded_prefixes.add(col)
            technical_cols = [col for col in X.columns if col not in excluded_prefixes]
            
            logger.info(f"  - Technical features: {len(technical_cols)} columns")
            logger.info(f"  - Symbol features: {len(symbol_cols)} columns")
            logger.info(f"  - Cross-symbol features: {len(cross_cols)} columns")
            logger.info(f"  - Market regime features: {len(regime_cols)} columns")
            logger.info(f"  - Sector features: {len(sector_cols)} columns")
            
            # Validate against expected feature count from Supabase validation report
            expected_features = 153
            actual_technical_features = len(technical_cols)
            
            if actual_technical_features < expected_features:
                logger.warning(f"  - ⚠️  FEATURE COUNT MISMATCH: Only {actual_technical_features} technical features found, expected {expected_features}")
                logger.warning(f"  - Missing {expected_features - actual_technical_features} features from expected count")
                logger.info(f"  - Technical feature names: {technical_cols[:10]}..." if len(technical_cols) > 10 else f"  - Technical feature names: {technical_cols}")
            elif actual_technical_features == expected_features:
                logger.info(f"  - ✅ FEATURE COUNT VALIDATED: {actual_technical_features} features matches expected {expected_features}")
            else:
                logger.info(f"  - ✅ FEATURE COUNT EXCEEDED: {actual_technical_features} features (expected {expected_features})")
            
            # Note: Total feature validation moved after feature selection to show accurate counts
        
        if X.empty or y.empty:
            logger.error("No training data available")
            return [], [], [], []
        
        # Extract symbol IDs and features separately
        # DEBUG: Check symbol_id values before extraction
        logger.info(f"DEBUG: symbol_id column unique values: {X['symbol_id'].unique()}")
        logger.info(f"DEBUG: symbol_id column min: {X['symbol_id'].min()}, max: {X['symbol_id'].max()}")
        logger.info(f"DEBUG: symbol_id column dtype: {X['symbol_id'].dtype}")
        logger.info(f"DEBUG: Expected symbol_id range: [0, {len(symbols)-1}] for symbols: {symbols}")
        
        # Validate symbol_id values are in expected range
        valid_symbol_ids = (X['symbol_id'] >= 0) & (X['symbol_id'] < len(symbols))
        invalid_count = (~valid_symbol_ids).sum()
        
        if invalid_count > 0:
            logger.error(f"CRITICAL: Found {invalid_count} invalid symbol_id values out of {len(X)} total rows")
            logger.error(f"Invalid symbol_id values: {X.loc[~valid_symbol_ids, 'symbol_id'].unique()}")
            logger.error(f"Valid range should be [0, {len(symbols)-1}] for symbols: {symbols}")
            
            # Fix invalid symbol_ids by mapping them to valid range
            logger.warning("Attempting to fix invalid symbol_ids by clipping to valid range...")
            X['symbol_id'] = X['symbol_id'].clip(0, len(symbols)-1)
            logger.info(f"After clipping - symbol_id unique values: {X['symbol_id'].unique()}")
        
        symbol_ids = X['symbol_id'].values.astype(np.int32)
        
        # Define actual symbol embedding columns to exclude (not cross-symbol or market regime features)
        # FIXED: Exclude all symbol columns that match the pattern to maintain consistency
        symbol_embedding_cols = [col for col in X.columns if (
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
        
        # Keep all features except actual symbol embedding columns
        feature_columns = [col for col in X.columns if col not in symbol_embedding_cols]
        
        # Store original feature columns for later use
        original_feature_columns = feature_columns.copy()
        
        # Apply feature selection if available
        if self.selected_features is not None and len(self.selected_features) > 0:
            logger.info(f"Feature selection available: {len(self.selected_features)} selected features")
            logger.info(f"Selected features format: {self.selected_features[:5]}...")
            logger.info(f"DataFrame columns format: {feature_columns[:5]}...")
            
            # Check if selected features are in temporal aggregation format (feature_X_aggregation)
            # If so, we need to map them back to original feature indices
            if any('_' in feat and feat.startswith('feature_') for feat in self.selected_features):
                logger.info("Detected temporal aggregation format in selected features")
                
                # Extract feature indices from temporal aggregation names
                # e.g., 'feature_5_max' -> index 5
                selected_indices = set()
                for feat_name in self.selected_features:
                    if feat_name.startswith('feature_') and '_' in feat_name:
                        try:
                            # Extract the number between 'feature_' and the aggregation type
                            parts = feat_name.split('_')
                            if len(parts) >= 3:  # feature_X_aggregation
                                idx = int(parts[1])
                                selected_indices.add(idx)
                        except (ValueError, IndexError):
                            logger.warning(f"Could not parse feature index from: {feat_name}")
                            continue
                
                logger.info(f"Extracted {len(selected_indices)} unique feature indices: {sorted(list(selected_indices))[:10]}...")
                
                # Map indices back to actual column names
                # FIXED: Use the original full column list (192 columns) for mapping, not just technical features
                # The selected feature indices were calculated from the original dataset before filtering
                original_feature_columns = [col for col in X.columns if col not in symbol_embedding_cols]
                logger.info(f"Original feature columns for mapping: {len(original_feature_columns)} (before any filtering)")
                logger.info(f"Technical features after filtering: {len([col for col in feature_columns if not any([col.startswith('corr_'), col.startswith('beta_'), col.startswith('relative_strength_'), col.startswith('market_dispersion_'), col.startswith('market_volatility'), col.startswith('vol_regime_'), col.startswith('vol_trend'), col.startswith('vol_correlation'), col.startswith('sector_')])])}")
                
                # Select features based on the indices using the original full column list
                selected_feature_columns = []
                sorted_indices = sorted(list(selected_indices))
                
                logger.info(f"Mapping {len(sorted_indices)} selected indices to column names:")
                for idx in sorted_indices:
                    if idx < len(original_feature_columns):
                        col_name = original_feature_columns[idx]
                        selected_feature_columns.append(col_name)
                        logger.info(f"  Index {idx} -> '{col_name}'")
                    else:
                        logger.warning(f"Feature index {idx} exceeds available original features ({len(original_feature_columns)})")
                
                # STRICT FEATURE SELECTION: Only use the selected features, no additional features
                # Removed automatic inclusion of cross-symbol and market regime features
                # to maintain consistency with the 65 selected features throughout training
                logger.info(f"Strictly using only {len(selected_feature_columns)} selected features (no additional cross-symbol/market regime features)")
                
                logger.info(f"Mapped to {len(selected_feature_columns)} actual columns (strictly selected features only):")
                logger.info(f"  - Selected features used: {len(selected_feature_columns)}")
                logger.info(f"  - Cross-symbol features: NOT automatically included (strict selection mode)")
                logger.info(f"  - Market regime features: NOT automatically included (strict selection mode)")
                
                feature_columns = selected_feature_columns
            else:
                # Direct column name matching (fallback)
                selected_feature_columns = [col for col in feature_columns if col in self.selected_features]
                logger.info(f"Direct column matching: {len(feature_columns)} -> {len(selected_feature_columns)} features")
                feature_columns = selected_feature_columns
            
            logger.info(f"Final feature selection result: {len(feature_columns)} features")
        
        # Log which columns are being excluded vs included for debugging
        excluded_cols = [col for col in X.columns if col in symbol_embedding_cols]
        cross_symbol_cols = [col for col in X.columns if any([
            col.startswith('corr_'),
            col.startswith('beta_'),
            col.startswith('relative_strength_'),
            col.startswith('market_dispersion_')
        ])]
        market_regime_cols = [col for col in X.columns if any([
            col.startswith('market_volatility'),
            col.startswith('vol_regime_'),
            col.startswith('vol_trend'),
            col.startswith('vol_correlation')
        ])]
        
        logger.info(f"Feature filtering results:")
        logger.info(f"  - Excluded symbol embedding columns ({len(excluded_cols)}): {excluded_cols}")
        logger.info(f"  - Included cross-symbol features ({len(cross_symbol_cols)}): {cross_symbol_cols}")
        logger.info(f"  - Included market regime features ({len(market_regime_cols)}): {market_regime_cols}")
        logger.info(f"  - Total feature columns kept: {len(feature_columns)}")
        if self.selected_features is not None:
            logger.info(f"  - Feature selection applied: {len(self.selected_features)} selected features")
        
        # Final validation for total feature count after feature selection
        logger.info(f"  - ✅ FINAL FEATURE COUNT VALIDATION: {len(feature_columns)} total columns after all filtering")
        
        # Log feature distribution for debugging (after feature selection)
        final_technical_cols = [col for col in feature_columns if not any([
            col.startswith('corr_'), col.startswith('beta_'), col.startswith('relative_strength_'),
            col.startswith('market_dispersion_'), col.startswith('market_volatility'),
            col.startswith('vol_regime_'), col.startswith('vol_trend'), col.startswith('vol_correlation'),
            col.startswith('sector_')
        ])]
        final_cross_cols = [col for col in feature_columns if any([
            col.startswith('corr_'), col.startswith('beta_'), col.startswith('relative_strength_'),
            col.startswith('market_dispersion_')
        ])]
        final_regime_cols = [col for col in feature_columns if any([
            col.startswith('market_volatility'), col.startswith('vol_regime_'),
            col.startswith('vol_trend'), col.startswith('vol_correlation')
        ])]
        final_sector_cols = [col for col in feature_columns if col.startswith('sector_')]
        
        logger.info(f"  - Final feature distribution breakdown:")
        logger.info(f"    * Technical: {len(final_technical_cols)} ({len(final_technical_cols)/len(feature_columns)*100:.1f}%)")
        logger.info(f"    * Cross-symbol: {len(final_cross_cols)} ({len(final_cross_cols)/len(feature_columns)*100:.1f}%)")
        logger.info(f"    * Market regime: {len(final_regime_cols)} ({len(final_regime_cols)/len(feature_columns)*100:.1f}%)")
        logger.info(f"    * Sector: {len(final_sector_cols)} ({len(final_sector_cols)/len(feature_columns)*100:.1f}%)")
        
        features = X[feature_columns].values.astype(np.float32)
        targets = y.values.astype(np.float32)
        
        logger.info(f"Extracted data: symbol_ids={len(symbol_ids)}, features={features.shape}, targets={len(targets)}")
        logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}...")  # Show first 5 columns
        logger.info(f"Step 4 - Feature extraction details:")
        logger.info(f"  - Total feature columns (excluding symbol_id): {len(feature_columns)}")
        logger.info(f"  - Feature column names: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"  - Feature column names: {feature_columns}")
        logger.info(f"  - Unique symbols in dataset: {np.unique(symbol_ids)}")
        logger.info(f"  - Features array shape: {features.shape}")
        logger.info(f"  - Targets array shape: {targets.shape}")
        
        if features.shape[1] != len(feature_columns):
            logger.error(f"  - MISMATCH: Expected {len(feature_columns)} features but got {features.shape[1]}")
        
        # Calculate expected unique feature count dynamically
        expected_unique_features = self._calculate_expected_unique_features()
        if features.shape[1] < expected_unique_features:
             logger.warning(f"  - WARNING: Only {features.shape[1]} features in final array, expected ~{expected_unique_features} (unique base features from selection)")
        
        # Check if we need to enforce 2D data pipeline for statistical models
        force_2d = getattr(config, 'force_2d_for_statistical', False) if config else False
        
        if force_2d:
            logger.info("Enforcing 2D data pipeline for statistical models - skipping sequence creation")
            # For statistical models, data is already in 2D format - no temporal aggregation needed
            # The features are already aggregated/flattened from the data pipeline
            logger.info(f"Using 2D features directly for statistical models: {features.shape}")
            
            # For 2D pipeline, no sequence creation - use direct train/val split
            validation_split = getattr(config, 'base_validation_split', 0.2) if config else 0.2
            split_idx = int(len(features) * (1 - validation_split))
            
            X_train_features = features[:split_idx]
            X_train_symbols = symbol_ids[:split_idx]
            y_train = targets[:split_idx]
            
            X_val_features = features[split_idx:]
            X_val_symbols = symbol_ids[split_idx:]
            y_val = targets[split_idx:]
            
            logger.info(f"2D Pipeline - Prepared dataset: {len(X_train_features)} training, {len(X_val_features)} validation samples")
            logger.info(f"2D Pipeline - Feature shape: {X_train_features.shape}")
        else:
            # Original 3D sequence pipeline
            # Reshape features for sequence models (assuming lookback_window)
            lookback_window = getattr(config, 'base_lookback_window', 30) if config else 30
            
            # Create sequences
            X_sequences = []
            X_symbols = []
            y_sequences = []
        
            logger.info(f"Creating sequences with lookback_window={lookback_window}, total_samples={len(features)}")
            
            for i in range(lookback_window, len(features)):
                X_sequences.append(features[i-lookback_window:i])
                X_symbols.append(symbol_ids[i])
                y_sequences.append(targets[i])
            
            logger.info(f"Created {len(X_sequences)} sequences")
            logger.info(f"Step 4 - Sequence details:")
            if len(X_sequences) > 0:
                X_sequences_array = np.array(X_sequences)
                logger.info(f"  - Input sequences shape: {X_sequences_array.shape} (samples, timesteps, features)")
                logger.info(f"  - Features per timestep: {X_sequences_array.shape[2] if len(X_sequences_array.shape) > 2 else 'N/A'}")
                logger.info(f"  - Target sequences shape: {np.array(y_sequences).shape}")
            
            if len(X_sequences) == 0:
                logger.error(f"Insufficient data for sequence creation: need at least {lookback_window} samples, got {len(features)}")
                return [], [], [], []
            
            X_sequences = np.array(X_sequences)
            X_symbols = np.array(X_symbols)
            y_sequences = np.array(y_sequences)
            
            # Split into training and validation
            split_idx = int(len(X_sequences) * (1 - self.config.base_validation_split))
            
            X_train_features = X_sequences[:split_idx]
            X_train_symbols = X_symbols[:split_idx]
            y_train = y_sequences[:split_idx]
            
            X_val_features = X_sequences[split_idx:]
            X_val_symbols = X_symbols[split_idx:]
            y_val = y_sequences[split_idx:]
        
        logger.info(f"Prepared dataset: {len(X_train_features)} training, {len(X_val_features)} validation samples")
        logger.info(f"Feature shape: {X_train_features.shape}, Symbol shape: {X_train_symbols.shape}")
        logger.info(f"Step 5 - Final dataset summary:")
        logger.info(f"  - Training samples: {X_train_features.shape[0]}")
        logger.info(f"  - Validation samples: {X_val_features.shape[0]}")
        logger.info(f"  - Features per sample: {X_train_features.shape[2] if len(X_train_features.shape) > 2 else 'N/A'}")
        logger.info(f"  - Timesteps per sample: {X_train_features.shape[1] if len(X_train_features.shape) > 1 else 'N/A'}")
        
        # Apply class imbalance mitigation if enabled (AFTER feature selection)
        if self.config.enable_imbalance_mitigation and self.imbalance_mitigator:
            logger.info("Step 6 - Applying class imbalance mitigation (with selected features only)")
            
            # Analyze original class distribution
            original_metrics = self.imbalance_mitigator.analyze_class_distribution(y_train)
            logger.info(f"Original class distribution: positive={original_metrics.get('positive_ratio', 0):.3f}, negative={original_metrics.get('negative_ratio', 0):.3f}")
            
            # Store original sample count for symbol replication
            original_sample_count = len(X_train_features)
            logger.info(f"Original sample count: {original_sample_count}")
            logger.info(f"Applying SMOTE to data with {X_train_features.shape[-1] if hasattr(X_train_features, 'shape') else 'unknown'} selected features")
            
            # Apply comprehensive balancing (SMOTE + class weights)
            X_train_balanced, y_train_balanced, imbalance_metrics = self.imbalance_mitigator.apply_comprehensive_balancing(
                X_train_features, y_train, apply_to_validation=False
            )
            
            # Update training data with balanced data
            X_train_features = X_train_balanced
            y_train = y_train_balanced
            
            # Verify that SMOTE maintained feature consistency
            if hasattr(X_train_features, 'shape') and len(X_train_features.shape) > 1:
                logger.info(f"SMOTE result verification: {X_train_features.shape[-1]} features maintained (expected: {len(feature_columns)})")
                if X_train_features.shape[-1] != len(feature_columns):
                    logger.warning(f"Feature count mismatch after SMOTE: expected {len(feature_columns)}, got {X_train_features.shape[-1]}")
            
            # Handle symbol data replication if SMOTE was applied
            if imbalance_metrics.smote_applied and len(X_train_features) > original_sample_count:
                new_sample_count = len(X_train_features)
                synthetic_samples_added = new_sample_count - original_sample_count
                
                logger.info(f"SMOTE increased samples from {original_sample_count} to {new_sample_count}")
                logger.info(f"Replicating symbol data for {synthetic_samples_added} synthetic samples")
                
                # Replicate symbol data to match SMOTE-enhanced feature data
                # For synthetic samples, we'll replicate symbols from the original samples
                # This is a reasonable approach since SMOTE creates synthetic samples by interpolating
                # between existing samples, so we can use the symbols from the original data
                
                # Create indices for replication - repeat original symbols cyclically
                original_indices = np.arange(original_sample_count)
                synthetic_indices = np.tile(original_indices, (synthetic_samples_added // original_sample_count) + 1)[:synthetic_samples_added]
                
                # Combine original and synthetic indices
                all_indices = np.concatenate([original_indices, synthetic_indices])
                
                # Replicate symbol data using the indices
                X_train_symbols_replicated = X_train_symbols[all_indices]
                X_train_symbols = X_train_symbols_replicated
                
                logger.info(f"Symbol data replicated: original shape {X_train_symbols.shape} -> new shape {X_train_symbols_replicated.shape}")
                
                # Validate data consistency
                if len(X_train_features) != len(X_train_symbols):
                    logger.error(f"Data cardinality mismatch after symbol replication: features={len(X_train_features)}, symbols={len(X_train_symbols)}")
                    raise ValueError(f"Sample count mismatch: features={len(X_train_features)}, symbols={len(X_train_symbols)}")
                else:
                    logger.info(f"Data cardinality validation passed: both features and symbols have {len(X_train_features)} samples")
            
            # Store class weights and metrics for later use in model training
            self.class_weights_keras = imbalance_metrics.class_weights
            self.class_weights_sklearn = self.imbalance_mitigator.get_sklearn_sample_weights(y_train)
            
            # Initialize imbalance_metrics list if it doesn't exist
            if not hasattr(self, 'imbalance_metrics'):
                self.imbalance_metrics = []
            self.imbalance_metrics.append(imbalance_metrics)
            
            logger.info(f"Applied SMOTE: {imbalance_metrics.smote_applied}")
            logger.info(f"New training samples: {len(X_train_features)}")
            logger.info(f"Final positive ratio: {imbalance_metrics.final_positive_ratio:.3f}")
            logger.info(f"Synthetic samples added: {imbalance_metrics.synthetic_samples_added}")
            logger.info(f"Class weights (Keras): {self.class_weights_keras}")
        else:
            # No imbalance mitigation - set default class weights
            self.class_weights_keras = None
            self.class_weights_sklearn = None
        
        return [X_train_features, X_train_symbols], y_train, [X_val_features, X_val_symbols], y_val
    
    async def perform_feature_selection(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        config: FeatureSelectionConfig = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive feature selection analysis.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            config: Feature selection configuration
            
        Returns:
            Dictionary containing feature selection results
        """
        logger.info("Starting comprehensive feature selection analysis")
        
        # Use default config if none provided
        if config is None:
            config = FeatureSelectionConfig()
        
        # Prepare dataset for feature selection
        X_train, y_train, X_val, y_val = await self.prepare_universal_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        if not X_train or len(X_train) == 0:
            logger.error("No training data available for feature selection")
            return {}
        
        # Extract features (exclude symbol embeddings for feature selection)
        if isinstance(X_train, list) and len(X_train) >= 2:
            features_train = X_train[0]  # Feature matrix
            features_val = X_val[0] if X_val else None
        else:
            features_train = X_train
            features_val = X_val
        
        # Handle 3D data by converting to 2D using temporal aggregation
        if isinstance(features_train, np.ndarray) and len(features_train.shape) == 3:
            logger.info("Detected 3D training data, applying temporal aggregation for feature selection")
            features_train = self.prepare_3d_for_feature_selection(features_train)
            
            if features_val is not None and isinstance(features_val, np.ndarray) and len(features_val.shape) == 3:
                logger.info("Detected 3D validation data, applying temporal aggregation for feature selection")
                features_val = self.prepare_3d_for_feature_selection(features_val)
        
        # Initialize feature selector
        self.feature_selector = UniversalFeatureSelector(config)
        
        # Perform feature selection
        results = await self.feature_selector.select_features(
            X_train=features_train,
            y_train=y_train,
            X_val=features_val,
            y_val=y_val,
            feature_names=None  # Will be inferred from data
        )
        
        # Store selected features for use in training
        self.selected_features = results.get('selected_features', [])
        self.selected_feature_indices = results.get('selected_feature_indices', [])
        
        logger.info(f"Feature selection completed. Selected {len(self.selected_features)} features")
        logger.info(f"Feature reduction: {results.get('original_feature_count', 0)} -> {len(self.selected_features)}")
        logger.info(f"Selected features: {self.selected_features}")
        logger.info(f"Selected feature indices: {self.selected_feature_indices}")
        
        return results
    
    def prepare_3d_for_feature_selection(self, data_3d: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Convert 3D temporal data to 2D aggregated DataFrame for feature selection.
        
        Args:
            data_3d: 3D numpy array with shape (samples, timesteps, features)
            feature_names: Optional list of feature names
            
        Returns:
            2D DataFrame with aggregated features suitable for feature selection
        """
        if not self.config.enable_temporal_aggregation or self.temporal_aggregator is None:
            logger.warning("Temporal aggregation is disabled, cannot convert 3D data")
            raise ValueError("3D data requires temporal aggregation to be enabled")
        
        logger.info(f"Converting 3D data {data_3d.shape} to 2D DataFrame for feature selection")
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data_3d.shape[2])]
        
        # Use temporal aggregator to convert 3D to 2D
        aggregated_df = self.temporal_aggregator.aggregate_3d_to_dataframe(
            data_3d=data_3d,
            feature_names=feature_names
        )
        
        logger.info(f"Successfully converted to 2D DataFrame: {aggregated_df.shape}")
        return aggregated_df
    
    async def phase1_train_base_models(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        model_types: List[str] = None
    ) -> Dict[str, UniversalTrainingResult]:
        """
        Phase 1: Train universal base models on all symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for training data
            end_date: End date for training data
            model_types: List of model types to train
            
        Returns:
            Dictionary of training results by model type
        """
        if model_types is None:
            model_types = ['lstm', 'cnn', 'transformer']
        
        logger.info(f"Phase 1: Training universal base models for {model_types}")
        
        # Log training configurations for quick reference
        logger.info("=== Universal Training Configurations ===")
        for model_type in model_types:
            if model_type in self.model_configs:
                config = self.model_configs[model_type]
                params = config.parameters
                logger.info(f"{model_type.upper()} Config: units/filters={params.get('units', params.get('filters', 'N/A'))}, "
                           f"dropout={params.get('dropout', 'N/A')}, epochs={params.get('epochs', 'N/A')}, "
                           f"batch_size={params.get('batch_size', 'N/A')}, lr={params.get('learning_rate', config.learning_rate)}, "
                           f"lookback={config.lookback_window}min, threshold={config.prediction_threshold}")
        logger.info("=============================================")
        
        # Initialize symbol mappings
        await self.initialize_symbol_mappings(symbols)
        
        # Prepare universal dataset
        X_train, y_train, X_val, y_val = await self.prepare_universal_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Check if data preparation succeeded
        if not X_train or len(X_train) == 0 or (isinstance(X_train, list) and len(X_train[0]) == 0):
            logger.error("No training data available after preparation - cannot train models")
            return {}
        
        logger.info(f"Data preparation successful: {len(X_train[0]) if isinstance(X_train, list) else len(X_train)} training samples")
        
        # Handle 2D aggregated data from SMOTE (post-aggregation)
        if isinstance(X_train, list) and len(X_train) >= 2:
            # Check if we have 2D aggregated data (from SMOTE temporal aggregation)
            # SMOTE returns pandas DataFrame for 3D->2D aggregated data
            if isinstance(X_train[0], pd.DataFrame):
                # 2D aggregated features from SMOTE temporal aggregation: (samples, aggregated_features)
                actual_feature_count = X_train[0].shape[-1]
                is_aggregated_data = True
                logger.info(f"Detected 2D aggregated DataFrame from SMOTE: {X_train[0].shape[0]} samples, {actual_feature_count} aggregated features")
            elif hasattr(X_train[0], 'shape') and len(X_train[0].shape) == 2:
                # 2D numpy array: (samples, aggregated_features)
                actual_feature_count = X_train[0].shape[-1]
                is_aggregated_data = True
                logger.info(f"Detected 2D aggregated numpy array: {X_train[0].shape[0]} samples, {actual_feature_count} aggregated features")
            else:
                # 3D sequence data: (samples, timesteps, features)
                actual_feature_count = X_train[0].shape[-1] if hasattr(X_train[0], 'shape') else len(X_train[0][0])
                is_aggregated_data = False
                logger.info(f"Detected 3D sequence data: using {actual_feature_count} features per timestep")
            
            # Check if feature selection is active for logging purposes
            if hasattr(self, 'selected_features') and self.selected_features is not None and len(self.selected_features) > 0:
                logger.info(f"Feature selection active: using {actual_feature_count} {'aggregated' if is_aggregated_data else 'sequence'} features (mapped from {len(self.selected_features)} selected features)")
            else:
                logger.info(f"No feature selection active: using actual feature count {actual_feature_count}")
            
            # Update all model configs with actual feature count (post-mapping)
            for model_type in model_types:
                if model_type in self.model_configs:
                    old_count = self.model_configs[model_type].feature_count
                    self.model_configs[model_type].feature_count = actual_feature_count
                    logger.info(f"Updated {model_type} feature_count from {old_count} to {actual_feature_count} (using {'aggregated' if is_aggregated_data else 'sequence'} features)")
        
        results = {}
        
        for model_type in model_types:
            start_time = datetime.now()
            logger.info(f"Training universal {model_type} model")
            
            try:
                # Get model configuration
                config = self.model_configs[model_type]
                
                # Create universal model - handle both 2D aggregated and 3D sequence data
                if is_aggregated_data:
                    # For 2D aggregated data, create dense neural networks
                    logger.info(f"Creating dense neural network for {model_type} with {config.feature_count} aggregated features")
                    
                    # Convert DataFrame to numpy array if needed for model training
                    if isinstance(X_train[0], pd.DataFrame):
                        logger.info("Converting DataFrame to numpy array for model training")
                        X_train_array = X_train[0].values.astype(np.float32)
                        X_train_symbols_array = X_train[1]
                        X_train = [X_train_array, X_train_symbols_array]
                        
                        if X_val and isinstance(X_val[0], pd.DataFrame):
                            X_val_array = X_val[0].values.astype(np.float32)
                            X_val_symbols_array = X_val[1]
                            X_val = [X_val_array, X_val_symbols_array]
                            logger.info(f"Converted validation DataFrame to numpy array: {X_val_array.shape}")
                        
                        logger.info(f"Converted training DataFrame to numpy array: {X_train_array.shape}")
                    
                    model = self.universal_architectures.create_universal_dense(
                        feature_dim=config.feature_count,
                        config=config.parameters,
                        model_name=f"universal_{model_type}_dense"
                    )
                else:
                    # For 3D sequence data, use original sequence-based models
                    if model_type == 'lstm':
                        model = self.universal_architectures.create_universal_lstm(
                            sequence_length=config.lookback_window,
                            feature_dim=config.feature_count,
                            config=config.parameters
                        )
                    elif model_type == 'cnn':
                        model = self.universal_architectures.create_universal_cnn(
                            sequence_length=config.lookback_window,
                            feature_dim=config.feature_count,
                            config=config.parameters
                        )
                    elif model_type == 'transformer':
                        model = self.universal_architectures.create_universal_transformer(
                            sequence_length=config.lookback_window,
                            feature_dim=config.feature_count,
                            config=config.parameters
                        )
                    else:
                        logger.warning(f"Unsupported model type: {model_type}")
                        continue
                
                # Setup callbacks
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        patience=self.config.early_stopping_patience,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        patience=self.config.reduce_lr_patience,
                        factor=0.5,
                        min_lr=1e-7,
                        verbose=1,
                        cooldown=1
                    ),
                    NaNDetectionCallback()
                ]
                
                # Train model with class weights if available
                # Use model-specific training parameters with base config as fallback
                model_epochs = config.parameters.get('epochs', self.config.base_epochs)
                model_batch_size = config.parameters.get('batch_size', self.config.base_batch_size)
                
                fit_kwargs = {
                    'x': X_train,
                    'y': y_train,
                    'validation_data': (X_val, y_val),
                    'epochs': model_epochs,
                    'batch_size': model_batch_size,
                    'callbacks': callbacks,
                    'verbose': 1
                }
                
                # Add class weights if imbalance mitigation is enabled
                if hasattr(self, 'class_weights_keras') and self.class_weights_keras is not None:
                    fit_kwargs['class_weight'] = self.class_weights_keras
                    logger.info(f"Training {model_type} with class weights: {self.class_weights_keras}")
                
                history = model.fit(**fit_kwargs)
                
                # Evaluate model
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                
                # Validate model outputs
                if np.isnan(val_loss) or np.isnan(val_accuracy):
                    logger.error(f"Model {model_type} produced NaN metrics: loss={val_loss}, accuracy={val_accuracy}")
                    continue
                
                # Test model predictions for NaN
                test_predictions = model.predict(X_val[:10], verbose=0)  # Test with small batch
                if np.any(np.isnan(test_predictions)) or np.any(np.isinf(test_predictions)):
                    logger.error(f"Model {model_type} produces NaN/Inf predictions")
                    continue
                
                logger.info(f"Model {model_type} validation passed: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")
                
                # Store base model
                self.base_models[model_type] = model
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Prepare metadata with training history and imbalance metrics
                metadata = {
                    'model_summary': self.universal_architectures.get_model_summary(model),
                    'training_history': {
                        'loss': [float(x) for x in history.history['loss']],
                        'val_loss': [float(x) for x in history.history['val_loss']],
                        'accuracy': [float(x) for x in history.history['accuracy']],
                        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
                    }
                }
                
                # Add imbalance mitigation metrics if available
                if hasattr(self, 'imbalance_metrics') and self.imbalance_metrics:
                    latest_metrics = self.imbalance_metrics[-1]
                    metadata['imbalance_mitigation'] = {
                        'original_positive_ratio': latest_metrics.original_positive_ratio,
                        'final_positive_ratio': latest_metrics.final_positive_ratio,
                        'improvement_ratio': latest_metrics.improvement_ratio,
                        'smote_applied': latest_metrics.smote_applied,
                        'synthetic_samples_added': latest_metrics.synthetic_samples_added,
                        'class_weights_applied': latest_metrics.class_weights_applied,
                        'class_weights': latest_metrics.class_weights,
                        'class_weights_keras': self.class_weights_keras,
                        'class_weights_sklearn': self.class_weights_sklearn
                    }
                
                # Create result
                result = UniversalTrainingResult(
                    phase="phase1_base_training",
                    model_name=model_type,
                    symbols_trained=symbols,
                    base_model_performance={
                        'validation_loss': float(val_loss),
                        'validation_accuracy': float(val_accuracy),
                        'final_lr': float(model.optimizer.learning_rate.numpy())
                    },
                    symbol_performances={},
                    ensemble_weights={},
                    training_time=training_time,
                    total_samples=len(X_train[0]) if isinstance(X_train, list) else len(X_train),
                    validation_accuracy=float(val_accuracy),
                    metadata=metadata
                )
                
                results[model_type] = result
                logger.info(f"Completed {model_type} base training: {val_accuracy:.4f} accuracy in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} base model: {e}")
                continue
        
        # Log imbalance mitigation summary if applied
        if self.config.enable_imbalance_mitigation and self.imbalance_metrics:
            self._log_imbalance_summary()
        
        logger.info(f"Phase 1 completed: {len(results)} base models trained")
        return results
    
    def _log_imbalance_summary(self):
        """Log summary of class imbalance mitigation effects"""
        if not self.imbalance_metrics:
            return
        
        latest_metrics = self.imbalance_metrics[-1]
        
        logger.info("=== Class Imbalance Mitigation Summary ===")
        logger.info(f"Original positive ratio: {latest_metrics.original_positive_ratio:.3f}")
        logger.info(f"Final positive ratio: {latest_metrics.final_positive_ratio:.3f}")
        logger.info(f"Improvement ratio: {latest_metrics.improvement_ratio:.2f}x")
        logger.info(f"SMOTE applied: {latest_metrics.smote_applied}")
        logger.info(f"Synthetic samples added: {latest_metrics.synthetic_samples_added}")
        logger.info(f"Class weights applied: {latest_metrics.class_weights_applied}")
        if latest_metrics.class_weights:
            logger.info(f"Class weights: {latest_metrics.class_weights}")
        logger.info("============================================")
    
    def get_imbalance_metrics(self) -> List[ImbalanceMetrics]:
        """Get all imbalance mitigation metrics from training"""
        return self.imbalance_metrics.copy() if hasattr(self, 'imbalance_metrics') else []
    
    async def phase2_symbol_specific_finetuning(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        model_types: List[str] = None
    ) -> Dict[str, UniversalTrainingResult]:
        """
        Phase 2: Fine-tune base models for specific symbols.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for fine-tuning data
            end_date: End date for fine-tuning data
            model_types: List of model types to fine-tune
            
        Returns:
            Dictionary of fine-tuning results by model type
        """
        if model_types is None:
            model_types = list(self.base_models.keys())
        
        logger.info(f"Phase 2: Fine-tuning models for {len(symbols)} symbols")
        
        results = {}
        
        for model_type in model_types:
            if model_type not in self.base_models:
                logger.warning(f"No base model found for {model_type}, skipping fine-tuning")
                continue
            
            start_time = datetime.now()
            base_model = self.base_models[model_type]
            symbol_performances = {}
            
            # Fine-tune for each symbol
            for symbol in symbols:
                try:
                    symbol_id = self.symbol_to_id[symbol]
                    
                    # Load symbol-specific data
                    # Convert dates to timezone-aware UTC datetime objects
                    # Handle both string and datetime inputs
                    if isinstance(start_date, str):
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(start_date, datetime):
                        # Ensure timezone-aware UTC
                        start_dt = start_date.replace(tzinfo=timezone.utc) if start_date.tzinfo is None else start_date.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"start_date must be str or datetime, got {type(start_date)}")
                        
                    if isinstance(end_date, str):
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(end_date, datetime):
                        # Ensure timezone-aware UTC
                        end_dt = end_date.replace(tzinfo=timezone.utc) if end_date.tzinfo is None else end_date.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"end_date must be str or datetime, got {type(end_date)}")
                    
                    symbol_data = await self.data_pipeline.load_market_data(
                        symbol=symbol,
                        start_date=start_dt,
                        end_date=end_dt
                    )
                    
                    if len(symbol_data) < self.config.min_samples_per_symbol:
                        logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} samples")
                        continue
                    
                    # Prepare symbol-specific features
                    features = await self.feature_engineering.engineer_features(
                        symbol=symbol,
                        start_date=start_dt,
                        end_date=end_dt
                    )
                    
                    # Create symbol-specific model
                    symbol_model = self.universal_architectures.create_symbol_specific_head(
                        base_model=base_model,
                        symbol_id=symbol_id,
                        config={
                            'layers_to_unfreeze': self.config.layers_to_unfreeze,
                            'fine_tune_lr': self.config.finetune_learning_rate,
                            'dropout': 0.2
                        }
                    )
                    
                    # Prepare training data using universal feature preparation for consistent dimensions
                    X_features = await self._prepare_universal_features_for_symbol(
                        symbol=symbol,
                        feature_set=features,
                        start_date=start_dt,
                        end_date=end_dt
                    )
                    y = self._extract_targets_from_market_data(symbol_data)
                    
                    # Align X_features and X_symbols with targets length (targets are shortened by 1)
                    X_features = X_features[:len(y)]
                    X_symbols = np.full(len(X_features), symbol_id)
                    
                    # Create sequences for all model types (LSTM, CNN, Transformer)
                    config = self.model_configs[model_type]
                    lookback_window = config.lookback_window
                    
                    logger.info(f"Creating sequences for {symbol} with lookback_window={lookback_window}")
                    
                    # Use helper method to create sequences
                    X_sequences, y_sequences = self.create_sequences(X_features, y, lookback_window)
                    
                    if len(X_sequences) == 0:
                        logger.warning(f"Insufficient data for {symbol}: need at least {lookback_window} samples, got {len(X_features)}")
                        continue
                    
                    # Create symbol sequences for the valid sequence length (matching create_sequences output)
                    X_symbols_seq = np.array([X_symbols[i] for i in range(lookback_window, len(X_features))])
                    
                    logger.info(f"Created {len(X_sequences)} sequences for {symbol} with shape {X_sequences.shape}")
                    
                    # Split for validation
                    split_idx = int(len(X_sequences) * 0.8)
                    X_train = [X_sequences[:split_idx], X_symbols_seq[:split_idx]]
                    y_train = y_sequences[:split_idx]
                    X_val = [X_sequences[split_idx:], X_symbols_seq[split_idx:]]
                    y_val = y_sequences[split_idx:]
                    
                    # Fine-tune model
                    history = symbol_model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=self.config.finetune_epochs,
                        batch_size=self.config.finetune_batch_size,
                        verbose=0
                    )
                    
                    # Evaluate symbol-specific model
                    val_loss, val_accuracy = symbol_model.evaluate(X_val, y_val, verbose=0)
                    
                    # Store symbol model
                    if model_type not in self.symbol_models:
                        self.symbol_models[model_type] = {}
                    self.symbol_models[model_type][symbol] = symbol_model
                    
                    symbol_performances[symbol] = {
                        'validation_loss': float(val_loss),
                        'validation_accuracy': float(val_accuracy),
                        'samples_used': len(X_train[0])
                    }
                    
                    logger.info(f"Fine-tuned {model_type} for {symbol}: {val_accuracy:.4f} accuracy")
                    
                except Exception as e:
                    logger.error(f"Failed to fine-tune {model_type} for {symbol}: {e}")
                    continue
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = UniversalTrainingResult(
                phase="phase2_symbol_finetuning",
                model_name=model_type,
                symbols_trained=list(symbol_performances.keys()),
                base_model_performance={},
                symbol_performances=symbol_performances,
                ensemble_weights={},
                training_time=training_time,
                total_samples=sum([perf['samples_used'] for perf in symbol_performances.values()]),
                validation_accuracy=np.mean([perf['validation_accuracy'] for perf in symbol_performances.values()]),
                metadata={
                    'symbols_finetuned': len(symbol_performances),
                    'avg_symbol_accuracy': np.mean([perf['validation_accuracy'] for perf in symbol_performances.values()])
                }
            )
            
            results[model_type] = result
            logger.info(f"Completed {model_type} fine-tuning: {len(symbol_performances)} symbols in {training_time:.2f}s")
        
        logger.info(f"Phase 2 completed: Fine-tuned {len(results)} model types")
        return results
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray, lookback_window: int) -> tuple:
        """
        Helper method to create sequences for time series models.
        Converts 2D features to 3D sequences with lookback window.
        
        Args:
            features: 2D array of features (samples, features)
            targets: 1D array of targets
            lookback_window: Number of timesteps to look back
            
        Returns:
            Tuple of (X_sequences, y_sequences) as numpy arrays
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(lookback_window, len(features)):
            X_sequences.append(features[i-lookback_window:i])
            y_sequences.append(targets[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    async def phase3_ensemble_optimization(
        self,
        symbols: List[str],
        validation_start: str,
        validation_end: str
    ) -> Dict[str, float]:
        """
        Phase 3: Optimize ensemble weights based on validation performance for statistical models.
        
        Args:
            symbols: List of trading symbols
            validation_start: Start date for validation period
            validation_end: End date for validation period
            
        Returns:
            Optimized ensemble weights by model type
        """
        logger.info("Phase 3: Optimizing ensemble weights for statistical models")
        
        # DEBUG: Log contents of self.base_models and self.symbol_models at START of phase3
        logger.info("=== DEBUG: Model availability at START of phase3 ===")
        logger.info(f"self.base_models keys: {list(self.base_models.keys())}")
        for key, model in self.base_models.items():
            logger.info(f"  - base_models[{key}]: {type(model).__name__} (id: {id(model)})")
        logger.info(f"self.symbol_models keys: {list(self.symbol_models.keys())}")
        for key, symbol_dict in self.symbol_models.items():
            logger.info(f"  - symbol_models[{key}]: {len(symbol_dict)} symbols")
        logger.info("=== END DEBUG phase3 start ===")
        
        # Collect predictions from all statistical models
        model_predictions = {}
        
        # Check if phase2 was skipped (symbol_models is empty)
        if not self.symbol_models:
            logger.info("Phase2 was skipped, using base models for ensemble optimization")
            models_to_use = self.base_models
            use_base_models = True
        else:
            logger.info("Using symbol-specific models from phase2")
            models_to_use = self.symbol_models
            use_base_models = False
        
        # DEBUG: Log models_to_use selection result
        logger.info(f"=== DEBUG: models_to_use selection ===")
        logger.info(f"use_base_models: {use_base_models}")
        logger.info(f"models_to_use keys: {list(models_to_use.keys())}")
        logger.info(f"models_to_use is empty: {len(models_to_use) == 0}")
        if len(models_to_use) == 0:
            logger.error("CRITICAL: models_to_use is empty! This will cause 'No model predictions available'")
            logger.error(f"self.base_models empty: {len(self.base_models) == 0}")
            logger.error(f"self.symbol_models empty: {len(self.symbol_models) == 0}")
        logger.info("=== END DEBUG models_to_use ===")
        
        for model_type in models_to_use.keys():
            model_predictions[model_type] = {}
            
            for symbol in symbols:
                if use_base_models:
                    # For base models, we use the same model for all symbols
                    model = models_to_use[model_type]
                else:
                    # For symbol-specific models, check if symbol exists
                    if symbol not in models_to_use[model_type]:
                        continue
                    model = models_to_use[model_type][symbol]
                
                try:
                    # Load validation data
                    # Convert dates to timezone-aware UTC datetime objects
                    if isinstance(validation_start, str):
                        validation_start_dt = datetime.strptime(validation_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(validation_start, datetime):
                        validation_start_dt = validation_start.replace(tzinfo=timezone.utc) if validation_start.tzinfo is None else validation_start.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"validation_start must be str or datetime, got {type(validation_start)}")
                        
                    if isinstance(validation_end, str):
                        validation_end_dt = datetime.strptime(validation_end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(validation_end, datetime):
                        validation_end_dt = validation_end.replace(tzinfo=timezone.utc) if validation_end.tzinfo is None else validation_end.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"validation_end must be str or datetime, got {type(validation_end)}")
                    
                    validation_data = await self.data_pipeline.load_market_data(
                        symbol=symbol,
                        start_date=validation_start_dt,
                        end_date=validation_end_dt
                    )
                    
                    # Get comprehensive features using the same DataFrame approach as phase1
                    # This ensures we have the full feature set with actual column names
                    logger.info(f"Loading comprehensive feature set for {symbol} using phase1 DataFrame approach")
                    
                    # Use the same data loading approach as phase1_universal_base_training
                    # Step 1: Load universal data
                    universal_data = await self.data_pipeline.load_universal_data(
                        symbols=[symbol],
                        start_date=validation_start_dt,
                        end_date=validation_end_dt
                    )
                    
                    # Step 2: Engineer universal features
                    universal_features = await self.feature_engineering.engineer_universal_features(
                        symbols=[symbol],
                        start_date=validation_start_dt,
                        end_date=validation_end_dt,
                        training_mode=True
                    )
                    
                    # Step 3: Get universal training data (returns DataFrame with actual column names)
                    X, y = await self.feature_engineering.prepare_universal_training_data(
                        universal_features=universal_features,
                        target_column='target'
                    )
                    
                    if X.empty or y.empty:
                        logger.warning(f"No validation data available for {symbol}")
                        continue
                    
                    logger.info(f"Loaded DataFrame for {symbol}: {X.shape} with actual column names")
                    logger.info(f"Sample column names: {list(X.columns)[:10]}...")
                    
                    # Extract symbol IDs and validate
                    symbol_ids = X['symbol_id'].values.astype(np.int32)
                    
                    # Define symbol embedding columns to exclude (same as phase1)
                    symbol_embedding_cols = [col for col in X.columns if (
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
                    
                    # Keep all features except symbol embedding columns (same as phase1)
                    feature_columns = [col for col in X.columns if col not in symbol_embedding_cols]
                    original_feature_columns = feature_columns.copy()
                    
                    logger.info(f"Feature columns for {symbol}: {len(feature_columns)} (excluding {len(symbol_embedding_cols)} symbol embedding cols)")
                    
                    # Apply feature selection using the same approach as phase1_universal_base_training
                    if hasattr(self, 'selected_features') and self.selected_features is not None:
                        logger.info(f"Phase3: Applying selected features for {symbol}: {len(self.selected_features)} features")
                        logger.info(f"Selected features format: {self.selected_features[:5]}... (showing first 5)")
                        logger.info(f"DataFrame columns format: {feature_columns[:5]}... (showing first 5)")
                        
                        # Check if selected features are in temporal aggregation format (feature_X_aggregation)
                        # If so, we need to map them back to original feature indices
                        if any('_' in feat and feat.startswith('feature_') for feat in self.selected_features):
                            logger.info("Detected temporal aggregation format in selected features")
                            
                            # Extract feature indices from temporal aggregation names
                            # e.g., 'feature_5_max' -> index 5
                            selected_indices = set()
                            for feat_name in self.selected_features:
                                if feat_name.startswith('feature_') and '_' in feat_name:
                                    try:
                                        # Extract the number between 'feature_' and the aggregation type
                                        parts = feat_name.split('_')
                                        if len(parts) >= 3:  # feature_X_aggregation
                                            idx = int(parts[1])
                                            selected_indices.add(idx)
                                    except (ValueError, IndexError):
                                        logger.warning(f"Could not parse feature index from: {feat_name}")
                                        continue
                            
                            logger.info(f"Extracted {len(selected_indices)} unique feature indices: {sorted(list(selected_indices))[:10]}...")
                            
                            # Map indices back to actual column names using original_feature_columns
                            logger.info(f"Original feature columns for mapping: {len(original_feature_columns)} (comprehensive feature set matching phase1)")
                            
                            # Select features based on the indices using the original full column list
                            selected_feature_columns = []
                            sorted_indices = sorted(list(selected_indices))
                            
                            logger.info(f"Mapping {len(sorted_indices)} selected indices to column names:")
                            for idx in sorted_indices:
                                if idx < len(original_feature_columns):
                                    col_name = original_feature_columns[idx]
                                    selected_feature_columns.append(col_name)
                                    logger.info(f"  Index {idx} -> '{col_name}'")
                                else:
                                    logger.warning(f"Feature index {idx} exceeds available original features ({len(original_feature_columns)})")
                            
                            # STRICT FEATURE SELECTION: Only use the selected features, no additional features
                            logger.info(f"Strictly using only {len(selected_feature_columns)} selected features (no additional cross-symbol/market regime features)")
                            
                            logger.info(f"Mapped to {len(selected_feature_columns)} actual columns (strictly selected features only):")
                            logger.info(f"  - Selected features used: {len(selected_feature_columns)}")
                            
                            feature_columns = selected_feature_columns
                        else:
                            # Direct column name matching (fallback)
                            selected_feature_columns = [col for col in feature_columns if col in self.selected_features]
                            logger.info(f"Direct column matching: {len(feature_columns)} -> {len(selected_feature_columns)} features")
                            feature_columns = selected_feature_columns
                        
                        logger.info(f"Final feature selection result: {len(feature_columns)} features")
                    
                    # Extract features using the selected column names (same as phase1)
                    features = X[feature_columns]
                    logger.info(f"Selected features for {symbol}: {features.shape} with actual column names")
                    logger.info(f"Feature column names: {list(features.columns)[:10]}... (showing first 10)")
                    
                    # Convert to numpy array for model input
                    X_features = features.values.astype(np.float32)
                    
                    # Use targets from DataFrame
                    y = y.values.astype(np.float32)
                    
                    # Ensure features and targets are aligned
                    min_length = min(len(X_features), len(y))
                    X_features = X_features[:min_length]
                    y = y[:min_length]
                    
                    if len(X_features) == 0:
                        logger.warning(f"No validation data available for {symbol}")
                        continue
                    
                    logger.info(f"Validation data for {symbol}: {len(X_features)} samples with {X_features.shape[1]} features")
                    
                    # === STATISTICAL MODEL DEBUGGING ===
                    logger.info(f"\n=== DEBUGGING {model_type.upper()}-{symbol} STATISTICAL MODEL ===")
                    
                    # 1. Log model object ID and type
                    model_id = id(model)
                    model_class = type(model).__name__
                    logger.info(f"Model object ID: {model_id}, Class: {model_class}")
                    
                    # 2. Log input data shapes and sample values
                    logger.info(f"Input data shape: {X_features.shape}")
                    logger.info(f"First 3 feature vectors (first 5 features): {X_features[:3, :5] if len(X_features) > 0 else 'None'}")
                    logger.info(f"First 10 targets: {y[:10] if len(y) > 0 else 'None'}")
                    
                    # 3. Log model-specific information
                    try:
                        if hasattr(model, 'n_features_in_'):
                            logger.info(f"Model expects {model.n_features_in_} features, got {X_features.shape[1]}")
                        if hasattr(model, 'classes_'):
                            logger.info(f"Model classes: {model.classes_}")
                        if hasattr(model, 'feature_importances_'):
                            top_features = np.argsort(model.feature_importances_)[-5:]
                            logger.info(f"Top 5 feature importances: {model.feature_importances_[top_features]}")
                    except Exception as model_info_e:
                        logger.warning(f"Could not log model info: {model_info_e}")
                    
                    # 4. Make predictions with statistical models
                    # Check if this is an ensemble model (stored as dictionary)
                    if isinstance(model, dict) and 'models' in model and 'weights' in model:
                        logger.info(f"Handling ensemble model for {model_type}-{symbol}")
                        # Handle ensemble model - extract predictions from component models
                        ensemble_models = model['models']
                        ensemble_weights = model['weights']
                        
                        # Get predictions from each component model
                        component_predictions = []
                        for component_name, component_model in ensemble_models.items():
                            logger.info(f"Getting predictions from ensemble component: {component_name}")
                            
                            if hasattr(component_model, 'predict_proba'):
                                # For models that support probability prediction
                                comp_predictions_proba = component_model.predict_proba(X_features)
                                if comp_predictions_proba.shape[1] > 1:
                                    comp_predictions = comp_predictions_proba[:, 1]  # Probability of positive class
                                else:
                                    comp_predictions = comp_predictions_proba.flatten()
                            else:
                                # For models that only support binary prediction
                                comp_predictions_binary = component_model.predict(X_features)
                                comp_predictions = comp_predictions_binary.astype(float)
                            
                            component_predictions.append(comp_predictions)
                            logger.info(f"Component {component_name} predictions shape: {comp_predictions.shape}")
                        
                        # Combine predictions using ensemble weights
                        if component_predictions:
                            # Stack predictions and apply weights
                            stacked_predictions = np.stack(component_predictions, axis=0)  # Shape: (n_models, n_samples)
                            weight_values = np.array([ensemble_weights.get(name, 0.0) for name in ensemble_models.keys()])
                            
                            # Weighted average of predictions
                            predictions_flat = np.average(stacked_predictions, axis=0, weights=weight_values)
                            logger.info(f"Ensemble prediction shape: {predictions_flat.shape}, weights: {weight_values}")
                        else:
                            logger.error(f"No component predictions available for ensemble {model_type}-{symbol}")
                            continue
                    else:
                        # Handle regular (non-ensemble) models
                        if hasattr(model, 'predict_proba'):
                            # For models that support probability prediction
                            predictions_proba = model.predict_proba(X_features)
                            if predictions_proba.shape[1] > 1:
                                predictions_flat = predictions_proba[:, 1]  # Probability of positive class
                            else:
                                predictions_flat = predictions_proba.flatten()
                        else:
                            # For models that only support binary prediction
                            predictions_binary = model.predict(X_features)
                            predictions_flat = predictions_binary.astype(float)
                    
                    # 5. Log prediction results
                    logger.info(f"Predictions shape: {predictions_flat.shape}")
                    logger.info(f"First 10 raw predictions: {predictions_flat[:10]}")
                    
                    # 6. Calculate accuracy
                    binary_predictions = (predictions_flat > 0.5).astype(int)
                    accuracy = accuracy_score(y, binary_predictions)
                    
                    # Additional metrics for debugging
                    unique_predictions = np.unique(predictions_flat)
                    unique_binary_predictions = np.unique(binary_predictions)
                    unique_targets = np.unique(y)
                    
                    logger.info(f"Accuracy: {accuracy:.6f}")
                    logger.info(f"Unique raw predictions count: {len(unique_predictions)}")
                    logger.info(f"Unique binary predictions: {unique_binary_predictions}")
                    logger.info(f"Unique targets: {unique_targets}")
                    logger.info(f"Prediction distribution: min={np.min(predictions_flat):.6f}, max={np.max(predictions_flat):.6f}, mean={np.mean(predictions_flat):.6f}")
                    
                    # 7. Check for prediction issues
                    if len(unique_predictions) == 1:
                        logger.error(f"WARNING: All predictions are identical ({unique_predictions[0]:.6f}) - model may not be learning!")
                    
                    logger.info(f"=== END DEBUGGING {model_type.upper()}-{symbol} ===")
                    
                    # Store prediction results
                    model_predictions[model_type][symbol] = {
                        'predictions': predictions_flat,
                        'targets': y,
                        'accuracy': accuracy,
                        'model_id': model_id,
                        'model_class': model_class
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to get predictions for {model_type}-{symbol}: {e}")
                    continue
        
        # Calculate ensemble weights based on performance
        logger.info("=== PHASE 3 ENSEMBLE WEIGHT CALCULATION DEBUG ===")
        
        # === CROSS-MODEL COMPARISON FOR IDENTICAL ACCURACY INVESTIGATION ===
        logger.info("\n=== CROSS-MODEL COMPARISON ANALYSIS ===")
        
        # 1. Compare model object IDs across types
        logger.info("\n--- MODEL OBJECT ID COMPARISON ---")
        for model_type in model_predictions.keys():
            for symbol, pred_data in model_predictions[model_type].items():
                model_id = pred_data.get('model_id', 'Unknown')
                weight_hash = pred_data.get('weight_hash', 'Unknown')
                logger.info(f"{model_type}-{symbol}: ID={model_id}, WeightHash={weight_hash}")
        
        # 2. Compare predictions across model types for same symbol
        logger.info("\n--- PREDICTION COMPARISON ACROSS MODEL TYPES ---")
        symbols_with_all_models = set()
        for model_type in model_predictions.keys():
            symbols_with_all_models.update(model_predictions[model_type].keys())
        
        for symbol in symbols_with_all_models:
            logger.info(f"\nSymbol {symbol} predictions comparison:")
            symbol_predictions = {}
            symbol_accuracies = {}
            
            for model_type in model_predictions.keys():
                if symbol in model_predictions[model_type]:
                    pred_data = model_predictions[model_type][symbol]
                    predictions = pred_data['predictions'][:5]  # First 5 predictions
                    accuracy = pred_data['accuracy']
                    symbol_predictions[model_type] = predictions
                    symbol_accuracies[model_type] = accuracy
                    logger.info(f"  {model_type}: accuracy={accuracy:.6f}, first_5_preds={predictions}")
            
            # Check if all accuracies are identical
            unique_accuracies = set(symbol_accuracies.values())
            if len(unique_accuracies) == 1:
                logger.error(f"  WARNING: All model types have IDENTICAL accuracy ({list(unique_accuracies)[0]:.6f}) for {symbol}!")
            
            # Check if all predictions are identical
            if len(symbol_predictions) > 1:
                pred_values = list(symbol_predictions.values())
                if all(np.allclose(pred_values[0], pred_val, atol=1e-6) for pred_val in pred_values[1:]):
                    logger.error(f"  WARNING: All model types have IDENTICAL predictions for {symbol}!")
        
        # 3. Overall statistics
        logger.info("\n--- OVERALL ACCURACY STATISTICS ---")
        all_accuracies = []
        for model_type in model_predictions.keys():
            for symbol, pred_data in model_predictions[model_type].items():
                all_accuracies.append(pred_data['accuracy'])
        
        # Check if we have any accuracies to analyze
        if not all_accuracies:
            logger.error("CRITICAL: No model predictions available for ensemble optimization!")
            logger.error("This indicates that no models were successfully trained in Phase 1.")
            logger.error("Returning empty ensemble weights.")
            return {}
        
        unique_accuracy_values = set(all_accuracies)
        logger.info(f"Total models evaluated: {len(all_accuracies)}")
        logger.info(f"Unique accuracy values: {len(unique_accuracy_values)}")
        logger.info(f"Accuracy range: {min(all_accuracies):.6f} to {max(all_accuracies):.6f}")
        
        if len(unique_accuracy_values) == 1:
            logger.error(f"CRITICAL: ALL MODELS HAVE IDENTICAL ACCURACY ({list(unique_accuracy_values)[0]:.6f})!")
        elif len(unique_accuracy_values) < len(all_accuracies) * 0.1:  # Less than 10% unique values
            logger.warning(f"WARNING: Very few unique accuracy values ({len(unique_accuracy_values)}) for {len(all_accuracies)} models!")
        
        logger.info("=== END CROSS-MODEL COMPARISON ANALYSIS ===")
        
        # Log individual model predictions and accuracies
        for model_type in model_predictions.keys():
            logger.info(f"\n--- {model_type.upper()} MODEL PERFORMANCE ---")
            for symbol, pred_data in model_predictions[model_type].items():
                accuracy = pred_data['accuracy']
                num_predictions = len(pred_data['predictions'])
                logger.info(f"  {symbol}: accuracy={accuracy:.4f}, predictions={num_predictions}")
        
        model_scores = {}
        for model_type in model_predictions.keys():
            accuracies = [pred['accuracy'] for pred in model_predictions[model_type].values()]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            model_scores[model_type] = float(avg_accuracy)  # Convert to Python float
            logger.info(f"{model_type} average accuracy: {avg_accuracy:.4f} (from {len(accuracies)} symbols)")
        
        logger.info(f"\nModel scores summary: {model_scores}")
        
        # Normalize weights
        total_score = sum(model_scores.values())
        logger.info(f"Total score for normalization: {total_score:.4f}")
        
        if total_score > 0:
            self.ensemble_weights = {model: float(score / total_score) for model, score in model_scores.items()}
            logger.info("Weights calculated using performance-based normalization")
        else:
            # Equal weights if no valid scores
            num_models = len(model_scores)
            self.ensemble_weights = {model: float(1.0 / num_models) for model in model_scores.keys()}
            logger.warning(f"Using equal weights ({1.0/num_models:.4f}) because total_score={total_score}")
        
        logger.info(f"\nFinal ensemble weights:")
        for model_type, weight in self.ensemble_weights.items():
            logger.info(f"  {model_type}: {weight:.6f}")
        
        logger.info("=== END PHASE 3 ENSEMBLE WEIGHT CALCULATION DEBUG ===")
        logger.info(f"Phase 3 completed: Ensemble weights = {self.ensemble_weights}")
        return self.ensemble_weights
    
    async def _prepare_2d_aggregated_features(self, features, symbol) -> np.ndarray:
        """
        Prepare 2D aggregated features for statistical models.
        
        Args:
            features: numpy array of features (2D or 3D) or FeatureSet object
            symbol: string symbol name for logging
            
        Returns:
            2D numpy array with aggregated features
        """
        try:
            # Validate input features
            if features is None:
                raise ValueError(f"No features provided for {symbol}")
            
            # Handle FeatureSet objects first
            if hasattr(features, 'symbol_features'):
                # This is a UniversalFeatureSet
                logger.info(f"[{symbol}] Converting UniversalFeatureSet in _prepare_2d_aggregated_features")
                features_df = self._combine_features_from_featureset(features.symbol_features.get(symbol, features))
                features = pd.DataFrame(features_df) if isinstance(features_df, np.ndarray) else features_df
            elif hasattr(features, 'technical_features'):
                # This is a regular FeatureSet
                logger.info(f"[{symbol}] Converting FeatureSet in _prepare_2d_aggregated_features")
                features_df = self._combine_features_from_featureset(features)
                features = pd.DataFrame(features_df) if isinstance(features_df, np.ndarray) else features_df
            
            # Convert to numpy array if needed
            if hasattr(features, 'values'):
                features = features.values
            
            # Apply temporal aggregation if enabled and data is 3D
            if (self.config.enable_temporal_aggregation and 
                self.temporal_aggregator is not None and 
                len(features.shape) == 3):
                
                logger.info(f"[{symbol}] Applying TemporalAggregator to 3D features {features.shape}")
                
                # Generate feature names for temporal aggregation
                n_features = features.shape[2]
                feature_names = [f"feature_{i}" for i in range(n_features)]
                
                # Use TemporalAggregator for proper temporal feature engineering
                aggregated_df = self.temporal_aggregator.aggregate_3d_to_dataframe(
                    data_3d=features,
                    feature_names=feature_names
                )
                
                # Convert DataFrame to numpy array
                aggregated_features = aggregated_df.values
                    
                logger.info(f"[{symbol}] TemporalAggregator result: {features.shape} -> {aggregated_features.shape}")
                return aggregated_features
            
            # If already 2D, return as-is
            if len(features.shape) == 2:
                logger.info(f"[{symbol}] Using 2D features directly: {features.shape}")
                return features
            elif len(features.shape) == 3:
                # Fallback temporal aggregation using direct numpy operations
                logger.info(f"[{symbol}] Applying fallback temporal aggregation to 3D features {features.shape}")
                
                # Calculate temporal statistics across the time dimension (axis=1)
                mean_features = np.nanmean(features, axis=1)  # Shape: (samples, features)
                max_features = np.nanmax(features, axis=1)    # Shape: (samples, features)
                std_features = np.nanstd(features, axis=1)    # Shape: (samples, features)
                
                # Concatenate all temporal statistics
                aggregated_features = np.concatenate([
                    mean_features,
                    max_features, 
                    std_features
                ], axis=1)
                
                logger.info(f"[{symbol}] Fallback temporal aggregation complete: {features.shape} -> {aggregated_features.shape}")
                return aggregated_features
            else:
                raise ValueError(f"Unexpected feature shape for {symbol}: {features.shape}")
                
        except Exception as e:
            logger.error(f"Error preparing 2D aggregated features for {symbol}: {e}")
            raise
    
    async def train_universal_models(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        model_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Execute complete 3-phase universal training process.
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for training data
            end_date: End date for training data
            model_types: List of model types to train
            
        Returns:
            Complete training results
        """
        logger.info(f"Starting universal training for {len(symbols)} symbols")
        
        # DEBUG: Check what's in self.base_models at the very start
        logger.info(f"DEBUG - START OF TRAINING: self.base_models keys: {list(self.base_models.keys())}")
        logger.info(f"DEBUG - START OF TRAINING: self.base_models empty: {len(self.base_models) == 0}")
        
        start_time = datetime.now()
        
        try:
            # Feature Selection: Perform feature selection before training
            logger.info("Performing feature selection...")
            await self.perform_feature_selection(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Phase 1: Train statistical base models using 2D aggregated features
            config = UniversalTrainingConfig(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                enable_smote=self.config.enable_imbalance_mitigation,
                force_2d_for_statistical=True  # Enforce 2D data pipeline for statistical models
            )
            phase1_results = await self.phase1_universal_base_training(
                data_loader=self.data_pipeline,
                config=config
            )
            
            # DEBUG: Check what's in self.base_models after phase1 statistical training
            logger.info(f"DEBUG - AFTER PHASE1: self.base_models keys: {list(self.base_models.keys())}")
            logger.info(f"DEBUG - AFTER PHASE1: self.base_models count: {len(self.base_models)}")
            for model_type, model in self.base_models.items():
                logger.info(f"DEBUG - AFTER PHASE1: {model_type} model type: {type(model).__name__}")
            
            # Phase 2: Symbol-specific fine-tuning (DISABLED)
            # Skipping phase2_symbol_specific_finetuning to eliminate dependency
            phase2_results = {}
            
            # Phase 3: Ensemble optimization
            # Use 5-day validation period for day trading (optimal for capturing recent patterns)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
            validation_start_dt = end_dt - timedelta(days=20)  # 30 days before end_date
            validation_end_dt = end_dt  # End at the training end date
            
            validation_start = validation_start_dt.strftime('%Y-%m-%d')
            validation_end = validation_end_dt.strftime('%Y-%m-%d')
            
            ensemble_weights = await self.phase3_ensemble_optimization(
                symbols=symbols,
                validation_start=validation_start,
                validation_end=validation_end
            )
            
            # Calculate total training time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Compile comprehensive statistical model training results
            results = {
                'training_completed': True,
                'total_training_time': total_time,
                'symbols_trained': symbols,
                'training_date_range': {'start': start_date, 'end': end_date},
                
                # Add models_trained field for main.py validation check
                'models_trained': list(self.base_models.keys()) if self.base_models else [],
                
                # Statistical Model Performance Metrics
                'statistical_models': {
                    'models_trained': phase1_results.get('models_trained', []),
                    'validation_metrics': phase1_results.get('validation_metrics', {}),
                    'model_configs': phase1_results.get('model_configs', {}),
                    'feature_importance': phase1_results.get('feature_importance', {}),
                    'phase1_training_time': phase1_results.get('training_time', 0)
                },
                
                # Feature Selection Information
                'feature_selection': {
                    'selected_feature_count': len(self.selected_features) if self.selected_features else 0,
                    'selected_feature_indices_count': len(self.selected_feature_indices) if self.selected_feature_indices else 0,
                    'feature_selection_method': 'mutual_info_regression',  # Based on UniversalFeatureSelector
                    'expected_unique_features': self._calculate_expected_unique_features() if hasattr(self, '_calculate_expected_unique_features') else 0
                },
                
                # Trading-Specific Metrics for Minute-to-Minute System
                'trading_performance': {
                    'avg_model_accuracy': np.mean([metrics.get('accuracy', 0) for metrics in phase1_results.get('validation_metrics', {}).values()]) if phase1_results.get('validation_metrics') else 0.0,
                    'best_performing_model': max(phase1_results.get('validation_metrics', {}).items(), key=lambda x: x[1].get('accuracy', 0))[0] if phase1_results.get('validation_metrics') else 'none',
                    'model_consistency': {
                        'accuracy_std': np.std([metrics.get('accuracy', 0) for metrics in phase1_results.get('validation_metrics', {}).values()]) if phase1_results.get('validation_metrics') else 0.0,
                        'loss_std': np.std([metrics.get('loss', 0) for metrics in phase1_results.get('validation_metrics', {}).values()]) if phase1_results.get('validation_metrics') else 0.0
                    },
                    'prediction_readiness': {
                        'models_ready_for_inference': len(self.base_models),
                        'ensemble_weights_available': len(ensemble_weights) > 0,
                        'feature_selection_applied': self.selected_features is not None or self.selected_feature_indices is not None
                    }
                },
                
                # Model Training Details
                'training_details': {
                    'imbalance_mitigation_enabled': self.config.enable_imbalance_mitigation,
                    'validation_split': self.config.base_validation_split,
                    'lookback_window_minutes': self.config.base_lookback_window,
                    'prediction_threshold': self.config.prediction_threshold,
                    'class_balance_info': {
                        'smote_applied': hasattr(self, 'imbalance_metrics') and len(self.imbalance_metrics) > 0,
                        'class_weights_available': hasattr(self, 'class_weights_sklearn') and self.class_weights_sklearn is not None
                    }
                },
                
                # Ensemble Information
                'ensemble_optimization': {
                    'weights': ensemble_weights,
                    'validation_period_days': 20,  # Based on validation window used
                    'optimization_completed': len(ensemble_weights) > 0
                },
                
                # Legacy fields for backward compatibility
                'phase1_results': phase1_results,
                'phase2_results': phase2_results,  # Empty as Phase 2 is disabled
                'ensemble_weights': ensemble_weights,
                'model_summary': {
                    'base_models': list(self.base_models.keys()),
                    'total_statistical_models': len(self.base_models),
                    'neural_network_models': 0  # No longer using neural networks
                }
            }
            
            logger.info(f"Universal training completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Universal training failed: {e}")
            raise
    
    async def save_universal_models(self, save_dir: Path) -> None:
        """
        Save all universal statistical models and training state using joblib.
        
        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting statistical model saving process to {save_dir}")
        
        # Save base models
        base_dir = save_dir / "base_models"
        base_dir.mkdir(exist_ok=True)
        
        for model_type, model in self.base_models.items():
            logger.info(f"Saving base {model_type} statistical model...")
            
            # Validate statistical model before saving
            try:
                logger.info(f"Base {model_type} - Model validation:")
                if hasattr(model, '__class__'):
                    logger.info(f"  - Model type: {model.__class__.__name__}")
                if hasattr(model, 'n_features_in_'):
                    logger.info(f"  - Features trained on: {model.n_features_in_}")
                if hasattr(model, 'classes_'):
                    logger.info(f"  - Classes: {model.classes_}")
                
                # Test model with dummy prediction for statistical models
                try:
                    if hasattr(model, 'n_features_in_'):
                        dummy_input = np.random.random((1, model.n_features_in_))
                        dummy_output = model.predict_proba(dummy_input) if hasattr(model, 'predict_proba') else model.predict(dummy_input)
                        logger.info(f"  - Prediction test: PASSED (output shape: {dummy_output.shape})")
                    else:
                        logger.info(f"  - Prediction test: SKIPPED (no feature info)")
                except Exception as prediction_error:
                    logger.error(f"  - Prediction test: FAILED ({prediction_error})")
                    
            except Exception as e:
                logger.error(f"Base {model_type} - Model validation FAILED: {e}")
                
            # Save the statistical model using joblib
            try:
                model_path = base_dir / f"{model_type}_base.joblib"
                self.universal_architectures.save_statistical_model(model, model_path)
                logger.info(f"✓ Successfully saved base {model_type} model to {model_path}")
            except Exception as e:
                logger.error(f"✗ Failed to save base {model_type} model: {e}")
                raise
        
        # Save symbol-specific models (if any exist)
        symbol_dir = save_dir / "symbol_models"
        symbol_dir.mkdir(exist_ok=True)
        
        if self.symbol_models:
            for model_type, symbol_models in self.symbol_models.items():
                type_dir = symbol_dir / model_type
                type_dir.mkdir(exist_ok=True)
                
                for symbol, model in symbol_models.items():
                    logger.info(f"Saving {model_type} statistical model for symbol {symbol}...")
                    
                    # Validate statistical model before saving
                    try:
                        logger.info(f"Symbol {symbol} {model_type} - Model validation:")
                        if hasattr(model, '__class__'):
                            logger.info(f"  - Model type: {model.__class__.__name__}")
                        if hasattr(model, 'n_features_in_'):
                            logger.info(f"  - Features trained on: {model.n_features_in_}")
                        if hasattr(model, 'classes_'):
                            logger.info(f"  - Classes: {model.classes_}")
                        
                        # Test model with dummy prediction for statistical models
                        try:
                            if hasattr(model, 'n_features_in_'):
                                dummy_input = np.random.random((1, model.n_features_in_))
                                dummy_output = model.predict_proba(dummy_input) if hasattr(model, 'predict_proba') else model.predict(dummy_input)
                                logger.info(f"  - Prediction test: PASSED (output shape: {dummy_output.shape})")
                            else:
                                logger.info(f"  - Prediction test: SKIPPED (no feature info)")
                        except Exception as prediction_error:
                            logger.error(f"  - Prediction test: FAILED ({prediction_error})")
                            
                    except Exception as e:
                        logger.error(f"Symbol {symbol} {model_type} - Model validation FAILED: {e}")
                        logger.error(f"  - This model may be corrupted and could cause loading issues")
                        
                    # Save the statistical model using joblib
                    try:
                        model_path = type_dir / f"{symbol}.joblib"
                        self.universal_architectures.save_statistical_model(model, model_path)
                        logger.info(f"✓ Successfully saved {model_type} model for {symbol} to {model_path}")
                    except Exception as e:
                        logger.error(f"✗ Failed to save {model_type} model for {symbol}: {e}")
                        raise
        else:
            logger.info("No symbol-specific models to save (Phase 2 was skipped)")
        
        # Save metadata
        metadata = {
            'symbol_mappings': {
                'symbol_to_id': self.symbol_to_id,
                'id_to_symbol': self.id_to_symbol
            },
            'ensemble_weights': self.ensemble_weights,
            'model_configs': {
                'xgboost': {
                    'n_estimators': getattr(self.config, 'xgb_n_estimators', 100),
                    'max_depth': getattr(self.config, 'xgb_max_depth', 6),
                    'learning_rate': getattr(self.config, 'xgb_learning_rate', 0.1),
                    'subsample': getattr(self.config, 'xgb_subsample', 0.8)
                },
                'random_forest': {
                    'n_estimators': getattr(self.config, 'rf_n_estimators', 100),
                    'max_depth': getattr(self.config, 'rf_max_depth', 10),
                    'min_samples_split': getattr(self.config, 'rf_min_samples_split', 2),
                    'min_samples_leaf': getattr(self.config, 'rf_min_samples_leaf', 1)
                },
                'svm': {
                    'C': getattr(self.config, 'svm_C', 1.0),
                    'kernel': getattr(self.config, 'svm_kernel', 'rbf'),
                    'gamma': getattr(self.config, 'svm_gamma', 'scale')
                }
            },
            'feature_selection': {
                'selected_features': getattr(self, 'selected_features', None),
                'selected_feature_indices': getattr(self, 'selected_feature_indices', None),
                'feature_importance': getattr(self, 'feature_importance', {}),
                'feature_mapping': getattr(self, 'feature_mapping', {}),
                'total_features': len(getattr(self, 'selected_features', [])) if hasattr(self, 'selected_features') and self.selected_features else 0
            },
            'model_performance': {
                'validation_accuracy': getattr(self, 'best_validation_accuracy', 0.0),
                'validation_precision': getattr(self, 'best_validation_precision', 0.0),
                'validation_recall': getattr(self, 'best_validation_recall', 0.0),
                'validation_f1': getattr(self, 'best_validation_f1', 0.0),
                'ensemble_accuracy': getattr(self, 'ensemble_accuracy', 0.0),
                'prediction_confidence_threshold': getattr(self.config, 'prediction_confidence_threshold', 0.6)
            },
            'training_data_stats': {
                'training_start_date': getattr(self, 'training_start_date', None),
                'training_end_date': getattr(self, 'training_end_date', None),
                'total_samples': getattr(self, 'total_training_samples', 0),
                'class_distribution': getattr(self, 'class_distribution', {}),
                'symbols_trained': list(self.symbol_to_id.keys()) if hasattr(self, 'symbol_to_id') else []
            },
            'model_files': {
                'xgboost_model_path': getattr(self, 'xgb_model_path', None),
                'random_forest_model_path': getattr(self, 'rf_model_path', None),
                'svm_model_path': getattr(self, 'svm_model_path', None),
                'ensemble_model_path': getattr(self, 'ensemble_model_path', None),
                'scaler_path': getattr(self, 'scaler_path', None)
            },
            'prediction_thresholds': {
                'buy_threshold': getattr(self.config, 'buy_threshold', 0.7),
                'sell_threshold': getattr(self.config, 'sell_threshold', 0.7),
                'hold_threshold': getattr(self.config, 'hold_threshold', 0.5)
            },
            'live_trading_config': {
                'prediction_window_minutes': getattr(self.config, 'prediction_window_minutes', 1),
                'feature_update_frequency': getattr(self.config, 'feature_update_frequency', 60),
                'model_version': getattr(self, 'model_version', '1.0.0'),
                'requires_feature_scaling': True,
                'max_prediction_latency_ms': getattr(self.config, 'max_prediction_latency_ms', 100)
            },
            'training_timestamp': datetime.now().isoformat(),
            'model_type': 'statistical_ensemble'
        }
        
        with open(save_dir / "universal_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved universal models to {save_dir}")
    
    async def train_statistical_model(self, model_type: ModelType, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray, config: ModelConfig) -> Tuple[object, float, Dict[str, float]]:
        """
        Train statistical models (XGBoost, Random Forest, SVM) optimized for 2D aggregated features.
        Returns model, validation loss, and comprehensive metrics dictionary.
        """
        logger.info(f"Training {model_type.value} model with {X_train.shape[1]} aggregated features")
        
        # Create progress bar for model training
        pbar = tqdm(total=100, desc=f"Training {model_type.value}", unit="%")
        pbar.update(10)  # Initial setup complete
        
        feature_dim = X_train.shape[1]
        
        if model_type == ModelType.XGBOOST:
            pbar.set_description(f"Creating {model_type.value} model")
            model = self.universal_architectures.create_universal_xgboost(
                feature_dim=feature_dim,
                config=config.parameters,
                model_name=f"universal_{model_type.value}"
            )
            pbar.update(20)  # Model creation complete
            
            # Train with early stopping
            pbar.set_description(f"Training {model_type.value} model")
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            pbar.update(60)  # Training complete
            
        elif model_type == ModelType.RANDOM_FOREST:
            pbar.set_description(f"Creating {model_type.value} model")
            model = self.universal_architectures.create_universal_random_forest(
                feature_dim=feature_dim,
                config=config.parameters,
                model_name=f"universal_{model_type.value}"
            )
            pbar.update(20)  # Model creation complete
            
            pbar.set_description(f"Training {model_type.value} model")
            model.fit(X_train, y_train)
            pbar.update(60)  # Training complete
            
        elif model_type == ModelType.SVM:
            pbar.set_description(f"Creating {model_type.value} model")
            model = self.universal_architectures.create_universal_svm(
                feature_dim=feature_dim,
                config=config.parameters,
                model_name=f"universal_{model_type.value}"
            )
            pbar.update(20)  # Model creation complete
            
            pbar.set_description(f"Training {model_type.value} model")
            model.fit(X_train, y_train)
            pbar.update(60)  # Training complete
            
        elif model_type == ModelType.ENSEMBLE:
            pbar.set_description(f"Creating {model_type.value} model")
            ensemble = self.universal_architectures.create_ensemble_model(
                feature_dim=feature_dim,
                config=config.parameters,
                model_name=f"universal_{model_type.value}"
            )
            pbar.update(15)  # Model creation complete
            
            # Train individual models
            models = ensemble['models']
            weights = ensemble['weights']
            
            # Train XGBoost
            pbar.set_description(f"Training {model_type.value} XGBoost")
            eval_set = [(X_train, y_train), (X_val, y_val)]
            models['xgboost'].fit(X_train, y_train, eval_set=eval_set, verbose=False)
            pbar.update(20)  # XGBoost training complete
            
            # Train Random Forest  
            pbar.set_description(f"Training {model_type.value} Random Forest")
            models['random_forest'].fit(X_train, y_train)
            pbar.update(20)  # Random Forest training complete
            
            # Train SVM
            pbar.set_description(f"Training {model_type.value} SVM")
            models['svm'].fit(X_train, y_train)
            pbar.update(20)  # SVM training complete
            
            model = ensemble
            
        else:
            raise ValueError(f"Unsupported statistical model type: {model_type}")
        
        # Evaluate model
        pbar.set_description(f"Evaluating {model_type.value} model")
        if model_type == ModelType.ENSEMBLE:
            # Ensemble predictions
            models = model['models']
            weights = model['weights']
            
            xgb_pred = models['xgboost'].predict_proba(X_val)[:, 1]
            rf_pred = models['random_forest'].predict_proba(X_val)[:, 1]
            svm_pred = models['svm'].predict_proba(X_val)[:, 1]
            
            val_predictions = (weights['xgboost'] * xgb_pred + 
                              weights['random_forest'] * rf_pred + 
                              weights['svm'] * svm_pred)
        else:
            val_predictions = model.predict_proba(X_val)[:, 1]
        
        # Calculate comprehensive metrics
        val_predictions_binary = (val_predictions > 0.5).astype(int)
        
        # Basic metrics
        val_accuracy = accuracy_score(y_val, val_predictions_binary)
        val_precision = precision_score(y_val, val_predictions_binary, zero_division=0)
        val_recall = recall_score(y_val, val_predictions_binary, zero_division=0)
        val_f1 = f1_score(y_val, val_predictions_binary, zero_division=0)
        val_roc_auc = roc_auc_score(y_val, val_predictions) if len(np.unique(y_val)) > 1 else 0.0
        val_loss = -np.mean(y_val * np.log(val_predictions + 1e-15) + (1 - y_val) * np.log(1 - val_predictions + 1e-15))
        
        # High confidence accuracy (predictions > 0.7)
        high_conf_mask = val_predictions > 0.7
        high_conf_accuracy = 0.0
        if np.sum(high_conf_mask) > 0:
            high_conf_predictions = val_predictions_binary[high_conf_mask]
            high_conf_targets = y_val[high_conf_mask]
            high_conf_accuracy = accuracy_score(high_conf_targets, high_conf_predictions)
        
        # Win rate by confidence levels
        confidence_intervals = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        win_rates_by_confidence = {}
        
        for low, high in confidence_intervals:
            mask = (val_predictions >= low) & (val_predictions < high)
            if np.sum(mask) > 0:
                conf_predictions = val_predictions_binary[mask]
                conf_targets = y_val[mask]
                win_rate = accuracy_score(conf_targets, conf_predictions)
                win_rates_by_confidence[f"{low:.1f}-{high:.1f}"] = win_rate
            else:
                win_rates_by_confidence[f"{low:.1f}-{high:.1f}"] = 0.0
        
        # Compile all metrics
        metrics = {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1,
            'roc_auc': val_roc_auc,
            'high_confidence_accuracy': high_conf_accuracy,
            'win_rate_0.5-0.6': win_rates_by_confidence.get('0.5-0.6', 0.0),
            'win_rate_0.6-0.7': win_rates_by_confidence.get('0.6-0.7', 0.0),
            'win_rate_0.7-0.8': win_rates_by_confidence.get('0.7-0.8', 0.0),
            'win_rate_0.8-0.9': win_rates_by_confidence.get('0.8-0.9', 0.0),
            'win_rate_0.9-1.0': win_rates_by_confidence.get('0.9-1.0', 0.0)
        }
        
        pbar.update(10)  # Evaluation complete
        
        # Close progress bar
        pbar.close()
        
        logger.info(f"Completed {model_type.value} training:")
        logger.info(f"  - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        logger.info(f"  - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        logger.info(f"  - ROC-AUC: {val_roc_auc:.4f}, High Conf Accuracy: {high_conf_accuracy:.4f}")
        logger.info(f"  - Win Rates: 0.5-0.6: {win_rates_by_confidence.get('0.5-0.6', 0.0):.3f}, 0.7-0.8: {win_rates_by_confidence.get('0.7-0.8', 0.0):.3f}, 0.9-1.0: {win_rates_by_confidence.get('0.9-1.0', 0.0):.3f}")
        
        return model, val_loss, metrics
    
    async def load_universal_models(self, load_dir: Path) -> bool:
        """
        Load universal models and training state.
        
        Args:
            load_dir: Directory to load models from
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            load_dir = Path(load_dir)
            
            # Load metadata
            metadata_path = load_dir / "universal_metadata.json"
            if not metadata_path.exists():
                logger.error(f"Universal metadata not found: {metadata_path}")
                return False
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.symbol_to_id = metadata['symbol_mappings']['symbol_to_id']
            self.id_to_symbol = {int(k): v for k, v in metadata['symbol_mappings']['id_to_symbol'].items()}
            self.ensemble_weights = metadata['ensemble_weights']
            
            # Initialize architectures
            self.universal_architectures = UniversalModelArchitectures(
                num_symbols=len(self.symbol_to_id),
                symbol_embedding_dim=metadata['config']['symbol_embedding_dim']
            )
            
            # Import joblib for loading statistical models
            import joblib
            
            # Load base models
            base_dir = load_dir / "base_models"
            base_models_loaded = 0
            if base_dir.exists():
                for model_file in base_dir.glob("*_base.joblib"):
                    model_type = model_file.stem.replace('_base', '')
                    try:
                        loaded_model = self.universal_architectures.load_statistical_model(model_file)
                        
                        # Perform integrity check after loading
                        try:
                            logger.info(f"Base {model_type} - Post-load integrity check:")
                            if hasattr(loaded_model, '__class__'):
                                logger.info(f"  - Model type: {loaded_model.__class__.__name__}")
                            if hasattr(loaded_model, 'n_features_in_'):
                                logger.info(f"  - Features trained on: {loaded_model.n_features_in_}")
                            if hasattr(loaded_model, 'classes_'):
                                logger.info(f"  - Classes: {loaded_model.classes_}")
                            
                            # Test model with dummy prediction for statistical models
                            try:
                                if hasattr(loaded_model, 'n_features_in_'):
                                    dummy_input = np.random.random((1, loaded_model.n_features_in_))
                                    dummy_output = loaded_model.predict_proba(dummy_input) if hasattr(loaded_model, 'predict_proba') else loaded_model.predict(dummy_input)
                                    logger.info(f"  - Post-load prediction test: PASSED (output shape: {dummy_output.shape})")
                                else:
                                    logger.info(f"  - Post-load prediction test: SKIPPED (no feature info)")
                            except Exception as pred_error:
                                logger.warning(f"  - Post-load prediction test: FAILED ({pred_error})")
                                
                            self.base_models[model_type] = loaded_model
                            base_models_loaded += 1
                            logger.info(f"✓ Loaded and verified base {model_type} statistical model")
                            
                        except Exception as integrity_error:
                            logger.error(f"Base {model_type} - Post-load integrity check FAILED: {integrity_error}")
                            logger.error(f"  - Model loaded but failed verification, skipping")
                            
                    except Exception as e:
                        logger.error(f"Failed to load base {model_type} statistical model: {e}")
            
            # Load symbol-specific models
            symbol_dir = load_dir / "symbol_models"
            symbol_models_loaded = 0
            corrupted_models = []
            if symbol_dir.exists():
                for type_dir in symbol_dir.iterdir():
                    if type_dir.is_dir():
                        model_type = type_dir.name
                        self.symbol_models[model_type] = {}
                        
                        for model_file in type_dir.glob("*.joblib"):
                            symbol = model_file.stem
                            try:
                                loaded_model = self.universal_architectures.load_statistical_model(model_file)
                                
                                # Perform integrity check after loading
                                try:
                                    logger.info(f"Symbol {symbol} {model_type} - Post-load integrity check:")
                                    if hasattr(loaded_model, '__class__'):
                                        logger.info(f"  - Model type: {loaded_model.__class__.__name__}")
                                    if hasattr(loaded_model, 'n_features_in_'):
                                        logger.info(f"  - Features trained on: {loaded_model.n_features_in_}")
                                    if hasattr(loaded_model, 'classes_'):
                                        logger.info(f"  - Classes: {loaded_model.classes_}")
                                    
                                    # Test model with dummy prediction for statistical models
                                    try:
                                        if hasattr(loaded_model, 'n_features_in_'):
                                            dummy_input = np.random.random((1, loaded_model.n_features_in_))
                                            dummy_output = loaded_model.predict_proba(dummy_input) if hasattr(loaded_model, 'predict_proba') else loaded_model.predict(dummy_input)
                                            logger.info(f"  - Post-load prediction test: PASSED (output shape: {dummy_output.shape})")
                                        else:
                                            logger.info(f"  - Post-load prediction test: SKIPPED (no feature info)")
                                    except Exception as pred_error:
                                        logger.warning(f"  - Post-load prediction test: FAILED ({pred_error})")
                                        
                                    self.symbol_models[model_type][symbol] = loaded_model
                                    symbol_models_loaded += 1
                                    logger.info(f"✓ Loaded and verified {model_type} statistical model for {symbol}")
                                    
                                except Exception as integrity_error:
                                    logger.error(f"Symbol {symbol} {model_type} - Post-load integrity check FAILED: {integrity_error}")
                                    logger.error(f"  - Model loaded but failed verification, marking as corrupted")
                                    corrupted_models.append(f"{model_type}/{symbol}")
                                    
                            except Exception as e:
                                logger.error(f"Failed to load {model_type} statistical model for {symbol}: {e}")
                                corrupted_models.append(f"{model_type}/{symbol}")
            
            total_models = base_models_loaded + symbol_models_loaded
            
            if corrupted_models:
                logger.info(f"Successfully loaded {base_models_loaded} base models and {symbol_models_loaded} symbol-specific models (total: {total_models})")
                logger.info(f"Found {len(corrupted_models)} corrupted symbol-specific models that will be regenerated: {corrupted_models}")
            else:
                logger.info(f"Successfully loaded {base_models_loaded} base models and {symbol_models_loaded} symbol-specific models (total: {total_models})")
            
            # Return True if we have at least base models (symbol models can be regenerated)
            return base_models_loaded > 0
            
        except Exception as e:
            logger.error(f"Error loading universal models: {e}")
            return False
    
    async def phase1_universal_base_training(self, data_loader: DataPipeline, config: UniversalTrainingConfig) -> Dict[str, Any]:
        """
        Phase 1: Train statistical models on 2D aggregated features.
        
        Args:
            data_loader: DataLoader instance for loading training data
            config: Training configuration
            
        Returns:
            Dict containing training results and model information
        """
        logger.info("Starting Phase 1: Universal Base Training with Statistical Models")
        
        # Initialize results dictionary
        results = {
            'models_trained': [],
            'training_metrics': {},
            'validation_metrics': {},
            'model_configs': {},
            'feature_importance': {},
            'training_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Step 2: Feature Selection → Use features already selected in train_universal_models
            logger.info("Step 2: Feature Selection - Using pre-selected features from train_universal_models...")
            
            # Verify selected features are available (should be set by train_universal_models)
            if self.selected_features is None and self.selected_feature_indices is None:
                logger.warning("No pre-existing selected features available, will use all features")
            else:
                logger.info(f"Using selected features: {len(self.selected_features) if self.selected_features else 0} feature names, {len(self.selected_feature_indices) if self.selected_feature_indices else 0} feature indices")
            
            # Step 1 & 3: Data Preparation using prepare_universal_dataset approach
            logger.info("Step 1 & 3: Data Preparation using comprehensive dataset preparation...")
            
            # Use the comprehensive prepare_universal_dataset approach
            # Calculate date range from config
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.base_training_window * 30)  # Convert months to days
            
            X_train_data, y_train, X_val_data, y_val = await self.prepare_universal_dataset(
                symbols=config.symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                config=config  # Pass config to enable force_2d_for_statistical
            )
            
            # Unpack the returned data lists
            X_train_features, X_train_symbols = X_train_data
            X_val_features, X_val_symbols = X_val_data
            
            logger.info(f"Dataset prepared: Training set {X_train_features.shape[0]} samples, Validation set {X_val_features.shape[0]} samples")
            logger.info(f"Feature count: {X_train_features.shape[1]} features")
            logger.info(f"Positive samples in training: {np.sum(y_train)} ({np.mean(y_train):.2%})")
            
            # Step 4: Validation - Calculate expected unique features and validate feature count
            logger.info("Step 4: Validation - Checking feature count consistency...")
            expected_unique_features = self._calculate_expected_unique_features()
            if X_train_features.shape[1] < expected_unique_features:
                logger.warning(f"  - WARNING: Only {X_train_features.shape[1]} features in final dataset, expected ~{expected_unique_features} (unique base features from selection)")
            else:
                logger.info(f"  - ✅ FEATURE COUNT VALIDATED: {X_train_features.shape[1]} features meets expected minimum of {expected_unique_features}")
            
            # Step 5: Training → phase1_universal_base_training applies selections
            logger.info("Step 5: Training - Training statistical models with selected features...")
            statistical_models = [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM, ModelType.ENSEMBLE]
            
            for model_type in statistical_models:
                try:
                    logger.info(f"Training {model_type.value} model...")
                    
                    # Train model
                    model, val_loss, metrics = await self.train_statistical_model(
                        model_type=model_type,
                        X_train=X_train_features,
                        y_train=y_train,
                        X_val=X_val_features,
                        y_val=y_val,
                        config=self.model_configs[model_type]
                    )
                    
                    # Store model
                    self.base_models[model_type.value] = model
                    
                    # Record results
                    results['models_trained'].append(model_type.value)
                    results['validation_metrics'][model_type.value] = {
                        'loss': val_loss,
                        **metrics  # Include all comprehensive metrics
                    }
                    
                    # Get feature importance for tree-based models
                    if model_type in [ModelType.XGBOOST, ModelType.RANDOM_FOREST]:
                        try:
                            if hasattr(model, 'feature_importances_'):
                                importance = model.feature_importances_
                            elif hasattr(model, 'get_score'):
                                importance_dict = model.get_score(importance_type='weight')
                                importance = np.array([importance_dict.get(f'f{i}', 0) for i in range(X_train_features.shape[1])])
                            else:
                                importance = None
                            
                            if importance is not None:
                                results['feature_importance'][model_type.value] = importance.tolist()
                                logger.info(f"Extracted feature importance for {model_type.value}")
                        except Exception as e:
                            logger.warning(f"Could not extract feature importance for {model_type.value}: {e}")
                    
                    logger.info(f"✓ {model_type.value} training completed: val_loss={val_loss:.4f}, val_accuracy={metrics['accuracy']:.4f}")
                    logger.info(f"  📊 Trading Metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
                    logger.info(f"  🎯 ROC-AUC: {metrics['roc_auc']:.4f}, High Confidence Accuracy (>0.7): {metrics['high_confidence_accuracy']:.4f}")
                    logger.info(f"  💰 Win Rates by Confidence - 0.5-0.6: {metrics['win_rate_0.5-0.6']:.3f}, 0.7-0.8: {metrics['win_rate_0.7-0.8']:.3f}, 0.9-1.0: {metrics['win_rate_0.9-1.0']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type.value} model: {e}")
                    continue
            
            # Calculate training time
            training_time = time.time() - start_time
            results['training_time'] = training_time
            
            # Store model configurations
            results['model_configs'] = {
                model_type: config_dict for model_type, config_dict in self.model_configs.items()
                if model_type in [mt.value for mt in statistical_models]
            }
            
            logger.info(f"Phase 1 Universal Base Training completed in {training_time:.2f} seconds")
            logger.info(f"Successfully trained {len(results['models_trained'])} statistical models")
            
            # DEBUG: Log contents of self.base_models at end of phase1
            logger.info("=== DEBUG: self.base_models contents at END of phase1 ===")
            logger.info(f"self.base_models keys: {list(self.base_models.keys())}")
            for key, model in self.base_models.items():
                logger.info(f"  - {key}: {type(model).__name__} (id: {id(model)})")
            logger.info("=== END DEBUG phase1 ===")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Phase 1 Universal Base Training: {e}")
            raise
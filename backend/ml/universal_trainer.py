import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timezone
from loguru import logger
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model

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
    """Configuration for universal training phases"""
    # Phase 1: Universal Base Model
    base_epochs: int = 100
    base_batch_size: int = 256
    base_learning_rate: float = 0.001
    base_validation_split: float = 0.2
    base_lookback_window: int = 30
    
    # Phase 2: Symbol-Specific Fine-tuning
    finetune_epochs: int = 50
    finetune_batch_size: int = 128
    finetune_learning_rate: float = 0.0001
    layers_to_unfreeze: int = 3
    
    # Phase 3: Ensemble Optimization
    ensemble_validation_periods: int = 10
    ensemble_rebalance_frequency: int = 5
    
    # General settings
    symbol_embedding_dim: int = 32
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 2  # Reduced from 5 to 2 for more responsive LR scheduling
    min_samples_per_symbol: int = 1000
    max_symbols_per_batch: int = 50

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
        
        # Initialize components
        self.symbol_to_id = {}
        self.id_to_symbol = {}
        self.universal_architectures = None
        
        # Training state
        self.base_models = {}
        self.symbol_models = {}
        self.ensemble_weights = {}
        self.training_history = []
        
        # Default model configurations (feature_count will be updated dynamically)
        self.model_configs = {
            'lstm': ModelConfig(
                name='universal_lstm',
                model_type='lstm',
                parameters={'units': 50, 'dropout': 0.2},
                training_window=252,
                validation_window=63,
                lookback_window=30,
                feature_count=None,  # Will be set dynamically
                learning_rate=0.001
            ),
            'cnn': ModelConfig(
                name='universal_cnn',
                model_type='cnn',
                parameters={'filters': 64, 'kernel_size': 3},
                training_window=252,
                validation_window=63,
                lookback_window=30,
                feature_count=None,  # Will be set dynamically
                learning_rate=0.001
            ),
            'transformer': ModelConfig(
                name='universal_transformer',
                model_type='transformer',
                parameters={'num_heads': 8, 'd_model': 64},
                training_window=252,
                validation_window=63,
                lookback_window=30,
                feature_count=None,  # Will be set dynamically
                learning_rate=0.001
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
    
    def _combine_features_from_featureset(self, feature_set) -> np.ndarray:
        """
        Combine all feature components from a FeatureSet into a single features array.
        
        Args:
            feature_set: FeatureSet object containing various feature categories
            
        Returns:
            Combined features as numpy array
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
        
        return combined_df.values
    
    def _validate_feature_dimensions(self, features_df: pd.DataFrame, context: str, expected_total: int = 184) -> bool:
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
        
        # Technical features: everything else
        special_feature_cols = set(cross_cols + regime_cols + sector_cols + symbol_cols)
        technical_cols = [col for col in features_df.columns if col not in special_feature_cols]
        
        total_features = len(features_df.columns)
        
        # Log detailed breakdown
        logger.info(f"  📊 Feature Breakdown:")
        logger.info(f"    • Technical features: {len(technical_cols)}")
        logger.info(f"    • Symbol embeddings: {len(symbol_cols)}")
        logger.info(f"    • Cross-symbol features: {len(cross_cols)}")
        logger.info(f"    • Market regime features: {len(regime_cols)}")
        logger.info(f"    • Sector features: {len(sector_cols)}")
        logger.info(f"    • TOTAL: {total_features}")
        
        # Log detailed feature column names for debugging
        logger.info(f"  🔍 Detailed Feature Analysis:")
        logger.info(f"    Technical columns ({len(technical_cols)}): {technical_cols[:10]}{'...' if len(technical_cols) > 10 else ''}")
        logger.info(f"    Symbol columns ({len(symbol_cols)}): {symbol_cols}")
        logger.info(f"    Cross-symbol columns ({len(cross_cols)}): {cross_cols}")
        logger.info(f"    Market regime columns ({len(regime_cols)}): {regime_cols}")
        logger.info(f"    Sector columns ({len(sector_cols)}): {sector_cols}")
        
        # Check for any unclassified columns
        all_classified = technical_cols + symbol_cols + cross_cols + regime_cols + sector_cols
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
            universal_features = await self.feature_engineering.engineer_universal_features(
                symbols=[symbol],  # Only this symbol, but we'll get universal features
                start_date=start_date,
                end_date=end_date,
                training_mode=False  # Fine-tuning mode
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
            symbol_embedding_count = len(self.symbol_to_id) + 1  # symbol_id + one-hot encodings
            total_features = total_individual_features + symbol_embedding_count + cross_symbol_count + regime_count + sector_count
            logger.info(f"[{symbol}] Total universal features: {total_features} (individual: {total_individual_features}, embeddings: {symbol_embedding_count}, cross: {cross_symbol_count}, regime: {regime_count}, sector: {sector_count})")
        
        except Exception as e:
            logger.warning(f"Failed to add universal features for {symbol}: {e}")
            logger.warning(f"Proceeding with individual features only for {symbol}")
        
        # Apply same symbol embedding exclusion logic as prepare_universal_dataset
        # Define actual symbol embedding columns to exclude (not cross-symbol or market regime features)
        symbol_embedding_cols = ['symbol_id'] + [col for col in symbol_df.columns if col.startswith('symbol_') and not any([
            col.startswith('corr_'),
            col.startswith('beta_'),
            col.startswith('relative_strength_'),
            col.startswith('market_dispersion_'),
            col.startswith('market_volatility'),
            col.startswith('vol_regime_'),
            col.startswith('vol_trend'),
            col.startswith('vol_correlation')
        ])]
        
        # Keep all features except actual symbol embedding columns
        feature_columns = [col for col in symbol_df.columns if col not in symbol_embedding_cols]
        
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
        
        logger.info(f"[{symbol}] Feature filtering results:")
        logger.info(f"  - Excluded symbol embedding columns ({len(excluded_cols)}): {excluded_cols}")
        logger.info(f"  - Included cross-symbol features ({len(cross_symbol_cols)}): {cross_symbol_cols}")
        logger.info(f"  - Included market regime features ({len(market_regime_cols)}): {market_regime_cols}")
        logger.info(f"  - Total feature columns kept: {len(feature_columns)}")
        
        # Filter to keep only non-symbol-embedding features
        symbol_df = symbol_df[feature_columns]
        
        # Handle NaN values
        symbol_df = symbol_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        symbol_df = symbol_df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Validate feature dimensions before returning (update expected_total to match training)
        validation_passed = self._validate_feature_dimensions(
            symbol_df, 
            f"Phase 2 Fine-tuning - {symbol}", 
            expected_total=184  # Updated to match training phase
        )
        
        if not validation_passed:
            logger.error(f"[{symbol}] Feature validation failed - proceeding with caution")
        
        logger.info(f"[{symbol}] Final feature shape after filtering: {symbol_df.shape}")
        return symbol_df.values
    
    def _extract_targets_from_market_data(self, market_data: pd.DataFrame, threshold: float = 0.001) -> np.ndarray:
        """
        Extract binary targets from market data based on future returns.
        
        Args:
            market_data: DataFrame with market data including 'close' prices
            threshold: Minimum return threshold for positive target (default 0.1%)
            
        Returns:
            Binary targets as numpy array (1 for positive return > threshold, 0 otherwise)
        """
        if 'close' not in market_data.columns:
            raise ValueError("Market data must contain 'close' column for target extraction")
        
        # Calculate next period returns
        returns = market_data['close'].pct_change().shift(-1)
        
        # Create binary targets based on threshold
        targets = (returns > threshold).astype(int)
        
        # Remove the last NaN value (no future return available)
        targets = targets[:-1]
        
        return targets.values
    
    async def prepare_universal_dataset(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
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
            
            # Additional validation for total feature count including all types
            total_features = len(X.columns)
            logger.info(f"  - Total feature validation: {total_features} total columns (technical + symbol + cross + regime + sector)")
            
            # Log feature distribution for debugging
            logger.info(f"  - Feature distribution breakdown:")
            logger.info(f"    * Technical: {len(technical_cols)} ({len(technical_cols)/total_features*100:.1f}%)")
            logger.info(f"    * Symbol: {len(symbol_cols)} ({len(symbol_cols)/total_features*100:.1f}%)")
            logger.info(f"    * Cross-symbol: {len(cross_cols)} ({len(cross_cols)/total_features*100:.1f}%)")
            logger.info(f"    * Market regime: {len(regime_cols)} ({len(regime_cols)/total_features*100:.1f}%)")
            logger.info(f"    * Sector: {len(sector_cols)} ({len(sector_cols)/total_features*100:.1f}%)")
        
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
        symbol_embedding_cols = ['symbol_id'] + [col for col in X.columns if col.startswith('symbol_') and not any([
            col.startswith('corr_'),
            col.startswith('beta_'),
            col.startswith('relative_strength_'),
            col.startswith('market_dispersion_'),
            col.startswith('market_volatility'),
            col.startswith('vol_regime_'),
            col.startswith('vol_trend'),
            col.startswith('vol_correlation')
        ])]
        
        # Keep all features except actual symbol embedding columns
        feature_columns = [col for col in X.columns if col not in symbol_embedding_cols]
        
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
        
        if features.shape[1] < 153:
             logger.warning(f"  - WARNING: Only {features.shape[1]} features in final array, expected ~153 (validated count)")
        
        # Reshape features for sequence models (assuming lookback_window)
        lookback_window = self.config.base_lookback_window if hasattr(self.config, 'base_lookback_window') else 30
        
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
        
        return [X_train_features, X_train_symbols], y_train, [X_val_features, X_val_symbols], y_val
    
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
        
        # Update feature_count dynamically based on actual data dimensions
        if isinstance(X_train, list) and len(X_train) >= 2:
            # X_train[0] contains features, get the feature dimension
            actual_feature_count = X_train[0].shape[-1] if hasattr(X_train[0], 'shape') else len(X_train[0][0])
            logger.info(f"Detected actual feature count: {actual_feature_count}")
            
            # Update all model configs with actual feature count
            for model_type in model_types:
                if model_type in self.model_configs:
                    old_count = self.model_configs[model_type].feature_count
                    self.model_configs[model_type].feature_count = actual_feature_count
                    logger.info(f"Updated {model_type} feature_count from {old_count} to {actual_feature_count}")
        
        results = {}
        
        for model_type in model_types:
            start_time = datetime.now()
            logger.info(f"Training universal {model_type} model")
            
            try:
                # Get model configuration
                config = self.model_configs[model_type]
                
                # Create universal model
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
                
                # Train model
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config.base_epochs,
                    batch_size=self.config.base_batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
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
                    metadata={
                        'model_summary': self.universal_architectures.get_model_summary(model),
                        'training_history': {
                            'loss': [float(x) for x in history.history['loss']],
                            'val_loss': [float(x) for x in history.history['val_loss']],
                            'accuracy': [float(x) for x in history.history['accuracy']],
                            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
                        }
                    }
                )
                
                results[model_type] = result
                logger.info(f"Completed {model_type} base training: {val_accuracy:.4f} accuracy in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} base model: {e}")
                continue
        
        logger.info(f"Phase 1 completed: {len(results)} base models trained")
        return results
    
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
                    X_symbols = np.full(len(X_features), symbol_id)
                    y = self._extract_targets_from_market_data(symbol_data)
                    
                    # Split for validation
                    split_idx = int(len(X_features) * 0.8)
                    X_train = [X_features[:split_idx], X_symbols[:split_idx]]
                    y_train = y[:split_idx]
                    X_val = [X_features[split_idx:], X_symbols[split_idx:]]
                    y_val = y[split_idx:]
                    
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
    
    async def phase3_ensemble_optimization(
        self,
        symbols: List[str],
        validation_start: str,
        validation_end: str
    ) -> Dict[str, float]:
        """
        Phase 3: Optimize ensemble weights based on validation performance.
        
        Args:
            symbols: List of trading symbols
            validation_start: Start date for validation period
            validation_end: End date for validation period
            
        Returns:
            Optimized ensemble weights by model type
        """
        logger.info("Phase 3: Optimizing ensemble weights")
        
        # Collect predictions from all models
        model_predictions = {}
        
        for model_type in self.symbol_models.keys():
            model_predictions[model_type] = {}
            
            for symbol in symbols:
                if symbol not in self.symbol_models[model_type]:
                    continue
                
                try:
                    # Load validation data
                    # Convert dates to timezone-aware UTC datetime objects
                    # Handle both string and datetime inputs
                    if isinstance(validation_start, str):
                        validation_start_dt = datetime.strptime(validation_start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(validation_start, datetime):
                        # Ensure timezone-aware UTC
                        validation_start_dt = validation_start.replace(tzinfo=timezone.utc) if validation_start.tzinfo is None else validation_start.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"validation_start must be str or datetime, got {type(validation_start)}")
                        
                    if isinstance(validation_end, str):
                        validation_end_dt = datetime.strptime(validation_end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    elif isinstance(validation_end, datetime):
                        # Ensure timezone-aware UTC
                        validation_end_dt = validation_end.replace(tzinfo=timezone.utc) if validation_end.tzinfo is None else validation_end.astimezone(timezone.utc)
                    else:
                        raise TypeError(f"validation_end must be str or datetime, got {type(validation_end)}")
                    
                    validation_data = await self.data_pipeline.load_market_data(
                        symbol=symbol,
                        start_date=validation_start_dt,
                        end_date=validation_end_dt
                    )
                    
                    # Get features
                    features = await self.feature_engineering.engineer_features(
                        symbol=symbol,
                        start_date=validation_start_dt,
                        end_date=validation_end_dt
                    )
                    
                    # Make predictions
                    symbol_id = self.symbol_to_id[symbol]
                    X_features = self._combine_features_from_featureset(features)
                    X = [X_features, np.full(len(X_features), symbol_id)]
                    
                    model = self.symbol_models[model_type][symbol]
                    predictions = model.predict(X, verbose=0)
                    
                    targets = self._extract_targets_from_market_data(validation_data)
                    model_predictions[model_type][symbol] = {
                        'predictions': predictions.flatten(),
                        'targets': targets,
                        'accuracy': accuracy_score(targets, (predictions > 0.5).astype(int))
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to get predictions for {model_type}-{symbol}: {e}")
                    continue
        
        # Calculate ensemble weights based on performance
        model_scores = {}
        for model_type in model_predictions.keys():
            accuracies = [pred['accuracy'] for pred in model_predictions[model_type].values()]
            model_scores[model_type] = np.mean(accuracies) if accuracies else 0.0
        
        # Normalize weights
        total_score = sum(model_scores.values())
        if total_score > 0:
            self.ensemble_weights = {model: score / total_score for model, score in model_scores.items()}
        else:
            # Equal weights if no valid scores
            num_models = len(model_scores)
            self.ensemble_weights = {model: 1.0 / num_models for model in model_scores.keys()}
        
        logger.info(f"Phase 3 completed: Ensemble weights = {self.ensemble_weights}")
        return self.ensemble_weights
    
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
        start_time = datetime.now()
        
        try:
            # Phase 1: Train base models
            phase1_results = await self.phase1_train_base_models(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                model_types=model_types
            )
            
            # Phase 2: Symbol-specific fine-tuning
            phase2_results = await self.phase2_symbol_specific_finetuning(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                model_types=list(phase1_results.keys())
            )
            
            # Phase 3: Ensemble optimization
            validation_start = end_date  # Use recent data for validation
            validation_end = datetime.now().strftime('%Y-%m-%d')
            
            ensemble_weights = await self.phase3_ensemble_optimization(
                symbols=symbols,
                validation_start=validation_start,
                validation_end=validation_end
            )
            
            # Calculate total training time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Compile final results
            results = {
                'training_completed': True,
                'total_training_time': total_time,
                'symbols_trained': symbols,
                'phase1_results': phase1_results,
                'phase2_results': phase2_results,
                'ensemble_weights': ensemble_weights,
                'model_summary': {
                    'base_models': list(self.base_models.keys()),
                    'symbol_models': {model: list(symbols.keys()) for model, symbols in self.symbol_models.items()},
                    'total_models': len(self.base_models) + sum(len(symbols) for symbols in self.symbol_models.values())
                },
                'performance_summary': {
                    'avg_base_accuracy': np.mean([r.validation_accuracy for r in phase1_results.values()]) if phase1_results else 0.0,
                    'avg_symbol_accuracy': np.mean([r.validation_accuracy for r in phase2_results.values()]) if phase2_results else 0.0,
                    'best_model': max(ensemble_weights.items(), key=lambda x: x[1])[0] if ensemble_weights else 'none'
                }
            }
            
            logger.info(f"Universal training completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Universal training failed: {e}")
            raise
    
    async def save_universal_models(self, save_dir: Path) -> None:
        """
        Save all universal models and training state.
        
        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        base_dir = save_dir / "base_models"
        base_dir.mkdir(exist_ok=True)
        
        for model_type, model in self.base_models.items():
            model.save(base_dir / f"{model_type}_base.h5")
        
        # Save symbol-specific models
        symbol_dir = save_dir / "symbol_models"
        symbol_dir.mkdir(exist_ok=True)
        
        for model_type, symbol_models in self.symbol_models.items():
            type_dir = symbol_dir / model_type
            type_dir.mkdir(exist_ok=True)
            
            for symbol, model in symbol_models.items():
                model.save(type_dir / f"{symbol}.h5")
        
        # Save metadata
        metadata = {
            'symbol_mappings': {
                'symbol_to_id': self.symbol_to_id,
                'id_to_symbol': self.id_to_symbol
            },
            'ensemble_weights': self.ensemble_weights,
            'config': {
                'symbol_embedding_dim': self.config.symbol_embedding_dim,
                'base_epochs': self.config.base_epochs,
                'finetune_epochs': self.config.finetune_epochs
            },
            'training_timestamp': datetime.now().isoformat()
        }
        
        with open(save_dir / "universal_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved universal models to {save_dir}")
    
    async def load_universal_models(self, load_dir: Path) -> None:
        """
        Load universal models and training state.
        
        Args:
            load_dir: Directory to load models from
        """
        load_dir = Path(load_dir)
        
        # Load metadata
        with open(load_dir / "universal_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.symbol_to_id = metadata['symbol_mappings']['symbol_to_id']
        self.id_to_symbol = {int(k): v for k, v in metadata['symbol_mappings']['id_to_symbol'].items()}
        self.ensemble_weights = metadata['ensemble_weights']
        
        # Initialize architectures
        self.universal_architectures = UniversalModelArchitectures(
            num_symbols=len(self.symbol_to_id),
            symbol_embedding_dim=metadata['config']['symbol_embedding_dim']
        )
        
        # Load base models
        base_dir = load_dir / "base_models"
        for model_file in base_dir.glob("*_base.h5"):
            model_type = model_file.stem.replace('_base', '')
            self.base_models[model_type] = tf.keras.models.load_model(model_file)
        
        # Load symbol-specific models
        symbol_dir = load_dir / "symbol_models"
        for type_dir in symbol_dir.iterdir():
            if type_dir.is_dir():
                model_type = type_dir.name
                self.symbol_models[model_type] = {}
                
                for model_file in type_dir.glob("*.h5"):
                    symbol = model_file.stem
                    self.symbol_models[model_type][symbol] = tf.keras.models.load_model(model_file)
        
        logger.info(f"Loaded universal models from {load_dir}")
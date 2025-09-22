#!/usr/bin/env python3
"""
Model Configuration Settings for Universal Trading System

This module contains configuration settings for all models in the trading system,
including feature selection parameters and model-specific settings.
"""

from typing import Dict, Any, List
from pathlib import Path
import json

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    # Target feature counts (reduced from ~262 to 50-75 range)
    'target_feature_count': 65,
    'min_feature_count': 50,
    'max_feature_count': 75,
    
    # Feature selection method
    'selection_method': 'mutual_info',  # mutual_info, correlation, variance, recursive, lasso
    
    # Feature categories to preserve (minimum from each)
    'min_technical_features': 20,
    'min_market_regime_features': 8,
    'min_cross_symbol_features': 5,
    
    # Scoring weights for composite score
    'correlation_weight': 0.15,
    'mutual_info_weight': 0.20,
    'random_forest_weight': 0.25,
    'lasso_weight': 0.15,
    'neural_network_weight': 0.15,
    'stability_weight': 0.10,
    
    # Stability analysis
    'stability_windows': 5,
    'min_stability_threshold': 0.3,
}

# Universal Model Configuration
UNIVERSAL_MODEL_CONFIG = {
    # Model architecture
    'model_type': 'ensemble',  # ensemble, lstm, cnn, transformer
    
    # Feature configuration
    'use_feature_selection': True,
    'selected_features_path': 'config/selected_features.json',
    'original_feature_count': 262,  # Original feature count before selection
    'target_feature_count': 60,     # Target after feature selection
    
    # Training configuration
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    
    # Model ensemble configuration
    'ensemble_models': {
        'lstm': {
            'enabled': True,
            'weight': 0.3,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2
        },
        'cnn': {
            'enabled': True,
            'weight': 0.3,
            'filters': [64, 128, 256],
            'kernel_sizes': [3, 5, 7],
            'dropout': 0.2
        },
        'transformer': {
            'enabled': True,
            'weight': 0.4,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
    },
    
    # Performance targets
    'target_daily_profit': 0.002,  # 0.2% daily profit target
    'max_drawdown': 0.05,          # 5% maximum drawdown
    'min_sharpe_ratio': 1.5,       # Minimum Sharpe ratio
}

# Signal Generator Configuration
SIGNAL_GENERATOR_CONFIG = {
    # Feature configuration
    'use_selected_features': True,
    'selected_features_path': 'config/selected_features.json',
    
    # Signal thresholds
    'buy_threshold': 0.6,
    'sell_threshold': 0.4,
    'confidence_threshold': 0.7,
    
    # Risk management
    'max_position_size': 0.1,      # 10% of portfolio per position
    'stop_loss_pct': 0.02,         # 2% stop loss
    'take_profit_pct': 0.04,       # 4% take profit
}

# Training Configuration
TRAINING_CONFIG = {
    # Data configuration
    'lookback_days': 30,
    'prediction_horizon': 1,       # 1 minute ahead prediction
    'min_samples_per_symbol': 1000,
    
    # Feature engineering
    'use_feature_selection': True,
    'feature_selection_config': FEATURE_SELECTION_CONFIG,
    
    # Cross-validation
    'cv_folds': 5,
    'time_series_split': True,
    
    # Model selection
    'hyperparameter_tuning': True,
    'optimization_metric': 'sharpe_ratio',
}

def load_selected_features() -> List[str]:
    """
    Load the selected features from the configuration file.
    
    Returns:
        List[str]: List of selected feature names
    """
    features_path = Path(UNIVERSAL_MODEL_CONFIG['selected_features_path'])
    
    if features_path.exists():
        with open(features_path, 'r') as f:
            return json.load(f)
    else:
        # Return empty list if no selection file exists
        return []

def get_feature_count() -> Dict[str, int]:
    """
    Get current feature count information.
    
    Returns:
        Dict[str, int]: Dictionary with feature count information
    """
    selected_features = load_selected_features()
    
    return {
        'original_count': UNIVERSAL_MODEL_CONFIG['original_feature_count'],
        'selected_count': len(selected_features),
        'target_count': UNIVERSAL_MODEL_CONFIG['target_feature_count'],
        'reduction_percentage': (
            (UNIVERSAL_MODEL_CONFIG['original_feature_count'] - len(selected_features)) / 
            UNIVERSAL_MODEL_CONFIG['original_feature_count'] * 100
        ) if selected_features else 0
    }

def validate_feature_selection() -> bool:
    """
    Validate that feature selection meets the requirements.
    
    Returns:
        bool: True if feature selection is valid, False otherwise
    """
    selected_features = load_selected_features()
    
    if not selected_features:
        return False
    
    feature_count = len(selected_features)
    min_count = FEATURE_SELECTION_CONFIG['min_feature_count']
    max_count = FEATURE_SELECTION_CONFIG['max_feature_count']
    
    return min_count <= feature_count <= max_count

# Export all configurations
__all__ = [
    'FEATURE_SELECTION_CONFIG',
    'UNIVERSAL_MODEL_CONFIG',
    'SIGNAL_GENERATOR_CONFIG',
    'TRAINING_CONFIG',
    'load_selected_features',
    'get_feature_count',
    'validate_feature_selection'
]
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnsembleWeights:
    """Ensemble weights configuration"""
    weights: Dict[str, float]
    performance_metrics: Dict[str, Dict[str, float]]
    optimization_timestamp: str
    validation_period: Dict[str, str]
    symbols_used: list
    samples_count: int
    sharpe_ratio: float
    version: str = "1.0"

class EnsembleConfigManager:
    """Unified ensemble configuration management"""
    
    def __init__(self, config_dir: str = "models/ensemble"):
        self.config_dir = config_dir
        self.weights_file = os.path.join(config_dir, "optimized_weights.json")
        self.performance_file = os.path.join(config_dir, "performance_history.json")
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Default weights for all models
        self.default_weights = {
            "lstm": 0.2,
            "cnn": 0.2,
            "random_forest": 0.2,
            "xgboost": 0.2,
            "transformer": 0.2
        }
    
    def save_optimized_weights(self, 
                             weights: Dict[str, float],
                             performance_metrics: Dict[str, Dict[str, float]],
                             validation_period: Dict[str, str],
                             symbols_used: list,
                             samples_count: int,
                             sharpe_ratio: float) -> bool:
        """Save optimized ensemble weights with metadata"""
        try:
            ensemble_config = EnsembleWeights(
                weights=weights,
                performance_metrics=performance_metrics,
                optimization_timestamp=datetime.now().isoformat(),
                validation_period=validation_period,
                symbols_used=symbols_used,
                samples_count=samples_count,
                sharpe_ratio=sharpe_ratio
            )
            
            # Save current weights
            with open(self.weights_file, 'w') as f:
                json.dump(asdict(ensemble_config), f, indent=2)
            
            # Update performance history
            self._update_performance_history(ensemble_config)
            
            logger.info(f"Saved optimized ensemble weights with Sharpe ratio: {sharpe_ratio:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving optimized weights: {e}")
            return False
    
    def load_optimized_weights(self) -> Dict[str, float]:
        """Load the latest optimized ensemble weights"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    config_data = json.load(f)
                
                weights = config_data.get('weights', self.default_weights)
                logger.info(f"Loaded optimized ensemble weights from {self.weights_file}")
                return weights
            else:
                logger.warning(f"No optimized weights found at {self.weights_file}, using default weights")
                return self.default_weights.copy()
                
        except Exception as e:
            logger.error(f"Error loading optimized weights: {e}, using default weights")
            return self.default_weights.copy()
    
    def get_ensemble_metadata(self) -> Optional[Dict[str, Any]]:
        """Get ensemble optimization metadata"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading ensemble metadata: {e}")
            return None
    
    def _update_performance_history(self, ensemble_config: EnsembleWeights) -> None:
        """Update performance history with new optimization results"""
        try:
            history = []
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    history = json.load(f)
            
            # Add new entry
            history.append({
                'timestamp': ensemble_config.optimization_timestamp,
                'sharpe_ratio': ensemble_config.sharpe_ratio,
                'samples_count': ensemble_config.samples_count,
                'symbols_used': ensemble_config.symbols_used,
                'weights': ensemble_config.weights,
                'validation_period': ensemble_config.validation_period
            })
            
            # Keep only last 50 entries
            history = history[-50:]
            
            with open(self.performance_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating performance history: {e}")
    
    def get_performance_history(self) -> list:
        """Get ensemble optimization performance history"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return []
    
    def compare_with_previous(self, current_sharpe: float) -> Dict[str, Any]:
        """Compare current optimization with previous results"""
        try:
            history = self.get_performance_history()
            if not history:
                return {
                    'is_improvement': True,
                    'previous_sharpe': None,
                    'improvement_pct': None,
                    'message': 'First optimization run'
                }
            
            previous_sharpe = history[-1]['sharpe_ratio']
            improvement_pct = ((current_sharpe - previous_sharpe) / abs(previous_sharpe)) * 100 if previous_sharpe != 0 else 0
            
            return {
                'is_improvement': current_sharpe > previous_sharpe,
                'previous_sharpe': previous_sharpe,
                'improvement_pct': improvement_pct,
                'message': f"{'Improved' if current_sharpe > previous_sharpe else 'Declined'} by {abs(improvement_pct):.2f}%"
            }
            
        except Exception as e:
            logger.error(f"Error comparing with previous results: {e}")
            return {
                'is_improvement': True,
                'previous_sharpe': None,
                'improvement_pct': None,
                'message': 'Error comparing results'
            }
    
    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """Validate ensemble weights"""
        try:
            # Check if weights sum to approximately 1
            total_weight = sum(weights.values())
            if not (0.95 <= total_weight <= 1.05):
                logger.warning(f"Weights sum to {total_weight:.3f}, not close to 1.0")
                return False
            
            # Check if any single weight exceeds 40% (as per requirements)
            max_weight = max(weights.values())
            if max_weight > 0.4:
                logger.warning(f"Maximum weight {max_weight:.3f} exceeds 40% limit")
                return False
            
            # Check for negative weights
            if any(w < 0 for w in weights.values()):
                logger.warning("Found negative weights")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating weights: {e}")
            return False
    
    def get_model_loading_config(self) -> Dict[str, Any]:
        """Get configuration for consistent model loading"""
        return {
            'model_types': ['lstm', 'cnn', 'random_forest', 'xgboost', 'transformer'],
            'model_dir': 'models',
            'scaler_dir': 'models/scalers',
            'metadata_file': 'models/training_metadata.json',
            'ensemble_config_dir': self.config_dir
        }
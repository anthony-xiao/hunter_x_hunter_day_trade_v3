import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

@dataclass
class ImbalanceConfig:
    """Configuration for class imbalance mitigation strategies"""
    # SMOTE Configuration
    enable_smote: bool = True
    smote_sampling_strategy: Union[str, float, Dict] = 'auto'  # 'auto', 'minority', float, or dict
    smote_k_neighbors: int = 5
    smote_random_state: int = 42
    min_samples_for_smote: int = 100  # Minimum samples required to apply SMOTE
    
    # Class Weighting Configuration
    enable_class_weights: bool = True
    class_weight_method: str = 'balanced'  # 'balanced', 'balanced_subsample', or custom dict
    false_negative_penalty: float = 2.0  # Additional penalty for missed profitable opportunities
    
    # Cross-validation Configuration
    cv_folds: int = 5
    stratified_cv: bool = True
    
    # Safety Configuration
    max_synthetic_ratio: float = 2.0  # Maximum ratio of synthetic to original samples
    target_positive_ratio: float = 0.35  # Target positive class ratio (35%)

@dataclass
class ImbalanceMetrics:
    """Metrics for tracking class imbalance mitigation effects"""
    original_positive_ratio: float
    original_negative_ratio: float
    original_total_samples: int
    
    smote_applied: bool
    post_smote_positive_ratio: float
    post_smote_negative_ratio: float
    post_smote_total_samples: int
    synthetic_samples_added: int
    
    class_weights_applied: bool
    class_weights: Dict[int, float]
    
    final_positive_ratio: float
    final_negative_ratio: float
    improvement_ratio: float

class ClassImbalanceMitigator:
    """
    Comprehensive class imbalance mitigation using SMOTE and class weighting.
    
    This class provides:
    1. SMOTE for synthetic minority class generation
    2. Class weighting with custom penalties
    3. Cross-validation aware resampling
    4. Comprehensive logging and metrics
    """
    
    def __init__(self, config: ImbalanceConfig = None):
        self.config = config or ImbalanceConfig()
        self.smote = None
        self.scaler = StandardScaler()
        self.metrics_history = []
        
        logger.info(f"Initialized ClassImbalanceMitigator with config: {self.config}")
    
    def analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the class distribution of target variable.
        
        Args:
            y: Target variable array
            
        Returns:
            Dictionary containing distribution analysis
        """
        unique, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        distribution = {
            'total_samples': total_samples,
            'unique_classes': unique.tolist(),
            'class_counts': counts.tolist(),
            'class_ratios': (counts / total_samples).tolist()
        }
        
        if len(unique) == 2:  # Binary classification
            positive_idx = np.argmax(unique)
            negative_idx = 1 - positive_idx
            
            distribution.update({
                'positive_class': unique[positive_idx],
                'negative_class': unique[negative_idx],
                'positive_count': counts[positive_idx],
                'negative_count': counts[negative_idx],
                'positive_ratio': counts[positive_idx] / total_samples,
                'negative_ratio': counts[negative_idx] / total_samples,
                'imbalance_ratio': counts[negative_idx] / counts[positive_idx]
            })
        
        return distribution
    
    def should_apply_smote(self, y: np.ndarray) -> bool:
        """
        Determine if SMOTE should be applied based on configuration and data characteristics.
        
        Args:
            y: Target variable array
            
        Returns:
            Boolean indicating whether to apply SMOTE
        """
        if not self.config.enable_smote:
            return False
        
        # Check minimum samples requirement
        if len(y) < self.config.min_samples_for_smote:
            logger.warning(f"Insufficient samples ({len(y)}) for SMOTE. Minimum required: {self.config.min_samples_for_smote}")
            return False
        
        # Check class distribution
        distribution = self.analyze_class_distribution(y)
        
        if len(distribution['unique_classes']) != 2:
            logger.warning("SMOTE only supports binary classification")
            return False
        
        # Check if minority class has enough samples for k-neighbors
        min_class_count = min(distribution['class_counts'])
        if min_class_count <= self.config.smote_k_neighbors:
            logger.warning(f"Minority class has {min_class_count} samples, insufficient for k_neighbors={self.config.smote_k_neighbors}")
            return False
        
        return True
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply SMOTE to balance the dataset.
        Handles both 2D and 3D (sequence) data by reshaping as needed.
        
        Args:
            X: Feature matrix (2D or 3D for sequences)
            y: Target variable array
            
        Returns:
            Tuple of (resampled_X, resampled_y, smote_info)
        """
        if not self.should_apply_smote(y):
            return X, y, {'applied': False, 'reason': 'Conditions not met'}
        
        try:
            # Initialize SMOTE
            self.smote = SMOTE(
                sampling_strategy=self.config.smote_sampling_strategy,
                k_neighbors=self.config.smote_k_neighbors,
                random_state=self.config.smote_random_state
            )
            
            # Get original distribution
            original_dist = self.analyze_class_distribution(y)
            
            # Handle 3D sequence data
            original_shape = X.shape
            is_3d = len(original_shape) == 3
            
            if is_3d:
                # Reshape 3D to 2D: (samples, timesteps, features) -> (samples, timesteps * features)
                n_samples, n_timesteps, n_features = original_shape
                logger.info(f"Detected 3D sequence data: {original_shape}. Reshaping for SMOTE.")
                X_2d = X.reshape(n_samples, n_timesteps * n_features)
                logger.info(f"Reshaped to 2D: {X_2d.shape}")
            else:
                X_2d = X
                logger.info(f"Using 2D data directly: {X_2d.shape}")
            
            # Apply SMOTE
            logger.info(f"Applying SMOTE with strategy: {self.config.smote_sampling_strategy}")
            X_resampled_2d, y_resampled = self.smote.fit_resample(X_2d, y)
            
            # Reshape back to 3D if needed
            if is_3d:
                # Calculate new number of samples after SMOTE
                n_new_samples = X_resampled_2d.shape[0]
                # Reshape back to 3D: (new_samples, timesteps * features) -> (new_samples, timesteps, features)
                X_resampled = X_resampled_2d.reshape(n_new_samples, n_timesteps, n_features)
                logger.info(f"Reshaped back to 3D: {X_resampled.shape}")
            else:
                X_resampled = X_resampled_2d
            
            # Get new distribution
            new_dist = self.analyze_class_distribution(y_resampled)
            
            # Calculate synthetic samples added
            synthetic_samples = len(y_resampled) - len(y)
            
            # Validate synthetic ratio
            synthetic_ratio = synthetic_samples / len(y)
            if synthetic_ratio > self.config.max_synthetic_ratio:
                logger.warning(f"Synthetic ratio ({synthetic_ratio:.2f}) exceeds maximum ({self.config.max_synthetic_ratio})")
            
            smote_info = {
                'applied': True,
                'original_samples': len(y),
                'resampled_samples': len(y_resampled),
                'synthetic_samples_added': synthetic_samples,
                'synthetic_ratio': synthetic_ratio,
                'original_distribution': original_dist,
                'new_distribution': new_dist,
                'improvement_factor': new_dist['positive_ratio'] / original_dist['positive_ratio'],
                'original_shape': original_shape,
                'final_shape': X_resampled.shape,
                'was_3d': is_3d
            }
            
            logger.info(f"SMOTE applied successfully. Original samples: {len(y)}, New samples: {len(y_resampled)}")
            logger.info(f"Positive class ratio improved from {original_dist['positive_ratio']:.3f} to {new_dist['positive_ratio']:.3f}")
            if is_3d:
                logger.info(f"Sequence data maintained: {original_shape} -> {X_resampled.shape}")
            
            return X_resampled, y_resampled, smote_info
            
        except Exception as e:
            logger.error(f"Error applying SMOTE: {e}")
            return X, y, {'applied': False, 'error': str(e)}
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced dataset.
        
        Args:
            y: Target variable array
            
        Returns:
            Dictionary mapping class labels to weights
        """
        if not self.config.enable_class_weights:
            return {}
        
        try:
            # Get unique classes
            classes = np.unique(y)
            
            # Compute base class weights using sklearn
            if self.config.class_weight_method == 'balanced':
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y
                )
            elif self.config.class_weight_method == 'balanced_subsample':
                weights = compute_class_weight(
                    class_weight='balanced_subsample',
                    classes=classes,
                    y=y
                )
            else:
                # Default balanced approach
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y
                )
            
            # Create class weight dictionary
            class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
            
            # Apply false negative penalty for positive class (assuming 1 is positive)
            if 1 in class_weights and self.config.false_negative_penalty > 1.0:
                class_weights[1] *= self.config.false_negative_penalty
                logger.info(f"Applied false negative penalty of {self.config.false_negative_penalty}x to positive class")
            
            logger.info(f"Computed class weights: {class_weights}")
            return class_weights
            
        except Exception as e:
            logger.error(f"Error computing class weights: {e}")
            return {}
    
    def get_tensorflow_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Get class weights formatted for TensorFlow/Keras models.
        
        Args:
            y: Target variable array
            
        Returns:
            Dictionary of class weights for TensorFlow
        """
        return self.compute_class_weights(y)
    
    def get_sklearn_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Get sample weights for scikit-learn models.
        
        Args:
            y: Target variable array
            
        Returns:
            Array of sample weights
        """
        class_weights = self.compute_class_weights(y)
        if not class_weights:
            return None
        
        # Map class weights to sample weights
        sample_weights = np.array([class_weights.get(int(label), 1.0) for label in y])
        return sample_weights
    
    def apply_comprehensive_balancing(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        apply_to_validation: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, ImbalanceMetrics]:
        """
        Apply comprehensive class imbalance mitigation including SMOTE and class weighting.
        
        Args:
            X: Feature matrix
            y: Target variable array
            apply_to_validation: Whether this is validation data (SMOTE should not be applied)
            
        Returns:
            Tuple of (balanced_X, balanced_y, metrics)
        """
        # Get original distribution
        original_dist = self.analyze_class_distribution(y)
        
        # Initialize metrics
        metrics = ImbalanceMetrics(
            original_positive_ratio=original_dist.get('positive_ratio', 0.0),
            original_negative_ratio=original_dist.get('negative_ratio', 0.0),
            original_total_samples=len(y),
            smote_applied=False,
            post_smote_positive_ratio=original_dist.get('positive_ratio', 0.0),
            post_smote_negative_ratio=original_dist.get('negative_ratio', 0.0),
            post_smote_total_samples=len(y),
            synthetic_samples_added=0,
            class_weights_applied=False,
            class_weights={},
            final_positive_ratio=original_dist.get('positive_ratio', 0.0),
            final_negative_ratio=original_dist.get('negative_ratio', 0.0),
            improvement_ratio=1.0
        )
        
        X_balanced, y_balanced = X.copy(), y.copy()
        
        # Apply SMOTE only to training data
        if not apply_to_validation:
            X_balanced, y_balanced, smote_info = self.apply_smote(X_balanced, y_balanced)
            
            if smote_info.get('applied', False):
                metrics.smote_applied = True
                metrics.post_smote_positive_ratio = smote_info['new_distribution']['positive_ratio']
                metrics.post_smote_negative_ratio = smote_info['new_distribution']['negative_ratio']
                metrics.post_smote_total_samples = len(y_balanced)
                metrics.synthetic_samples_added = smote_info['synthetic_samples_added']
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_balanced)
        if class_weights:
            metrics.class_weights_applied = True
            metrics.class_weights = class_weights
        
        # Update final metrics
        final_dist = self.analyze_class_distribution(y_balanced)
        metrics.final_positive_ratio = final_dist.get('positive_ratio', 0.0)
        metrics.final_negative_ratio = final_dist.get('negative_ratio', 0.0)
        metrics.improvement_ratio = metrics.final_positive_ratio / max(metrics.original_positive_ratio, 0.001)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        logger.info(f"Class imbalance mitigation completed:")
        logger.info(f"  Original positive ratio: {metrics.original_positive_ratio:.3f}")
        logger.info(f"  Final positive ratio: {metrics.final_positive_ratio:.3f}")
        logger.info(f"  Improvement factor: {metrics.improvement_ratio:.2f}x")
        logger.info(f"  SMOTE applied: {metrics.smote_applied}")
        logger.info(f"  Class weights applied: {metrics.class_weights_applied}")
        
        return X_balanced, y_balanced, metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all imbalance mitigation metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            'total_applications': len(self.metrics_history),
            'latest_metrics': {
                'original_positive_ratio': latest_metrics.original_positive_ratio,
                'final_positive_ratio': latest_metrics.final_positive_ratio,
                'improvement_ratio': latest_metrics.improvement_ratio,
                'smote_applied': latest_metrics.smote_applied,
                'synthetic_samples_added': latest_metrics.synthetic_samples_added,
                'class_weights_applied': latest_metrics.class_weights_applied,
                'class_weights': latest_metrics.class_weights
            },
            'average_improvement': np.mean([m.improvement_ratio for m in self.metrics_history]),
            'smote_application_rate': np.mean([m.smote_applied for m in self.metrics_history]),
            'class_weight_application_rate': np.mean([m.class_weights_applied for m in self.metrics_history])
        }
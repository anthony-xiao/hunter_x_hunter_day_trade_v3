import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AggregationConfig:
    """Configuration for temporal aggregation"""
    
    # Aggregation methods to include
    include_mean: bool = True          # Average value over time
    include_current: bool = True       # Latest timestep value
    include_trend: bool = True         # Linear trend (slope)
    include_volatility: bool = True    # Standard deviation
    include_min_max: bool = True       # Min and max values
    include_momentum: bool = True      # Rate of change
    include_acceleration: bool = False # Second derivative (optional)
    
    # Advanced aggregation options
    include_quantiles: bool = False    # 25th, 50th, 75th percentiles
    quantile_levels: List[float] = None  # [0.25, 0.5, 0.75]
    
    # Trend calculation options
    trend_method: str = 'linear'       # 'linear', 'polynomial'
    polynomial_degree: int = 2         # For polynomial trends
    
    # Minimum valid data points for calculations
    min_valid_points: int = 3          # Need at least 3 points for trend
    
    # Feature naming
    separator: str = '_'               # Feature name separator
    
    def __post_init__(self):
        if self.quantile_levels is None:
            self.quantile_levels = [0.25, 0.5, 0.75]

class TemporalAggregator:
    """
    Convert 3D temporal data to meaningful 2D aggregated features
    Preserves temporal patterns through statistical summarization
    """
    
    def __init__(self, config: AggregationConfig = None):
        self.config = config or AggregationConfig()
        logger.info(f"Initialized TemporalAggregator with config: {self.config}")
    
    def aggregate_3d_to_dataframe(self, 
                                data_3d: np.ndarray, 
                                feature_names: List[str],
                                sample_ids: Optional[List] = None) -> pd.DataFrame:
        """
        Convert 3D temporal data to aggregated DataFrame
        
        Args:
            data_3d: (samples, timesteps, features) numpy array
            feature_names: List of base feature names
            sample_ids: Optional sample identifiers
            
        Returns:
            DataFrame with aggregated features
        """
        
        logger.info(f"Aggregating 3D data: {data_3d.shape} -> DataFrame")
        
        if len(feature_names) != data_3d.shape[2]:
            raise ValueError(f"Feature names length {len(feature_names)} doesn't match "
                           f"features dimension {data_3d.shape[2]}")
        
        samples, timesteps, n_features = data_3d.shape
        aggregated_features = {}
        
        # Process each base feature
        for feat_idx, base_name in enumerate(feature_names):
            feature_data = data_3d[:, :, feat_idx]  # Shape: (samples, timesteps)
            
            # Calculate aggregations for this feature
            feature_aggregations = self._calculate_feature_aggregations(
                feature_data, base_name, timesteps
            )
            
            # Add to aggregated features
            aggregated_features.update(feature_aggregations)
        
        # Create DataFrame
        df = pd.DataFrame(aggregated_features, index=sample_ids)
        
        logger.info(f"Created aggregated DataFrame: {df.shape} "
                   f"({n_features} base features -> {df.shape[1]} aggregated features)")
        
        # Log aggregation statistics
        self._log_aggregation_stats(n_features, df.shape[1])
        
        return df
    
    def _calculate_feature_aggregations(self, 
                                      feature_data: np.ndarray, 
                                      base_name: str,
                                      timesteps: int) -> Dict[str, np.ndarray]:
        """Calculate all aggregations for a single feature"""
        
        aggregations = {}
        samples = feature_data.shape[0]
        
        # 1. Mean (central tendency)
        if self.config.include_mean:
            aggregations[f"{base_name}{self.config.separator}mean"] = np.nanmean(feature_data, axis=1)
        
        # 2. Current value (latest state)
        if self.config.include_current:
            aggregations[f"{base_name}{self.config.separator}current"] = feature_data[:, -1]
        
        # 3. Trend analysis (direction over time)
        if self.config.include_trend:
            trends = self._calculate_trends(feature_data, timesteps)
            aggregations[f"{base_name}{self.config.separator}trend"] = trends
        
        # 4. Volatility (stability measure)
        if self.config.include_volatility:
            aggregations[f"{base_name}{self.config.separator}volatility"] = np.nanstd(feature_data, axis=1)
        
        # 5. Min/Max values (range)
        if self.config.include_min_max:
            aggregations[f"{base_name}{self.config.separator}min"] = np.nanmin(feature_data, axis=1)
            aggregations[f"{base_name}{self.config.separator}max"] = np.nanmax(feature_data, axis=1)
        
        # 6. Momentum (rate of change)
        if self.config.include_momentum:
            momentum = self._calculate_momentum(feature_data, timesteps)
            aggregations[f"{base_name}{self.config.separator}momentum"] = momentum
        
        # 7. Acceleration (second derivative)
        if self.config.include_acceleration:
            acceleration = self._calculate_acceleration(feature_data, timesteps)
            aggregations[f"{base_name}{self.config.separator}acceleration"] = acceleration
        
        # 8. Quantiles (distribution shape)
        if self.config.include_quantiles:
            for q in self.config.quantile_levels:
                q_name = f"q{int(q*100)}"
                aggregations[f"{base_name}{self.config.separator}{q_name}"] = np.nanquantile(
                    feature_data, q, axis=1
                )
        
        return aggregations
    
    def _calculate_trends(self, feature_data: np.ndarray, timesteps: int) -> np.ndarray:
        """Calculate linear trends (slopes) for each sample"""
        
        samples = feature_data.shape[0]
        trends = np.zeros(samples)
        time_points = np.arange(timesteps)
        
        for sample_idx in range(samples):
            sample_data = feature_data[sample_idx, :]
            
            # Check for sufficient valid data points
            valid_mask = ~np.isnan(sample_data)
            valid_count = np.sum(valid_mask)
            
            if valid_count >= self.config.min_valid_points:
                try:
                    if self.config.trend_method == 'linear':
                        # Linear trend (slope)
                        valid_times = time_points[valid_mask]
                        valid_values = sample_data[valid_mask]
                        
                        if len(valid_values) > 1:
                            slope, _ = np.polyfit(valid_times, valid_values, 1)
                            trends[sample_idx] = slope
                        
                    elif self.config.trend_method == 'polynomial':
                        # Polynomial trend
                        valid_times = time_points[valid_mask]
                        valid_values = sample_data[valid_mask]
                        
                        if len(valid_values) > self.config.polynomial_degree:
                            poly_coeffs = np.polyfit(valid_times, valid_values, self.config.polynomial_degree)
                            # Use the linear coefficient as trend
                            trends[sample_idx] = poly_coeffs[-2] if len(poly_coeffs) > 1 else 0
                
                except (np.linalg.LinAlgError, np.RankWarning):
                    trends[sample_idx] = 0.0
            else:
                trends[sample_idx] = 0.0
        
        return trends
    
    def _calculate_momentum(self, feature_data: np.ndarray, timesteps: int) -> np.ndarray:
        """Calculate momentum (recent rate of change)"""
        
        samples = feature_data.shape[0]
        momentum = np.zeros(samples)
        
        # Use last 1/3 of the time window for momentum calculation
        momentum_window = max(3, timesteps // 3)
        
        for sample_idx in range(samples):
            recent_data = feature_data[sample_idx, -momentum_window:]
            
            if not np.all(np.isnan(recent_data)) and len(recent_data) > 1:
                # Simple momentum: (current - previous) / time
                start_val = recent_data[0] if not np.isnan(recent_data[0]) else np.nanmean(recent_data[:3])
                end_val = recent_data[-1] if not np.isnan(recent_data[-1]) else np.nanmean(recent_data[-3:])
                
                if not (np.isnan(start_val) or np.isnan(end_val)):
                    momentum[sample_idx] = (end_val - start_val) / momentum_window
        
        return momentum
    
    def _calculate_acceleration(self, feature_data: np.ndarray, timesteps: int) -> np.ndarray:
        """Calculate acceleration (second derivative)"""
        
        samples = feature_data.shape[0]
        acceleration = np.zeros(samples)
        
        for sample_idx in range(samples):
            sample_data = feature_data[sample_idx, :]
            
            if timesteps >= 3:
                # Calculate second derivative using finite differences
                first_diff = np.diff(sample_data)
                second_diff = np.diff(first_diff)
                
                # Use mean of second differences as acceleration
                valid_second_diff = second_diff[~np.isnan(second_diff)]
                if len(valid_second_diff) > 0:
                    acceleration[sample_idx] = np.mean(valid_second_diff)
        
        return acceleration
    
    def _log_aggregation_stats(self, base_features: int, aggregated_features: int):
        """Log aggregation statistics"""
        
        aggregation_count = aggregated_features // base_features if base_features > 0 else 0
        
        logger.info(f"Aggregation Statistics:")
        logger.info(f"  - Base features: {base_features}")
        logger.info(f"  - Aggregated features: {aggregated_features}")
        logger.info(f"  - Aggregation ratio: {aggregation_count}:1")
        logger.info(f"  - Enabled aggregations: {self._get_enabled_aggregations()}")
    
    def _get_enabled_aggregations(self) -> List[str]:
        """Get list of enabled aggregation methods"""
        enabled = []
        
        if self.config.include_mean: enabled.append('mean')
        if self.config.include_current: enabled.append('current')
        if self.config.include_trend: enabled.append('trend')
        if self.config.include_volatility: enabled.append('volatility')
        if self.config.include_min_max: enabled.extend(['min', 'max'])
        if self.config.include_momentum: enabled.append('momentum')
        if self.config.include_acceleration: enabled.append('acceleration')
        if self.config.include_quantiles: 
            enabled.extend([f"q{int(q*100)}" for q in self.config.quantile_levels])
        
        return enabled
    
    def get_feature_mapping(self, original_features: List[str]) -> Dict[str, List[str]]:
        """Get mapping from original features to aggregated features"""
        
        mapping = {}
        enabled_aggs = self._get_enabled_aggregations()
        
        for base_feature in original_features:
            aggregated = []
            for agg_type in enabled_aggs:
                aggregated.append(f"{base_feature}{self.config.separator}{agg_type}")
            mapping[base_feature] = aggregated
        
        return mapping
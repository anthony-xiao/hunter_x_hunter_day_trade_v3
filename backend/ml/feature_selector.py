import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import logging
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportanceScore:
    """Individual feature importance score from different methods"""
    feature_name: str
    correlation_score: float = 0.0      # Correlation with target
    mutual_info_score: float = 0.0      # Mutual information
    random_forest_score: float = 0.0    # RF feature importance
    lasso_score: float = 0.0           # Lasso coefficient magnitude
    lstm_attention_score: float = 0.0   # LSTM attention weights
    cnn_activation_score: float = 0.0   # CNN activation importance
    transformer_attention_score: float = 0.0  # Transformer attention
    stability_score: float = 0.0        # Stability across time windows
    composite_score: float = 0.0        # Weighted composite score
    rank: int = 0                       # Final ranking

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    target_feature_count: int = 65      # Target number of features (50-75 range)
    min_feature_count: int = 50         # Minimum acceptable features
    max_feature_count: int = 75         # Maximum acceptable features
    
    # Scoring weights for composite score
    correlation_weight: float = 0.15
    mutual_info_weight: float = 0.20
    random_forest_weight: float = 0.25
    lasso_weight: float = 0.15
    neural_network_weight: float = 0.15  # Combined NN attention scores
    stability_weight: float = 0.10
    
    # Stability analysis
    stability_windows: int = 5          # Number of time windows for stability
    min_stability_threshold: float = 0.3  # Minimum stability score
    
    # Feature categories to preserve (minimum from each)
    min_technical_features: int = 20    # Minimum technical indicators
    min_market_regime_features: int = 8  # Minimum market regime features
    min_cross_symbol_features: int = 5   # Minimum cross-symbol features
    
    # Output paths
    selected_features_path: str = "config/selected_features.json"
    feature_rankings_path: str = "config/feature_rankings.json"
    selection_metadata_path: str = "config/feature_selection_metadata.json"
    
    # Selection method
    selection_method: str = "mutual_info"  # mutual_info, correlation, variance, recursive, lasso

class UniversalFeatureSelector:
    """Universal feature selection for trading models"""
    
    def __init__(self, config: FeatureSelectionConfig = None):
        self.config = config or FeatureSelectionConfig()
        self.feature_scores: Dict[str, FeatureImportanceScore] = {}
        self.selected_features: List[str] = []
        self.feature_rankings: List[Tuple[str, float]] = []
        self.selection_metadata: Dict[str, Any] = {}
        
        # Create output directories
        Path(self.config.selected_features_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FeatureSelector targeting {self.config.target_feature_count} features")
    
    async def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze features and return comprehensive statistics"""
        logger.info("Starting feature analysis...")
        
        # Calculate basic statistics
        feature_stats = {
            'total_features': len(X.columns),
            'samples': len(X),
            'missing_values': X.isnull().sum().sum(),
            'high_correlation_count': 0,
            'low_variance_count': 0
        }
        
        # Calculate correlations
        correlations = {}
        for feature in X.columns:
            try:
                corr, _ = spearmanr(X[feature].fillna(0), y)
                correlations[feature] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlations[feature] = 0.0
        
        # Count high correlation features
        feature_stats['high_correlation_count'] = sum(1 for corr in correlations.values() if corr > 0.3)
        
        # Calculate variance
        variances = X.var()
        feature_stats['low_variance_count'] = sum(1 for var in variances if var < 0.01)
        
        # Calculate mutual information
        try:
            X_filled = X.fillna(0)
            mi_scores = mutual_info_classif(X_filled, y, random_state=42)
            importance_scores = dict(zip(X.columns, mi_scores))
        except Exception as e:
            logger.warning(f"Could not calculate mutual information: {e}")
            importance_scores = correlations
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        analysis_results = {
            'feature_stats': feature_stats,
            'correlations': correlations,
            'importance_scores': sorted_importance,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Feature analysis completed for {feature_stats['total_features']} features")
        return analysis_results
    
    async def select_features(
        self, 
        X_train: pd.DataFrame = None, 
        y_train: pd.Series = None,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        feature_names: List[str] = None,
        # Legacy support for old signature
        X: pd.DataFrame = None,
        y: pd.Series = None
    ) -> Dict[str, Any]:
        """Select features using the specified method
        
        Args:
            X_train: Training features (preferred)
            y_train: Training labels (preferred)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names (optional)
            X: Legacy parameter for features
            y: Legacy parameter for labels
        """
        # Handle legacy parameters for backward compatibility
        if X_train is None and X is not None:
            X_train = X
        if y_train is None and y is not None:
            y_train = y
            
        if X_train is None or y_train is None:
            raise ValueError("Either (X_train, y_train) or (X, y) must be provided")
        logger.info(f"Starting feature selection using method: {self.config.selection_method}")
        
        try:
            if self.config.selection_method == "mutual_info":
                selected_features = await self._select_by_mutual_info(X_train, y_train)
            elif self.config.selection_method == "correlation":
                selected_features = await self._select_by_correlation(X_train, y_train)
            elif self.config.selection_method == "variance":
                selected_features = await self._select_by_variance(X_train, y_train)
            elif self.config.selection_method == "recursive":
                selected_features = await self._select_by_recursive_elimination(X_train, y_train)
            elif self.config.selection_method == "lasso":
                selected_features = await self._select_by_lasso(X_train, y_train)
            else:
                logger.warning(f"Unknown selection method: {self.config.selection_method}, using mutual_info")
                selected_features = await self._select_by_mutual_info(X_train, y_train)
            
            self.selected_features = selected_features
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(X_train, y_train, selected_features)
            
            # Save results
            await self._save_selection_results()
            
            selection_results = {
                'method': self.config.selection_method,
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'original_feature_count': len(X_train.columns),
                'reduction_percentage': ((len(X_train.columns) - len(selected_features)) / len(X_train.columns)) * 100,
                'performance_metrics': performance_metrics,
                'selection_score': performance_metrics.get('mutual_info_score', 0.0),
                'selection_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Feature selection completed: {len(X_train.columns)} -> {len(selected_features)} features")
            return selection_results
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise
    
    async def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using mutual information"""
        X_filled = X.fillna(0)
        selector = SelectKBest(score_func=mutual_info_classif, k=self.config.target_feature_count)
        selector.fit(X_filled, y)
        
        selected_indices = selector.get_support(indices=True)
        return [X.columns[i] for i in selected_indices]
    
    async def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using correlation with target"""
        correlations = {}
        for feature in X.columns:
            try:
                corr, _ = spearmanr(X[feature].fillna(0), y)
                correlations[feature] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                correlations[feature] = 0.0
        
        # Sort by correlation and select top features
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:self.config.target_feature_count]]
    
    async def _select_by_variance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using variance threshold"""
        from sklearn.feature_selection import VarianceThreshold
        
        X_filled = X.fillna(0)
        
        # First remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = variance_selector.fit_transform(X_filled)
        variance_features = X.columns[variance_selector.get_support()]
        
        # Then select top features by mutual information from remaining
        if len(variance_features) > self.config.target_feature_count:
            X_variance_df = pd.DataFrame(X_variance_filtered, columns=variance_features)
            selector = SelectKBest(score_func=mutual_info_classif, k=self.config.target_feature_count)
            selector.fit(X_variance_df, y)
            selected_indices = selector.get_support(indices=True)
            return [variance_features[i] for i in selected_indices]
        else:
            return list(variance_features)
    
    async def _select_by_recursive_elimination(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using recursive feature elimination"""
        X_filled = X.fillna(0)
        
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=self.config.target_feature_count)
        selector.fit(X_filled, y)
        
        selected_indices = selector.get_support(indices=True)
        return [X.columns[i] for i in selected_indices]
    
    async def _select_by_lasso(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Lasso regularization"""
        from sklearn.preprocessing import StandardScaler
        
        X_filled = X.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Fit Lasso
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        # Select features with non-zero coefficients
        selected_features = []
        feature_importance = [(feature, abs(coef)) for feature, coef in zip(X.columns, lasso.coef_) if abs(coef) > 1e-6]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Take top features up to target count
        for feature, _ in feature_importance[:self.config.target_feature_count]:
            selected_features.append(feature)
        
        return selected_features
    
    async def _calculate_performance_metrics(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> Dict[str, float]:
        """Calculate performance metrics for selected features"""
        try:
            X_selected = X[selected_features].fillna(0)
            
            # Calculate mutual information score
            mi_scores = mutual_info_classif(X_selected, y, random_state=42)
            avg_mutual_info = np.mean(mi_scores)
            
            # Calculate average correlation
            correlations = []
            for feature in selected_features:
                try:
                    corr, _ = spearmanr(X[feature].fillna(0), y)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except Exception:
                    continue
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            # Calculate feature diversity (average pairwise correlation)
            feature_correlations = X_selected.corr().abs()
            upper_triangle = np.triu(feature_correlations, k=1)
            avg_feature_correlation = np.mean(upper_triangle[upper_triangle > 0]) if np.any(upper_triangle > 0) else 0.0
            
            return {
                'mutual_info_score': float(avg_mutual_info),
                'avg_correlation': float(avg_correlation),
                'feature_diversity': float(1.0 - avg_feature_correlation),  # Higher is better
                'feature_count': len(selected_features)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate performance metrics: {e}")
            return {
                'mutual_info_score': 0.0,
                'avg_correlation': 0.0,
                'feature_diversity': 0.0,
                'feature_count': len(selected_features)
            }
    
    async def _save_selection_results(self):
        """Save feature selection results to files"""
        try:
            # Save selected features
            with open(self.config.selected_features_path, 'w') as f:
                json.dump(self.selected_features, f, indent=2)
            
            # Save feature rankings (if available)
            if self.feature_rankings:
                rankings_data = [
                    {
                        'feature': name,
                        'score': score,
                        'rank': rank + 1
                    }
                    for rank, (name, score) in enumerate(self.feature_rankings)
                ]
                
                with open(self.config.feature_rankings_path, 'w') as f:
                    json.dump(rankings_data, f, indent=2)
            
            # Save selection metadata
            category_counts = {}
            for feature in self.selected_features:
                category = self._categorize_feature(feature)
                category_counts[category] = category_counts.get(category, 0) + 1
            
            metadata = {
                'selection_timestamp': datetime.now().isoformat(),
                'selected_feature_count': len(self.selected_features),
                'target_feature_count': self.config.target_feature_count,
                'selection_method': self.config.selection_method,
                'category_distribution': category_counts,
                'config': {
                    'target_feature_count': self.config.target_feature_count,
                    'min_feature_count': self.config.min_feature_count,
                    'max_feature_count': self.config.max_feature_count,
                    'selection_method': self.config.selection_method
                }
            }
            
            with open(self.config.selection_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Feature selection results saved to {self.config.selected_features_path}")
            
        except Exception as e:
            logger.error(f"Failed to save selection results: {e}")
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature name"""
        feature_lower = feature_name.lower()
        
        if any(pattern in feature_lower for pattern in ['corr_', 'beta_', 'relative_strength', 'market_dispersion']):
            return 'cross_symbol'
        elif any(pattern in feature_lower for pattern in ['market_volatility', 'vol_regime', 'vol_trend', 'vol_correlation']):
            return 'market_regime'
        elif any(pattern in feature_lower for pattern in ['rsi', 'macd', 'sma', 'ema', 'bb', 'stoch', 'atr', 'momentum', 'trend']):
            return 'technical'
        else:
            return 'other'
    
    def load_selected_features(self) -> List[str]:
        """Load previously selected features"""
        try:
            with open(self.config.selected_features_path, 'r') as f:
                self.selected_features = json.load(f)
            logger.info(f"Loaded {len(self.selected_features)} selected features")
            return self.selected_features
        except FileNotFoundError:
            logger.warning("No saved feature selection found")
            return []
        except Exception as e:
            logger.error(f"Error loading selected features: {e}")
            return []
    
    async def calculate_comprehensive_importance(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any] = None) -> Dict[str, FeatureImportanceScore]:
        """Calculate feature importance using multiple methods"""
        logger.info("Calculating comprehensive feature importance scores...")
        feature_names = X.columns.tolist()
        
        # Initialize feature scores
        self.feature_scores = {
            name: FeatureImportanceScore(feature_name=name) 
            for name in feature_names
        }
        
        # 1. Correlation-based scoring
        await self._calculate_correlation_scores(X, y)
        
        # 2. Mutual information scoring
        await self._calculate_mutual_info_scores(X, y)
        
        # 3. Random Forest importance
        await self._calculate_random_forest_scores(X, y)
        
        # 4. Lasso regularization scoring
        await self._calculate_lasso_scores(X, y)
        
        # 5. Neural network attention scores (if models provided)
        if models:
            await self._calculate_neural_network_scores(X, y, models)
        
        # 6. Stability scoring across time windows
        await self._calculate_stability_scores(X, y)
        
        # 7. Calculate composite scores
        self._calculate_composite_scores()
        
        logger.info(f"Calculated importance scores for {len(self.feature_scores)} features")
        return self.feature_scores
    
    async def _calculate_correlation_scores(self, X: pd.DataFrame, y: pd.Series):
        """Calculate correlation-based feature importance"""
        try:
            logger.debug("Calculating correlation scores...")
            for feature in X.columns:
                try:
                    # Use Spearman correlation (robust to non-linear relationships)
                    corr, _ = spearmanr(X[feature].fillna(0), y)
                    self.feature_scores[feature].correlation_score = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    logger.warning(f"Error calculating correlation for {feature}: {e}")
                    self.feature_scores[feature].correlation_score = 0.0
        except Exception as e:
            logger.error(f"Error in correlation scoring: {e}")
    
    async def _calculate_mutual_info_scores(self, X: pd.DataFrame, y: pd.Series):
        """Calculate mutual information scores"""
        try:
            logger.debug("Calculating mutual information scores...")
            X_filled = X.fillna(0)
            mi_scores = mutual_info_classif(X_filled, y, random_state=42)
            
            for i, feature in enumerate(X.columns):
                self.feature_scores[feature].mutual_info_score = mi_scores[i]
        except Exception as e:
            logger.error(f"Error in mutual information scoring: {e}")
    
    async def _calculate_random_forest_scores(self, X: pd.DataFrame, y: pd.Series):
        """Calculate Random Forest feature importance"""
        try:
            logger.debug("Calculating Random Forest importance scores...")
            X_filled = X.fillna(0)
            
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                n_jobs=-1
            )
            rf.fit(X_filled, y)
            
            for i, feature in enumerate(X.columns):
                self.feature_scores[feature].random_forest_score = rf.feature_importances_[i]
        except Exception as e:
            logger.error(f"Error in Random Forest scoring: {e}")
    
    async def _calculate_lasso_scores(self, X: pd.DataFrame, y: pd.Series):
        """Calculate Lasso regularization scores"""
        try:
            logger.debug("Calculating Lasso coefficient scores...")
            X_filled = X.fillna(0)
            
            # Standardize features for Lasso
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filled)
            
            lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
            lasso.fit(X_scaled, y)
            
            for i, feature in enumerate(X.columns):
                self.feature_scores[feature].lasso_score = abs(lasso.coef_[i])
        except Exception as e:
            logger.error(f"Error in Lasso scoring: {e}")
    
    async def _calculate_neural_network_scores(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any]):
        """Calculate neural network attention/importance scores"""
        try:
            logger.debug("Calculating neural network importance scores...")
            
            # This would integrate with your existing models
            # For now, placeholder implementation
            for feature in X.columns:
                self.feature_scores[feature].lstm_attention_score = 0.5  # Placeholder
                self.feature_scores[feature].cnn_activation_score = 0.5  # Placeholder
                self.feature_scores[feature].transformer_attention_score = 0.5  # Placeholder
                
        except Exception as e:
            logger.error(f"Error in neural network scoring: {e}")
    
    async def _calculate_stability_scores(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature stability across time windows"""
        try:
            logger.debug("Calculating stability scores...")
            
            window_size = len(X) // self.config.stability_windows
            if window_size < 50:  # Minimum window size
                logger.warning("Dataset too small for stability analysis")
                return
            
            feature_rankings = []
            
            for window in range(self.config.stability_windows):
                start_idx = window * window_size
                end_idx = min((window + 1) * window_size, len(X))
                
                X_window = X.iloc[start_idx:end_idx]
                y_window = y.iloc[start_idx:end_idx]
                
                if len(X_window) < 20:  # Skip very small windows
                    continue
                
                # Calculate mutual info for this window
                X_filled = X_window.fillna(0)
                try:
                    mi_scores = mutual_info_classif(X_filled, y_window, random_state=42)
                    window_rankings = [(feat, score) for feat, score in zip(X.columns, mi_scores)]
                    window_rankings.sort(key=lambda x: x[1], reverse=True)
                    feature_rankings.append(window_rankings)
                except Exception as e:
                    logger.warning(f"Error in stability window {window}: {e}")
                    continue
            
            # Calculate stability as rank consistency across windows
            if feature_rankings:
                for feature in X.columns:
                    ranks = []
                    for ranking in feature_rankings:
                        for i, (feat, _) in enumerate(ranking):
                            if feat == feature:
                                ranks.append(i)
                                break
                    
                    if ranks:
                        rank_stability = 1.0 - (np.std(ranks) / len(X.columns))
                        self.feature_scores[feature].stability_score = max(0.0, rank_stability)
                    else:
                        self.feature_scores[feature].stability_score = 0.0
                        
        except Exception as e:
            logger.error(f"Error in stability scoring: {e}")
    
    def _calculate_composite_scores(self):
        """Calculate weighted composite scores"""
        logger.debug("Calculating composite scores...")
        
        # Normalize scores to 0-1 range
        for score_type in ['correlation_score', 'mutual_info_score', 'random_forest_score', 
                          'lasso_score', 'stability_score']:
            scores = [getattr(score, score_type) for score in self.feature_scores.values()]
            max_score = max(scores) if scores else 1.0
            if max_score > 0:
                for feature_score in self.feature_scores.values():
                    normalized = getattr(feature_score, score_type) / max_score
                    setattr(feature_score, score_type, normalized)
        
        # Calculate composite score
        for feature_score in self.feature_scores.values():
            neural_avg = (feature_score.lstm_attention_score + 
                         feature_score.cnn_activation_score + 
                         feature_score.transformer_attention_score) / 3.0
            
            composite = (
                feature_score.correlation_score * self.config.correlation_weight +
                feature_score.mutual_info_score * self.config.mutual_info_weight +
                feature_score.random_forest_score * self.config.random_forest_weight +
                feature_score.lasso_score * self.config.lasso_weight +
                neural_avg * self.config.neural_network_weight +
                feature_score.stability_score * self.config.stability_weight
            )
            
            feature_score.composite_score = composite
    
    async def select_optimal_features(self, X: pd.DataFrame, y: pd.Series, models: Dict[str, Any] = None) -> List[str]:
        """Select optimal feature subset using comprehensive analysis"""
        logger.info("Starting comprehensive feature selection...")
        
        # Calculate importance scores
        await self.calculate_comprehensive_importance(X, y, models)
        
        # Rank features by composite score
        feature_rankings = [(name, score.composite_score) 
                          for name, score in self.feature_scores.items()]
        feature_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Apply ranking
        for rank, (feature_name, score) in enumerate(feature_rankings):
            self.feature_scores[feature_name].rank = rank + 1
        
        self.feature_rankings = feature_rankings
        
        # Select features with category constraints
        selected_features = self._apply_category_constraints(feature_rankings)
        
        # Validate feature count
        if len(selected_features) < self.config.min_feature_count:
            logger.warning(f"Selected only {len(selected_features)} features, below minimum {self.config.min_feature_count}")
            # Add more features from top rankings
            for feature_name, _ in feature_rankings:
                if feature_name not in selected_features:
                    selected_features.append(feature_name)
                    if len(selected_features) >= self.config.min_feature_count:
                        break
        
        elif len(selected_features) > self.config.max_feature_count:
            selected_features = selected_features[:self.config.max_feature_count]
        
        self.selected_features = selected_features
        
        # Save results
        await self._save_selection_results()
        
        logger.info(f"Selected {len(selected_features)} optimal features")
        return selected_features
    
    def _apply_category_constraints(self, feature_rankings: List[Tuple[str, float]]) -> List[str]:
        """Apply category-based constraints to feature selection"""
        selected_features = []
        category_counts = {
            'technical': 0,
            'market_regime': 0,
            'cross_symbol': 0,
            'other': 0
        }
        
        # First pass: ensure minimum category requirements
        for feature_name, score in feature_rankings:
            category = self._categorize_feature(feature_name)
            
            # Check if we need more features from this category
            needs_category = (
                (category == 'technical' and category_counts['technical'] < self.config.min_technical_features) or
                (category == 'market_regime' and category_counts['market_regime'] < self.config.min_market_regime_features) or
                (category == 'cross_symbol' and category_counts['cross_symbol'] < self.config.min_cross_symbol_features)
            )
            
            if needs_category or len(selected_features) < self.config.target_feature_count:
                selected_features.append(feature_name)
                category_counts[category] += 1
                
                if len(selected_features) >= self.config.target_feature_count:
                    # Check if we've met minimum requirements
                    if (category_counts['technical'] >= self.config.min_technical_features and
                        category_counts['market_regime'] >= self.config.min_market_regime_features and
                        category_counts['cross_symbol'] >= self.config.min_cross_symbol_features):
                        break
        
        return selected_features
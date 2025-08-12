import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
# SQLAlchemy imports removed - using Supabase only
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Concept drift detection metrics"""
    accuracy_drift: float
    precision_drift: float
    recall_drift: float
    sharpe_drift: float
    return_drift: float
    volatility_drift: float
    feature_drift: float
    prediction_drift: float
    overall_drift_score: float
    drift_detected: bool
    confidence_level: float

@dataclass
class DriftAlert:
    """Drift detection alert"""
    timestamp: datetime
    model_name: str
    drift_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    drift_score: float
    affected_metrics: List[str]
    recommended_action: str
    details: Dict[str, Any]

class ConceptDriftDetector:
    """Advanced concept drift detection system"""
    
    def __init__(self, db_url: str = None, supabase_client=None):
        # Only support Supabase client approach
        if supabase_client:
            self.supabase_client = supabase_client
        else:
            raise ValueError("supabase_client must be provided")
        
        # No SQLAlchemy support
        self.engine = None
        self.Session = None
        
        # Drift detection thresholds
        self.thresholds = {
            'accuracy_threshold': 0.05,      # 5% accuracy drop
            'sharpe_threshold': 0.3,         # 0.3 Sharpe ratio drop
            'return_threshold': 0.1,         # 10% return drop
            'volatility_threshold': 0.2,     # 20% volatility increase
            'feature_threshold': 0.15,       # 15% feature distribution change
            'prediction_threshold': 0.1,     # 10% prediction distribution change
            'critical_threshold': 0.7,       # Overall drift score for critical alert
            'high_threshold': 0.5,           # Overall drift score for high alert
            'medium_threshold': 0.3          # Overall drift score for medium alert
        }
        
        # Detection windows
        self.windows = {
            'baseline_days': 30,     # Baseline performance window
            'current_days': 7,       # Current performance window
            'feature_days': 14,      # Feature drift detection window
            'prediction_days': 7     # Prediction drift detection window
        }
        
        # Model performance history
        self.performance_history = {}
        self.drift_alerts = []
        
        # Baseline initialization flag
        self.baseline_initialized = False
        self.baseline_data = {}
    
    async def initialize_baseline(self) -> bool:
        """Initialize baseline performance data for all active models"""
        try:
            logger.info("Initializing baseline performance data for drift detection")
            
            # Get list of active models
            active_models = await self._get_active_models()
            
            if not active_models:
                logger.warning("No active models found for baseline initialization")
                return False
            
            # Initialize baseline for each model
            for model_name in active_models:
                baseline_data = await self._get_baseline_performance(model_name)
                if baseline_data:
                    self.baseline_data[model_name] = baseline_data
                    logger.info(f"Baseline initialized for model: {model_name}")
                else:
                    logger.warning(f"Failed to initialize baseline for model: {model_name}")
            
            self.baseline_initialized = True
            logger.info(f"Baseline initialization completed for {len(self.baseline_data)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize baseline: {e}")
            self.baseline_initialized = False
            return False
    
    async def detect_drift(self, model_name: str, force_check: bool = False) -> DriftMetrics:
        """Detect concept drift for a specific model"""
        try:
            logger.info(f"Starting drift detection for model: {model_name}")
            
            # Get baseline and current performance data
            baseline_data = await self._get_baseline_performance(model_name)
            current_data = await self._get_current_performance(model_name)
            
            if not baseline_data or not current_data:
                logger.warning(f"Insufficient data for drift detection: {model_name}")
                return self._get_empty_drift_metrics()
            
            # Calculate drift metrics
            drift_metrics = await self._calculate_drift_metrics(
                model_name, baseline_data, current_data
            )
            
            # Check for drift alerts
            if drift_metrics.drift_detected or force_check:
                alert = self._generate_drift_alert(model_name, drift_metrics)
                self.drift_alerts.append(alert)
                logger.warning(f"Drift detected for {model_name}: {alert.severity}")
            
            # Update performance history
            self._update_performance_history(model_name, drift_metrics)
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Drift detection failed for {model_name}: {e}")
            return self._get_empty_drift_metrics()
    
    async def detect_all_models_drift(self) -> Dict[str, DriftMetrics]:
        """Detect drift for all active models"""
        try:
            # Get list of active models
            active_models = await self._get_active_models()
            
            drift_results = {}
            
            for model_name in active_models:
                drift_metrics = await self.detect_drift(model_name)
                drift_results[model_name] = drift_metrics
            
            # Generate system-wide drift summary
            await self._generate_system_drift_summary(drift_results)
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to detect drift for all models: {e}")
            return {}
    
    async def _get_baseline_performance(self, model_name: str) -> Optional[Dict]:
        """Get baseline performance data for comparison"""
        try:
            if not self.supabase_client:
                logger.error("Supabase client not available for baseline performance")
                return None
                
            end_date = datetime.now(timezone.utc) - timedelta(days=self.windows['current_days'])
            start_date = end_date - timedelta(days=self.windows['baseline_days'])
            
            # Get model predictions
            predictions_response = self.supabase_client.table('model_predictions').select(
                'timestamp, symbol, prediction, confidence'
            ).eq('model_name', model_name).gte(
                'timestamp', start_date.isoformat()
            ).lte('timestamp', end_date.isoformat()).order('timestamp').execute()
            
            if not predictions_response.data:
                return None
            
            # Get market data for the same period and symbols
            symbols = list(set(pred['symbol'] for pred in predictions_response.data))
            market_data_response = self.supabase_client.table('market_data').select(
                'timestamp, symbol, close'
            ).in_('symbol', symbols).gte(
                'timestamp', start_date.isoformat()
            ).lte('timestamp', end_date.isoformat()).order('timestamp').execute()
            
            if not market_data_response.data:
                return None
            
            # Create market data lookup by symbol and date
            market_lookup = {}
            for md in market_data_response.data:
                date_key = md['timestamp'][:10]  # Extract date part
                key = f"{md['symbol']}_{date_key}"
                market_lookup[key] = md['close']
            
            # Process baseline data with actual returns
            baseline = {
                'predictions': [],
                'confidences': [],
                'actual_returns': [],
                'timestamps': []
            }
            
            # Sort market data by symbol and timestamp for prev_price calculation
            market_by_symbol = {}
            for md in sorted(market_data_response.data, key=lambda x: (x['symbol'], x['timestamp'])):
                if md['symbol'] not in market_by_symbol:
                    market_by_symbol[md['symbol']] = []
                market_by_symbol[md['symbol']].append(md)
            
            for pred in predictions_response.data:
                pred_date = pred['timestamp'][:10]
                market_key = f"{pred['symbol']}_{pred_date}"
                
                if market_key in market_lookup:
                    baseline['predictions'].append(float(pred['prediction']))
                    baseline['confidences'].append(float(pred['confidence']))
                    baseline['timestamps'].append(pred['timestamp'])
                    
                    # Calculate actual return
                    current_price = market_lookup[market_key]
                    symbol_data = market_by_symbol.get(pred['symbol'], [])
                    
                    # Find previous price
                    prev_price = None
                    for i, md in enumerate(symbol_data):
                        if md['timestamp'][:10] == pred_date and i > 0:
                            prev_price = symbol_data[i-1]['close']
                            break
                    
                    if prev_price and prev_price > 0:
                        actual_return = (current_price - prev_price) / prev_price
                        baseline['actual_returns'].append(actual_return)
            
            return baseline if baseline['predictions'] else None
                
        except Exception as e:
            logger.error(f"Failed to get baseline performance for {model_name}: {e}")
            return None
    
    async def _get_current_performance(self, model_name: str) -> Optional[Dict]:
        """Get current performance data for comparison"""
        try:
            if not self.supabase_client:
                logger.error("Supabase client not available for current performance")
                return None
                
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.windows['current_days'])
            
            # Get model predictions
            predictions_response = self.supabase_client.table('model_predictions').select(
                'timestamp, symbol, prediction, confidence'
            ).eq('model_name', model_name).gte(
                'timestamp', start_date.isoformat()
            ).lte('timestamp', end_date.isoformat()).order('timestamp').execute()
            
            if not predictions_response.data:
                return None
            
            # Get market data for the same period and symbols
            symbols = list(set(pred['symbol'] for pred in predictions_response.data))
            market_data_response = self.supabase_client.table('market_data').select(
                'timestamp, symbol, close'
            ).in_('symbol', symbols).gte(
                'timestamp', start_date.isoformat()
            ).lte('timestamp', end_date.isoformat()).order('timestamp').execute()
            
            if not market_data_response.data:
                return None
            
            # Create market data lookup by symbol and date
            market_lookup = {}
            for md in market_data_response.data:
                date_key = md['timestamp'][:10]  # Extract date part
                key = f"{md['symbol']}_{date_key}"
                market_lookup[key] = md['close']
            
            # Process current data with actual returns
            current = {
                'predictions': [],
                'confidences': [],
                'actual_returns': [],
                'timestamps': []
            }
            
            # Sort market data by symbol and timestamp for prev_price calculation
            market_by_symbol = {}
            for md in sorted(market_data_response.data, key=lambda x: (x['symbol'], x['timestamp'])):
                if md['symbol'] not in market_by_symbol:
                    market_by_symbol[md['symbol']] = []
                market_by_symbol[md['symbol']].append(md)
            
            for pred in predictions_response.data:
                pred_date = pred['timestamp'][:10]
                market_key = f"{pred['symbol']}_{pred_date}"
                
                if market_key in market_lookup:
                    current['predictions'].append(float(pred['prediction']))
                    current['confidences'].append(float(pred['confidence']))
                    current['timestamps'].append(pred['timestamp'])
                    
                    # Calculate actual return
                    current_price = market_lookup[market_key]
                    symbol_data = market_by_symbol.get(pred['symbol'], [])
                    
                    # Find previous price
                    prev_price = None
                    for i, md in enumerate(symbol_data):
                        if md['timestamp'][:10] == pred_date and i > 0:
                            prev_price = symbol_data[i-1]['close']
                            break
                    
                    if prev_price and prev_price > 0:
                        actual_return = (current_price - prev_price) / prev_price
                        current['actual_returns'].append(actual_return)
            
            return current if current['predictions'] else None
                
        except Exception as e:
            logger.error(f"Failed to get current performance for {model_name}: {e}")
            return None
    
    async def _calculate_drift_metrics(self, model_name: str, baseline: Dict, current: Dict) -> DriftMetrics:
        """Calculate comprehensive drift metrics"""
        try:
            # Convert to numpy arrays for analysis
            baseline_preds = np.array(baseline['predictions'])
            current_preds = np.array(current['predictions'])
            baseline_returns = np.array(baseline['actual_returns'])
            current_returns = np.array(current['actual_returns'])
            
            # 1. Accuracy drift (using directional accuracy)
            baseline_accuracy = self._calculate_directional_accuracy(
                baseline_preds, baseline_returns
            )
            current_accuracy = self._calculate_directional_accuracy(
                current_preds, current_returns
            )
            accuracy_drift = baseline_accuracy - current_accuracy
            
            # 2. Precision/Recall drift
            baseline_precision, baseline_recall = self._calculate_precision_recall(
                baseline_preds, baseline_returns
            )
            current_precision, current_recall = self._calculate_precision_recall(
                current_preds, current_returns
            )
            precision_drift = baseline_precision - current_precision
            recall_drift = baseline_recall - current_recall
            
            # 3. Sharpe ratio drift
            baseline_sharpe = self._calculate_sharpe_ratio(baseline_returns)
            current_sharpe = self._calculate_sharpe_ratio(current_returns)
            sharpe_drift = baseline_sharpe - current_sharpe
            
            # 4. Return drift
            baseline_return = np.mean(baseline_returns)
            current_return = np.mean(current_returns)
            return_drift = (baseline_return - current_return) / abs(baseline_return) if baseline_return != 0 else 0
            
            # 5. Volatility drift
            baseline_vol = np.std(baseline_returns)
            current_vol = np.std(current_returns)
            volatility_drift = (current_vol - baseline_vol) / baseline_vol if baseline_vol != 0 else 0
            
            # 6. Feature drift (using prediction distribution)
            feature_drift = self._calculate_distribution_drift(baseline_preds, current_preds)
            
            # 7. Prediction drift (using confidence distribution)
            baseline_conf = np.array(baseline['confidences'])
            current_conf = np.array(current['confidences'])
            prediction_drift = self._calculate_distribution_drift(baseline_conf, current_conf)
            
            # 8. Overall drift score (weighted combination)
            drift_weights = {
                'accuracy': 0.25,
                'sharpe': 0.20,
                'return': 0.15,
                'volatility': 0.15,
                'feature': 0.15,
                'prediction': 0.10
            }
            
            overall_drift_score = (
                drift_weights['accuracy'] * min(abs(accuracy_drift) / self.thresholds['accuracy_threshold'], 1.0) +
                drift_weights['sharpe'] * min(abs(sharpe_drift) / self.thresholds['sharpe_threshold'], 1.0) +
                drift_weights['return'] * min(abs(return_drift) / self.thresholds['return_threshold'], 1.0) +
                drift_weights['volatility'] * min(abs(volatility_drift) / self.thresholds['volatility_threshold'], 1.0) +
                drift_weights['feature'] * min(feature_drift / self.thresholds['feature_threshold'], 1.0) +
                drift_weights['prediction'] * min(prediction_drift / self.thresholds['prediction_threshold'], 1.0)
            )
            
            # Determine if drift is detected
            drift_detected = (
                abs(accuracy_drift) > self.thresholds['accuracy_threshold'] or
                abs(sharpe_drift) > self.thresholds['sharpe_threshold'] or
                abs(return_drift) > self.thresholds['return_threshold'] or
                abs(volatility_drift) > self.thresholds['volatility_threshold'] or
                feature_drift > self.thresholds['feature_threshold'] or
                prediction_drift > self.thresholds['prediction_threshold']
            )
            
            # Calculate confidence level
            confidence_level = min(len(baseline_returns), len(current_returns)) / 100.0
            confidence_level = min(confidence_level, 1.0)
            
            return DriftMetrics(
                accuracy_drift=accuracy_drift,
                precision_drift=precision_drift,
                recall_drift=recall_drift,
                sharpe_drift=sharpe_drift,
                return_drift=return_drift,
                volatility_drift=volatility_drift,
                feature_drift=feature_drift,
                prediction_drift=prediction_drift,
                overall_drift_score=overall_drift_score,
                drift_detected=drift_detected,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate drift metrics: {e}")
            return self._get_empty_drift_metrics()
    
    def _calculate_directional_accuracy(self, predictions: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate directional accuracy (correct prediction of up/down movement)"""
        if len(predictions) == 0 or len(actual_returns) == 0:
            return 0.0
        
        # Convert to directional signals
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actual_returns > 0).astype(int)
        
        # Ensure same length
        min_len = min(len(pred_direction), len(actual_direction))
        pred_direction = pred_direction[:min_len]
        actual_direction = actual_direction[:min_len]
        
        if min_len == 0:
            return 0.0
        
        return accuracy_score(actual_direction, pred_direction)
    
    def _calculate_precision_recall(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate precision and recall for directional predictions"""
        if len(predictions) == 0 or len(actual_returns) == 0:
            return 0.0, 0.0
        
        # Convert to directional signals
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actual_returns > 0).astype(int)
        
        # Ensure same length
        min_len = min(len(pred_direction), len(actual_direction))
        pred_direction = pred_direction[:min_len]
        actual_direction = actual_direction[:min_len]
        
        if min_len == 0:
            return 0.0, 0.0
        
        try:
            precision = precision_score(actual_direction, pred_direction, zero_division=0)
            recall = recall_score(actual_direction, pred_direction, zero_division=0)
            return precision, recall
        except:
            return 0.0, 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_distribution_drift(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Calculate distribution drift using Kolmogorov-Smirnov test"""
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(baseline, current)
            
            # Convert to drift score (higher = more drift)
            drift_score = ks_statistic
            
            return drift_score
            
        except Exception as e:
            logger.error(f"Failed to calculate distribution drift: {e}")
            return 0.0
    
    def _generate_drift_alert(self, model_name: str, drift_metrics: DriftMetrics) -> DriftAlert:
        """Generate drift alert based on metrics"""
        # Determine severity
        if drift_metrics.overall_drift_score >= self.thresholds['critical_threshold']:
            severity = 'CRITICAL'
            recommended_action = 'Immediate model retraining required'
        elif drift_metrics.overall_drift_score >= self.thresholds['high_threshold']:
            severity = 'HIGH'
            recommended_action = 'Schedule model retraining within 24 hours'
        elif drift_metrics.overall_drift_score >= self.thresholds['medium_threshold']:
            severity = 'MEDIUM'
            recommended_action = 'Monitor closely and consider retraining'
        else:
            severity = 'LOW'
            recommended_action = 'Continue monitoring'
        
        # Identify affected metrics
        affected_metrics = []
        if abs(drift_metrics.accuracy_drift) > self.thresholds['accuracy_threshold']:
            affected_metrics.append('accuracy')
        if abs(drift_metrics.sharpe_drift) > self.thresholds['sharpe_threshold']:
            affected_metrics.append('sharpe_ratio')
        if abs(drift_metrics.return_drift) > self.thresholds['return_threshold']:
            affected_metrics.append('returns')
        if abs(drift_metrics.volatility_drift) > self.thresholds['volatility_threshold']:
            affected_metrics.append('volatility')
        if drift_metrics.feature_drift > self.thresholds['feature_threshold']:
            affected_metrics.append('features')
        if drift_metrics.prediction_drift > self.thresholds['prediction_threshold']:
            affected_metrics.append('predictions')
        
        # Determine drift type
        if 'accuracy' in affected_metrics or 'sharpe_ratio' in affected_metrics:
            drift_type = 'PERFORMANCE_DRIFT'
        elif 'features' in affected_metrics:
            drift_type = 'FEATURE_DRIFT'
        elif 'predictions' in affected_metrics:
            drift_type = 'PREDICTION_DRIFT'
        else:
            drift_type = 'GENERAL_DRIFT'
        
        return DriftAlert(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            drift_type=drift_type,
            severity=severity,
            drift_score=drift_metrics.overall_drift_score,
            affected_metrics=affected_metrics,
            recommended_action=recommended_action,
            details={
                'accuracy_drift': drift_metrics.accuracy_drift,
                'sharpe_drift': drift_metrics.sharpe_drift,
                'return_drift': drift_metrics.return_drift,
                'volatility_drift': drift_metrics.volatility_drift,
                'feature_drift': drift_metrics.feature_drift,
                'prediction_drift': drift_metrics.prediction_drift,
                'confidence_level': drift_metrics.confidence_level
            }
        )
    
    async def _get_active_models(self) -> List[str]:
        """Get list of active models"""
        try:
            if not self.supabase_client:
                logger.error("Supabase client not available for getting active models")
                return ['lstm', 'cnn', 'random_forest', 'xgboost', 'transformer']  # Default models
                
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
            
            response = self.supabase_client.table('model_predictions').select(
                'model_name'
            ).gte('timestamp', cutoff_date.isoformat()).execute()
            
            if response.data:
                # Get unique model names
                model_names = list(set(row['model_name'] for row in response.data))
                return model_names
            else:
                return ['lstm', 'cnn', 'random_forest', 'xgboost', 'transformer']  # Default models
                
        except Exception as e:
            logger.error(f"Failed to get active models: {e}")
            return ['lstm', 'cnn', 'random_forest', 'xgboost', 'transformer']  # Default models
    
    async def _generate_system_drift_summary(self, drift_results: Dict[str, DriftMetrics]) -> None:
        """Generate system-wide drift summary"""
        try:
            total_models = len(drift_results)
            models_with_drift = sum(1 for metrics in drift_results.values() if metrics.drift_detected)
            avg_drift_score = np.mean([metrics.overall_drift_score for metrics in drift_results.values()])
            
            logger.info(f"System Drift Summary: {models_with_drift}/{total_models} models with drift, avg score: {avg_drift_score:.3f}")
            
            # Log critical alerts
            critical_models = [
                name for name, metrics in drift_results.items() 
                if metrics.overall_drift_score >= self.thresholds['critical_threshold']
            ]
            
            if critical_models:
                logger.critical(f"CRITICAL DRIFT DETECTED in models: {critical_models}")
            
        except Exception as e:
            logger.error(f"Failed to generate system drift summary: {e}")
    
    def _update_performance_history(self, model_name: str, drift_metrics: DriftMetrics) -> None:
        """Update performance history for trend analysis"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        self.performance_history[model_name].append({
            'timestamp': datetime.now(timezone.utc),
            'drift_score': drift_metrics.overall_drift_score,
            'drift_detected': drift_metrics.drift_detected
        })
        
        # Keep only last 100 records
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def _get_empty_drift_metrics(self) -> DriftMetrics:
        """Get empty drift metrics for error cases"""
        return DriftMetrics(
            accuracy_drift=0.0,
            precision_drift=0.0,
            recall_drift=0.0,
            sharpe_drift=0.0,
            return_drift=0.0,
            volatility_drift=0.0,
            feature_drift=0.0,
            prediction_drift=0.0,
            overall_drift_score=0.0,
            drift_detected=False,
            confidence_level=0.0
        )
    
    async def get_drift_alerts(self, severity_filter: Optional[str] = None) -> List[DriftAlert]:
        """Get recent drift alerts"""
        alerts = self.drift_alerts.copy()
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:50]  # Return last 50 alerts
    
    async def should_retrain_model(self, model_name: str) -> Tuple[bool, str]:
        """Determine if a model should be retrained based on drift"""
        try:
            drift_metrics = await self.detect_drift(model_name)
            
            if drift_metrics.overall_drift_score >= self.thresholds['critical_threshold']:
                return True, "Critical drift detected - immediate retraining required"
            elif drift_metrics.overall_drift_score >= self.thresholds['high_threshold']:
                return True, "High drift detected - retraining recommended"
            elif drift_metrics.drift_detected:
                return True, "Drift detected - consider retraining"
            else:
                return False, "No significant drift detected"
                
        except Exception as e:
            logger.error(f"Failed to determine retraining need for {model_name}: {e}")
            return False, f"Error in drift detection: {e}"
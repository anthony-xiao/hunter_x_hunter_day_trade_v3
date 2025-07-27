import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import required modules for walk-forward testing
from data.data_pipeline import DataPipeline
from ml.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for validation"""
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_per_day: float
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    total_trades: int
    profitable_trades: int
    avg_profit_per_trade: float
    max_consecutive_losses: int
    recovery_factor: float

@dataclass
class ValidationResult:
    """Result of performance validation"""
    passed: bool
    metrics: PerformanceMetrics
    validation_details: Dict[str, Any]
    recommendations: List[str]
    risk_warnings: List[str]

@dataclass
class WalkForwardValidationResult:
    """Result of walk-forward testing validation for system readiness"""
    system_ready: bool
    ml_results: Dict[str, Any]  # Results from model_trainer.walk_forward_test
    trading_validation: Dict[str, Any]  # Trading system validation results
    overall_score: float
    recommendations: List[str]
    risk_warnings: List[str]
    data_period: Tuple[datetime, datetime]
    models_tested: List[str]

class PerformanceValidator:
    """Advanced performance validation system"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize data pipeline and model trainer for walk-forward testing
        self.data_pipeline = DataPipeline()
        self.model_trainer = ModelTrainer(create_model_dir=False)  # Don't create directories for walk-forward testing
        
        # Performance targets from requirements
        self.targets = {
            'min_annual_return': 0.30,      # 30% minimum annual return
            'target_annual_return': 0.45,   # 45% target annual return
            'max_annual_return': 0.60,      # 60% maximum expected return
            'min_sharpe_ratio': 2.0,        # 2.0 minimum Sharpe ratio
            'target_sharpe_ratio': 2.75,    # 2.75 target Sharpe ratio
            'max_sharpe_ratio': 3.5,        # 3.5 maximum expected Sharpe
            'max_drawdown': 0.15,           # 15% maximum drawdown
            'min_win_rate': 0.55,           # 55% minimum win rate
            'min_trades_per_day': 200,      # 200 minimum trades/day
            'max_trades_per_day': 400,      # 400 maximum trades/day
            'min_profit_factor': 1.5,       # 1.5 minimum profit factor
            'max_volatility': 0.25,         # 25% maximum annual volatility
            'min_calmar_ratio': 2.0,        # 2.0 minimum Calmar ratio
            'min_sortino_ratio': 3.0        # 3.0 minimum Sortino ratio
        }
    
    async def validate_system_performance(self, 
                                        start_date: datetime,
                                        end_date: datetime,
                                        model_predictions: Optional[Dict] = None) -> ValidationResult:
        """Comprehensive system performance validation"""
        try:
            logger.info(f"Starting performance validation from {start_date} to {end_date}")
            
            # Get trading data for the period
            trades_data = await self._get_trades_data(start_date, end_date)
            returns_data = await self._get_returns_data(start_date, end_date)
            
            if not trades_data or not returns_data:
                logger.warning("Insufficient data for performance validation")
                return ValidationResult(
                    passed=False,
                    metrics=self._get_empty_metrics(),
                    validation_details={'error': 'Insufficient data'},
                    recommendations=['Collect more trading data before validation'],
                    risk_warnings=['Cannot validate performance without sufficient data']
                )
            
            # Calculate comprehensive metrics
            metrics = await self._calculate_comprehensive_metrics(trades_data, returns_data)
            
            # Validate against targets
            validation_details = self._validate_against_targets(metrics)
            
            # Generate recommendations and warnings
            recommendations = self._generate_recommendations(metrics, validation_details)
            risk_warnings = self._generate_risk_warnings(metrics, validation_details)
            
            # Determine overall pass/fail
            passed = self._determine_validation_result(validation_details)
            
            logger.info(f"Performance validation complete. Passed: {passed}")
            
            return ValidationResult(
                passed=passed,
                metrics=metrics,
                validation_details=validation_details,
                recommendations=recommendations,
                risk_warnings=risk_warnings
            )
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return ValidationResult(
                passed=False,
                metrics=self._get_empty_metrics(),
                validation_details={'error': str(e)},
                recommendations=['Fix validation system errors'],
                risk_warnings=['Performance validation system failure']
            )
    
    async def walk_forward_test(self, start_date: datetime, end_date: datetime, 
                               symbol: str = 'AAPL') -> WalkForwardValidationResult:
        """Walk-forward testing for system readiness validation
        
        This method bridges execution_engine and model_trainer by:
        1. Loading data from PostgreSQL using data_pipeline.load_features_from_db()
        2. Calling model_trainer.walk_forward_test() with the loaded dataframes
        3. Validating results against trading system requirements (30-60% returns, 2.0-3.5 Sharpe)
        4. Returning system readiness assessment for live trading
        
        Args:
            start_date: Start date for walk-forward testing
            end_date: End date for walk-forward testing
            symbol: Symbol to test (default: AAPL)
            
        Returns:
            WalkForwardValidationResult with system readiness assessment
        """
        try:
            logger.info(f"Starting walk-forward testing validation from {start_date} to {end_date} for {symbol}")
            
            # Step 1: Load data from PostgreSQL using data_pipeline
            logger.info("Loading features and market data from PostgreSQL...")
            features_df = await self.data_pipeline.load_features_from_db(symbol, start_date, end_date)
            
            if features_df.empty:
                logger.error(f"No features data found for {symbol} in the specified period")
                return WalkForwardValidationResult(
                    system_ready=False,
                    ml_results={},
                    trading_validation={'error': 'No features data available'},
                    overall_score=0.0,
                    recommendations=['Ensure features are engineered and stored for the specified period'],
                    risk_warnings=['Cannot perform walk-forward testing without features data'],
                    data_period=(start_date, end_date),
                    models_tested=[]
                )
            
            # Extract targets from features data (predict next period price direction)
            logger.info("Extracting targets from features data...")
            targets_df = self._extract_targets_from_features(features_df)
            
            if targets_df.empty:
                logger.error("Failed to extract valid targets from features data")
                return WalkForwardValidationResult(
                    system_ready=False,
                    ml_results={},
                    trading_validation={'error': 'Failed to extract targets'},
                    overall_score=0.0,
                    recommendations=['Check data quality and target extraction logic'],
                    risk_warnings=['Cannot perform walk-forward testing without valid targets'],
                    data_period=(start_date, end_date),
                    models_tested=[]
                )
            
            logger.info(f"Loaded {len(features_df)} feature records and {len(targets_df)} targets")
            
            # Step 2: Call model_trainer.walk_forward_test() with loaded dataframes
            logger.info("Running ML walk-forward testing...")
            ml_results = await self.model_trainer.walk_forward_test(features_df, targets_df)
            
            if not ml_results:
                logger.error("ML walk-forward testing returned no results")
                return WalkForwardValidationResult(
                    system_ready=False,
                    ml_results={},
                    trading_validation={'error': 'ML testing failed'},
                    overall_score=0.0,
                    recommendations=['Check model training configuration and data quality'],
                    risk_warnings=['ML models failed walk-forward testing'],
                    data_period=(start_date, end_date),
                    models_tested=[]
                )
            
            # Step 3: Validate results against trading system requirements
            logger.info("Validating ML results against trading system requirements...")
            trading_validation = self._validate_trading_system_requirements(ml_results)
            
            # Step 4: Generate system readiness assessment
            system_readiness = self._assess_system_readiness(ml_results, trading_validation)
            
            logger.info(f"Walk-forward testing validation complete. System ready: {system_readiness['ready']}")
            
            return WalkForwardValidationResult(
                system_ready=system_readiness['ready'],
                ml_results=ml_results,
                trading_validation=trading_validation,
                overall_score=system_readiness['score'],
                recommendations=system_readiness['recommendations'],
                risk_warnings=system_readiness['warnings'],
                data_period=(start_date, end_date),
                models_tested=list(ml_results.keys())
            )
            
        except Exception as e:
            logger.error(f"Walk-forward testing validation failed: {e}")
            return WalkForwardValidationResult(
                system_ready=False,
                ml_results={},
                trading_validation={'error': str(e)},
                overall_score=0.0,
                recommendations=['Fix walk-forward testing system errors'],
                risk_warnings=['Walk-forward testing system failure'],
                data_period=(start_date, end_date),
                models_tested=[]
            )
    
    def _extract_targets_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract trading targets from features dataframe"""
        try:
            # Ensure we have close price data
            if 'close' not in features_df.columns:
                logger.error("No 'close' price column found in features data")
                return pd.DataFrame()
            
            # Create target: predict if next period's close will be higher than current close
            targets = (features_df['close'].shift(-1) > features_df['close']).astype(int)
            targets = targets.dropna()
            
            # Return as DataFrame with proper index alignment
            targets_df = pd.DataFrame({'target': targets}, index=targets.index)
            
            logger.info(f"Extracted {len(targets_df)} targets from features data")
            return targets_df
            
        except Exception as e:
            logger.error(f"Failed to extract targets: {e}")
            return pd.DataFrame()
    
    def _validate_trading_system_requirements(self, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ML results against trading system requirements (30-60% returns, 2.0-3.5 Sharpe)"""
        try:
            validation_results = {
                'models_validated': {},
                'overall_validation': {},
                'requirements_met': False,
                'critical_failures': []
            }
            
            models_passing = 0
            total_models = len(ml_results)
            
            for model_name, walk_forward_result in ml_results.items():
                model_validation = {
                    'sharpe_ratio': walk_forward_result.avg_sharpe,
                    'consistency_score': walk_forward_result.consistency_score,
                    'total_periods': walk_forward_result.total_periods,
                    'meets_sharpe_requirement': False,
                    'meets_consistency_requirement': False,
                    'overall_pass': False
                }
                
                # Check Sharpe ratio requirement (2.0-3.5)
                if self.targets['min_sharpe_ratio'] <= walk_forward_result.avg_sharpe <= self.targets['max_sharpe_ratio']:
                    model_validation['meets_sharpe_requirement'] = True
                else:
                    validation_results['critical_failures'].append(
                        f"{model_name}: Sharpe ratio {walk_forward_result.avg_sharpe:.2f} outside target range {self.targets['min_sharpe_ratio']}-{self.targets['max_sharpe_ratio']}"
                    )
                
                # Check consistency requirement (at least 70% of periods should pass)
                if walk_forward_result.consistency_score >= 0.70:
                    model_validation['meets_consistency_requirement'] = True
                else:
                    validation_results['critical_failures'].append(
                        f"{model_name}: Consistency score {walk_forward_result.consistency_score:.1%} below 70% threshold"
                    )
                
                # Overall model pass/fail
                model_validation['overall_pass'] = (
                    model_validation['meets_sharpe_requirement'] and 
                    model_validation['meets_consistency_requirement']
                )
                
                if model_validation['overall_pass']:
                    models_passing += 1
                
                validation_results['models_validated'][model_name] = model_validation
            
            # Overall system validation
            passing_percentage = models_passing / total_models if total_models > 0 else 0
            validation_results['overall_validation'] = {
                'models_passing': models_passing,
                'total_models': total_models,
                'passing_percentage': passing_percentage,
                'min_models_required': 3  # At least 3 models should pass
            }
            
            # System requirements met if at least 3 models pass (60% of 5 models)
            validation_results['requirements_met'] = models_passing >= 3
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate trading system requirements: {e}")
            return {
                'models_validated': {},
                'overall_validation': {},
                'requirements_met': False,
                'critical_failures': [f'Validation error: {str(e)}']
            }
    
    def _assess_system_readiness(self, ml_results: Dict[str, Any], 
                                trading_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system readiness for live trading"""
        try:
            recommendations = []
            warnings = []
            
            # Calculate overall score based on multiple factors
            score_components = {
                'models_passing': 0.0,
                'sharpe_quality': 0.0,
                'consistency': 0.0,
                'data_sufficiency': 0.0
            }
            
            # Models passing score (40% weight)
            if trading_validation['requirements_met']:
                passing_pct = trading_validation['overall_validation']['passing_percentage']
                score_components['models_passing'] = min(passing_pct * 1.2, 1.0) * 0.4
            else:
                warnings.append("Insufficient models meeting performance requirements")
            
            # Sharpe ratio quality score (30% weight)
            sharpe_scores = []
            for model_name, result in ml_results.items():
                # Normalize Sharpe ratio to 0-1 scale (target range 2.0-3.5)
                normalized_sharpe = min(max((result.avg_sharpe - 2.0) / 1.5, 0), 1)
                sharpe_scores.append(normalized_sharpe)
            
            if sharpe_scores:
                avg_sharpe_score = float(np.mean(sharpe_scores))
                score_components['sharpe_quality'] = avg_sharpe_score * 0.3
            
            # Consistency score (20% weight)
            consistency_scores = [result.consistency_score for result in ml_results.values()]
            if consistency_scores:
                avg_consistency = float(np.mean(consistency_scores))
                score_components['consistency'] = avg_consistency * 0.2
            
            # Data sufficiency score (10% weight)
            total_periods = [result.total_periods for result in ml_results.values()]
            if total_periods:
                avg_periods = float(np.mean(total_periods))
                # Target: at least 50 periods for robust testing
                data_score = min(avg_periods / 50.0, 1.0)
                score_components['data_sufficiency'] = data_score * 0.1
            
            # Calculate overall score
            overall_score = float(sum(score_components.values()))
            
            # Determine system readiness
            system_ready = bool(
                overall_score >= 0.75 and  # At least 75% overall score
                trading_validation['requirements_met'] and  # Core requirements met
                len(trading_validation['critical_failures']) == 0  # No critical failures
            )
            
            # Generate recommendations
            if not system_ready:
                if overall_score < 0.75:
                    recommendations.append(f"Improve overall system performance (current score: {overall_score:.1%})")
                
                if not trading_validation['requirements_met']:
                    recommendations.append("Ensure at least 3 models meet Sharpe ratio and consistency requirements")
                
                if trading_validation['critical_failures']:
                    recommendations.extend([
                        "Address critical model failures:",
                        *trading_validation['critical_failures']
                    ])
            else:
                recommendations.append("System ready for live trading deployment")
                recommendations.append("Monitor performance continuously during live trading")
            
            # Generate warnings
            if overall_score < 0.6:
                warnings.append("CRITICAL: Overall system score below 60% - high risk deployment")
            
            if len(ml_results) < 5:
                warnings.append(f"WARNING: Only {len(ml_results)} models tested, recommend testing all 5 models")
            
            return {
                'ready': system_ready,
                'score': overall_score,
                'score_components': score_components,
                'recommendations': recommendations,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Failed to assess system readiness: {e}")
            return {
                'ready': False,
                'score': 0.0,
                'score_components': {},
                'recommendations': ['Fix system readiness assessment'],
                'warnings': ['System readiness assessment failed']
            }
    
    async def _get_trades_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get trading data for performance analysis"""
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT 
                        symbol,
                        entry_time,
                        exit_time,
                        entry_price,
                        exit_price,
                        quantity,
                        side,
                        pnl,
                        commission,
                        strategy
                    FROM trades
                    WHERE entry_time >= :start_date
                    AND exit_time <= :end_date
                    AND exit_time IS NOT NULL
                    ORDER BY entry_time
                """), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                trades = []
                for row in result.fetchall():
                    trades.append({
                        'symbol': row.symbol,
                        'entry_time': row.entry_time,
                        'exit_time': row.exit_time,
                        'entry_price': float(row.entry_price),
                        'exit_price': float(row.exit_price),
                        'quantity': float(row.quantity),
                        'side': row.side,
                        'pnl': float(row.pnl),
                        'commission': float(row.commission),
                        'strategy': row.strategy
                    })
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to get trades data: {e}")
            return []
    
    async def _get_returns_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get daily returns data for performance analysis"""
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT 
                        date,
                        daily_pnl,
                        total_trades,
                        winning_trades,
                        portfolio_value
                    FROM daily_performance
                    WHERE date >= :start_date
                    AND date <= :end_date
                    ORDER BY date
                """), {
                    'start_date': start_date.date(),
                    'end_date': end_date.date()
                })
                
                returns = []
                for row in result.fetchall():
                    returns.append({
                        'date': row.date,
                        'daily_pnl': float(row.daily_pnl),
                        'total_trades': int(row.total_trades),
                        'winning_trades': int(row.winning_trades),
                        'portfolio_value': float(row.portfolio_value)
                    })
                
                return returns
                
        except Exception as e:
            logger.error(f"Failed to get returns data: {e}")
            return []
    
    async def _calculate_comprehensive_metrics(self, trades_data: List[Dict], returns_data: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Convert to DataFrames for easier analysis
            trades_df = pd.DataFrame(trades_data)
            returns_df = pd.DataFrame(returns_data)
            
            # Basic trade metrics
            total_trades = len(trades_df)
            profitable_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = trades_df['pnl'].sum()
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            # Time-based metrics
            if not returns_df.empty:
                returns_df['date'] = pd.to_datetime(returns_df['date'])
                period_days = (returns_df['date'].max() - returns_df['date'].min()).days
                trades_per_day = total_trades / period_days if period_days > 0 else 0
                
                # Calculate daily returns
                returns_df['daily_return'] = returns_df['daily_pnl'] / returns_df['portfolio_value'].shift(1)
                daily_returns = returns_df['daily_return'].dropna()
                
                # Annual return
                total_return = (returns_df['portfolio_value'].iloc[-1] / returns_df['portfolio_value'].iloc[0]) - 1
                annual_return = (1 + total_return) ** (365 / period_days) - 1 if period_days > 0 else 0
                
                # Volatility (annualized)
                volatility = daily_returns.std() * np.sqrt(252)
                
                # Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                excess_return = annual_return - risk_free_rate
                sharpe_ratio = excess_return / volatility if volatility > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = abs(drawdowns.min())
                
                # Calmar ratio
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
                
                # Sortino ratio
                downside_returns = daily_returns[daily_returns < 0]
                downside_deviation = downside_returns.std() * np.sqrt(252)
                sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
                
            else:
                trades_per_day = 0
                annual_return = 0
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                calmar_ratio = 0
                sortino_ratio = 0
            
            # Trade duration analysis
            if not trades_df.empty:
                trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) - 
                                       pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 3600
                avg_trade_duration = trades_df['duration'].mean()
                
                # Consecutive losses
                trades_df['is_loss'] = trades_df['pnl'] < 0
                consecutive_losses = self._calculate_max_consecutive_losses(trades_df['is_loss'].tolist())
            else:
                avg_trade_duration = 0
                consecutive_losses = 0
            
            # Recovery factor
            recovery_factor = total_pnl / max_drawdown if max_drawdown > 0 else 0
            
            return PerformanceMetrics(
                annual_return=annual_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades_per_day=trades_per_day,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                avg_profit_per_trade=avg_profit_per_trade,
                max_consecutive_losses=consecutive_losses,
                recovery_factor=recovery_factor
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return self._get_empty_metrics()
    
    def _calculate_max_consecutive_losses(self, losses: List[bool]) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for is_loss in losses:
            if is_loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _validate_against_targets(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Validate metrics against performance targets"""
        validation = {
            'annual_return': {
                'value': metrics.annual_return,
                'target': self.targets['target_annual_return'],
                'min_threshold': self.targets['min_annual_return'],
                'max_threshold': self.targets['max_annual_return'],
                'passed': self.targets['min_annual_return'] <= metrics.annual_return <= self.targets['max_annual_return'],
                'score': min(metrics.annual_return / self.targets['target_annual_return'], 2.0)
            },
            'sharpe_ratio': {
                'value': metrics.sharpe_ratio,
                'target': self.targets['target_sharpe_ratio'],
                'min_threshold': self.targets['min_sharpe_ratio'],
                'passed': metrics.sharpe_ratio >= self.targets['min_sharpe_ratio'],
                'score': min(metrics.sharpe_ratio / self.targets['target_sharpe_ratio'], 2.0)
            },
            'max_drawdown': {
                'value': metrics.max_drawdown,
                'target': self.targets['max_drawdown'],
                'passed': metrics.max_drawdown <= self.targets['max_drawdown'],
                'score': max(0, 2.0 - (metrics.max_drawdown / self.targets['max_drawdown']))
            },
            'win_rate': {
                'value': metrics.win_rate,
                'target': self.targets['min_win_rate'],
                'passed': metrics.win_rate >= self.targets['min_win_rate'],
                'score': min(metrics.win_rate / self.targets['min_win_rate'], 2.0)
            },
            'trades_per_day': {
                'value': metrics.trades_per_day,
                'min_target': self.targets['min_trades_per_day'],
                'max_target': self.targets['max_trades_per_day'],
                'passed': self.targets['min_trades_per_day'] <= metrics.trades_per_day <= self.targets['max_trades_per_day'],
                'score': 1.0 if self.targets['min_trades_per_day'] <= metrics.trades_per_day <= self.targets['max_trades_per_day'] else 0.5
            },
            'profit_factor': {
                'value': metrics.profit_factor,
                'target': self.targets['min_profit_factor'],
                'passed': metrics.profit_factor >= self.targets['min_profit_factor'],
                'score': min(metrics.profit_factor / self.targets['min_profit_factor'], 2.0)
            },
            'volatility': {
                'value': metrics.volatility,
                'target': self.targets['max_volatility'],
                'passed': metrics.volatility <= self.targets['max_volatility'],
                'score': max(0, 2.0 - (metrics.volatility / self.targets['max_volatility']))
            },
            'calmar_ratio': {
                'value': metrics.calmar_ratio,
                'target': self.targets['min_calmar_ratio'],
                'passed': metrics.calmar_ratio >= self.targets['min_calmar_ratio'],
                'score': min(metrics.calmar_ratio / self.targets['min_calmar_ratio'], 2.0)
            },
            'sortino_ratio': {
                'value': metrics.sortino_ratio,
                'target': self.targets['min_sortino_ratio'],
                'passed': metrics.sortino_ratio >= self.targets['min_sortino_ratio'],
                'score': min(metrics.sortino_ratio / self.targets['min_sortino_ratio'], 2.0)
            }
        }
        
        # Calculate overall score
        total_score = sum(v['score'] for v in validation.values())
        validation['overall_score'] = total_score / len(validation)
        validation['critical_failures'] = sum(1 for v in validation.values() if isinstance(v, dict) and not v.get('passed', True))
        
        return validation
    
    def _generate_recommendations(self, metrics: PerformanceMetrics, validation: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not validation['annual_return']['passed']:
            if metrics.annual_return < self.targets['min_annual_return']:
                recommendations.append("Increase position sizing or improve signal quality to boost returns")
            else:
                recommendations.append("Returns are too high - consider reducing risk to ensure sustainability")
        
        if not validation['sharpe_ratio']['passed']:
            recommendations.append("Improve risk-adjusted returns by reducing volatility or enhancing signal accuracy")
        
        if not validation['max_drawdown']['passed']:
            recommendations.append("Implement stricter risk controls to reduce maximum drawdown")
        
        if not validation['win_rate']['passed']:
            recommendations.append("Improve model accuracy or adjust entry/exit criteria to increase win rate")
        
        if not validation['trades_per_day']['passed']:
            if metrics.trades_per_day < self.targets['min_trades_per_day']:
                recommendations.append("Increase trading frequency to meet capacity targets")
            else:
                recommendations.append("Reduce trading frequency to avoid overtrading")
        
        if metrics.max_consecutive_losses > 10:
            recommendations.append("Implement circuit breakers to limit consecutive losses")
        
        if metrics.avg_trade_duration > 4:  # More than 4 hours
            recommendations.append("Consider reducing average trade duration for day trading strategy")
        
        if validation['overall_score'] < 1.0:
            recommendations.append("Overall performance below targets - comprehensive strategy review needed")
        
        return recommendations
    
    def _generate_risk_warnings(self, metrics: PerformanceMetrics, validation: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on performance"""
        warnings = []
        
        if metrics.max_drawdown > 0.20:  # 20% drawdown
            warnings.append("CRITICAL: Maximum drawdown exceeds 20% - high risk of significant losses")
        
        if metrics.sharpe_ratio < 1.0:
            warnings.append("WARNING: Sharpe ratio below 1.0 indicates poor risk-adjusted performance")
        
        if metrics.max_consecutive_losses > 15:
            warnings.append("CRITICAL: Excessive consecutive losses detected - strategy may be broken")
        
        if metrics.volatility > 0.30:  # 30% annual volatility
            warnings.append("WARNING: High volatility detected - consider reducing position sizes")
        
        if metrics.profit_factor < 1.2:
            warnings.append("WARNING: Low profit factor indicates marginal profitability")
        
        if validation['critical_failures'] > 3:
            warnings.append("CRITICAL: Multiple performance targets failed - strategy not ready for live trading")
        
        if metrics.trades_per_day > 500:
            warnings.append("WARNING: Extremely high trading frequency may lead to execution issues")
        
        return warnings
    
    def _determine_validation_result(self, validation: Dict[str, Any]) -> bool:
        """Determine overall validation pass/fail"""
        # Critical requirements that must pass
        critical_checks = [
            validation['annual_return']['passed'],
            validation['sharpe_ratio']['passed'],
            validation['max_drawdown']['passed'],
            validation['win_rate']['passed']
        ]
        
        # Must pass all critical checks and have overall score > 0.8
        return all(critical_checks) and validation['overall_score'] > 0.8
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty metrics for error cases"""
        return PerformanceMetrics(
            annual_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            trades_per_day=0.0,
            avg_trade_duration=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            total_trades=0,
            profitable_trades=0,
            avg_profit_per_trade=0.0,
            max_consecutive_losses=0,
            recovery_factor=0.0
        )
    
    async def validate_live_readiness(self) -> ValidationResult:
        """Validate if system is ready for live trading"""
        # Validate last 3 months of performance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        result = await self.validate_system_performance(start_date, end_date)
        
        # Additional live trading checks
        if result.passed:
            # Check recent performance consistency
            recent_validation = await self._validate_recent_consistency()
            if not recent_validation:
                result.passed = False
                result.risk_warnings.append("CRITICAL: Recent performance inconsistency detected")
        
        return result
    
    async def validate_model_performance(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Validate current model performance for trading readiness"""
        try:
            logger.info(f"Validating model performance over last {lookback_days} days")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get model predictions and actual results
            predictions_data = await self._get_predictions_data(start_date, end_date)
            trades_data = await self._get_trades_data(start_date, end_date)
            
            if not predictions_data or not trades_data:
                logger.warning("Insufficient data for model validation")
                return {
                    'is_valid': False,
                    'reason': 'Insufficient data',
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'sharpe_ratio': 0.0,
                    'total_predictions': 0
                }
            
            # Calculate model accuracy metrics
            accuracy_metrics = self._calculate_model_accuracy(predictions_data, trades_data)
            
            # Determine if model is performing adequately
            is_valid = (
                accuracy_metrics['accuracy'] >= 0.55 and  # 55% minimum accuracy
                accuracy_metrics['sharpe_ratio'] >= 1.5 and  # 1.5 minimum Sharpe
                accuracy_metrics['total_predictions'] >= 50  # Minimum sample size
            )
            
            result = {
                'is_valid': is_valid,
                'accuracy': accuracy_metrics['accuracy'],
                'precision': accuracy_metrics['precision'],
                'recall': accuracy_metrics['recall'],
                'sharpe_ratio': accuracy_metrics['sharpe_ratio'],
                'total_predictions': accuracy_metrics['total_predictions'],
                'validation_period': f"{start_date.date()} to {end_date.date()}"
            }
            
            if not is_valid:
                result['reason'] = 'Model performance below minimum thresholds'
            
            logger.info(f"Model validation result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {str(e)}',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'sharpe_ratio': 0.0,
                'total_predictions': 0
            }
    
    async def _get_predictions_data(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get model predictions data for validation"""
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT 
                        symbol,
                        prediction_time,
                        predicted_return,
                        confidence,
                        model_type,
                        actual_return
                    FROM predictions
                    WHERE prediction_time >= :start_date
                    AND prediction_time <= :end_date
                    AND actual_return IS NOT NULL
                    ORDER BY prediction_time
                """), {
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                predictions = []
                for row in result.fetchall():
                    predictions.append({
                        'symbol': row.symbol,
                        'prediction_time': row.prediction_time,
                        'predicted_return': float(row.predicted_return),
                        'confidence': float(row.confidence),
                        'model_type': row.model_type,
                        'actual_return': float(row.actual_return)
                    })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Failed to get predictions data: {e}")
            return []
    
    def _calculate_model_accuracy(self, predictions_data: List[Dict], trades_data: List[Dict]) -> Dict[str, float]:
        """Calculate model accuracy metrics"""
        try:
            if not predictions_data:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'sharpe_ratio': 0.0,
                    'total_predictions': 0
                }
            
            predictions_df = pd.DataFrame(predictions_data)
            
            # Calculate directional accuracy
            predictions_df['predicted_direction'] = predictions_df['predicted_return'] > 0
            predictions_df['actual_direction'] = predictions_df['actual_return'] > 0
            predictions_df['correct_prediction'] = predictions_df['predicted_direction'] == predictions_df['actual_direction']
            
            # Basic metrics
            total_predictions = len(predictions_df)
            correct_predictions = predictions_df['correct_prediction'].sum()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Precision and Recall for positive predictions
            positive_predictions = predictions_df[predictions_df['predicted_direction'] == True]
            true_positives = positive_predictions['correct_prediction'].sum()
            false_positives = len(positive_predictions) - true_positives
            
            actual_positives = predictions_df[predictions_df['actual_direction'] == True]
            false_negatives = len(actual_positives) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate Sharpe ratio of predictions
            returns = predictions_df['actual_return'].values
            if len(returns) > 1:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'sharpe_ratio': sharpe_ratio,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate model accuracy: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'sharpe_ratio': 0.0,
                'total_predictions': 0
            }
    
    async def _validate_recent_consistency(self) -> bool:
        """Validate recent performance consistency"""
        try:
            # Check last 30 days for consistency
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            returns_data = await self._get_returns_data(start_date, end_date)
            
            if len(returns_data) < 20:  # Need at least 20 trading days
                return False
            
            returns_df = pd.DataFrame(returns_data)
            daily_returns = returns_df['daily_pnl'] / returns_df['portfolio_value'].shift(1)
            
            # Check for consistent positive performance
            positive_days = (daily_returns > 0).sum()
            total_days = len(daily_returns)
            
            # At least 60% positive days in recent period
            return (positive_days / total_days) >= 0.6
            
        except Exception as e:
            logger.error(f"Failed to validate recent consistency: {e}")
            return False
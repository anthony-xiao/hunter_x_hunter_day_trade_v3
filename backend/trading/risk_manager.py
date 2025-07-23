from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import asyncio
from enum import Enum

from .execution_engine import TradeSignal, Position
from data.data_pipeline import DataPipeline

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    portfolio_value: float
    total_exposure: float
    cash_balance: float
    leverage: float
    var_1d: float  # 1-day Value at Risk
    var_5d: float  # 5-day Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_spy: float
    concentration_risk: float
    sector_exposure: Dict[str, float]
    position_count: int
    avg_position_size: float
    largest_position_pct: float
    risk_level: RiskLevel
    timestamp: datetime

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    position_size: float
    market_value: float
    portfolio_weight: float
    volatility: float
    beta: float
    var_contribution: float
    correlation_portfolio: float
    risk_score: float
    max_loss_1d: float
    max_loss_5d: float

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_portfolio_leverage: float = 2.0
    max_position_size_pct: float = 0.05  # 5% of portfolio
    max_sector_exposure_pct: float = 0.30  # 30% per sector
    max_daily_loss_pct: float = 0.02  # 2% daily loss limit
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    min_cash_reserve_pct: float = 0.10  # 10% cash reserve
    max_correlation_threshold: float = 0.70
    max_var_pct: float = 0.05  # 5% VaR limit
    max_positions: int = 20
    min_liquidity_volume: float = 1000000  # $1M daily volume

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.risk_limits = RiskLimits()
        self.portfolio_history: List[RiskMetrics] = []
        self.position_risks: Dict[str, PositionRisk] = {}
        self.sector_mappings: Dict[str, str] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional[pd.Series] = None
        
        # Initialize sector mappings (simplified)
        self._initialize_sector_mappings()
        
        logger.info("Risk Manager initialized")
    
    def _initialize_sector_mappings(self):
        """Initialize sector mappings for stocks"""
        self.sector_mappings = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'AMD': 'Technology',
            'NFLX': 'Communication Services',
            'CRM': 'Technology',
            'UBER': 'Technology',
            'LYFT': 'Technology',
            'SHOP': 'Technology',
            'SQ': 'Technology',
            'PYPL': 'Technology',
            'ROKU': 'Communication Services',
            'ZM': 'Technology',
            'DOCU': 'Technology',
            'SNOW': 'Technology',
            'PLTR': 'Technology'
        }
    
    async def calculate_position_size(
        self,
        signal: TradeSignal,
        market_data: Optional[pd.DataFrame] = None,
        portfolio_value: float = 100000.0,
        current_positions: Optional[List[Position]] = None
    ) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            if market_data is None or len(market_data) < 20:
                logger.warning(f"Insufficient market data for {signal.symbol}")
                return 0.0
            
            # Calculate volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Base position size using Kelly Criterion (simplified)
            win_rate = signal.confidence
            avg_win = 0.02  # Assume 2% average win
            avg_loss = 0.015  # Assume 1.5% average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Adjust for volatility
            volatility_adjustment = min(1.0, 0.20 / volatility)  # Target 20% volatility
            
            # Base position size
            base_size = portfolio_value * kelly_fraction * volatility_adjustment
            
            # Apply risk limits
            max_position_value = portfolio_value * self.risk_limits.max_position_size_pct
            position_size = min(base_size, max_position_value)
            
            # Check sector concentration
            sector = self.sector_mappings.get(signal.symbol, 'Unknown')
            if current_positions:
                sector_exposure = self._calculate_sector_exposure(current_positions, sector)
                max_sector_value = portfolio_value * self.risk_limits.max_sector_exposure_pct
                
                if sector_exposure + position_size > max_sector_value:
                    position_size = max(0, max_sector_value - sector_exposure)
            
            # Check correlation risk
            if current_positions and len(current_positions) > 0:
                correlation_risk = await self._assess_correlation_risk(
                    signal.symbol, current_positions, market_data
                )
                
                if correlation_risk > self.risk_limits.max_correlation_threshold:
                    position_size *= 0.5  # Reduce size for high correlation
            
            # Ensure minimum liquidity
            if market_data is not None and len(market_data) > 0:
                avg_volume = market_data['volume'].tail(20).mean()
                avg_price = market_data['close'].tail(20).mean()
                daily_dollar_volume = avg_volume * avg_price
                
                if daily_dollar_volume < self.risk_limits.min_liquidity_volume:
                    position_size *= 0.5  # Reduce for low liquidity
            
            # Convert to number of shares
            current_price = market_data['close'].iloc[-1]
            shares = int(position_size / current_price)
            
            logger.info(
                f"Position sizing for {signal.symbol}: "
                f"${position_size:.2f} ({shares} shares) - "
                f"Volatility: {volatility:.2%}, Kelly: {kelly_fraction:.2%}"
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return 0.0
    
    def _calculate_sector_exposure(
        self,
        positions: List[Position],
        target_sector: str
    ) -> float:
        """Calculate current exposure to a specific sector"""
        sector_value = 0.0
        
        for position in positions:
            position_sector = self.sector_mappings.get(position.symbol, 'Unknown')
            if position_sector == target_sector:
                sector_value += position.market_value
        
        return sector_value
    
    async def _assess_correlation_risk(
        self,
        symbol: str,
        current_positions: List[Position],
        market_data: pd.DataFrame
    ) -> float:
        """Assess correlation risk with existing positions"""
        try:
            if len(current_positions) == 0:
                return 0.0
            
            # Get returns for the new symbol
            new_returns = market_data['close'].pct_change().dropna().tail(60)
            
            correlations = []
            
            for position in current_positions:
                # In a real implementation, you would fetch market data for each position
                # For now, we'll use a simplified correlation estimate
                sector_new = self.sector_mappings.get(symbol, 'Unknown')
                sector_existing = self.sector_mappings.get(position.symbol, 'Unknown')
                
                if sector_new == sector_existing:
                    correlations.append(0.7)  # High correlation within sector
                else:
                    correlations.append(0.3)  # Lower correlation across sectors
            
            return max(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return 0.5  # Conservative estimate
    
    async def assess_portfolio_risk(
        self,
        positions: List[Position],
        portfolio_value: float,
        market_data: Dict[str, pd.DataFrame]
    ) -> RiskMetrics:
        """Assess overall portfolio risk"""
        try:
            if not positions:
                return RiskMetrics(
                    portfolio_value=portfolio_value,
                    total_exposure=0.0,
                    cash_balance=portfolio_value,
                    leverage=0.0,
                    var_1d=0.0,
                    var_5d=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    beta=0.0,
                    correlation_spy=0.0,
                    concentration_risk=0.0,
                    sector_exposure={},
                    position_count=0,
                    avg_position_size=0.0,
                    largest_position_pct=0.0,
                    risk_level=RiskLevel.LOW,
                    timestamp=datetime.now()
                )
            
            # Calculate basic metrics
            total_exposure = sum(abs(pos.market_value) for pos in positions)
            cash_balance = portfolio_value - total_exposure
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate sector exposure
            sector_exposure = {}
            for position in positions:
                sector = self.sector_mappings.get(position.symbol, 'Unknown')
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position.market_value
            
            # Convert to percentages
            sector_exposure_pct = {
                sector: value / portfolio_value
                for sector, value in sector_exposure.items()
            }
            
            # Calculate concentration risk
            position_weights = [pos.market_value / portfolio_value for pos in positions]
            concentration_risk = max(position_weights) if position_weights else 0
            
            # Calculate VaR (simplified)
            portfolio_returns = []
            for position in positions:
                if position.symbol in market_data:
                    returns = market_data[position.symbol]['close'].pct_change().dropna().tail(60)
                    weight = position.market_value / portfolio_value
                    portfolio_returns.append(returns * weight)
            
            if portfolio_returns:
                combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
                var_1d = np.percentile(combined_returns, 5) * portfolio_value
                var_5d = var_1d * np.sqrt(5)
            else:
                var_1d = var_5d = 0.0
            
            # Calculate other metrics (simplified)
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            beta = self._calculate_portfolio_beta(positions, market_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                leverage, concentration_risk, abs(var_1d) / portfolio_value
            )
            
            metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                cash_balance=cash_balance,
                leverage=leverage,
                var_1d=var_1d,
                var_5d=var_5d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_spy=0.0,  # Would need SPY data
                concentration_risk=concentration_risk,
                sector_exposure=sector_exposure_pct,
                position_count=len(positions),
                avg_position_size=total_exposure / len(positions) if positions else 0,
                largest_position_pct=concentration_risk,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.portfolio_history.append(metrics)
            
            # Keep only last 100 records
            if len(self.portfolio_history) > 100:
                self.portfolio_history = self.portfolio_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            raise
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        values = [metric.portfolio_value for metric in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from portfolio history"""
        if len(self.portfolio_history) < 10:
            return 0.0
        
        values = [metric.portfolio_value for metric in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Assume 2% risk-free rate
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_portfolio_beta(
        self,
        positions: List[Position],
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Calculate portfolio beta (simplified)"""
        # In a real implementation, you would calculate beta against a benchmark
        # For now, return a simplified estimate
        tech_weight = 0.0
        total_weight = 0.0
        
        for position in positions:
            sector = self.sector_mappings.get(position.symbol, 'Unknown')
            weight = abs(position.market_value)
            total_weight += weight
            
            if sector == 'Technology':
                tech_weight += weight
        
        # Tech stocks typically have higher beta
        if total_weight > 0:
            tech_ratio = tech_weight / total_weight
            return 1.0 + (tech_ratio * 0.5)  # Simplified beta calculation
        
        return 1.0
    
    def _determine_risk_level(
        self,
        leverage: float,
        concentration: float,
        var_pct: float
    ) -> RiskLevel:
        """Determine overall risk level"""
        risk_score = 0
        
        # Leverage risk
        if leverage > 1.5:
            risk_score += 2
        elif leverage > 1.0:
            risk_score += 1
        
        # Concentration risk
        if concentration > 0.20:
            risk_score += 2
        elif concentration > 0.10:
            risk_score += 1
        
        # VaR risk
        if var_pct > 0.05:
            risk_score += 2
        elif var_pct > 0.03:
            risk_score += 1
        
        if risk_score >= 4:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def check_risk_limits(
        self,
        signal: TradeSignal,
        positions: List[Position],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Check if a trade violates risk limits"""
        try:
            # Check maximum positions
            if len(positions) >= self.risk_limits.max_positions:
                return False, f"Maximum positions limit reached ({self.risk_limits.max_positions})"
            
            # Check if we already have a position in this symbol
            existing_position = next(
                (pos for pos in positions if pos.symbol == signal.symbol),
                None
            )
            
            if existing_position and signal.action in ['BUY', 'SELL']:
                return False, f"Already have position in {signal.symbol}"
            
            # Check daily loss limit
            daily_pnl = sum(pos.unrealized_pnl for pos in positions)
            max_daily_loss = portfolio_value * self.risk_limits.max_daily_loss_pct
            
            if daily_pnl < -max_daily_loss:
                return False, f"Daily loss limit exceeded: ${daily_pnl:.2f}"
            
            # Check cash reserve
            total_exposure = sum(abs(pos.market_value) for pos in positions)
            min_cash = portfolio_value * self.risk_limits.min_cash_reserve_pct
            available_cash = portfolio_value - total_exposure
            
            if available_cash < min_cash:
                return False, f"Insufficient cash reserve: ${available_cash:.2f} < ${min_cash:.2f}"
            
            # Check sector concentration
            sector = self.sector_mappings.get(signal.symbol, 'Unknown')
            sector_exposure = self._calculate_sector_exposure(positions, sector)
            max_sector_exposure = portfolio_value * self.risk_limits.max_sector_exposure_pct
            
            if sector_exposure > max_sector_exposure:
                return False, f"Sector exposure limit exceeded for {sector}"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False, f"Risk check error: {str(e)}"
    
    async def get_portfolio_risk_metrics(self) -> Dict:
        """Get current portfolio risk metrics"""
        try:
            if not self.portfolio_history:
                return {
                    "status": "no_data",
                    "message": "No portfolio history available"
                }
            
            latest_metrics = self.portfolio_history[-1]
            
            return {
                "portfolio_value": latest_metrics.portfolio_value,
                "total_exposure": latest_metrics.total_exposure,
                "cash_balance": latest_metrics.cash_balance,
                "leverage": latest_metrics.leverage,
                "var_1d": latest_metrics.var_1d,
                "var_5d": latest_metrics.var_5d,
                "max_drawdown": latest_metrics.max_drawdown,
                "sharpe_ratio": latest_metrics.sharpe_ratio,
                "beta": latest_metrics.beta,
                "concentration_risk": latest_metrics.concentration_risk,
                "sector_exposure": latest_metrics.sector_exposure,
                "position_count": latest_metrics.position_count,
                "risk_level": latest_metrics.risk_level.value,
                "risk_limits": {
                    "max_leverage": self.risk_limits.max_portfolio_leverage,
                    "max_position_size_pct": self.risk_limits.max_position_size_pct,
                    "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                    "max_drawdown_pct": self.risk_limits.max_drawdown_pct,
                    "max_positions": self.risk_limits.max_positions
                },
                "timestamp": latest_metrics.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio risk metrics: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def update_risk_limits(self, new_limits: Dict):
        """Update risk management limits"""
        try:
            for key, value in new_limits.items():
                if hasattr(self.risk_limits, key):
                    setattr(self.risk_limits, key, value)
                    logger.info(f"Updated risk limit {key} to {value}")
                else:
                    logger.warning(f"Unknown risk limit parameter: {key}")
        
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
    
    def get_position_risk_analysis(self, symbol: str) -> Optional[PositionRisk]:
        """Get risk analysis for a specific position"""
        return self.position_risks.get(symbol)
    
    async def emergency_risk_check(self, positions: List[Position], portfolio_value: float) -> bool:
        """Emergency risk check - returns True if emergency action needed"""
        try:
            # Check for extreme losses
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            loss_pct = abs(total_pnl) / portfolio_value if portfolio_value > 0 else 0
            
            if loss_pct > self.risk_limits.max_daily_loss_pct * 2:  # 2x daily limit
                logger.critical(f"Emergency: Portfolio loss {loss_pct:.2%} exceeds emergency threshold")
                return True
            
            # Check for extreme leverage
            total_exposure = sum(abs(pos.market_value) for pos in positions)
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            if leverage > self.risk_limits.max_portfolio_leverage * 1.5:  # 1.5x leverage limit
                logger.critical(f"Emergency: Portfolio leverage {leverage:.2f} exceeds emergency threshold")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in emergency risk check: {e}")
            return True  # Conservative: trigger emergency if check fails
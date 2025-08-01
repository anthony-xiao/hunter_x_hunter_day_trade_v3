import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
import json
from pathlib import Path
import pytz

# Alpaca API (for trading only)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest, 
    StopLimitOrderRequest, TrailingStopOrderRequest,
    TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.common.exceptions import APIError

# Polygon API (for market data)
from polygon import RESTClient

from config import settings
from ml.performance_validator import PerformanceValidator
from ml.concept_drift_detector import ConceptDriftDetector
from data.polygon_websocket import websocket_manager, start_real_time_data, get_current_price as ws_get_current_price

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    NEW = "new"
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # "long" or "short"
    cost_basis: float
    market_price: float
    pnl_percentage: float
    timestamp: datetime
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Order:
    id: str
    symbol: str
    quantity: float
    side: str
    order_type: OrderType
    status: OrderStatus
    filled_price: Optional[float]
    filled_quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    trail_amount: Optional[float]
    timestamp: datetime
    updated_at: datetime
    commission: float = 0.0

@dataclass
class TradeSignal:
    symbol: str
    action: str  # "buy", "sell", "hold", "close"
    confidence: float
    predicted_return: float
    risk_score: float
    quantity: Optional[float] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None
    model_predictions: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.model_predictions is None:
            self.model_predictions = {}

@dataclass
class RiskMetrics:
    portfolio_value: float
    cash_available: float
    buying_power: float
    total_exposure: float
    leverage_ratio: float
    daily_pnl: float
    daily_pnl_percentage: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    beta: float
    correlation_risk: float
    concentration_risk: float
    timestamp: datetime

@dataclass
class PositionSizing:
    base_size: float
    volatility_adjusted: float
    confidence_adjusted: float
    risk_adjusted: float
    final_size: float
    max_allowed: float
    reasoning: str

class ExecutionEngine:
    def __init__(self):
        # Get API keys based on trading mode
        if settings.trading_mode == "paper":
            api_key = settings.alpaca_paper_api_key
            secret_key = settings.alpaca_paper_secret_key
        else:
            api_key = settings.alpaca_live_api_key
            secret_key = settings.alpaca_live_secret_key
        
        # Initialize Alpaca client for trading only
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=settings.trading_mode == "paper"
        )
        
        # Initialize Polygon client for market data
        self.polygon_client = RESTClient(settings.polygon_api_key)
        
        # Initialize WebSocket for real-time data
        self.websocket_manager = websocket_manager
        self.websocket_active = False
        
        # Initialize ML validation components
        from database import db_manager
        db_url = f"postgresql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"
        self.performance_validator = PerformanceValidator(db_url)
        self.drift_detector = ConceptDriftDetector(db_url)
        
        self.is_trading = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        
        # System validation state
        self.system_validated = False
        self.last_validation_time = None
        self.last_drift_check = None
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.total_trades = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Risk management parameters (as per requirements)
        self.max_position_size = 0.02      # 2% of portfolio per position
        self.max_daily_loss = 0.03         # 3% daily loss limit
        self.max_portfolio_risk = 0.10     # 10% total portfolio risk
        self.max_correlation = 0.7         # Maximum correlation between positions
        self.max_sector_exposure = 0.25    # 25% max exposure per sector
        self.min_liquidity = 1000000       # Minimum $1M daily volume
        
        # Position sizing parameters
        self.base_position_size = 0.015    # 1.5% base position
        self.volatility_target = 0.15      # 15% volatility target
        self.confidence_multiplier = 2.0   # Confidence scaling factor
        
        # Stop loss and take profit
        self.default_stop_loss = 0.02      # 2% stop loss
        self.default_take_profit = 0.04    # 4% take profit (2:1 ratio)
        self.trailing_stop_distance = 0.015  # 1.5% trailing stop
        
        # Market data cache
        self.price_cache: Dict[str, Dict] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        # Create directories
        Path('logs/trades').mkdir(parents=True, exist_ok=True)
        Path('logs/risk').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExecutionEngine initialized in {settings.trading_mode} mode")
    
    async def start_trading(self) -> bool:
        """Start the trading engine with comprehensive checks"""
        try:
            # Verify account access and status
            account = self.trading_client.get_account()
            
            if account.trading_blocked:
                logger.error("Trading is blocked on this account")
                return False
            
            if account.account_blocked:
                logger.error("Account is blocked")
                return False
            
            if float(account.equity) < 25000 and settings.trading_mode == "live":
                logger.warning(f"Account equity ${float(account.equity):,.2f} below PDT minimum")
            
            # Check market hours
            clock = self.trading_client.get_clock()
            if not clock.is_open and settings.trading_mode == "live":
                logger.warning("Market is currently closed")
            
            self.is_trading = True
            
            # Load existing positions and orders
            await self._load_positions()
            await self._load_orders()
            
            # Initialize risk monitoring
            await self._initialize_risk_monitoring()
            
            # Initialize ML validation systems
            await self._initialize_ml_validation()
            
            # Log startup information
            account_info = self.get_account_info()
            logger.info(f"Trading started successfully")
            logger.info(f"Account equity: ${account_info['equity']:,.2f}")
            logger.info(f"Buying power: ${account_info['buying_power']:,.2f}")
            logger.info(f"Positions: {len(self.positions)}")
            logger.info(f"Open orders: {len(self.orders)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return False
    
    async def stop_trading(self) -> bool:
        """Stop trading with proper cleanup"""
        try:
            self.is_trading = False
            
            # Cancel all open orders
            cancelled_orders = await self._cancel_all_orders()
            
            # Close all positions if requested (optional)
            # await self._close_all_positions()
            
            # Save trading session data
            await self._save_session_data()
            
            logger.info(f"Trading stopped. Cancelled {cancelled_orders} orders")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            return False
    
    async def execute_signal(self, signal: TradeSignal) -> Optional[Order]:
        """Execute trading signal with comprehensive risk management"""
        if not self.is_trading:
            logger.warning("Trading engine is not active")
            return None
        
        try:
            logger.info(f"Processing signal: {signal.symbol} {signal.action} confidence={signal.confidence:.3f}")
            
            # Check system validation status (skip in paper mode for testing)
            if settings.trading_mode == 'live':
                if not await self._check_system_validation():
                    logger.warning("System validation failed, skipping signal execution")
                    return None
            else:
                logger.info("Skipping system validation in paper trading mode for testing")
            
            # Pre-execution validation
            if not await self._validate_signal(signal):
                return None
            
            # Risk assessment
            risk_level = await self._assess_signal_risk(signal)
            if risk_level == RiskLevel.CRITICAL:
                logger.warning(f"Critical risk level for {signal.symbol}, skipping")
                return None
            
            # Position sizing
            sizing = await self._calculate_position_sizing(signal)
            if sizing.final_size <= 0:
                logger.warning(f"Invalid position size for {signal.symbol}")
                return None
            
            # Check for end-of-day position prevention (only for buy signals)
            if signal.action == "buy" and self.should_prevent_new_positions():
                logger.warning(f"Preventing new buy position for {signal.symbol} - market closes within 15 minutes")
                return None
            
            # Execute based on action
            order = None
            if signal.action == "buy":
                order = await self._execute_buy_signal(signal, sizing)
            elif signal.action == "sell":
                order = await self._execute_sell_signal(signal, sizing)
            elif signal.action == "close":
                order = await self._execute_close_signal(signal)
            else:
                logger.warning(f"Unknown action: {signal.action}")
                return None
            
            if order:
                self.orders[order.id] = order
                self.daily_trades += 1
                self.total_trades += 1
                
                # Log trade
                await self._log_trade(signal, order, sizing)
                
                logger.info(f"Order executed: {order.symbol} {order.side} {order.quantity} @ {order.limit_price or 'market'}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute signal for {signal.symbol}: {e}")
            return None
    
    async def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate signal before execution"""
        try:
            # Basic validation
            if not signal.symbol or signal.confidence <= 0:
                return False
            
            # Check if symbol is tradeable
            try:
                asset = self.trading_client.get_asset(signal.symbol)
                if not asset.tradable or not asset.fractionable:
                    logger.warning(f"{signal.symbol} is not tradeable")
                    return False
            except:
                logger.warning(f"Could not verify asset {signal.symbol}")
                return False
            
            # Check liquidity
            if not await self._check_liquidity(signal.symbol):
                return False
            
            # Check market hours for live trading
            if settings.trading_mode == "live":
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    logger.warning(f"Market closed, skipping {signal.symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    async def _assess_signal_risk(self, signal: TradeSignal) -> RiskLevel:
        """Assess risk level of signal"""
        try:
            risk_score = 0
            
            # Confidence risk
            if signal.confidence < 0.6:
                risk_score += 2
            elif signal.confidence < 0.7:
                risk_score += 1
            
            # Volatility risk
            volatility = await self._get_volatility(signal.symbol)
            if volatility > 0.4:  # 40% annualized volatility
                risk_score += 3
            elif volatility > 0.3:
                risk_score += 2
            elif volatility > 0.2:
                risk_score += 1
            
            # Correlation risk
            correlation = await self._check_correlation_risk(signal.symbol)
            if correlation > 0.8:
                risk_score += 2
            elif correlation > 0.7:
                risk_score += 1
            
            # Portfolio concentration risk
            if len(self.positions) > 0:
                max_position_pct = max(abs(pos.market_value) for pos in self.positions.values()) / self._get_portfolio_value()
                if max_position_pct > 0.15:  # 15% concentration
                    risk_score += 2
            
            # Map score to risk level
            if risk_score >= 6:
                return RiskLevel.CRITICAL
            elif risk_score >= 4:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return RiskLevel.HIGH
    
    async def _calculate_position_sizing(self, signal: TradeSignal) -> PositionSizing:
        """Calculate optimal position size using multiple factors"""
        try:
            portfolio_value = self._get_portfolio_value()
            
            # Base size (1.5% of portfolio)
            base_size = portfolio_value * self.base_position_size
            
            # Volatility adjustment
            volatility = await self._get_volatility(signal.symbol)
            vol_adjustment = self.volatility_target / max(volatility, 0.05)  # Avoid division by zero
            volatility_adjusted = base_size * min(vol_adjustment, 2.0)  # Cap at 2x
            
            # Confidence adjustment
            confidence_adjusted = volatility_adjusted * (signal.confidence * self.confidence_multiplier)
            
            # Risk adjustment based on signal risk score
            risk_multiplier = 1.0
            if hasattr(signal, 'risk_score'):
                risk_multiplier = max(0.5, 1.0 - signal.risk_score)
            
            risk_adjusted = confidence_adjusted * risk_multiplier
            
            # Apply maximum position size limit
            max_allowed = portfolio_value * self.max_position_size
            final_size = min(risk_adjusted, max_allowed)
            
            # Convert to shares
            current_price = await self._get_current_price(signal.symbol)
            if current_price <= 0:
                raise ValueError(f"Invalid price for {signal.symbol}")
            
            shares = final_size / current_price
            final_shares = max(1, int(shares))  # At least 1 share
            
            reasoning = f"Base: ${base_size:.0f}, Vol adj: {vol_adjustment:.2f}, Conf adj: {signal.confidence:.2f}, Risk adj: {risk_multiplier:.2f}"
            
            return PositionSizing(
                base_size=base_size,
                volatility_adjusted=volatility_adjusted,
                confidence_adjusted=confidence_adjusted,
                risk_adjusted=risk_adjusted,
                final_size=final_shares,
                max_allowed=max_allowed / current_price,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return PositionSizing(0, 0, 0, 0, 0, 0, f"Error: {e}")
    
    async def _execute_buy_signal(self, signal: TradeSignal, sizing: PositionSizing) -> Optional[Order]:
        """Execute buy signal with bracket order (stop loss and take profit)"""
        try:
            current_price = await self._get_current_price(signal.symbol)
            quantity = int(sizing.final_size)
            
            # Calculate stop loss and take profit with proper rounding
            stop_loss = signal.stop_loss or (current_price * (1 - self.default_stop_loss))
            take_profit = signal.take_profit or (current_price * (1 + self.default_take_profit))
            
            # Round prices to nearest penny
            stop_loss = self._round_price(stop_loss)
            take_profit = self._round_price(take_profit)
            
            # Create bracket order with stop loss and take profit
            stop_loss_request = StopLossRequest(stop_price=stop_loss)
            take_profit_request = TakeProfitRequest(limit_price=take_profit)
            
            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=stop_loss_request,
                take_profit=take_profit_request
            )
            
            alpaca_order = self.trading_client.submit_order(request)
            
            # Create order object
            order = Order(
                id=str(alpaca_order.id),
                symbol=signal.symbol,
                quantity=quantity,
                side="buy",
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                filled_price=None,
                filled_quantity=0,
                limit_price=None,
                stop_price=stop_loss,
                trail_amount=None,
                timestamp=datetime.now(),
                updated_at=datetime.now()
            )
            
            logger.info(f"Bracket buy order submitted for {signal.symbol}: qty={quantity}, stop_loss=${stop_loss:.2f}, take_profit=${take_profit:.2f}")
            
            return order
            
        except APIError as e:
            logger.error(f"Alpaca API error placing bracket buy order for {signal.symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error executing buy signal: {e}")
            return None
    
    async def _execute_sell_signal(self, signal: TradeSignal, sizing: PositionSizing) -> Optional[Order]:
        """Execute sell signal (short position) with bracket order"""
        try:
            current_price = await self._get_current_price(signal.symbol)
            quantity = int(sizing.final_size)
            
            # Check if shorting is allowed
            try:
                asset = self.trading_client.get_asset(signal.symbol)
                if not asset.shortable:
                    logger.warning(f"{signal.symbol} is not shortable")
                    return None
            except:
                logger.warning(f"Could not verify shortability of {signal.symbol}")
                return None
            
            # Calculate stop loss and take profit for short with proper rounding
            stop_loss = signal.stop_loss or (current_price * (1 + self.default_stop_loss))
            take_profit = signal.take_profit or (current_price * (1 - self.default_take_profit))
            
            # Round prices to nearest penny
            stop_loss = self._round_price(stop_loss)
            take_profit = self._round_price(take_profit)
            
            # Create bracket order with stop loss and take profit for short position
            stop_loss_request = StopLossRequest(stop_price=stop_loss)
            take_profit_request = TakeProfitRequest(limit_price=take_profit)
            
            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=stop_loss_request,
                take_profit=take_profit_request
            )
            
            alpaca_order = self.trading_client.submit_order(request)
            
            order = Order(
                id=str(alpaca_order.id),
                symbol=signal.symbol,
                quantity=quantity,
                side="sell",
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                filled_price=None,
                filled_quantity=0,
                limit_price=None,
                stop_price=stop_loss,
                trail_amount=None,
                timestamp=datetime.now(),
                updated_at=datetime.now()
            )
            
            logger.info(f"Bracket sell order submitted for {signal.symbol}: qty={quantity}, stop_loss=${stop_loss:.2f}, take_profit=${take_profit:.2f}")
            
            return order
            
        except APIError as e:
            logger.error(f"Alpaca API error placing bracket sell order for {signal.symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error executing sell signal: {e}")
            return None
    
    async def _execute_close_signal(self, signal: TradeSignal) -> Optional[Order]:
        """Close existing position"""
        try:
            if signal.symbol not in self.positions:
                logger.warning(f"No position to close for {signal.symbol}")
                return None
            
            position = self.positions[signal.symbol]
            
            # Determine order side (opposite of position)
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            quantity = abs(position.quantity)
            
            request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            alpaca_order = self.trading_client.submit_order(request)
            
            order = Order(
                id=str(alpaca_order.id),
                symbol=signal.symbol,
                quantity=quantity,
                side="sell" if side == OrderSide.SELL else "buy",
                order_type=OrderType.MARKET,
                status=OrderStatus.PENDING,
                filled_price=None,
                filled_quantity=0,
                limit_price=None,
                stop_price=None,
                trail_amount=None,
                timestamp=datetime.now(),
                updated_at=datetime.now()
            )
            
            return order
            
        except APIError as e:
            logger.error(f"Alpaca API error closing position for {signal.symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error executing close signal: {e}")
            return None
    
    # NOTE: These methods are no longer used as we now use bracket orders
    # to avoid wash trade detection issues
    
    # async def _submit_stop_loss_order(self, symbol: str, quantity: float, stop_price: float) -> None:
    #     """Submit stop loss order - DEPRECATED: Use bracket orders instead"""
    #     try:
    #         side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
    #         
    #         # Round stop price to nearest penny
    #         rounded_stop_price = self._round_price(stop_price)
    #         
    #         request = StopOrderRequest(
    #             symbol=symbol,
    #             qty=abs(quantity),
    #             side=side,
    #             stop_price=rounded_stop_price,
    #             time_in_force=TimeInForce.GTC  # Good till cancelled
    #         )
    #         
    #         self.trading_client.submit_order(request)
    #         logger.info(f"Stop loss order submitted for {symbol} at ${stop_price:.2f}")
    #         
    #     except Exception as e:
    #         logger.error(f"Error submitting stop loss for {symbol}: {e}")
    # 
    # async def _submit_take_profit_order(self, symbol: str, quantity: float, limit_price: float) -> None:
    #     """Submit take profit order - DEPRECATED: Use bracket orders instead"""
    #     try:
    #         side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
    #         
    #         # Round limit price to nearest penny
    #         rounded_limit_price = self._round_price(limit_price)
    #         
    #         request = LimitOrderRequest(
    #             symbol=symbol,
    #             qty=abs(quantity),
    #             side=side,
    #             limit_price=rounded_limit_price,
    #             time_in_force=TimeInForce.GTC
    #         )
    #         
    #         self.trading_client.submit_order(request)
    #         logger.info(f"Take profit order submitted for {symbol} at ${limit_price:.2f}")
    #         
    #     except Exception as e:
    #         logger.error(f"Error submitting take profit for {symbol}: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol using WebSocket (primary) or REST API (fallback)"""
        try:
            # First try to get price from WebSocket if active
            if self.websocket_active:
                ws_price = ws_get_current_price(symbol)
                if ws_price is not None:
                    logger.debug(f"Got price for {symbol} from WebSocket: ${ws_price:.2f}")
                    return ws_price
                logger.debug(f"No WebSocket price available for {symbol}, falling back to REST API")
            
            # Fallback to REST API with caching
            if symbol in self.price_cache:
                cache_time = self.price_cache[symbol].get('timestamp', datetime.min)
                if datetime.now() - cache_time < timedelta(seconds=30):  # 30-second cache
                    return self.price_cache[symbol]['price']
            
            # Get latest quote from Polygon REST API
            quote = self.polygon_client.get_last_quote(symbol)
            
            if quote and hasattr(quote, 'bid') and hasattr(quote, 'ask'):
                price = float(quote.bid + quote.ask) / 2
                
                # Update cache
                self.price_cache[symbol] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                logger.debug(f"Got price for {symbol} from Polygon REST quote: ${price:.2f}")
                return price
            else:
                # Fallback to last trade if quote unavailable
                trade = self.polygon_client.get_last_trade(symbol)
                if trade and hasattr(trade, 'price'):
                    price = float(trade.price)
                    
                    # Update cache
                    self.price_cache[symbol] = {
                        'price': price,
                        'timestamp': datetime.now()
                    }
                    
                    logger.debug(f"Got price for {symbol} from Polygon REST trade: ${price:.2f}")
                    return price
                else:
                    raise ValueError(f"No price data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    async def _get_volatility(self, symbol: str) -> float:
        """Calculate 20-day volatility for symbol with fallback data sources"""
        try:
            # Check cache
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            # Try primary data source (Alpaca/Polygon)
            volatility = await self._get_volatility_primary(symbol)
            if volatility is not None:
                self.volatility_cache[symbol] = volatility
                return volatility
            
            # Fallback to database historical data
            volatility = await self._get_volatility_from_db(symbol)
            if volatility is not None:
                self.volatility_cache[symbol] = volatility
                return volatility
            
            # Fallback to symbol-based estimates
            volatility = self._get_volatility_estimate(symbol)
            self.volatility_cache[symbol] = volatility
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return self._get_volatility_estimate(symbol)
    
    async def _get_volatility_primary(self, symbol: str) -> Optional[float]:
        """Get volatility from primary data source (Polygon)"""
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            # Get daily bars from Polygon
            bars = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            if bars and hasattr(bars, 'results') and bars.results and len(bars.results) >= 10:
                prices = [float(bar.close) for bar in bars.results]
                returns = np.diff(np.log(prices))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                
                logger.info(f"Calculated volatility for {symbol} from Polygon: {volatility:.4f}")
                return volatility
            
            return None
            
        except Exception as e:
            logger.error(f"Polygon volatility calculation failed for {symbol}: {e}")
            return None
    
    async def _get_volatility_from_db(self, symbol: str) -> Optional[float]:
        """Get volatility from stored database data"""
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import sessionmaker
            
            engine = create_engine(f"postgresql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}")
            Session = sessionmaker(bind=engine)
            
            with Session() as session:
                # Get last 30 days of market data
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=30)
                
                result = session.execute(text("""
                    SELECT close, timestamp
                    FROM market_data
                    WHERE symbol = :symbol
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
                    ORDER BY timestamp
                """), {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                prices = [float(row.close) for row in result.fetchall()]
                
                if len(prices) >= 10:
                    returns = np.diff(np.log(prices))
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    
                    logger.info(f"Calculated volatility for {symbol} from database: {volatility:.4f}")
                    return volatility
            
            return None
            
        except Exception as e:
            logger.error(f"Database volatility calculation failed for {symbol}: {e}")
            return None
    
    def _get_volatility_estimate(self, symbol: str) -> float:
        """Get estimated volatility based on symbol characteristics"""
        try:
            # Symbol-based volatility estimates (based on historical patterns)
            volatility_estimates = {
                # Technology stocks (higher volatility)
                'AAPL': 0.28, 'MSFT': 0.25, 'GOOGL': 0.30, 'AMZN': 0.35, 'NVDA': 0.45,
                'TSLA': 0.55, 'META': 0.40, 'NFLX': 0.45, 'AMD': 0.50, 'INTC': 0.35,
                
                # Biotechnology (very high volatility)
                'MRNA': 0.65, 'GILD': 0.35, 'BIIB': 0.40, 'VRTX': 0.35, 'AMGN': 0.30,
                
                # Energy (moderate-high volatility)
                'XOM': 0.30, 'CVX': 0.28, 'SLB': 0.40, 'HAL': 0.45,
                
                # Financial (moderate volatility)
                'JPM': 0.25, 'BAC': 0.30, 'WFC': 0.35, 'GS': 0.30, 'MS': 0.35,
                
                # Consumer (lower volatility)
                'WMT': 0.20, 'PG': 0.18, 'KO': 0.16, 'PEP': 0.18, 'JNJ': 0.15
            }
            
            if symbol in volatility_estimates:
                volatility = volatility_estimates[symbol]
                logger.info(f"Using estimated volatility for {symbol}: {volatility:.4f}")
                return volatility
            
            # Default estimates based on symbol patterns
            if any(tech in symbol for tech in ['TECH', 'SOFT', 'DATA', 'CLOUD']):
                return 0.35  # Tech-related
            elif any(bio in symbol for bio in ['BIO', 'GENE', 'PHARM', 'DRUG']):
                return 0.45  # Biotech-related
            elif any(energy in symbol for energy in ['OIL', 'GAS', 'ENERGY']):
                return 0.35  # Energy-related
            else:
                return 0.25  # Default moderate volatility
                
        except Exception as e:
            logger.error(f"Error getting volatility estimate for {symbol}: {e}")
            return 0.25  # Conservative default
    
    async def _check_liquidity(self, symbol: str) -> bool:
        """Check if symbol meets minimum liquidity requirements using Polygon"""
        try:
            # In paper trading mode, be more permissive for testing
            if settings.trading_mode == "paper":
                logger.info(f"Skipping liquidity check for {symbol} in paper trading mode")
                return True
            
            # Get recent volume data from Polygon
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=5)
            
            bars = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )
            
            if bars and hasattr(bars, 'results') and bars.results and len(bars.results) > 0:
                avg_volume = np.mean([float(bar.volume) for bar in bars.results])
                avg_dollar_volume = avg_volume * np.mean([float(bar.close) for bar in bars.results])
                
                logger.info(f"Liquidity check for {symbol}: avg_dollar_volume=${avg_dollar_volume:,.0f}, min_required=${self.min_liquidity:,.0f}")
                return avg_dollar_volume >= self.min_liquidity
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking liquidity for {symbol}: {e}")
            # In paper trading mode, return True even on error for testing
            if settings.trading_mode == "paper":
                logger.info(f"Allowing {symbol} trade despite liquidity check error in paper mode")
                return True
            return False
    
    async def _check_correlation_risk(self, symbol: str) -> float:
        """Check correlation with existing positions"""
        try:
            if not self.positions:
                return 0.0
            
            # This is a simplified correlation check
            # In practice, you'd calculate actual correlations using historical data
            
            # For now, return a placeholder
            return 0.3  # Assume moderate correlation
            
        except Exception as e:
            logger.error(f"Error checking correlation for {symbol}: {e}")
            return 0.5
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 100000.0  # Fallback value
    
    async def _load_positions(self) -> None:
        """Load current positions from Alpaca"""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            
            self.positions.clear()
            
            for pos in alpaca_positions:
                current_price = await self._get_current_price(pos.symbol)
                
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pnl=float(pos.unrealized_pl),
                    realized_pnl=0.0,  # Would need to track separately
                    side="long" if float(pos.qty) > 0 else "short",
                    cost_basis=float(pos.cost_basis),
                    market_price=current_price,
                    pnl_percentage=float(pos.unrealized_plpc),
                    timestamp=datetime.now(),
                    entry_time=datetime.now()  # Would need to track separately
                )
                
                self.positions[pos.symbol] = position
            
            logger.info(f"Loaded {len(self.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def _load_orders(self) -> None:
        """Load current orders from Alpaca"""
        try:
            alpaca_orders = self.trading_client.get_orders()
            
            self.orders.clear()
            
            for alpaca_order in alpaca_orders:
                order = Order(
                    id=str(alpaca_order.id),
                    symbol=alpaca_order.symbol,
                    quantity=float(alpaca_order.qty),
                    side=alpaca_order.side.value,
                    order_type=OrderType(alpaca_order.order_type.value),
                    status=OrderStatus(alpaca_order.status.value),
                    filled_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price is not None else None,
                    filled_quantity=float(alpaca_order.filled_qty) if alpaca_order.filled_qty is not None else 0,
                    limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price is not None else None,
                    stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price is not None else None,
                    trail_amount=float(getattr(alpaca_order, 'trail_price', None) or getattr(alpaca_order, 'trail_percent', None)) if hasattr(alpaca_order, 'trail_price') or hasattr(alpaca_order, 'trail_percent') else None,
                    timestamp=alpaca_order.created_at,
                    updated_at=alpaca_order.updated_at
                )
                
                self.orders[order.id] = order
            
            logger.info(f"Loaded {len(self.orders)} orders")
            
        except Exception as e:
            logger.error(f"Error loading orders: {e}")
    
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring systems"""
        try:
            # Calculate initial risk metrics
            await self.update_risk_metrics()
            
            # Set up risk alerts
            # This would integrate with monitoring systems
            
            logger.info("Risk monitoring initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk monitoring: {e}")
    
    async def _initialize_ml_validation(self) -> None:
        """Initialize ML validation and drift detection systems"""
        try:
            # Validate model performance
            validation_result = await self.performance_validator.validate_model_performance()
            
            if validation_result.get('is_valid', False):
                self.system_validated = True
                self.last_validation_time = datetime.now()
                logger.info(f"Model validation passed: {validation_result}")
            else:
                logger.warning(f"Model validation failed: {validation_result}")
                self.system_validated = False
            
            # Initialize drift detection
            await self.drift_detector.initialize_baseline()
            self.last_drift_check = datetime.now()
            
            logger.info("ML validation systems initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML validation: {e}")
            self.system_validated = False
    
    async def start_websocket_data_stream(self, symbols: List[str]) -> bool:
        """Start WebSocket data streaming for given symbols"""
        try:
            success = await start_real_time_data(symbols)
            if success:
                self.websocket_active = True
                logger.info(f"WebSocket data streaming started for {len(symbols)} symbols")
            else:
                logger.warning("Failed to start WebSocket data streaming, will use REST API fallback")
            return success
        except Exception as e:
            logger.error(f"Error starting WebSocket data stream: {e}")
            return False
    
    async def stop_websocket_data_stream(self):
        """Stop WebSocket data streaming"""
        try:
            await self.websocket_manager.disconnect()
            self.websocket_active = False
            logger.info("WebSocket data streaming stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket data stream: {e}")
    
    async def _cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        try:
            cancelled_count = 0
            
            for order_id in list(self.orders.keys()):
                try:
                    self.trading_client.cancel_order_by_id(order_id)
                    cancelled_count += 1
                except:
                    pass  # Order might already be filled/cancelled
            
            self.orders.clear()
            logger.info(f"Cancelled {cancelled_count} orders")
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return 0
    
    async def _save_session_data(self) -> None:
        """Save trading session data"""
        try:
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'positions': {k: asdict(v) for k, v in self.positions.items()},
                'performance_metrics': {
                    'total_trades': self.total_trades,
                    'win_rate': self.win_rate,
                    'sharpe_ratio': self.sharpe_ratio,
                    'max_drawdown': self.max_drawdown
                }
            }
            
            filename = f"logs/trades/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"Session data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}")
    
    async def _log_trade(self, signal: TradeSignal, order: Order, sizing: PositionSizing) -> None:
        """Log trade details"""
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'signal': asdict(signal),
                'order': asdict(order),
                'sizing': asdict(sizing),
                'market_conditions': {
                    'portfolio_value': self._get_portfolio_value(),
                    'daily_pnl': self.daily_pnl,
                    'positions_count': len(self.positions)
                }
            }
            
            self.trade_history.append(trade_log)
            
            # Save to file
            filename = f"logs/trades/trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order.symbol}.json"
            with open(filename, 'w') as f:
                json.dump(trade_log, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    async def update_positions(self) -> None:
        """Update position information and P&L"""
        try:
            await self._load_positions()
            
            # Update daily P&L
            self.daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Update performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        try:
            if self.trade_history:
                # Calculate win rate
                profitable_trades = sum(1 for trade in self.trade_history 
                                      if trade.get('realized_pnl', 0) > 0)
                self.win_rate = profitable_trades / len(self.trade_history) if self.trade_history else 0
                
                # Calculate Sharpe ratio (simplified)
                returns = [trade.get('realized_pnl', 0) for trade in self.trade_history]
                if len(returns) > 1:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                
                # Calculate max drawdown
                cumulative_pnl = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
                self.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_system_validation(self) -> bool:
        """Check if system validation is current and models are performing well"""
        try:
            current_time = datetime.now()
            
            # Check if validation is recent (within last 24 hours)
            if (self.last_validation_time is None or 
                current_time - self.last_validation_time > timedelta(hours=24)):
                
                logger.info("Running periodic model validation")
                # Use a recent 30-day period for validation
                end_date = current_time
                start_date = end_date - timedelta(days=30)
                
                validation_result = await self.performance_validator.validate_system_performance(
                    start_date, end_date
                )
                
                self.system_validated = validation_result.passed
                self.last_validation_time = current_time
                
                if not self.system_validated:
                    logger.warning(f"Model validation failed: {validation_result.validation_details}")
                    return False
            
            # Check for concept drift (every 4 hours)
            if (self.last_drift_check is None or 
                current_time - self.last_drift_check > timedelta(hours=4)):
                
                logger.info("Checking for concept drift")
                drift_detected = await self.drift_detector.detect_drift()
                
                self.last_drift_check = current_time
                
                if drift_detected:
                    logger.warning("Concept drift detected - reducing trading activity")
                    # Could implement drift response strategies here
                    return False
            
            return self.system_validated
            
        except Exception as e:
            logger.error(f"Error checking system validation: {e}")
            return False
    
    async def update_risk_metrics(self) -> RiskMetrics:
        """Calculate and update risk metrics"""
        try:
            account = self.trading_client.get_account()
            portfolio_value = float(account.equity)
            
            # Calculate exposure
            total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
            leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate daily P&L percentage
            daily_pnl_pct = self.daily_pnl / portfolio_value if portfolio_value > 0 else 0
            
            # Simplified VaR calculation (95% confidence)
            var_95 = portfolio_value * 0.02  # 2% of portfolio value
            
            # Simplified beta (would need market data for accurate calculation)
            beta = 1.0
            
            # Concentration risk
            if self.positions:
                max_position_value = max(abs(pos.market_value) for pos in self.positions.values())
                concentration_risk = max_position_value / portfolio_value
            else:
                concentration_risk = 0.0
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                cash_available=float(account.cash),
                buying_power=float(account.buying_power),
                total_exposure=total_exposure,
                leverage_ratio=leverage_ratio,
                daily_pnl=self.daily_pnl,
                daily_pnl_percentage=daily_pnl_pct,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=self.sharpe_ratio,
                var_95=var_95,
                beta=beta,
                correlation_risk=0.3,  # Simplified
                concentration_risk=concentration_risk,
                timestamp=datetime.now()
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=0, cash_available=0, buying_power=0,
                total_exposure=0, leverage_ratio=0, daily_pnl=0,
                daily_pnl_percentage=0, max_drawdown=0, sharpe_ratio=0,
                var_95=0, beta=1, correlation_risk=0, concentration_risk=0,
                timestamp=datetime.now()
            )
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_orders(self) -> Dict[str, Order]:
        """Get current orders"""
        return self.orders.copy()
    
    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        try:
            account = self.trading_client.get_account()
            
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "daily_pnl": self.daily_pnl,
                "total_pnl": self.total_pnl,
                "daily_trades": self.daily_trades,
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "is_trading": self.is_trading,
                "positions_count": len(self.positions),
                "orders_count": len(self.orders),
                "account_status": {
                    "trading_blocked": account.trading_blocked,
                    "account_blocked": account.account_blocked,
                    "pattern_day_trader": account.pattern_day_trader
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_trading_status(self) -> Dict:
        """Get comprehensive trading engine status"""
        return {
            "is_trading": self.is_trading,
            "positions": len(self.positions),
            "orders": len(self.orders),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "performance": {
                "win_rate": self.win_rate,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown
            },
            "risk_limits": {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "max_portfolio_risk": self.max_portfolio_risk,
                "max_correlation": self.max_correlation,
                "max_sector_exposure": self.max_sector_exposure
            },
            "trading_mode": settings.trading_mode
        }
    
    async def emergency_stop(self) -> bool:
        """Emergency stop - close all positions immediately"""
        try:
            logger.warning("EMERGENCY STOP - Closing all positions")
            
            # Close all positions
            positions = self.trading_client.get_all_positions()
            for position in positions:
                try:
                    self.trading_client.close_position(position.symbol)
                    logger.info(f"Closed position: {position.symbol}")
                except APIError as e:
                    logger.error(f"Failed to close position {position.symbol}: {e}")
            
            # Cancel all open orders
            await self._cancel_all_orders()
            
            self.is_trading = False
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    async def close_all_positions_eod(self) -> bool:
        """End-of-day liquidation - close all positions for daily operations"""
        try:
            logger.info("END-OF-DAY LIQUIDATION - Closing all positions before market close")
            
            # Get all current positions
            positions = self.trading_client.get_all_positions()
            
            if not positions:
                logger.info("No positions to close for end-of-day liquidation")
                return True
            
            closed_positions = 0
            failed_positions = 0
            
            # Close all positions
            for position in positions:
                try:
                    self.trading_client.close_position(position.symbol)
                    logger.info(f"EOD: Closed position {position.symbol} (qty: {position.qty}, value: ${float(position.market_value):.2f})")
                    closed_positions += 1
                except APIError as e:
                    logger.error(f"EOD: Failed to close position {position.symbol}: {e}")
                    failed_positions += 1
            
            # Cancel all open orders to prevent new positions
            await self._cancel_all_orders()
            logger.info("EOD: Cancelled all open orders")
            
            # Log summary
            logger.info(f"EOD LIQUIDATION COMPLETE - Closed: {closed_positions}, Failed: {failed_positions}")
            
            return failed_positions == 0
            
        except Exception as e:
            logger.error(f"End-of-day liquidation failed: {e}")
            return False
    
    def get_market_clock(self) -> Optional[Dict[str, Any]]:
        """Get market clock from Alpaca and convert timestamps to UTC"""
        try:
            # Get market clock from Alpaca (returns Eastern Time)
            clock = self.trading_client.get_clock()
            
            # Convert Eastern Time to UTC
            eastern = pytz.timezone('US/Eastern')
            utc = pytz.UTC
            
            # Convert timestamps to UTC
            market_clock = {
                'timestamp': datetime.now(utc),
                'is_open': clock.is_open,
                'next_open': eastern.localize(clock.next_open.replace(tzinfo=None)).astimezone(utc),
                'next_close': eastern.localize(clock.next_close.replace(tzinfo=None)).astimezone(utc),
                'raw_clock': clock  # Keep original for debugging
            }
            
            return market_clock
            
        except Exception as e:
            logger.error(f"Failed to get market clock: {e}")
            return None
    
    def is_market_near_close(self, minutes_before_close: int = 10) -> bool:
        """Check if market closes within specified minutes"""
        try:
            market_clock = self.get_market_clock()
            if not market_clock:
                logger.warning("Could not get market clock, assuming market not near close")
                return False
            
            # If market is closed, return False
            if not market_clock['is_open']:
                return False
            
            # Calculate time until market close
            now_utc = datetime.now(timezone.utc)
            next_close_utc = market_clock['next_close']
            
            # Handle timezone-aware datetime comparison
            if next_close_utc.tzinfo is None:
                next_close_utc = next_close_utc.replace(tzinfo=timezone.utc)
            
            time_until_close = next_close_utc - now_utc
            minutes_until_close = time_until_close.total_seconds() / 60
            
            logger.debug(f"Minutes until market close: {minutes_until_close:.1f}")
            
            return minutes_until_close <= minutes_before_close
            
        except Exception as e:
            logger.error(f"Error checking if market near close: {e}")
            return False
    
    def should_prevent_new_positions(self) -> bool:
        """Check if new positions should be prevented (15 minutes before market close)"""
        return self.is_market_near_close(minutes_before_close=15)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            account_info = self.get_account_info()
            risk_metrics = await self.update_risk_metrics()
            
            # Get validation status
            validation_status = {
                'system_validated': self.system_validated,
                'last_validation': self.last_validation_time.isoformat() if self.last_validation_time else None,
                'last_drift_check': self.last_drift_check.isoformat() if self.last_drift_check else None
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'trading_active': self.is_trading,
                'account_info': account_info,
                'risk_metrics': {
                    'daily_pnl': risk_metrics.daily_pnl,
                    'portfolio_value': risk_metrics.portfolio_value,
                    'total_exposure': risk_metrics.total_exposure,
                    'leverage_ratio': risk_metrics.leverage_ratio,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'var_95': risk_metrics.var_95,
                    'concentration_risk': risk_metrics.concentration_risk
                },
                'positions': {
                    'total_positions': len(self.positions),
                    'long_positions': len([p for p in self.positions.values() if p.side == 'long']),
                    'short_positions': len([p for p in self.positions.values() if p.side == 'short'])
                },
                'performance': {
                    'win_rate': self.win_rate,
                    'sharpe_ratio': self.sharpe_ratio,
                    'max_drawdown': self.max_drawdown,
                    'total_trades': self.total_trades
                },
                'daily_trades': self.daily_trades,
                'validation_status': validation_status
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def run_walk_forward_test(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run comprehensive walk-forward testing as specified in requirements"""
        try:
            logger.info(f"Starting walk-forward test from {start_date} to {end_date}")
            
            # Run walk-forward testing on all models
            walk_forward_results = await self.performance_validator.walk_forward_test(
                start_date=start_date,
                end_date=end_date
            )
            
            # Validate performance against targets
            validation_result = await self.performance_validator.validate_system_performance(
                start_date=start_date,
                end_date=end_date
            )
            
            # Check for concept drift during the period
            drift_results = await self.drift_detector.detect_all_models_drift()
            
            # Compile comprehensive results
            test_results = {
                'test_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'duration_days': (end_date - start_date).days
                },
                'walk_forward_results': walk_forward_results,
                'performance_validation': {
                    'passed': validation_result.passed if validation_result else False,
                    'metrics': {
                        'annual_return': validation_result.metrics.annual_return if validation_result else 0,
                        'sharpe_ratio': validation_result.metrics.sharpe_ratio if validation_result else 0,
                        'max_drawdown': validation_result.metrics.max_drawdown if validation_result else 0,
                        'win_rate': validation_result.metrics.win_rate if validation_result else 0,
                        'trades_per_day': validation_result.metrics.trades_per_day if validation_result else 0
                    },
                    'recommendations': validation_result.recommendations if validation_result else [],
                    'risk_warnings': validation_result.risk_warnings if validation_result else []
                },
                'drift_analysis': drift_results,
                'system_readiness': {
                    'ready_for_live_trading': (
                        (validation_result.passed if validation_result else False) and 
                        (walk_forward_results.overall_score > 0.6 if walk_forward_results else False) and
                        not any(metrics.get('drift_detected', False) for metrics in drift_results.values())
                    ),
                    'confidence_score': self._calculate_confidence_score(
                        walk_forward_results, validation_result, drift_results
                    )
                }
            }
            
            logger.info(f"Walk-forward test completed. Ready for live trading: {test_results['system_readiness']['ready_for_live_trading']}")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Walk-forward test failed: {e}")
            return {
                'error': str(e),
                'system_readiness': {
                    'ready_for_live_trading': False,
                    'confidence_score': 0.0
                }
            }
    
    async def validate_live_readiness(self) -> Dict[str, Any]:
        """Comprehensive validation for live trading readiness"""
        try:
            logger.info("Starting comprehensive live trading readiness validation")
            
            # 1. Run recent walk-forward test (last 3 months)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            walk_forward_results = await self.run_walk_forward_test(start_date, end_date)
            
            # 2. Validate current model performance
            current_validation = await self.performance_validator.validate_live_readiness()
            
            # 3. Check for recent concept drift
            drift_check = await self.drift_detector.detect_all_models_drift()
            
            # 4. Validate trading infrastructure
            infrastructure_check = await self._validate_trading_infrastructure()
            
            # 5. Check risk management systems
            risk_check = await self._validate_risk_systems()
            
            # 6. Validate data pipeline
            data_check = await self._validate_data_pipeline()
            
            # Compile readiness assessment
            readiness_checks = {
                'walk_forward_test': walk_forward_results.get('system_readiness', {}).get('ready_for_live_trading', False) if walk_forward_results else False,
                'current_performance': current_validation.passed if current_validation else False,
                'drift_status': not any(metrics.get('drift_detected', False) for metrics in drift_check.values()),
                'infrastructure': infrastructure_check,
                'risk_management': risk_check,
                'data_pipeline': data_check
            }
            
            overall_ready = all(readiness_checks.values())
            
            readiness_result = {
                'timestamp': datetime.now().isoformat(),
                'overall_ready': overall_ready,
                'readiness_checks': readiness_checks,
                'detailed_results': {
                    'walk_forward_test': walk_forward_results,
                    'performance_validation': current_validation,
                    'drift_analysis': drift_check
                },
                'recommendations': self._generate_readiness_recommendations(readiness_checks),
                'next_validation_due': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            if overall_ready:
                logger.info("✅ System is READY for live trading")
                self.system_validated = True
                self.last_validation_time = datetime.now()
            else:
                logger.warning("❌ System is NOT READY for live trading")
                self.system_validated = False
            
            return readiness_result
            
        except Exception as e:
            logger.error(f"Live readiness validation failed: {e}")
            return {
                'overall_ready': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_confidence_score(self, walk_forward_results, validation_result, drift_results: Dict) -> float:
        """Calculate overall confidence score for system readiness"""
        try:
            scores = []
            
            # Walk-forward test score - handle WalkForwardValidationResult object
            if walk_forward_results and hasattr(walk_forward_results, 'overall_score'):
                overall_score = walk_forward_results.overall_score
                if overall_score > 0.8:
                    scores.append(1.0)
                elif overall_score > 0.6:
                    scores.append(0.8)
                else:
                    scores.append(0.4)
            elif walk_forward_results and isinstance(walk_forward_results, dict):
                # Handle dictionary case - extract from nested structure
                confidence_score = walk_forward_results.get('system_readiness', {}).get('confidence_score', 0.4)
                scores.append(confidence_score)
            else:
                scores.append(0.4)
            
            # Performance validation score - handle ValidationResult object
            if validation_result and hasattr(validation_result, 'passed') and validation_result.passed:
                scores.append(1.0)
            else:
                scores.append(0.3)
            
            # Drift score
            drift_detected = any(metrics.get('drift_detected', False) for metrics in drift_results.values())
            scores.append(0.2 if drift_detected else 1.0)
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    async def _validate_trading_infrastructure(self) -> bool:
        """Validate trading infrastructure readiness"""
        try:
            # Check Alpaca API connection
            account = self.trading_client.get_account()
            if not account:
                return False
            
            # Check account status
            if account.trading_blocked or account.account_blocked:
                return False
            
            # Check buying power
            if float(account.buying_power) < 10000:  # Minimum $10k
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Infrastructure validation failed: {e}")
            return False
    
    async def _validate_risk_systems(self) -> bool:
        """Validate risk management systems"""
        try:
            # Test basic risk calculations
            portfolio_value = self._get_portfolio_value()
            if portfolio_value <= 0:
                return False
            
            # Test position sizing calculation
            test_signal = TradeSignal(
                symbol='AAPL',
                action='buy',
                confidence=0.8,
                predicted_return=0.05,
                risk_score=0.3
            )
            
            sizing = await self._calculate_position_sizing(test_signal)
            if sizing.final_size <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk system validation failed: {e}")
            return False
    
    async def _validate_data_pipeline(self) -> bool:
        """Validate data pipeline functionality"""
        try:
            # Test data client connectivity
            test_symbols = ['AAPL', 'MSFT']
            
            for symbol in test_symbols:
                try:
                    price = await self._get_current_price(symbol)
                    if price <= 0:
                        return False
                except:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data pipeline validation failed: {e}")
            return False
    
    def _generate_readiness_recommendations(self, readiness_checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on readiness check results"""
        recommendations = []
        
        if not readiness_checks.get('walk_forward_test', True):
            recommendations.append("Improve model performance - walk-forward test failed")
        
        if not readiness_checks.get('current_performance', True):
            recommendations.append("Current performance below targets - retrain models")
        
        if not readiness_checks.get('drift_status', True):
            recommendations.append("Concept drift detected - update models with recent data")
        
        if not readiness_checks.get('infrastructure', True):
            recommendations.append("Fix trading infrastructure issues - check API connectivity")
        
        if not readiness_checks.get('risk_management', True):
            recommendations.append("Risk management system issues - review risk parameters")
        
        if not readiness_checks.get('data_pipeline', True):
            recommendations.append("Data pipeline issues - ensure fresh, complete data")
        
        if not recommendations:
            recommendations.append("System ready for live trading - all checks passed")
        
        return recommendations
    
    def _round_price(self, price: float) -> float:
        """Round price to nearest penny to avoid sub-penny pricing errors"""
        return round(price, 2)
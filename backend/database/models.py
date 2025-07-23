from sqlalchemy import Column, Integer, String, DateTime, Numeric, BigInteger, Text, Index, UniqueConstraint, create_engine, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Any
from loguru import logger

Base = declarative_base()

# Database engine and session
engine = None
SessionLocal = None

class MarketData(Base):
    """Market data model for storing OHLCV data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Numeric(10, 4), nullable=False)
    high = Column(Numeric(10, 4), nullable=False)
    low = Column(Numeric(10, 4), nullable=False)
    close = Column(Numeric(10, 4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    vwap = Column(Numeric(10, 4), nullable=True)
    transactions = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uq_market_data_symbol_timestamp'),
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_timestamp', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': self.volume,
            'vwap': float(self.vwap) if self.vwap else None,
            'transactions': self.transactions,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Features(Base):
    """Features model for storing engineered features"""
    __tablename__ = 'features'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    features = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uq_features_symbol_timestamp'),
        Index('idx_features_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'features': self.features,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Predictions(Base):
    """Predictions model for storing ML model predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    prediction = Column(Numeric(6, 4), nullable=False)
    confidence = Column(Numeric(6, 4), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_predictions_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_predictions_model_timestamp', 'model_name', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'model_name': self.model_name,
            'prediction': float(self.prediction),
            'confidence': float(self.confidence),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Trades(Base):
    """Trades model for storing executed trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(10, 4), nullable=False)
    order_id = Column(String(100), nullable=True, index=True)
    status = Column(String(20), nullable=False)  # 'pending', 'filled', 'cancelled', 'rejected'
    commission = Column(Numeric(8, 4), nullable=True)
    pnl = Column(Numeric(10, 4), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_trades_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_trades_order_id', 'order_id'),
        Index('idx_trades_status', 'status'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'side': self.side,
            'quantity': self.quantity,
            'price': float(self.price),
            'order_id': self.order_id,
            'status': self.status,
            'commission': float(self.commission) if self.commission else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Positions(Base):
    """Positions model for tracking current positions"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    quantity = Column(Integer, nullable=False)
    avg_price = Column(Numeric(10, 4), nullable=False)
    market_value = Column(Numeric(12, 4), nullable=False)
    unrealized_pnl = Column(Numeric(10, 4), nullable=False)
    realized_pnl = Column(Numeric(10, 4), nullable=False, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': float(self.avg_price),
            'market_value': float(self.market_value),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=True, index=True)  # None for overall performance
    date = Column(DateTime, nullable=False, index=True)
    accuracy = Column(Numeric(6, 4), nullable=True)
    precision = Column(Numeric(6, 4), nullable=True)
    recall = Column(Numeric(6, 4), nullable=True)
    f1_score = Column(Numeric(6, 4), nullable=True)
    sharpe_ratio = Column(Numeric(6, 4), nullable=True)
    total_return = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(6, 4), nullable=True)
    win_rate = Column(Numeric(6, 4), nullable=True)
    avg_win = Column(Numeric(8, 4), nullable=True)
    avg_loss = Column(Numeric(8, 4), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_model_performance_model_date', 'model_name', 'date'),
        Index('idx_model_performance_symbol_date', 'symbol', 'date'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_name': self.model_name,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'accuracy': float(self.accuracy) if self.accuracy else None,
            'precision': float(self.precision) if self.precision else None,
            'recall': float(self.recall) if self.recall else None,
            'f1_score': float(self.f1_score) if self.f1_score else None,
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'total_return': float(self.total_return) if self.total_return else None,
            'max_drawdown': float(self.max_drawdown) if self.max_drawdown else None,
            'win_rate': float(self.win_rate) if self.win_rate else None,
            'avg_win': float(self.avg_win) if self.avg_win else None,
            'avg_loss': float(self.avg_loss) if self.avg_loss else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TradingSignals(Base):
    """Trading signals model for storing generated signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(10), nullable=False)  # 'buy', 'sell', 'hold'
    strength = Column(Numeric(6, 4), nullable=False)  # Signal strength 0-1
    confidence = Column(Numeric(6, 4), nullable=False)  # Model confidence 0-1
    price = Column(Numeric(10, 4), nullable=False)  # Price when signal generated
    volume = Column(BigInteger, nullable=True)
    features_snapshot = Column(JSONB, nullable=True)  # Key features at signal time
    model_ensemble = Column(JSONB, nullable=True)  # Individual model predictions
    executed = Column(String(20), default='pending')  # 'pending', 'executed', 'ignored', 'expired'
    execution_price = Column(Numeric(10, 4), nullable=True)
    execution_time = Column(DateTime, nullable=True)
    pnl = Column(Numeric(10, 4), nullable=True)  # Realized PnL if position closed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_trading_signals_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_trading_signals_executed', 'executed'),
        Index('idx_trading_signals_signal_type', 'signal_type'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'signal_type': self.signal_type,
            'strength': float(self.strength),
            'confidence': float(self.confidence),
            'price': float(self.price),
            'volume': self.volume,
            'features_snapshot': self.features_snapshot,
            'model_ensemble': self.model_ensemble,
            'executed': self.executed,
            'execution_price': float(self.execution_price) if self.execution_price else None,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class RiskMetrics(Base):
    """Risk metrics tracking"""
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    portfolio_value = Column(Numeric(12, 4), nullable=False)
    total_exposure = Column(Numeric(12, 4), nullable=False)
    cash_balance = Column(Numeric(12, 4), nullable=False)
    daily_pnl = Column(Numeric(10, 4), nullable=False)
    unrealized_pnl = Column(Numeric(10, 4), nullable=False)
    realized_pnl = Column(Numeric(10, 4), nullable=False)
    max_drawdown = Column(Numeric(6, 4), nullable=False)
    var_95 = Column(Numeric(10, 4), nullable=True)  # Value at Risk 95%
    sharpe_ratio = Column(Numeric(6, 4), nullable=True)
    volatility = Column(Numeric(6, 4), nullable=True)
    beta = Column(Numeric(6, 4), nullable=True)
    active_positions = Column(Integer, nullable=False)
    sector_exposure = Column(JSONB, nullable=True)  # Exposure by sector
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_risk_metrics_timestamp', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'portfolio_value': float(self.portfolio_value),
            'total_exposure': float(self.total_exposure),
            'cash_balance': float(self.cash_balance),
            'daily_pnl': float(self.daily_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'max_drawdown': float(self.max_drawdown),
            'var_95': float(self.var_95) if self.var_95 else None,
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'volatility': float(self.volatility) if self.volatility else None,
            'beta': float(self.beta) if self.beta else None,
            'active_positions': self.active_positions,
            'sector_exposure': self.sector_exposure,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SystemLogs(Base):
    """System logs for monitoring and debugging"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    level = Column(String(10), nullable=False, index=True)  # 'INFO', 'WARNING', 'ERROR', 'DEBUG'
    component = Column(String(50), nullable=False, index=True)  # 'data_pipeline', 'ml_engine', 'trading_engine'
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)  # Additional structured data
    
    __table_args__ = (
        Index('idx_system_logs_level_timestamp', 'level', 'timestamp'),
        Index('idx_system_logs_component_timestamp', 'component', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level,
            'component': self.component,
            'message': self.message,
            'details': self.details
        }

async def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        # Use the DatabaseManager from __init__.py
        from . import db_manager
        
        engine = db_manager.get_engine()
        SessionLocal = db_manager.SessionLocal
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
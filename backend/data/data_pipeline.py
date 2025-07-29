import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import json
from polygon import RESTClient
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import settings

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    transactions: Optional[int] = None

@dataclass
class DataQuality:
    symbol: str
    total_bars: int
    missing_bars: int
    data_completeness: float
    last_update: datetime
    avg_volume: float
    price_gaps: int

class DataPipeline:
    def __init__(self):
        self.polygon_client = RESTClient(settings.polygon_api_key)
        self.engine = self._create_db_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.quality_metrics: Dict[str, DataQuality] = {}
        
        # Hybrid Feature Storage Strategy
        self.feature_cache: Dict[str, Dict[datetime, Dict]] = {}  # In-memory cache for recent features
        self.cache_duration_hours = 2  # Cache last 2 hours of features
        self.cache_max_size = 10000  # Maximum cached feature records per symbol
        
        # Trading universe
        self.trading_universe = [
            'AAPL', 'NVDA'
            # Technology
            # 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'META',
            # Biotechnology
            # 'MRNA', 'GILD', 'BIIB', 'VRTX',
            # Energy
            # 'XOM', 'CVX', 'SLB', 'HAL',
            # Crypto-Related
            # 'MARA', 'COIN', 'RIOT',
            # Consumer Discretionary
            # 'AMZN', 'NFLX', 'DIS'
        ]
    
    def get_ticker_universe(self) -> List[str]:
        """Get the list of trading symbols"""
        return self.trading_universe
        
    def _create_db_engine(self):
        """Create database engine"""
        from database import db_manager
        return db_manager.get_engine()
    
    async def initialize_database(self):
        """Initialize database tables"""
        try:
            with self.engine.connect() as conn:
                # Create market data table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        open DECIMAL(10,4) NOT NULL,
                        high DECIMAL(10,4) NOT NULL,
                        low DECIMAL(10,4) NOT NULL,
                        close DECIMAL(10,4) NOT NULL,
                        volume BIGINT NOT NULL,
                        vwap DECIMAL(10,4),
                        transactions INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """))
                
                # Create index for faster queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                    ON market_data(symbol, timestamp DESC)
                """))
                
                # Create features table with time-series optimizations
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS features (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        features JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timestamp)
                    )
                """))
                
                # Create optimized indexes for hybrid storage strategy
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
                    ON features(symbol, timestamp DESC)
                """))
                
                # Create JSONB GIN index for fast feature queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_jsonb 
                    ON features USING GIN (features)
                """))
                
                # Create index for features (removed time-based WHERE clause to avoid IMMUTABLE function requirement)
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_features_recent 
                    ON features(symbol, timestamp DESC)
                """))
                
                # Create predictions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        model_name VARCHAR(50) NOT NULL,
                        prediction DECIMAL(6,4) NOT NULL,
                        confidence DECIMAL(6,4) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create trades table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        quantity INTEGER NOT NULL,
                        price DECIMAL(10,4) NOT NULL,
                        order_id VARCHAR(100),
                        status VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def download_historical_data(self, symbol: str, 
                                     start_date: datetime, 
                                     end_date: datetime) -> pd.DataFrame:
        """Download historical minute data from Polygon.io with proper pagination
        
        First checks for existing data in database and only downloads missing data if needed.
        This prevents redundant downloads and maintains consistency with training endpoint logic.
        """
        try:
            logger.info(f"Starting download of historical data for {symbol} from {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # First, try to load existing data from database (same logic as train_symbol_models)
            existing_data = await self.load_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Check if we have sufficient existing data
            if existing_data is not None and len(existing_data) >= 100:
                # Calculate expected data points (approximately 390 minutes per trading day)
                expected_days = (end_date - start_date).days
                expected_data_points = expected_days * 390  # Rough estimate
                coverage_percentage = (len(existing_data) / expected_data_points * 100) if expected_data_points > 0 else 0
                
                logger.info(f"Found {len(existing_data)} existing data points for {symbol} (estimated {coverage_percentage:.1f}% coverage)")
                
                # If we have good coverage (>80%), return existing data
                if coverage_percentage > 80.0:
                    logger.info(f"Using existing data for {symbol} - sufficient coverage ({coverage_percentage:.1f}%)")
                    # Cache the data
                    self.data_cache[symbol] = existing_data
                    return existing_data
                else:
                    logger.info(f"Existing data coverage insufficient ({coverage_percentage:.1f}%), downloading fresh data...")
            else:
                logger.info(f"No existing data or insufficient data for {symbol} (found {len(existing_data) if existing_data is not None else 0} points), downloading...")
            
            # Get minute bars from Polygon using date range with pagination
            bars = []
            page_count = 0
            
            # Use list_aggs for automatic pagination
            logger.info(f"Fetching data for {symbol} from Polygon API...")
            
            try:
                for agg in self.polygon_client.list_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="minute",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=50000
                ):
                    bars.append(MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                        open=agg.open,
                        high=agg.high,
                        low=agg.low,
                        close=agg.close,
                        volume=agg.volume,
                        vwap=getattr(agg, 'vwap', None),
                        transactions=getattr(agg, 'transactions', None)
                    ))
                    
                    # Log progress every 10,000 bars
                    if len(bars) % 10000 == 0:
                        logger.info(f"Downloaded {len(bars)} bars for {symbol}...")
                
                logger.info(f"Completed data fetch for {symbol} - total bars: {len(bars)}")
                
            except Exception as e:
                logger.error(f"Error during data fetch for {symbol}: {e}")
                # Fallback to get_aggs if list_aggs fails
                logger.info(f"Falling back to single get_aggs call for {symbol}")
                
                resp = self.polygon_client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="minute",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=50000
                )
                
                if resp and hasattr(resp, 'results') and resp.results:
                    for bar in resp.results:
                        bars.append(MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(bar.timestamp / 1000, tz=timezone.utc),
                            open=bar.open,
                            high=bar.high,
                            low=bar.low,
                            close=bar.close,
                            volume=bar.volume,
                            vwap=getattr(bar, 'vwap', None),
                            transactions=getattr(bar, 'transactions', None)
                        ))
                    
                    logger.info(f"Fallback completed - downloaded {len(bars)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol} in the specified date range")
            
            if not bars:
                logger.warning(f"No data retrieved for {symbol} in the specified date range")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(bars)} total bars for {symbol}...")
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'symbol': bar.symbol,
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                    'transactions': bar.transactions
                }
                for bar in bars
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            logger.info(f"Storing {len(df)} bars in database for {symbol}...")
            # Store in database
            await self._store_market_data(df, symbol)
            
            # Cache data
            self.data_cache[symbol] = df
            
            logger.info(f"Successfully downloaded and stored {len(df)} bars for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time market data"""
        try:
            # Get latest trade
            trade = self.polygon_client.get_last_trade(symbol)
            
            # Get latest quote
            quote = self.polygon_client.get_last_quote(symbol)
            
            if trade and quote:
                return MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(trade.timestamp / 1000, tz=timezone.utc),
                    open=trade.price,  # Simplified - would need more logic
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size,
                    vwap=None,
                    transactions=1
                )
            
        except Exception as e:
            logger.error(f"Failed to get real-time data for {symbol}: {e}")
        
        return None
    
    async def _store_market_data(self, df: pd.DataFrame, symbol: str):
        """Store market data in database"""
        try:
            with self.Session() as session:
                for timestamp, row in df.iterrows():
                    session.execute(text("""
                        INSERT INTO market_data 
                        (symbol, timestamp, open, high, low, close, volume, vwap, transactions)
                        VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :vwap, :transactions)
                        ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        vwap = EXCLUDED.vwap,
                        transactions = EXCLUDED.transactions
                    """), {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']),
                        'vwap': float(row['vwap']) if pd.notna(row['vwap']) else None,
                        'transactions': int(row['transactions']) if pd.notna(row['transactions']) else None
                    })
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store market data for {symbol}: {e}")
    
    async def load_market_data(self, symbol: str, 
                             start_date: datetime, 
                             end_date: datetime) -> pd.DataFrame:
        """Load market data from database"""
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT timestamp, open, high, low, close, volume, vwap, transactions
                    FROM market_data
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_date AND :end_date
                    ORDER BY timestamp
                """), {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                data = result.fetchall()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to load market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    async def store_features(self, symbol: str, timestamp: datetime, features: Dict):
        """Store engineered features using hybrid strategy: PostgreSQL + in-memory cache"""
        try:
            # Convert features to JSON-serializable format
            serializable_features = self._make_json_serializable(features)
            
            # 1. Store in PostgreSQL for persistence
            with self.Session() as session:
                session.execute(text("""
                    INSERT INTO features (symbol, timestamp, features)
                    VALUES (:symbol, :timestamp, :features)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                    features = EXCLUDED.features
                """), {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'features': json.dumps(serializable_features)
                })
                
                session.commit()
            
            # 2. Store in in-memory cache for ultra-low latency access
            await self._cache_features(symbol, timestamp, features)
            
            logger.debug(f"Stored features for {symbol} at {timestamp} (DB + Cache)")
                
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
    
    async def _cache_features(self, symbol: str, timestamp: datetime, features: Dict):
        """Cache features in memory for ultra-low latency access"""
        try:
            # Initialize symbol cache if not exists
            if symbol not in self.feature_cache:
                self.feature_cache[symbol] = {}
            
            # Add features to cache
            self.feature_cache[symbol][timestamp] = features.copy()
            
            # Clean old cache entries (keep only last 2 hours)
            cutoff_time = timestamp - timedelta(hours=self.cache_duration_hours)
            timestamps_to_remove = [
                ts for ts in self.feature_cache[symbol].keys() 
                if ts < cutoff_time
            ]
            
            for ts in timestamps_to_remove:
                del self.feature_cache[symbol][ts]
            
            # Limit cache size per symbol
            if len(self.feature_cache[symbol]) > self.cache_max_size:
                # Remove oldest entries
                sorted_timestamps = sorted(self.feature_cache[symbol].keys())
                excess_count = len(self.feature_cache[symbol]) - self.cache_max_size
                
                for ts in sorted_timestamps[:excess_count]:
                    del self.feature_cache[symbol][ts]
            
            logger.debug(f"Cached features for {symbol}, cache size: {len(self.feature_cache[symbol])}")
            
        except Exception as e:
            logger.error(f"Failed to cache features for {symbol}: {e}")
    
    async def get_cached_features(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get features from in-memory cache for ultra-low latency"""
        try:
            if symbol in self.feature_cache and timestamp in self.feature_cache[symbol]:
                return self.feature_cache[symbol][timestamp].copy()
            return None
        except Exception as e:
            logger.error(f"Failed to get cached features for {symbol}: {e}")
            return None
    
    async def bootstrap_feature_cache(self, symbol: str, minutes: int = 120) -> int:
        """Bootstrap feature cache from database for a symbol
        
        Args:
            symbol: Stock symbol to bootstrap
            minutes: Number of minutes to load from database
            
        Returns:
            int: Number of features loaded into cache
        """
        try:
            # Calculate time range for bootstrap
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=minutes)
            
            logger.info(f"Bootstrapping feature cache for {symbol} from {start_time} to {end_time}")
            
            # Load features from database
            features_df = await self.load_features_from_db(symbol, start_time, end_time)
            
            if features_df is None or len(features_df) == 0:
                logger.debug(f"No existing features found in database for {symbol}")
                return 0
            
            # Load features into cache
            loaded_count = 0
            for timestamp, row in features_df.iterrows():
                # Convert row to dictionary, excluding NaN values
                feature_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
                
                # Cache the features
                await self._cache_features(symbol, timestamp, feature_dict)
                loaded_count += 1
            
            logger.info(f"Bootstrapped {loaded_count} features for {symbol} into cache")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to bootstrap feature cache for {symbol}: {e}")
            return 0
    
    async def get_recent_cached_features(self, symbol: str, minutes: int = 60) -> Dict[datetime, Dict]:
        """Get recent cached features for a symbol (last N minutes)
        
        Smart timestamp handling:
        1. Use most recent cached feature timestamp as reference point
        2. Handle market gaps (overnight, weekends) intelligently
        3. Ensure sufficient features for signal generation (60 minimum)
        4. Auto-bootstrap from database if cache is empty
        """
        try:
            # Check if cache is empty and try to bootstrap from database
            if symbol not in self.feature_cache or not self.feature_cache[symbol]:
                logger.debug(f"No cached features found for {symbol}, attempting bootstrap from database")
                
                # Try to bootstrap from database
                bootstrap_count = await self.bootstrap_feature_cache(symbol, minutes * 2)  # Load more for better coverage
                
                if bootstrap_count == 0:
                    logger.debug(f"No features available in database for {symbol}")
                    return {}
                
                logger.info(f"Bootstrapped {bootstrap_count} features for {symbol} from database")
            
            # Get all cached timestamps for this symbol
            cached_timestamps = list(self.feature_cache[symbol].keys())
            
            if not cached_timestamps:
                return {}
            
            # Sort timestamps to find the most recent
            cached_timestamps.sort()
            most_recent_timestamp = cached_timestamps[-1]
            
            logger.debug(f"Most recent cached feature for {symbol}: {most_recent_timestamp}")
            
            # Strategy 1: Try to get last N minutes from most recent timestamp
            cutoff_time = most_recent_timestamp - timedelta(minutes=minutes)
            recent_features = {
                ts: features for ts, features in self.feature_cache[symbol].items()
                if ts >= cutoff_time
            }
            
            # Check if we have sufficient features (minimum 60 for signal generation)
            min_required_features = 60
            
            if len(recent_features) >= min_required_features:
                logger.debug(f"Found {len(recent_features)} recent features for {symbol} using {minutes}-minute window")
                return recent_features
            
            # Strategy 2: If insufficient, get the most recent N features regardless of time gap
            # This handles market gaps (overnight, weekends, holidays)
            if len(cached_timestamps) >= min_required_features:
                # Take the most recent N features
                recent_timestamps = cached_timestamps[-min_required_features:]
                recent_features = {
                    ts: self.feature_cache[symbol][ts] for ts in recent_timestamps
                }
                
                time_span = recent_timestamps[-1] - recent_timestamps[0]
                logger.info(f"Using {len(recent_features)} most recent features for {symbol} "
                           f"spanning {time_span} (market gap detected)")
                return recent_features
            
            # Strategy 3: Return all available features if we have less than minimum
            logger.warning(f"Only {len(cached_timestamps)} cached features available for {symbol}, "
                          f"less than minimum required {min_required_features}")
            
            recent_features = {
                ts: features for ts, features in self.feature_cache[symbol].items()
            }
            
            return recent_features
            
        except Exception as e:
            logger.error(f"Failed to get recent cached features for {symbol}: {e}")
            return {}
    
    async def check_existing_features(self, symbol: str, start_time: datetime, end_time: datetime) -> List[datetime]:
        """Check which timestamps already have features stored in the database
        
        Args:
            symbol: Stock symbol to check
            start_time: Start of time range to check
            end_time: End of time range to check
            
        Returns:
            List of timestamps that already have features stored
        """
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT timestamp
                    FROM features
                    WHERE symbol = :symbol
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp
                """), {
                    'symbol': symbol,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                existing_timestamps = []
                for row in result.fetchall():
                    timestamp = row.timestamp
                    # Ensure timestamp is timezone-aware (UTC)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    existing_timestamps.append(timestamp)
                
                logger.info(f"Found {len(existing_timestamps)} existing feature timestamps for {symbol} in range {start_time} to {end_time}")
                return existing_timestamps
                
        except Exception as e:
            logger.error(f"Failed to check existing features for {symbol}: {e}")
            return []
    
    async def load_features_from_db(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load features from PostgreSQL for historical analysis, including basic OHLCV data"""
        try:
            with self.Session() as session:
                # Join features with market_data to get both engineered features and basic OHLCV columns
                result = session.execute(text("""
                    SELECT 
                        f.timestamp, 
                        f.features,
                        m.open,
                        m.high,
                        m.low,
                        m.close,
                        m.volume,
                        m.vwap,
                        m.transactions
                    FROM features f
                    LEFT JOIN market_data m ON f.symbol = m.symbol AND f.timestamp = m.timestamp
                    WHERE f.symbol = :symbol
                    AND f.timestamp BETWEEN :start_time AND :end_time
                    ORDER BY f.timestamp
                """), {
                    'symbol': symbol,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                rows = result.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    # JSONB column returns dict directly, no need for json.loads()
                    feature_dict = row.features if isinstance(row.features, dict) else json.loads(row.features)
                    
                    # Ensure timestamp is timezone-aware (UTC)
                    timestamp = row.timestamp
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    
                    # Add basic OHLCV columns
                    feature_dict.update({
                        'timestamp': timestamp,
                        'open': float(row.open) if row.open is not None else None,
                        'high': float(row.high) if row.high is not None else None,
                        'low': float(row.low) if row.low is not None else None,
                        'close': float(row.close) if row.close is not None else None,
                        'volume': int(row.volume) if row.volume is not None else None,
                        'vwap': float(row.vwap) if row.vwap is not None else None,
                        'transactions': int(row.transactions) if row.transactions is not None else None
                    })
                    
                    data.append(feature_dict)
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Loaded {len(df)} feature records with OHLCV data for {symbol} from database")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load features from database for {symbol}: {e}")
            return pd.DataFrame()
    
    async def store_prediction(self, symbol: str, timestamp: datetime, 
                             model_name: str, prediction: float, confidence: float):
        """Store model prediction in database"""
        try:
            with self.Session() as session:
                session.execute(text("""
                    INSERT INTO predictions (symbol, timestamp, model_name, prediction, confidence)
                    VALUES (:symbol, :timestamp, :model_name, :prediction, :confidence)
                """), {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'model_name': model_name,
                    'prediction': prediction,
                    'confidence': confidence
                })
                
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store prediction for {symbol}: {e}")
    
    async def check_data_quality(self, symbol: str) -> DataQuality:
        """Check data quality for a symbol"""
        try:
            with self.Session() as session:
                # Get basic statistics
                result = session.execute(text("""
                    SELECT 
                        COUNT(*) as total_bars,
                        AVG(volume) as avg_volume,
                        MAX(timestamp) as last_update
                    FROM market_data
                    WHERE symbol = :symbol
                    AND timestamp >= :start_date
                """), {
                    'symbol': symbol,
                    'start_date': datetime.now(timezone.utc) - timedelta(days=30)
                })
                
                stats = result.fetchone()
                
                if not stats or stats.total_bars == 0:
                    return DataQuality(
                        symbol=symbol,
                        total_bars=0,
                        missing_bars=0,
                        data_completeness=0.0,
                        last_update=datetime.min,
                        avg_volume=0.0,
                        price_gaps=0
                    )
                
                # Calculate expected bars (market hours: 9:30-16:00, 390 minutes per day)
                days_in_period = 30
                expected_bars = days_in_period * 390  # Simplified
                missing_bars = max(0, expected_bars - stats.total_bars)
                data_completeness = stats.total_bars / expected_bars if expected_bars > 0 else 0
                
                # Check for price gaps
                gap_result = session.execute(text("""
                    SELECT COUNT(*) as price_gaps
                    FROM (
                        SELECT 
                            ABS(open - LAG(close) OVER (ORDER BY timestamp)) / LAG(close) OVER (ORDER BY timestamp) as gap_pct
                        FROM market_data
                        WHERE symbol = :symbol
                        AND timestamp >= :start_date
                    ) gaps
                    WHERE gap_pct > 0.02  -- 2% gap threshold
                """), {
                    'symbol': symbol,
                    'start_date': datetime.now(timezone.utc) - timedelta(days=30)
                })
                
                price_gaps = gap_result.fetchone().price_gaps or 0
                
                quality = DataQuality(
                    symbol=symbol,
                    total_bars=stats.total_bars,
                    missing_bars=missing_bars,
                    data_completeness=data_completeness,
                    last_update=stats.last_update,
                    avg_volume=float(stats.avg_volume or 0),
                    price_gaps=price_gaps
                )
                
                self.quality_metrics[symbol] = quality
                return quality
                
        except Exception as e:
            logger.error(f"Failed to check data quality for {symbol}: {e}")
            return DataQuality(
                symbol=symbol,
                total_bars=0,
                missing_bars=0,
                data_completeness=0.0,
                last_update=datetime.min,
                avg_volume=0.0,
                price_gaps=0
            )
    
    async def update_trading_universe(self) -> List[str]:
        """Complete dynamic ticker selection with ATR, beta, and market cap filtering"""
        try:
            logger.info("Starting comprehensive ticker screening")
            screened_symbols = []
            
            # Selection criteria from requirements
            criteria = {
                'min_volume': 10_000_000,      # 10M+ average daily volume
                'min_atr_pct': 2.0,            # ATR > 2% for sufficient price movement
                'min_beta': 1.5,               # Beta > 1.5 for higher volatility
                'market_cap_range': (1e9, 100e9)  # $1B - $100B market cap
            }
            
            for symbol in self.trading_universe:
                try:
                    # Check basic data quality first
                    quality = await self.check_data_quality(symbol)
                    
                    if quality.data_completeness < 0.8:
                        continue
                    
                    # Calculate ATR (Average True Range)
                    atr_pct = await self._calculate_atr_percentage(symbol)
                    if atr_pct < criteria['min_atr_pct']:
                        continue
                    
                    # Calculate Beta vs SPY
                    beta = await self._calculate_beta(symbol)
                    if beta < criteria['min_beta']:
                        continue
                    
                    # Check market cap (simplified - would need fundamental data API)
                    market_cap = await self._get_market_cap(symbol)
                    if not (criteria['market_cap_range'][0] <= market_cap <= criteria['market_cap_range'][1]):
                        continue
                    
                    # Volume check
                    if quality.avg_volume < criteria['min_volume']:
                        continue
                    
                    # All criteria passed
                    screened_symbols.append(symbol)
                    logger.info(f"{symbol}: ATR={atr_pct:.2f}%, Beta={beta:.2f}, Volume={quality.avg_volume:,.0f}")
                    
                except Exception as e:
                    logger.error(f"Failed to screen {symbol}: {e}")
                    continue
            
            logger.info(f"Screening complete: {len(screened_symbols)}/{len(self.trading_universe)} symbols passed")
            
            # Update trading universe with screened symbols
            if screened_symbols:
                self.trading_universe = screened_symbols
            
            return screened_symbols
            
        except Exception as e:
            logger.error(f"Failed to update trading universe: {e}")
            return self.trading_universe
    
    async def _calculate_atr_percentage(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range as percentage of price"""
        try:
            with self.Session() as session:
                # Get recent OHLC data
                result = session.execute(text("""
                    SELECT high, low, close, 
                           LAG(close) OVER (ORDER BY timestamp) as prev_close
                    FROM market_data
                    WHERE symbol = :symbol
                    AND timestamp >= :start_date
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), {
                    'symbol': symbol,
                    'start_date': datetime.now(timezone.utc) - timedelta(days=30),
                    'limit': period + 5
                })
                
                data = result.fetchall()
                
                if len(data) < period:
                    return 0.0
                
                # Calculate True Range for each period
                true_ranges = []
                for row in data[1:]:  # Skip first row (no prev_close)
                    high, low, close, prev_close = row
                    
                    tr1 = high - low
                    tr2 = abs(high - prev_close) if prev_close else 0
                    tr3 = abs(low - prev_close) if prev_close else 0
                    
                    true_range = max(tr1, tr2, tr3)
                    true_ranges.append(true_range)
                
                if len(true_ranges) < period:
                    return 0.0
                
                # Calculate ATR (simple moving average of True Range)
                atr = sum(true_ranges[:period]) / period
                
                # Get current price for percentage calculation
                current_price = data[0].close
                atr_percentage = (atr / current_price) * 100 if current_price > 0 else 0
                
                return atr_percentage
                
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return 0.0
    
    async def _calculate_beta(self, symbol: str, benchmark: str = 'SPY', period_days: int = 252) -> float:
        """Calculate beta vs benchmark (default SPY)"""
        try:
            with self.Session() as session:
                # Get returns for both symbol and benchmark
                result = session.execute(text("""
                    WITH symbol_returns AS (
                        SELECT 
                            timestamp,
                            (close - LAG(close) OVER (ORDER BY timestamp)) / LAG(close) OVER (ORDER BY timestamp) as return
                        FROM market_data
                        WHERE symbol = :symbol
                        AND timestamp >= :start_date
                        ORDER BY timestamp
                    ),
                    benchmark_returns AS (
                        SELECT 
                            timestamp,
                            (close - LAG(close) OVER (ORDER BY timestamp)) / LAG(close) OVER (ORDER BY timestamp) as return
                        FROM market_data
                        WHERE symbol = :benchmark
                        AND timestamp >= :start_date
                        ORDER BY timestamp
                    )
                    SELECT s.return as symbol_return, b.return as benchmark_return
                    FROM symbol_returns s
                    JOIN benchmark_returns b ON s.timestamp = b.timestamp
                    WHERE s.return IS NOT NULL AND b.return IS NOT NULL
                    ORDER BY s.timestamp DESC
                    LIMIT :limit
                """), {
                    'symbol': symbol,
                    'benchmark': benchmark,
                    'start_date': datetime.now(timezone.utc) - timedelta(days=period_days + 30),
                    'limit': period_days
                })
                
                data = result.fetchall()
                
                if len(data) < 50:  # Need minimum data points
                    return 1.0  # Default beta
                
                # Extract returns
                symbol_returns = [float(row.symbol_return) for row in data]
                benchmark_returns = [float(row.benchmark_return) for row in data]
                
                # Calculate beta using covariance and variance
                symbol_returns = np.array(symbol_returns)
                benchmark_returns = np.array(benchmark_returns)
                
                covariance = np.cov(symbol_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                else:
                    beta = 1.0
                
                return beta
                
        except Exception as e:
            logger.error(f"Failed to calculate beta for {symbol}: {e}")
            return 1.0  # Default beta
    
    async def _get_market_cap(self, symbol: str) -> float:
        """Get market capitalization (simplified - would need fundamental data)"""
        try:
            # Simplified market cap estimation based on symbol
            # In production, this would use a fundamental data API like Alpha Vantage or Polygon
            
            # Predefined market caps for major symbols (in billions)
            market_caps = {
                'AAPL': 3000, 'MSFT': 2800, 'GOOGL': 1700, 'AMZN': 1500, 'NVDA': 1800,
                'TSLA': 800, 'META': 800, 'BRK.B': 700, 'UNH': 500, 'JNJ': 450,
                'V': 500, 'WMT': 400, 'XOM': 350, 'JPM': 450, 'PG': 350,
                'MA': 350, 'CVX': 300, 'HD': 350, 'ABBV': 300, 'PFE': 250,
                'KO': 250, 'AVGO': 600, 'PEP': 230, 'TMO': 200, 'COST': 250,
                'DIS': 180, 'ABT': 180, 'ACN': 200, 'VZ': 150, 'ADBE': 200,
                'NFLX': 180, 'CRM': 200, 'NKE': 150, 'MRK': 250, 'ORCL': 300,
                'AMD': 200, 'INTC': 150, 'QCOM': 150, 'TXN': 150, 'INTU': 120,
                'IBM': 120, 'AMGN': 150, 'HON': 140, 'UPS': 140, 'LOW': 140,
                'SPGI': 120, 'GS': 120, 'CAT': 130, 'BA': 120, 'AXP': 120,
                'GILD': 80, 'BIIB': 35, 'VRTX': 80, 'MRNA': 50,
                'SLB': 60, 'HAL': 25, 'MARA': 5, 'COIN': 15, 'RIOT': 2
            }
            
            # Convert to actual values (billions to actual)
            market_cap = market_caps.get(symbol, 50) * 1e9  # Default 50B if not found
            
            return market_cap
            
        except Exception as e:
            logger.error(f"Failed to get market cap for {symbol}: {e}")
            return 50e9  # Default 50B market cap
    
    async def get_sector_allocation(self) -> Dict[str, List[str]]:
        """Get sector allocation for portfolio diversification"""
        # Sector mapping based on requirements
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'QCOM'],
            'Biotechnology': ['MRNA', 'GILD', 'BIIB', 'VRTX', 'AMGN'],
            'Energy': ['XOM', 'CVX', 'SLB', 'HAL'],
            'Crypto-Related': ['MARA', 'COIN', 'RIOT'],
            'Consumer Discretionary': ['TSLA', 'NFLX', 'DIS', 'NKE', 'HD', 'LOW'],
            'Financial': ['JPM', 'V', 'MA', 'GS', 'AXP'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'MRK'],
            'Industrial': ['HON', 'CAT', 'BA', 'UPS'],
            'Consumer Staples': ['WMT', 'PG', 'KO', 'PEP', 'COST']
        }
        
        # Filter sectors to only include symbols in current trading universe
        filtered_sectors = {}
        for sector, symbols in sectors.items():
            sector_symbols = [s for s in symbols if s in self.trading_universe]
            if sector_symbols:
                filtered_sectors[sector] = sector_symbols
        
        return filtered_sectors
    
    async def get_pipeline_status(self) -> Dict:
        """Get data pipeline status"""
        try:
            status = {
                "trading_universe_size": len(self.trading_universe),
                "cached_symbols": list(self.data_cache.keys()),
                "data_quality": {
                    symbol: {
                        "total_bars": quality.total_bars,
                        "data_completeness": quality.data_completeness,
                        "last_update": quality.last_update.isoformat() if quality.last_update != datetime.min else None,
                        "avg_volume": quality.avg_volume
                    }
                    for symbol, quality in self.quality_metrics.items()
                },
                "database_connection": "connected" if self.engine else "disconnected"
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {"error": str(e)}
    
    async def start_real_time_feed(self):
        """Start real-time data feed (simplified implementation)"""
        logger.info("Starting real-time data feed")
        
        while True:
            try:
                for symbol in self.trading_universe:
                    data = await self.get_real_time_data(symbol)
                    if data:
                        # Store in cache for immediate use
                        if symbol not in self.data_cache:
                            self.data_cache[symbol] = pd.DataFrame()
                        
                        # Add to cache (keep last 1000 bars)
                        new_row = pd.DataFrame({
                            'open': [data.open],
                            'high': [data.high],
                            'low': [data.low],
                            'close': [data.close],
                            'volume': [data.volume]
                        }, index=[data.timestamp])
                        
                        self.data_cache[symbol] = pd.concat([
                            self.data_cache[symbol], new_row
                        ]).tail(1000)
                
                # Wait before next update
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in real-time feed: {e}")
                await asyncio.sleep(10)
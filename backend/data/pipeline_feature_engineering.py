import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
# import talib  # Temporarily commented out
from loguru import logger

@dataclass
class FeatureConfig:
    symbol: str
    lookback_periods: List[int]
    technical_indicators: List[str]
    market_features: bool = True
    volume_features: bool = True
    price_features: bool = True

class FeatureEngineer:
    def __init__(self, data_pipeline=None):
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.feature_columns: List[str] = []
        self.data_pipeline = data_pipeline  # Reference to DataPipeline for hybrid storage
        
    async def engineer_features(self, ohlcv_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer comprehensive features from OHLCV data"""
        logger.info(f"Engineering features for {symbol}")
        
        df = ohlcv_data.copy()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Convert all numeric columns to float to avoid Decimal/float type errors
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
        # Price-based features
        logger.info(f"Adding price features for {symbol}...")
        df = self._add_price_features(df)
        
        # Technical indicators
        logger.info(f"Adding technical indicators for {symbol}...")
        df = self._add_technical_indicators(df)
        
        # Volume features
        logger.info(f"Adding volume features for {symbol}...")
        df = self._add_volume_features(df)
        
        # Market microstructure features
        logger.info(f"Adding microstructure features for {symbol}...")
        df = self._add_microstructure_features(df)
        
        # Time-based features
        logger.info(f"Adding time features for {symbol}...")
        df = self._add_time_features(df)
        
        # Volatility features
        logger.info(f"Adding volatility features for {symbol}...")
        df = self._add_volatility_features(df)
        
        # Momentum features
        logger.info(f"Adding momentum features for {symbol}...")
        df = self._add_momentum_features(df)
        
        # Statistical features
        logger.info(f"Adding statistical features for {symbol}...")
        df = self._add_statistical_features(df)
        
        # Clean data
        logger.info(f"Cleaning features for {symbol}...")
        df = self._clean_features(df)
        
        # Cache features
        self.feature_cache[symbol] = df
        
        # Update feature columns list
        feature_cols = [col for col in df.columns if col not in required_cols]
        self.feature_columns = feature_cols
        
        # Store features using hybrid strategy (PostgreSQL + in-memory cache)
        if self.data_pipeline:
            await self._store_features_hybrid(df, symbol, feature_cols)
        
        logger.info(f"Generated {len(feature_cols)} features for {symbol}")
        
        return df
    
    async def _store_features_hybrid(self, df: pd.DataFrame, symbol: str, feature_cols: List[str]):
        """Store features using hybrid strategy: PostgreSQL + in-memory cache"""
        try:
            if not hasattr(df.index, 'to_pydatetime'):
                logger.warning(f"DataFrame index is not datetime for {symbol}, skipping feature storage")
                return
            
            stored_count = 0
            
            # Store each row of features
            for timestamp, row in df.iterrows():
                if pd.isna(timestamp):
                    continue
                
                # Convert timestamp to datetime if needed
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime()
                elif not isinstance(timestamp, datetime):
                    continue
                
                # Extract feature values (exclude OHLCV columns)
                features = {}
                for col in feature_cols:
                    if col in row and not pd.isna(row[col]):
                        value = row[col]
                        # Convert numpy types to Python types for JSON serialization
                        if hasattr(value, 'item'):
                            value = value.item()
                        features[col] = float(value) if isinstance(value, (int, float)) else value
                
                # Only store if we have features
                if features:
                    await self.data_pipeline.store_features(symbol, timestamp, features)
                    stored_count += 1
            
            logger.info(f"Stored {stored_count} feature records for {symbol} using hybrid strategy")
            
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Basic price features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = df['price_change'] / df['open']
        
        # Price position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # Gap features
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Price ratios
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using pandas calculations (talib temporarily disabled)"""
        try:
            # Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
                df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
            
            # Bollinger Bands
            for period in [10, 20]:
                middle = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                upper = middle + (std * 2)
                lower = middle - (std * 2)
                df[f'bb_upper_{period}'] = upper
                df[f'bb_middle_{period}'] = middle
                df[f'bb_lower_{period}'] = lower
                df[f'bb_width_{period}'] = (upper - lower) / middle
                df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-8)
            
            # RSI
            for period in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
            
            # Average True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # Commodity Channel Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=14).mean()
            mad = typical_price.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Money Flow Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=14).sum()
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # On Balance Volume
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Accumulation/Distribution Line
            df['ad'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # Volume-price features
        df['volume_price'] = df['volume'] * df['close']
        df['vwap'] = (df['volume_price'].rolling(20).sum() / 
                     df['volume'].rolling(20).sum())
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # Volume oscillator
        df['volume_oscillator'] = ((df['volume'].rolling(5).mean() - 
                                   df['volume'].rolling(10).mean()) / 
                                  df['volume'].rolling(10).mean())
        
        # Price Volume Trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / 
                    df['close'].shift(1) * df['volume']).cumsum()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced market microstructure features using Polygon WebSocket fields (Priority 3)"""
        # Basic microstructure features
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['price_impact'] = abs(df['close'] - df['open']) / (df['volume'] + 1e-8)
        
        # Tick direction and momentum
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        
        # Consecutive ticks
        df['consecutive_up'] = (df['tick_direction'] == 1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (df['tick_direction'] == -1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        
        # Order flow imbalance proxy
        df['flow_imbalance'] = (df['close'] - (df['high'] + df['low']) / 2) / \
                              (df['high'] - df['low'] + 1e-8)
        
        # Enhanced microstructure features using Polygon WebSocket fields
        if 'vwap' in df.columns:
            # VWAP-based features
            df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-8)
            df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-8)
            df['vwap_momentum'] = df['vwap'].pct_change()
            
            # VWAP trend strength
            for period in [5, 10, 20]:
                vwap_sma = df['vwap'].rolling(period).mean()
                df[f'vwap_trend_{period}'] = (df['vwap'] - vwap_sma) / (vwap_sma + 1e-8)
        
        if 'transactions' in df.columns:
            # Transaction-based microstructure indicators
            df['avg_trade_size'] = df['volume'] / (df['transactions'] + 1e-8)
            df['trade_intensity'] = df['transactions'] / df['volume'].rolling(20).mean().fillna(1)
            
            # Transaction momentum and volatility
            df['transaction_momentum'] = df['transactions'].pct_change()
            df['transaction_volatility'] = df['transactions'].rolling(10).std() / (df['transactions'].rolling(10).mean() + 1e-8)
            
            # Market activity indicators
            df['high_frequency_ratio'] = df['transactions'] / (df['volume'] + 1e-8)  # Transactions per unit volume
            
            # Rolling transaction statistics
            for period in [5, 10, 20]:
                df[f'transactions_ma_{period}'] = df['transactions'].rolling(period).mean()
                df[f'transactions_ratio_{period}'] = df['transactions'] / (df[f'transactions_ma_{period}'] + 1e-8)
        
        if 'accumulated_volume' in df.columns:
            # Accumulated volume features
            df['volume_acceleration'] = df['accumulated_volume'].diff().diff()  # Second derivative
            df['volume_momentum'] = df['accumulated_volume'].pct_change()
            
            # Volume distribution analysis
            df['volume_concentration'] = df['volume'] / (df['accumulated_volume'] + 1e-8)
            
            # Intraday volume patterns
            df['volume_profile'] = df['accumulated_volume'] / df['accumulated_volume'].rolling(20).max().fillna(1)
        
        # Advanced microstructure indicators combining multiple fields
        if all(col in df.columns for col in ['vwap', 'transactions', 'volume']):
            # Market efficiency indicators
            df['price_efficiency'] = abs(df['close'] - df['vwap']) / (df['transactions'] + 1e-8)
            df['liquidity_proxy'] = df['volume'] / (abs(df['close'] - df['vwap']) + 1e-8)
            
            # Order flow toxicity (Kyle's lambda proxy)
            price_impact_per_trade = abs(df['close'] - df['close'].shift(1)) / (df['transactions'] + 1e-8)
            df['order_flow_toxicity'] = price_impact_per_trade.rolling(10).mean()
            
            # Market depth proxy
            df['market_depth_proxy'] = df['volume'] / (df['spread_proxy'] + 1e-8)
        
        # Real-time volatility indicators (Priority 3: Real-time volatility calculations)
        if 'vwap' in df.columns:
            # Intraday volatility using VWAP
            df['intraday_vol_vwap'] = abs(df['close'] - df['vwap']).rolling(10).std()
            df['vwap_volatility_ratio'] = df['intraday_vol_vwap'] / (df['close'].rolling(10).std() + 1e-8)
        
        # Microstructure momentum indicators
        if 'transactions' in df.columns and 'volume' in df.columns:
            # Trade size momentum
            trade_size = df['volume'] / (df['transactions'] + 1e-8)
            df['trade_size_momentum'] = trade_size.pct_change()
            df['trade_size_acceleration'] = df['trade_size_momentum'].diff()
            
            # Activity-adjusted price momentum
            df['activity_adj_momentum'] = df['close'].pct_change() * np.log1p(df['transactions'])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if df.index.dtype.kind == 'M':  # datetime index
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            # Market session features
            df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
            df['is_pre_market'] = ((df['hour'] >= 4) & (df['hour'] < 9)).astype(int)
            df['is_after_hours'] = ((df['hour'] >= 16) | (df['hour'] < 4)).astype(int)
            
            # Time since market open
            market_open_minutes = (df['hour'] - 9) * 60 + df['minute'] - 30
            df['minutes_since_open'] = np.where(df['is_market_open'], market_open_minutes, 0)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced real-time volatility features (Priority 3)"""
        # Historical volatility
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(390)  # Annualized
        
        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            np.log(df['high'] / df['low']).rolling(20).apply(lambda x: (x**2).sum())
        )
        
        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * (np.log(df['high'] / df['low']))**2 - 
            (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']))**2
        ).rolling(20).mean()
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
        df['vol_ratio_10_20'] = df['volatility_10'] / (df['volatility_20'] + 1e-8)
        
        # Real-time volatility enhancements using Polygon WebSocket fields
        if 'vwap' in df.columns:
            # VWAP-based volatility
            vwap_returns = df['vwap'].pct_change()
            for period in [5, 10, 20]:
                df[f'vwap_volatility_{period}'] = vwap_returns.rolling(period).std() * np.sqrt(390)
            
            # Cross-volatility between price and VWAP
            price_vwap_spread = (df['close'] - df['vwap']) / df['vwap']
            df['price_vwap_vol'] = price_vwap_spread.rolling(10).std()
        
        if 'transactions' in df.columns:
            # Transaction-weighted volatility
            transaction_weights = df['transactions'] / df['transactions'].rolling(20).sum()
            weighted_returns = df['close'].pct_change() * transaction_weights
            df['transaction_weighted_vol'] = weighted_returns.rolling(10).std() * np.sqrt(390)
            
            # Volatility per transaction
            df['vol_per_transaction'] = df['volatility_10'] / (np.log1p(df['transactions']) + 1e-8)
        
        if 'volume' in df.columns:
            # Volume-weighted volatility
            volume_weights = df['volume'] / df['volume'].rolling(20).sum()
            volume_weighted_returns = df['close'].pct_change() * volume_weights
            df['volume_weighted_vol'] = volume_weighted_returns.rolling(10).std() * np.sqrt(390)
            
            # Realized volatility using volume
            df['realized_vol'] = (df['close'].pct_change() * np.sqrt(df['volume'])).rolling(10).std()
        
        # Intraday volatility patterns
        if df.index.dtype.kind == 'M':  # datetime index
            # Time-of-day volatility
            hour_vol = df.groupby(df.index.hour)['close'].pct_change().rolling(5).std()
            df['hour_volatility'] = hour_vol.reindex(df.index, method='ffill')
            
            # Volatility clustering
            df['vol_clustering'] = df['volatility_5'].rolling(5).std()
        
        # Advanced volatility measures
        # Jump detection
        returns = df['close'].pct_change()
        vol_threshold = returns.rolling(20).std() * 3  # 3-sigma threshold
        df['jump_indicator'] = (abs(returns) > vol_threshold).astype(int)
        df['jump_intensity'] = df['jump_indicator'].rolling(10).sum()
        
        # Volatility regime detection
        short_vol = df['volatility_5']
        long_vol = df['volatility_20']
        df['vol_regime'] = np.where(short_vol > long_vol * 1.5, 1,  # High vol regime
                                   np.where(short_vol < long_vol * 0.5, -1, 0))  # Low vol regime
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced real-time momentum features (Priority 3)"""
        # Rate of Change
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Acceleration
        df['acceleration_5'] = df['roc_5'] - df['roc_5'].shift(1)
        df['acceleration_10'] = df['roc_10'] - df['roc_10'].shift(1)
        
        # Trend strength
        for period in [10, 20]:
            df[f'trend_strength_{period}'] = abs(df[f'roc_{period}']) / \
                                            (df[f'volatility_{period}'] + 1e-8)
        
        # Real-time momentum enhancements using Polygon WebSocket fields
        if 'vwap' in df.columns:
            # VWAP momentum
            for period in [5, 10, 20]:
                df[f'vwap_momentum_{period}'] = df['vwap'].pct_change(period)
            
            # Price vs VWAP momentum
            df['price_vwap_momentum'] = (df['close'] - df['vwap']) / df['vwap']
            df['price_vwap_momentum_change'] = df['price_vwap_momentum'].diff()
            
            # VWAP trend strength
            vwap_sma_5 = df['vwap'].rolling(5).mean()
            vwap_sma_20 = df['vwap'].rolling(20).mean()
            df['vwap_trend_strength'] = (vwap_sma_5 - vwap_sma_20) / vwap_sma_20
        
        if 'transactions' in df.columns:
            # Transaction momentum
            for period in [5, 10, 20]:
                df[f'transaction_momentum_{period}'] = df['transactions'].pct_change(period)
            
            # Transaction-weighted price momentum
            transaction_weights = df['transactions'] / df['transactions'].rolling(10).sum()
            df['transaction_weighted_momentum'] = (df['close'].pct_change() * transaction_weights).rolling(5).sum()
            
            # Activity-adjusted momentum
            df['activity_adjusted_momentum'] = df['momentum_10'] * np.log1p(df['transactions'])
            
            # Transaction acceleration
            df['transaction_acceleration'] = df['transactions'].diff().diff()
        
        if 'volume' in df.columns:
            # Volume momentum
            for period in [5, 10, 20]:
                df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
            
            # Volume-weighted momentum
            volume_weights = df['volume'] / df['volume'].rolling(10).sum()
            df['volume_weighted_momentum'] = (df['close'].pct_change() * volume_weights).rolling(5).sum()
            
            # Volume-price momentum divergence
            price_momentum_norm = (df['momentum_10'] - df['momentum_10'].rolling(20).mean()) / df['momentum_10'].rolling(20).std()
            volume_momentum_norm = (df['volume_momentum_10'] - df['volume_momentum_10'].rolling(20).mean()) / df['volume_momentum_10'].rolling(20).std()
            df['momentum_volume_divergence'] = price_momentum_norm - volume_momentum_norm
        
        if 'accumulated_volume' in df.columns:
            # Accumulated volume momentum
            df['accumulated_volume_momentum'] = df['accumulated_volume'].pct_change()
            df['accumulated_volume_acceleration'] = df['accumulated_volume_momentum'].diff()
            
            # Volume accumulation rate
            df['volume_accumulation_rate'] = df['accumulated_volume'] / (df.index.to_series().diff().dt.total_seconds() / 60).fillna(1)
        
        # Cross-asset momentum indicators
        if all(col in df.columns for col in ['vwap', 'transactions', 'volume']):
            # Composite momentum score
            momentum_components = [
                df['momentum_10'].fillna(0),
                df['vwap_momentum_10'].fillna(0),
                df['transaction_momentum_10'].fillna(0),
                df['volume_momentum_10'].fillna(0)
            ]
            df['composite_momentum'] = np.mean(momentum_components, axis=0)
            
            # Momentum consistency
            momentum_signs = [np.sign(comp) for comp in momentum_components]
            df['momentum_consistency'] = np.mean([np.sum(signs) / len(signs) for signs in zip(*momentum_signs)])
            
            # Market efficiency momentum
            price_efficiency = abs(df['close'] - df['vwap']) / df['vwap']
            df['efficiency_momentum'] = price_efficiency.rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0])
        
        # Advanced momentum patterns
        # Momentum persistence
        df['momentum_persistence'] = df['momentum_10'].rolling(5).apply(
            lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / (len(x) - 1)
        )
        
        # Momentum acceleration
        df['momentum_acceleration'] = df['momentum_10'].diff()
        df['momentum_jerk'] = df['momentum_acceleration'].diff()
        
        # Momentum regime detection
        momentum_ma = df['momentum_10'].rolling(20).mean()
        momentum_std = df['momentum_10'].rolling(20).std()
        df['momentum_regime'] = np.where(
            df['momentum_10'] > momentum_ma + momentum_std, 1,  # Strong positive momentum
            np.where(df['momentum_10'] < momentum_ma - momentum_std, -1, 0)  # Strong negative momentum
        )
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for period in [10, 20]:
            df[f'skewness_{period}'] = df['close'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['close'].rolling(period).kurt()
        
        # Percentile features
        for period in [10, 20]:
            df[f'percentile_rank_{period}'] = df['close'].rolling(period).rank(pct=True)
        
        # Z-score
        for period in [10, 20]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-8)
        
        # Autocorrelation
        for lag in [1, 5]:
            df[f'autocorr_lag_{lag}'] = df['close'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill (using modern pandas syntax)
        df = df.ffill().bfill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        return df
    
    async def create_target_variable(self, df: pd.DataFrame, 
                                   prediction_horizon: int = 1,
                                   threshold: float = 0.001) -> pd.Series:
        """Create target variable for classification"""
        # Calculate future returns
        future_returns = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Create binary target (1 for buy, 0 for sell/hold)
        target = (future_returns > threshold).astype(int)
        
        return target
    
    async def get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importance = abs(model.coef_[0])
            else:
                # Neural networks - use permutation importance
                return {}
            
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features)
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    async def select_features(self, df: pd.DataFrame, target: pd.Series, 
                            top_k: int = 30) -> List[str]:
        """Select top features using correlation and mutual information"""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import StandardScaler
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        X = df[feature_cols].fillna(0)
        y = target.fillna(0)
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        
        # Create feature scores
        feature_scores = dict(zip(feature_cols, mi_scores))
        
        # Sort and select top features
        sorted_features = sorted(feature_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        selected_features = [feat[0] for feat in sorted_features[:top_k]]
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)}")
        
        return selected_features
    
    async def get_feature_statistics(self, symbol: str) -> Dict:
        """Get statistics about engineered features"""
        if symbol not in self.feature_cache:
            return {"error": "No features cached for symbol"}
        
        df = self.feature_cache[symbol]
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        stats = {
            "total_features": len(feature_cols),
            "data_points": len(df),
            "feature_categories": {
                "price_features": len([col for col in feature_cols if 'price' in col]),
                "volume_features": len([col for col in feature_cols if 'volume' in col]),
                "technical_indicators": len([col for col in feature_cols 
                                           if any(indicator in col for indicator in 
                                                ['sma', 'ema', 'rsi', 'macd', 'bb'])]),
                "volatility_features": len([col for col in feature_cols if 'vol' in col]),
                "momentum_features": len([col for col in feature_cols 
                                        if any(mom in col for mom in ['roc', 'momentum'])]),
            },
            "missing_values": df[feature_cols].isnull().sum().sum(),
            "infinite_values": np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
        }
        
        return stats
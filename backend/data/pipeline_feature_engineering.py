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
        """Add market microstructure features"""
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price impact
        df['price_impact'] = abs(df['close'] - df['open']) / df['volume']
        
        # Tick direction
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        
        # Consecutive ticks
        df['consecutive_up'] = (df['tick_direction'] == 1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (df['tick_direction'] == -1).astype(int).groupby(
            (df['tick_direction'] != df['tick_direction'].shift()).cumsum()).cumsum()
        
        # Order flow imbalance proxy
        df['flow_imbalance'] = (df['close'] - (df['high'] + df['low']) / 2) / \
                              (df['high'] - df['low'] + 1e-8)
        
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
        """Add volatility features"""
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
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
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
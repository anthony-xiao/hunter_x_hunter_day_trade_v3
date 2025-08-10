import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Complete feature set for ML models"""
    technical_features: pd.DataFrame
    market_microstructure: pd.DataFrame
    sentiment_features: pd.DataFrame
    macro_features: pd.DataFrame
    cross_asset_features: pd.DataFrame
    engineered_features: pd.DataFrame
    feature_importance: Dict[str, float]
    feature_metadata: Dict[str, Any]

class FeatureEngineering:
    """Advanced feature engineering for algorithmic trading"""
    
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50, 100, 200]
        self.volatility_windows = [5, 10, 20, 30]
        self.momentum_periods = [3, 5, 10, 15, 20]
        self.mean_reversion_periods = [5, 10, 20]
        
        # Scalers for different feature types
        self.price_scaler = RobustScaler()
        self.volume_scaler = StandardScaler()
        self.technical_scaler = StandardScaler()
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        
        logger.info("Feature Engineering initialized")
    
    def calculate_required_lookback(self) -> int:
        """
        Calculate the maximum lookback period required by all technical indicators.
        This ensures we have sufficient historical data for feature engineering.
        
        Returns:
            int: Maximum lookback period in minutes (with safety buffer)
        """
        # Define all lookback periods used in feature engineering
        lookback_periods = {
            # Moving averages (SMA, EMA)
            'moving_averages': [5, 10, 20, 50, 100, 200],
            
            # Bollinger Bands
            'bollinger_bands': [20, 50],
            
            # RSI
            'rsi': [14, 21, 30],
            
            # MACD components
            'macd': [12, 26, 9],  # EMA12, EMA26, Signal line
            
            # Stochastic, Williams %R, ATR, CCI, MFI
            'oscillators': [14],
            
            # Volume features
            'volume_features': [5, 10, 20],
            
            # VWAP and microstructure
            'vwap_features': [5, 10, 20],
            
            # Volatility features
            'volatility': [5, 10, 20, 30],
            
            # Momentum features
            'momentum': [3, 5, 10, 15, 20],
            
            # Statistical features
            'statistical': [10, 20],
            
            # Mean reversion features
            'mean_reversion': [5, 10, 20],
        }
        
        # Find maximum lookback across all categories
        max_lookback = 0
        for category, periods in lookback_periods.items():
            category_max = max(periods)
            max_lookback = max(max_lookback, category_max)
            logger.debug(f"Category {category}: max lookback = {category_max}")
        
        # Add safety buffer (20% extra) to ensure all indicators have sufficient data
        safety_buffer = int(max_lookback * 0.2)
        total_lookback = max_lookback + safety_buffer
        
        logger.info(f"Calculated required lookback: {max_lookback} minutes + {safety_buffer} buffer = {total_lookback} minutes")
        
        return total_lookback
    
    async def engineer_features(self, 
                              symbol: str, 
                              start_date: datetime, 
                              end_date: datetime,
                              include_cross_asset: bool = True) -> FeatureSet:
        """Engineer comprehensive feature set for a symbol"""
        try:
            logger.info(f"Engineering features for {symbol} from {start_date} to {end_date}")
            
            # Get base market data
            market_data = await self._get_market_data(symbol, start_date, end_date)
            
            if market_data is None or len(market_data) < 200:
                logger.warning(f"Insufficient data for {symbol}")
                return self._get_empty_feature_set()
            
            # Engineer different feature categories
            technical_features = await self._engineer_technical_features(market_data)
            microstructure_features = await self._engineer_microstructure_features(market_data)
            sentiment_features = await self._engineer_sentiment_features(symbol, market_data)
            macro_features = await self._engineer_macro_features(market_data)
            
            # Cross-asset features (if requested)
            cross_asset_features = pd.DataFrame()
            if include_cross_asset:
                cross_asset_features = await self._engineer_cross_asset_features(symbol, market_data)
            
            # Advanced engineered features
            engineered_features = await self._engineer_advanced_features(
                market_data, technical_features, microstructure_features
            )
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(
                technical_features, microstructure_features, engineered_features
            )
            
            # Generate feature metadata
            feature_metadata = self._generate_feature_metadata(
                technical_features, microstructure_features, engineered_features
            )
            
            return FeatureSet(
                technical_features=technical_features,
                market_microstructure=microstructure_features,
                sentiment_features=sentiment_features,
                macro_features=macro_features,
                cross_asset_features=cross_asset_features,
                engineered_features=engineered_features,
                feature_importance=feature_importance,
                feature_metadata=feature_metadata
            )
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")
            return self._get_empty_feature_set()
    
    async def _get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Get market data for feature engineering"""
        try:
            with self.Session() as session:
                result = session.execute(text("""
                    SELECT 
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        vwap,
                        transactions
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
                
                data = result.fetchall()
                
                if not data:
                    return None
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions'
                ])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert to numeric
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _engineer_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive technical indicators"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Moving averages
            for period in self.lookback_periods:
                features[f'sma_{period}'] = talib.SMA(data['close'].values.astype(np.float64), timeperiod=period)
                features[f'ema_{period}'] = talib.EMA(data['close'].values.astype(np.float64), timeperiod=period)
                features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
                features[f'price_to_ema_{period}'] = data['close'] / features[f'ema_{period}']
            
            # Volatility indicators
            for window in self.volatility_windows:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()
                features[f'realized_vol_{window}'] = features['log_returns'].rolling(window).std() * np.sqrt(252)
            
            # Momentum indicators
            for period in self.momentum_periods:
                features[f'roc_{period}'] = talib.ROC(data['close'].values.astype(np.float64), timeperiod=period)
                features[f'momentum_{period}'] = talib.MOM(data['close'].values.astype(np.float64), timeperiod=period)
            
            # RSI with multiple periods
            for period in [14, 21, 30]:
                features[f'rsi_{period}'] = talib.RSI(data['close'].values.astype(np.float64), timeperiod=period)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(data['close'].values.astype(np.float64))
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values.astype(np.float64), timeperiod=period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                features[f'bb_position_{period}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # Williams %R
            features['williams_r'] = talib.WILLR(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            
            # Average True Range
            features['atr'] = talib.ATR(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            features['atr_pct'] = features['atr'] / data['close']
            
            # Commodity Channel Index
            features['cci'] = talib.CCI(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            
            # Money Flow Index
            features['mfi'] = talib.MFI(data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64))
            
            # On Balance Volume
            features['obv'] = talib.OBV(data['close'].values.astype(np.float64), data['volume'].values.astype(np.float64))
            features['obv_sma'] = features['obv'].rolling(20).mean()
            
            # Volume indicators
            features['volume_sma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma_20']
            
            # Price patterns
            features['doji'] = talib.CDLDOJI(data['open'].values.astype(np.float64), data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            features['hammer'] = talib.CDLHAMMER(data['open'].values.astype(np.float64), data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            features['engulfing'] = talib.CDLENGULFING(data['open'].values.astype(np.float64), data['high'].values.astype(np.float64), data['low'].values.astype(np.float64), data['close'].values.astype(np.float64))
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logger.error(f"Technical feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    async def _engineer_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer market microstructure features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Calculate accumulated_volume as cumulative sum of volume
            data = data.copy()
            data['accumulated_volume'] = data['volume'].cumsum()
            
            # Basic microstructure features
            features['spread_proxy'] = (data['high'] - data['low']) / data['close']
            features['price_impact'] = abs(data['close'] - data['open']) / (data['volume'] + 1e-8)
            
            # Tick direction and momentum
            features['tick_direction'] = np.sign(data['close'] - data['close'].shift(1))
            
            # Consecutive ticks
            features['consecutive_up'] = (features['tick_direction'] == 1).astype(int).groupby(
                (features['tick_direction'] != features['tick_direction'].shift()).cumsum()).cumsum()
            features['consecutive_down'] = (features['tick_direction'] == -1).astype(int).groupby(
                (features['tick_direction'] != features['tick_direction'].shift()).cumsum()).cumsum()
            
            # Order flow imbalance proxy
            features['flow_imbalance'] = (data['close'] - (data['high'] + data['low']) / 2) / \
                                        (data['high'] - data['low'] + 1e-8)
            
            # Enhanced microstructure features using VWAP
            if 'vwap' in data.columns:
                # VWAP-based features
                features['price_vwap_ratio'] = data['close'] / (data['vwap'] + 1e-8)
                features['vwap_deviation'] = (data['close'] - data['vwap']) / (data['vwap'] + 1e-8)
                features['vwap_momentum'] = data['vwap'].pct_change()
                
                # VWAP trend strength
                for period in [5, 10, 20]:
                    vwap_sma = data['vwap'].rolling(period).mean()
                    features[f'vwap_trend_{period}'] = (data['vwap'] - vwap_sma) / (vwap_sma + 1e-8)
            
            # Transaction-based microstructure indicators
            if 'transactions' in data.columns:
                # Transaction-based microstructure indicators
                features['avg_trade_size'] = data['volume'] / (data['transactions'] + 1e-8)
                features['trade_intensity'] = data['transactions'] / data['volume'].rolling(20).mean().fillna(1)
                
                # Transaction momentum and volatility
                features['transaction_momentum'] = data['transactions'].pct_change()
                features['transaction_volatility'] = data['transactions'].rolling(10).std() / (data['transactions'].rolling(10).mean() + 1e-8)
                
                # Market activity indicators
                features['high_frequency_ratio'] = data['transactions'] / (data['volume'] + 1e-8)  # Transactions per unit volume
                
                # Rolling transaction statistics
                for period in [5, 10, 20]:
                    features[f'transactions_ma_{period}'] = data['transactions'].rolling(period).mean()
                    features[f'transactions_ratio_{period}'] = data['transactions'] / (features[f'transactions_ma_{period}'] + 1e-8)
            
            # Accumulated volume features
            if 'accumulated_volume' in data.columns:
                # Accumulated volume features
                features['volume_acceleration'] = data['accumulated_volume'].diff().diff()  # Second derivative
                features['volume_momentum'] = data['accumulated_volume'].pct_change()
                
                # Volume distribution analysis
                features['volume_concentration'] = data['volume'] / (data['accumulated_volume'] + 1e-8)
                
                # Intraday volume patterns
                features['volume_profile'] = data['accumulated_volume'] / data['accumulated_volume'].rolling(20).max().fillna(1)
            
            # Advanced microstructure indicators combining multiple fields
            if all(col in data.columns for col in ['vwap', 'transactions', 'volume']):
                # Market efficiency indicators
                features['price_efficiency'] = abs(data['close'] - data['vwap']) / (data['transactions'] + 1e-8)
                features['liquidity_proxy'] = data['volume'] / (abs(data['close'] - data['vwap']) + 1e-8)
                
                # Order flow toxicity (Kyle's lambda proxy)
                price_impact_per_trade = abs(data['close'] - data['close'].shift(1)) / (data['transactions'] + 1e-8)
                features['order_flow_toxicity'] = price_impact_per_trade.rolling(10).mean()
                
                # Market depth proxy
                features['market_depth_proxy'] = data['volume'] / (features['spread_proxy'] + 1e-8)
            
            # Legacy features for backward compatibility
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_to_high'] = data['close'] / data['high']
            features['close_to_low'] = data['close'] / data['low']
            features['open_to_close'] = (data['close'] - data['open']) / data['open']
            features['high_to_close'] = (data['high'] - data['close']) / data['close']
            features['low_to_close'] = (data['close'] - data['low']) / data['close']
            # Custom Volume Price Trend calculation (VPT = previous_VPT + volume * ((close - previous_close) / previous_close))
            vpt = np.zeros(len(data))
            for i in range(1, len(data)):
                price_change_pct = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                vpt[i] = vpt[i-1] + data['volume'].iloc[i] * price_change_pct
            features['volume_price_trend'] = vpt
            
            # Price efficiency measures
            for window in [5, 10, 20]:
                features[f'price_efficiency_{window}'] = self._calculate_price_efficiency(data['close'], window)
            
            # Bid-ask spread features
            features['spread_ma'] = features['spread_proxy'].rolling(20).mean()
            
            # Volume clustering
            features['volume_clusters'] = self._identify_volume_clusters(data['volume'])
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logger.error(f"Microstructure feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_price_efficiency(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate price efficiency measure"""
        try:
            returns = prices.pct_change()
            rolling_returns = returns.rolling(window)
            
            # Calculate the ratio of actual price change to sum of absolute returns
            actual_change = prices.pct_change(window)
            sum_abs_returns = rolling_returns.apply(lambda x: x.abs().sum())
            
            efficiency = actual_change.abs() / sum_abs_returns
            return efficiency.fillna(0)
            
        except Exception as e:
            logger.error(f"Price efficiency calculation failed: {e}")
            return pd.Series(0, index=prices.index)
    
    def _identify_volume_clusters(self, volume: pd.Series) -> pd.Series:
        """Identify volume clustering patterns"""
        try:
            # Calculate rolling quantiles properly
            q25 = volume.rolling(50).quantile(0.25)
            q50 = volume.rolling(50).quantile(0.5)
            q75 = volume.rolling(50).quantile(0.75)
            
            clusters = pd.Series(1, index=volume.index)  # Default to cluster 1
            
            # Classify volumes based on quantiles
            clusters = pd.Series(1, index=volume.index)  # Low volume
            clusters[volume > q25] = 2  # Medium-low volume
            clusters[volume > q50] = 3  # Medium-high volume
            clusters[volume > q75] = 4  # High volume
            
            # Fill NaN values (first 49 observations) with default cluster
            clusters = clusters.fillna(1)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Volume clustering failed: {e}")
            return pd.Series(1, index=volume.index)
    
    async def _engineer_sentiment_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer sentiment-based features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price momentum sentiment
            for period in [5, 10, 20]:
                returns = data['close'].pct_change(period)
                features[f'momentum_sentiment_{period}'] = np.where(returns > 0, 1, -1)
            
            # Volume sentiment
            volume_ma = data['volume'].rolling(20).mean()
            features['volume_sentiment'] = np.where(data['volume'] > volume_ma, 1, -1)
            
            # Volatility sentiment
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            vol_ma = volatility.rolling(50).mean()
            features['volatility_sentiment'] = np.where(volatility > vol_ma, -1, 1)  # High vol = negative sentiment
            
            # Trend strength sentiment
            for period in [10, 20, 50]:
                sma = data['close'].rolling(period).mean()
                features[f'trend_sentiment_{period}'] = np.where(data['close'] > sma, 1, -1)
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Sentiment feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    async def _engineer_macro_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer macro-economic features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Time-based features
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Market session features
            features['is_market_open'] = ((data.index.hour >= 9) & (data.index.hour < 16)).astype(int)
            features['is_pre_market'] = ((data.index.hour >= 4) & (data.index.hour < 9)).astype(int)
            features['is_after_hours'] = ((data.index.hour >= 16) | (data.index.hour < 4)).astype(int)
            
            # Seasonal patterns
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            features['is_month_end'] = (data.index.day > 25).astype(int)
            features['is_quarter_end'] = ((data.index.month % 3 == 0) & (data.index.day > 25)).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Macro feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    async def _engineer_cross_asset_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer cross-asset correlation features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Get SPY data for market correlation
            spy_data = await self._get_market_data('SPY', data.index[0], data.index[-1])
            
            if spy_data is not None and len(spy_data) > 0:
                # Align data
                aligned_data = pd.concat([data['close'], spy_data['close']], axis=1, keys=[symbol, 'SPY'])
                aligned_data = aligned_data.fillna(method='ffill').dropna()
                
                if len(aligned_data) > 20:
                    # Rolling correlations
                    for window in [20, 50, 100]:
                        if len(aligned_data) >= window:
                            corr = aligned_data[symbol].rolling(window).corr(aligned_data['SPY'])
                            features[f'spy_correlation_{window}'] = corr
                    
                    # Beta calculation
                    returns_symbol = aligned_data[symbol].pct_change()
                    returns_spy = aligned_data['SPY'].pct_change()
                    
                    for window in [50, 100]:
                        if len(aligned_data) >= window:
                            covariance = returns_symbol.rolling(window).cov(returns_spy)
                            variance_spy = returns_spy.rolling(window).var()
                            beta = covariance / variance_spy
                            features[f'beta_{window}'] = beta
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            logger.error(f"Cross-asset feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    async def _engineer_advanced_features(self, 
                                        data: pd.DataFrame, 
                                        technical: pd.DataFrame, 
                                        microstructure: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced composite features"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Composite momentum score
            momentum_features = [col for col in technical.columns if 'roc_' in col or 'momentum_' in col]
            if momentum_features:
                features['composite_momentum'] = technical[momentum_features].mean(axis=1)
            
            # Composite volatility score
            volatility_features = [col for col in technical.columns if 'volatility_' in col]
            if volatility_features:
                features['composite_volatility'] = technical[volatility_features].mean(axis=1)
            
            # Composite trend strength
            trend_features = [col for col in technical.columns if 'price_to_' in col]
            if trend_features:
                features['composite_trend'] = technical[trend_features].mean(axis=1)
            
            # Risk-adjusted returns
            if 'returns' in technical.columns and 'volatility_20' in technical.columns:
                features['risk_adjusted_returns'] = technical['returns'] / (technical['volatility_20'] + 1e-8)
            
            # Volume-price divergence
            if 'volume_ratio' in technical.columns and 'returns' in technical.columns:
                features['volume_price_divergence'] = technical['volume_ratio'] * technical['returns']
            
            # Regime detection features
            if 'volatility_20' in technical.columns:
                vol_ma = technical['volatility_20'].rolling(100).mean()
                features['volatility_regime'] = np.where(technical['volatility_20'] > vol_ma, 1, 0)
            
            # Mean reversion signals
            for period in self.mean_reversion_periods:
                if f'price_to_sma_{period}' in technical.columns:
                    price_ratio = technical[f'price_to_sma_{period}']
                    features[f'mean_reversion_{period}'] = np.where(
                        (price_ratio > 1.02) | (price_ratio < 0.98), 1, 0
                    )
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Advanced feature engineering failed: {e}")
            return pd.DataFrame(index=data.index)
    
    async def _calculate_feature_importance(self, 
                                          technical: pd.DataFrame, 
                                          microstructure: pd.DataFrame, 
                                          engineered: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores"""
        try:
            # Combine all features
            all_features = pd.concat([technical, microstructure, engineered], axis=1)
            all_features = all_features.fillna(0)
            
            if len(all_features) == 0:
                return {}
            
            # Calculate correlation with future returns as proxy for importance
            returns = technical['returns'].shift(-1) if 'returns' in technical.columns else None
            
            if returns is None:
                return {col: 1.0 for col in all_features.columns}
            
            importance = {}
            for col in all_features.columns:
                try:
                    corr = all_features[col].corr(returns)
                    importance[col] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    importance[col] = 0.0
            
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def _generate_feature_metadata(self, 
                                 technical: pd.DataFrame, 
                                 microstructure: pd.DataFrame, 
                                 engineered: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata for features"""
        try:
            metadata = {
                'total_features': len(technical.columns) + len(microstructure.columns) + len(engineered.columns),
                'technical_features': len(technical.columns),
                'microstructure_features': len(microstructure.columns),
                'engineered_features': len(engineered.columns),
                'feature_categories': {
                    'momentum': [col for col in technical.columns if any(x in col for x in ['roc', 'momentum', 'macd'])],
                    'volatility': [col for col in technical.columns if 'volatility' in col or 'atr' in col],
                    'trend': [col for col in technical.columns if any(x in col for x in ['sma', 'ema', 'bb'])],
                    'oscillators': [col for col in technical.columns if any(x in col for x in ['rsi', 'stoch', 'williams'])],
                    'volume': [col for col in technical.columns if 'volume' in col or 'obv' in col],
                    'microstructure': list(microstructure.columns),
                    'composite': list(engineered.columns)
                },
                'generation_time': datetime.now().isoformat()
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            return {}
    
    def _get_empty_feature_set(self) -> FeatureSet:
        """Return empty feature set for error cases"""
        empty_df = pd.DataFrame()
        return FeatureSet(
            technical_features=empty_df,
            market_microstructure=empty_df,
            sentiment_features=empty_df,
            macro_features=empty_df,
            cross_asset_features=empty_df,
            engineered_features=empty_df,
            feature_importance={},
            feature_metadata={}
        )
    
    async def get_feature_statistics(self, features: FeatureSet) -> Dict[str, Any]:
        """Get comprehensive feature statistics"""
        try:
            all_features = pd.concat([
                features.technical_features,
                features.market_microstructure,
                features.sentiment_features,
                features.macro_features,
                features.cross_asset_features,
                features.engineered_features
            ], axis=1)
            
            stats = {
                'total_features': len(all_features.columns),
                'total_observations': len(all_features),
                'missing_values': all_features.isnull().sum().to_dict(),
                'feature_correlations': all_features.corr().to_dict(),
                'feature_distributions': {
                    col: {
                        'mean': float(all_features[col].mean()),
                        'std': float(all_features[col].std()),
                        'min': float(all_features[col].min()),
                        'max': float(all_features[col].max()),
                        'skewness': float(all_features[col].skew()),
                        'kurtosis': float(all_features[col].kurtosis())
                    }
                    for col in all_features.columns
                },
                'top_features_by_importance': dict(
                    sorted(features.feature_importance.items(), 
                          key=lambda x: x[1], reverse=True)[:20]
                )
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Feature statistics calculation failed: {e}")
            return {}
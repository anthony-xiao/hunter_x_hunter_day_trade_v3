import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .ml_feature_engineering import FeatureEngineering, FeatureSet

logger = logging.getLogger(__name__)

@dataclass
class UniversalFeatureSet:
    """Universal feature set for multi-symbol ML models"""
    symbol_features: Dict[str, FeatureSet]  # Individual symbol features
    cross_symbol_features: pd.DataFrame  # Cross-symbol correlation features
    market_regime_features: pd.DataFrame  # Market regime indicators
    sector_features: pd.DataFrame  # Sector-based features
    universal_embeddings: pd.DataFrame  # Symbol embeddings for neural networks
    feature_importance: Dict[str, float]  # Universal feature importance
    feature_metadata: Dict[str, Any]  # Universal feature metadata
    symbol_mappings: Dict[str, int]  # Symbol to ID mappings

class UniversalFeatureEngineering(FeatureEngineering):
    """Advanced universal feature engineering for multi-symbol algorithmic trading"""
    
    def __init__(self, supabase_client=None):
        super().__init__(supabase_client)
        
        # Universal feature engineering parameters
        self.correlation_windows = [5, 10, 20, 50]
        self.regime_detection_window = 100
        self.sector_correlation_threshold = 0.7
        
        # Symbol clustering for sector analysis
        self.symbol_clusterer = KMeans(n_clusters=5, random_state=42)
        
        # Cross-symbol scalers
        self.cross_symbol_scaler = StandardScaler()
        self.regime_scaler = RobustScaler()
        
        logger.info("Universal Feature Engineering initialized")
    
    async def engineer_universal_features(self, 
                                        symbols: List[str], 
                                        start_date: datetime, 
                                        end_date: datetime,
                                        include_cross_asset: bool = True,
                                        training_mode: bool = False) -> UniversalFeatureSet:
        """Engineer comprehensive universal feature set for multiple symbols
        
        Args:
            symbols: List of stock symbols to engineer features for
            start_date: Start date for feature engineering
            end_date: End date for feature engineering
            include_cross_asset: Whether to include cross-asset features
            training_mode: If True, only generate features for timestamps with market data
        """
        try:
            logger.info(f"Engineering universal features for {len(symbols)} symbols: {symbols}")
            logger.info(f"Date range: {start_date} to {end_date} (training_mode={training_mode})")
            
            # Create symbol mappings for embeddings
            symbol_mappings = {symbol: idx for idx, symbol in enumerate(symbols)}
            
            # Engineer individual symbol features concurrently
            symbol_features = await self._engineer_individual_symbol_features(
                symbols, start_date, end_date, include_cross_asset, training_mode
            )
            
            if not symbol_features:
                logger.error("No symbol features generated")
                return self._get_empty_universal_feature_set(symbols)
            
            # Filter symbols to only include those with actual features
            available_symbols = [s for s in symbols if s in symbol_features]
            logger.info(f"Available symbols for feature engineering: {available_symbols} (out of {len(symbols)} requested)")
            
            # Engineer cross-symbol features
            cross_symbol_features = await self._engineer_cross_symbol_features(
                symbol_features, available_symbols, start_date, end_date
            )
            
            # Engineer market regime features
            market_regime_features = await self._engineer_market_regime_features(
                symbol_features, available_symbols
            )
            
            # Engineer sector-based features
            sector_features = await self._engineer_sector_features(
                symbol_features, available_symbols
            )
            
            # Create universal embeddings
            universal_embeddings = await self._create_universal_embeddings(
                symbol_features, symbols, symbol_mappings
            )
            
            # Calculate universal feature importance
            feature_importance = await self._calculate_universal_feature_importance(
                symbol_features, cross_symbol_features, market_regime_features
            )
            
            # Generate universal feature metadata
            feature_metadata = self._generate_universal_feature_metadata(
                symbol_features, cross_symbol_features, market_regime_features, symbols
            )
            
            logger.info(f"Successfully engineered universal features for {len(symbols)} symbols")
            
            return UniversalFeatureSet(
                symbol_features=symbol_features,
                cross_symbol_features=cross_symbol_features,
                market_regime_features=market_regime_features,
                sector_features=sector_features,
                universal_embeddings=universal_embeddings,
                feature_importance=feature_importance,
                feature_metadata=feature_metadata,
                symbol_mappings=symbol_mappings
            )
            
        except Exception as e:
            logger.error(f"Universal feature engineering failed: {e}")
            return self._get_empty_universal_feature_set(symbols)
    
    async def _engineer_individual_symbol_features(self, 
                                                 symbols: List[str], 
                                                 start_date: datetime, 
                                                 end_date: datetime,
                                                 include_cross_asset: bool,
                                                 training_mode: bool) -> Dict[str, FeatureSet]:
        """Engineer features for individual symbols concurrently"""
        try:
            logger.info(f"Engineering individual features for {len(symbols)} symbols")
            
            # Create tasks for concurrent feature engineering
            tasks = []
            for symbol in symbols:
                task = self.engineer_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    include_cross_asset=include_cross_asset,
                    training_mode=training_mode
                )
                tasks.append((symbol, task))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results
            symbol_features = {}
            for i, (symbol, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to engineer features for {symbol}: {result}")
                    continue
                
                if result and hasattr(result, 'technical_features') and len(result.technical_features) > 0:
                    symbol_features[symbol] = result
                    logger.info(f"Engineered {len(result.technical_features)} feature records for {symbol}")
                else:
                    logger.warning(f"No features generated for {symbol}")
            
            logger.info(f"Successfully engineered features for {len(symbol_features)} symbols")
            return symbol_features
            
        except Exception as e:
            logger.error(f"Failed to engineer individual symbol features: {e}")
            return {}
    
    async def _engineer_cross_symbol_features(self, 
                                            symbol_features: Dict[str, FeatureSet], 
                                            symbols: List[str],
                                            start_date: datetime,
                                            end_date: datetime) -> pd.DataFrame:
        """Engineer cross-symbol correlation and interaction features"""
        try:
            # symbols parameter now contains only symbols with actual features
            available_symbols = symbols
            logger.info(f"Engineering cross-symbol features for {len(available_symbols)} available symbols")
            logger.info(f"Available symbols: {available_symbols}")
            
            if len(available_symbols) < 2:
                logger.warning("Insufficient symbols with features - generating fallback cross-symbol features")
                logger.info(f"CROSS_SYMBOL_DEBUG: available_symbols={available_symbols}, symbol_features keys={list(symbol_features.keys())}")
                
                # Create fallback features for single-symbol processing
                if available_symbols:
                    symbol = available_symbols[0]
                    features = symbol_features[symbol]
                    logger.info(f"CROSS_SYMBOL_DEBUG: Found features for {symbol}, type={type(features)}")
                    
                    if hasattr(features, 'technical_features'):
                        logger.info(f"CROSS_SYMBOL_DEBUG: {symbol} has technical_features, type={type(features.technical_features)}")
                        if features.technical_features is not None:
                            logger.info(f"CROSS_SYMBOL_DEBUG: {symbol} technical_features columns={list(features.technical_features.columns)}")
                            
                            if 'close' in features.technical_features.columns:
                                close_prices = features.technical_features['close']
                                logger.info(f"CROSS_SYMBOL_DEBUG: {symbol} close_prices length={len(close_prices)}, empty={close_prices.empty}")
                                
                                fallback_features = pd.DataFrame(index=close_prices.index)
                                logger.info(f"CROSS_SYMBOL_DEBUG: Created fallback_features with {len(fallback_features)} rows")
                                
                                # Create dummy cross-symbol features with neutral values
                                feature_count = 0
                                for window in self.correlation_windows:
                                    fallback_features[f'corr_{symbol}_market_{window}'] = 0.5  # Neutral correlation
                                    fallback_features[f'beta_{symbol}_market_{window}'] = 1.0   # Market beta
                                    fallback_features[f'relative_strength_{symbol}_market_{window}'] = 1.0  # Neutral relative strength
                                    feature_count += 3
                                    
                                # Market dispersion features (3 windows: 5, 10, 20)
                                for window in [5, 10, 20]:
                                    fallback_features[f'market_dispersion_{window}'] = close_prices.pct_change().rolling(window).std().fillna(0)
                                    feature_count += 1
                                    
                                logger.info(f'CROSS_SYMBOL_DEBUG: Expected 15 features: {len(self.correlation_windows)} correlation windows * 3 features + 3 dispersion features = {len(self.correlation_windows) * 3 + 3}')
                                    
                                logger.info(f"CROSS_SYMBOL_DEBUG: Generated {feature_count} individual features, total columns={len(fallback_features.columns)}")
                                logger.info(f"CROSS_SYMBOL_DEBUG: Fallback feature columns: {list(fallback_features.columns)}")
                                
                                result = fallback_features.fillna(0)
                                logger.info(f"CROSS_SYMBOL_DEBUG: Returning fallback features with shape {result.shape}")
                                return result
                            else:
                                logger.warning(f"CROSS_SYMBOL_DEBUG: {symbol} technical_features missing 'close' column")
                        else:
                            logger.warning(f"CROSS_SYMBOL_DEBUG: {symbol} technical_features is None")
                    else:
                        logger.warning(f"CROSS_SYMBOL_DEBUG: {symbol} features object has no technical_features attribute")
                else:
                    logger.warning(f"CROSS_SYMBOL_DEBUG: No available symbols with features")
                
                logger.warning(f"CROSS_SYMBOL_DEBUG: Returning empty DataFrame")
                return pd.DataFrame()
            
            # Collect price data for all symbols
            price_data = {}
            for symbol, features in symbol_features.items():
                if hasattr(features, 'technical_features') and 'close' in features.technical_features.columns:
                    price_data[symbol] = features.technical_features['close']
            
            if len(price_data) < 2:
                logger.warning("Insufficient price data for cross-symbol features")
                return pd.DataFrame()
            
            # Create aligned DataFrame
            price_df = pd.DataFrame(price_data)
            
            # Debug logging to understand data shapes
            logger.info(f"Price data shapes before alignment:")
            for symbol, data in price_data.items():
                logger.info(f"  {symbol}: {len(data)} rows, NaN count: {data.isna().sum()}")
            
            logger.info(f"Combined price_df shape: {price_df.shape}")
            logger.info(f"NaN counts per column: {price_df.isna().sum().to_dict()}")
            
            # Don't drop all NaN rows - instead handle missing data per symbol pair
            # Only check if we have any data at all
            if price_df.empty or len(price_df.columns) == 0 or all(price_df[col].isna().all() for col in price_df.columns):
                logger.warning("No price data available for any symbols")
                return pd.DataFrame()
            
            # Create cross_features DataFrame with the full price_df index
            cross_features = pd.DataFrame(index=price_df.index)
            logger.info(f"Initialized cross_features with {len(cross_features)} rows")
            
            # Rolling correlations between symbols
            for window in self.correlation_windows:
                for i, symbol1 in enumerate(available_symbols):
                    for j, symbol2 in enumerate(available_symbols[i+1:], i+1):
                        if symbol1 in price_df.columns and symbol2 in price_df.columns:
                            corr_col = f'corr_{symbol1}_{symbol2}_{window}'
                            
                            # Create a subset with only these two symbols and drop rows where both are NaN
                            pair_data = price_df[[symbol1, symbol2]].dropna(how='all')
                            
                            logger.info(f"Correlation calculation for {symbol1}-{symbol2}:")
                            logger.info(f"  Original data: {len(price_df)} rows")
                            logger.info(f"  Pair data after dropping rows with both NaN: {len(pair_data)} rows")
                            logger.info(f"  Window size: {window}")
                            
                            if len(pair_data) >= window:
                                # Calculate correlation on the pair data
                                correlation = pair_data[symbol1].rolling(window, min_periods=window//2).corr(
                                    pair_data[symbol2]
                                )
                                # Reindex to match the original price_df index
                                cross_features[corr_col] = correlation.reindex(price_df.index)
                                logger.info(f"  Generated {correlation.notna().sum()} valid correlation values")
                            else:
                                logger.warning(f"  Insufficient overlapping data between {symbol1} and {symbol2} for rolling correlation calculations")
                                logger.warning(f"  Need at least {window} rows, but only have {len(pair_data)} rows")
                                cross_features[corr_col] = np.nan
            
            # Market beta calculations (proper beta formula: covariance/variance)
            # Calculate market average using only available data at each timestamp
            market_avg = price_df.mean(axis=1, skipna=True)
            market_returns = market_avg.pct_change()
            
            logger.info(f"Market returns calculated: {market_returns.notna().sum()} valid values out of {len(market_returns)}")
            
            for symbol in available_symbols:
                if symbol in price_df.columns:
                    symbol_returns = price_df[symbol].pct_change()
                    
                    # Create aligned data for this symbol and market
                    beta_data = pd.DataFrame({
                        'symbol': symbol_returns,
                        'market': market_returns
                    }).dropna()
                    
                    logger.info(f"Beta calculation for {symbol}: {len(beta_data)} aligned return pairs")
                    
                    for window in self.correlation_windows:
                        beta_col = f'beta_{symbol}_market_{window}'
                        
                        if len(beta_data) >= window:
                            # Calculate proper beta: covariance(symbol, market) / variance(market)
                            covariance = beta_data['symbol'].rolling(window, min_periods=window//2).cov(beta_data['market'])
                            market_variance = beta_data['market'].rolling(window, min_periods=window//2).var()
                            
                            # Avoid division by zero and handle edge cases
                            beta_values = np.where(
                                (market_variance > 1e-8) & (np.isfinite(covariance)) & (np.isfinite(market_variance)),
                                covariance / market_variance,
                                np.nan
                            )
                            
                            # Create series with beta_data index and reindex to price_df
                            beta_series = pd.Series(beta_values, index=beta_data.index)
                            cross_features[beta_col] = beta_series.reindex(price_df.index)
                            
                            logger.info(f"  {beta_col}: {pd.Series(beta_values).notna().sum()} valid beta values")
                        else:
                            logger.warning(f"  Insufficient data for {beta_col}: need {window}, have {len(beta_data)}")
                            cross_features[beta_col] = np.nan
            
            # Relative strength between symbols
            for i, symbol1 in enumerate(available_symbols):
                for j, symbol2 in enumerate(available_symbols[i+1:], i+1):
                    if symbol1 in price_df.columns and symbol2 in price_df.columns:
                        rs_col = f'relative_strength_{symbol1}_{symbol2}'
                        
                        # Create pair data for relative strength calculation
                        rs_pair_data = price_df[[symbol1, symbol2]].dropna(how='any')
                        
                        logger.info(f"Relative strength for {symbol1}-{symbol2}: {len(rs_pair_data)} valid pairs")
                        
                        if len(rs_pair_data) > 0:
                            # Safe division to avoid division by zero
                            denominator = rs_pair_data[symbol2].replace(0, np.nan)
                            rs_values = rs_pair_data[symbol1] / denominator
                            cross_features[rs_col] = rs_values.reindex(price_df.index)
                            
                            # Rolling relative strength momentum
                            for window in [5, 10, 20]:
                                rs_mom_col = f'rs_momentum_{symbol1}_{symbol2}_{window}'
                                if len(rs_pair_data) >= window:
                                    momentum = rs_values.pct_change(window)
                                    cross_features[rs_mom_col] = momentum.reindex(price_df.index)
                                else:
                                    cross_features[rs_mom_col] = np.nan
                        else:
                            cross_features[rs_col] = np.nan
                            for window in [5, 10, 20]:
                                rs_mom_col = f'rs_momentum_{symbol1}_{symbol2}_{window}'
                                cross_features[rs_mom_col] = np.nan
            
            # Market dispersion (volatility of cross-symbol returns)
            returns_df = price_df.pct_change()
            
            logger.info(f"Market dispersion calculation: returns_df shape {returns_df.shape}")
            logger.info(f"Returns NaN counts: {returns_df.isna().sum().to_dict()}")
            
            for window in [5, 10, 20]:
                dispersion_col = f'market_dispersion_{window}'
                # Calculate standard deviation across symbols at each timestamp (skipna=True)
                cross_symbol_std = returns_df.std(axis=1, skipna=True)
                # Then calculate rolling mean of the dispersion
                dispersion = cross_symbol_std.rolling(window, min_periods=window//2).mean()
                cross_features[dispersion_col] = dispersion
                
                logger.info(f"  {dispersion_col}: {dispersion.notna().sum()} valid values")
            
            # Data validation and cleaning
            # Replace infinite values with NaN
            cross_features = cross_features.replace([np.inf, -np.inf], np.nan)
            
            # Cap extremely large values (beyond 3 standard deviations)
            for col in cross_features.columns:
                if cross_features[col].dtype in ['float64', 'float32']:
                    mean_val = cross_features[col].mean()
                    std_val = cross_features[col].std()
                    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                        upper_bound = mean_val + 3 * std_val
                        lower_bound = mean_val - 3 * std_val
                        cross_features[col] = cross_features[col].clip(lower_bound, upper_bound)
            
            # Fill NaN values with forward fill, then backward fill, then 0
            cross_features = cross_features.ffill().bfill().fillna(0)
            
            # Final validation - ensure no infinite or extremely large values remain
            cross_features = cross_features.replace([np.inf, -np.inf], 0)
            cross_features = cross_features.clip(-1e6, 1e6)  # Cap at reasonable bounds
            
            logger.info(f"Generated {len(cross_features.columns)} cross-symbol features")
            return cross_features
            
        except Exception as e:
            logger.error(f"Failed to engineer cross-symbol features: {e}")
            return pd.DataFrame()
    
    async def _engineer_market_regime_features(self, 
                                             symbol_features: Dict[str, FeatureSet], 
                                             symbols: List[str]) -> pd.DataFrame:
                                             
        """Engineer market regime detection features"""
        try:
            logger.info(f"MARKET_REGIME_DEBUG: Engineering market regime features for {len(symbols)} symbols")
            logger.info(f"MARKET_REGIME_DEBUG: symbols={symbols}, symbol_features keys={list(symbol_features.keys())}")
            
            # Collect volatility data for regime detection
            volatility_data = {}
            for symbol, features in symbol_features.items():
                logger.info(f"MARKET_REGIME_DEBUG: Processing symbol {symbol}, type={type(features)}")
                if hasattr(features, 'technical_features'):
                    tech_features = features.technical_features
                    logger.info(f"MARKET_REGIME_DEBUG: {symbol} has technical_features, type={type(tech_features)}")
                    if tech_features is not None and 'close' in tech_features.columns:
                        logger.info(f"MARKET_REGIME_DEBUG: {symbol} has close column, length={len(tech_features['close'])}")
                        returns = tech_features['close'].pct_change()
                        volatility = returns.rolling(20).std()
                        volatility_data[symbol] = volatility
                        logger.info(f"MARKET_REGIME_DEBUG: {symbol} volatility calculated, length={len(volatility)}, NaN count={volatility.isnull().sum()}")
                    else:
                        logger.warning(f"MARKET_REGIME_DEBUG: {symbol} missing close column or tech_features is None")
                else:
                    logger.warning(f"MARKET_REGIME_DEBUG: {symbol} has no technical_features attribute")
            
            logger.info(f"MARKET_REGIME_DEBUG: Collected volatility data for {len(volatility_data)} symbols")
            
            if not volatility_data:
                logger.warning("MARKET_REGIME_DEBUG: No volatility data for regime detection")
                return pd.DataFrame()
            
            # Create aligned DataFrame
            vol_df = pd.DataFrame(volatility_data)
            logger.info(f"MARKET_REGIME_DEBUG: Created vol_df with shape {vol_df.shape}, columns={list(vol_df.columns)}")
            
            vol_df = vol_df.dropna(how='all')  # Only drop rows where ALL symbols have NaN values
            logger.info(f"MARKET_REGIME_DEBUG: After dropna, vol_df shape {vol_df.shape}")
            
            if len(vol_df) == 0:
                logger.warning("MARKET_REGIME_DEBUG: vol_df is empty after dropna")
                return pd.DataFrame()
            
            regime_features = pd.DataFrame(index=vol_df.index)
            logger.info(f"MARKET_REGIME_DEBUG: Created regime_features with {len(regime_features)} rows")
            
            # Market-wide volatility regime
            market_vol = vol_df.mean(axis=1)
            regime_features['market_volatility'] = market_vol
            
            # Volatility regime classification (low, medium, high)
            # Use smaller window if insufficient data
            effective_window = min(self.regime_detection_window, len(market_vol) // 2, 50)
            if effective_window >= 10:  # Minimum window for meaningful quantiles
                # Calculate quantiles separately to avoid 'must be real number, not list' error
                q33 = market_vol.rolling(effective_window).quantile(0.33)
                q67 = market_vol.rolling(effective_window).quantile(0.67)
                
                if not q33.empty and not q67.empty:
                    # Use the calculated quantiles directly
                    
                    regime_features['vol_regime_low'] = (market_vol <= q33).astype(int)
                    regime_features['vol_regime_high'] = (market_vol >= q67).astype(int)
                    regime_features['vol_regime_medium'] = ((market_vol > q33) & (market_vol < q67)).astype(int)
                else:
                    # Fallback to simple classification
                    median_vol = market_vol.median()
                    regime_features['vol_regime_low'] = (market_vol <= median_vol * 0.8).astype(int)
                    regime_features['vol_regime_high'] = (market_vol >= median_vol * 1.2).astype(int)
                    regime_features['vol_regime_medium'] = ((market_vol > median_vol * 0.8) & (market_vol < median_vol * 1.2)).astype(int)
            else:
                # Insufficient data - use simple binary classification
                median_vol = market_vol.median()
                regime_features['vol_regime_low'] = (market_vol <= median_vol).astype(int)
                regime_features['vol_regime_high'] = (market_vol > median_vol).astype(int)
                regime_features['vol_regime_medium'] = 0
            
            # Volatility trend with error handling
            def safe_linregress(x):
                try:
                    if len(x) >= 3 and x.notna().sum() > 0:
                        x_clean = x.dropna()
                        if len(x_clean) >= 3:
                            slope, _, _, _, _ = stats.linregress(range(len(x_clean)), x_clean)
                            return slope if np.isfinite(slope) else 0
                except:
                    pass
                return 0
            
            trend_window = min(20, len(market_vol) // 3, len(market_vol))
            if trend_window >= 3:
                regime_features['vol_trend'] = market_vol.rolling(trend_window).apply(safe_linregress)
            else:
                regime_features['vol_trend'] = 0
            
            # Cross-symbol volatility correlation with error handling
            corr_window = min(50, len(vol_df) // 2, len(vol_df))
            if corr_window >= 5 and vol_df.shape[1] > 1:
                try:
                    vol_corr = vol_df.rolling(corr_window).corr()
                    if not vol_corr.empty:
                        # Calculate mean correlation excluding self-correlations
                        corr_means = []
                        for idx in vol_corr.index.get_level_values(0).unique():
                            corr_matrix = vol_corr.loc[idx]
                            if isinstance(corr_matrix, pd.DataFrame) and corr_matrix.shape[0] > 1:
                                # Get upper triangle excluding diagonal
                                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                                upper_triangle = corr_matrix.where(mask)
                                mean_corr = upper_triangle.stack().mean()
                                corr_means.append(mean_corr if pd.notna(mean_corr) else 0)
                            else:
                                corr_means.append(0)
                        regime_features['vol_correlation'] = pd.Series(corr_means, index=vol_df.index[:len(corr_means)])
                        # Fill remaining values
                        if len(corr_means) < len(vol_df):
                            regime_features['vol_correlation'] = regime_features['vol_correlation'].reindex(vol_df.index).fillna(0)
                    else:
                        regime_features['vol_correlation'] = 0
                except Exception as e:
                    logger.warning(f"Error calculating volatility correlation: {e}")
                    regime_features['vol_correlation'] = 0
            else:
                regime_features['vol_correlation'] = 0
            
            # Fill NaN values
            regime_features = regime_features.ffill().fillna(0)
            
            logger.info(f"MARKET_REGIME_DEBUG: Generated {len(regime_features.columns)} market regime features")
            logger.info(f"MARKET_REGIME_DEBUG: Final regime feature columns: {list(regime_features.columns)}")
            logger.info(f"MARKET_REGIME_DEBUG: Final regime features shape: {regime_features.shape}")
            return regime_features
            
        except Exception as e:
            logger.error(f"MARKET_REGIME_DEBUG: Failed to engineer market regime features: {e}")
            logger.exception("MARKET_REGIME_DEBUG: Full exception traceback:")
            return pd.DataFrame()
    
    async def _engineer_sector_features(self, 
                                       symbol_features: Dict[str, FeatureSet], 
                                       symbols: List[str]) -> pd.DataFrame:
        """Engineer sector-based features using symbol clustering"""
        try:
            logger.info(f"Engineering sector features for {len(symbols)} symbols")
            
            # For now, create basic sector features
            # In a real implementation, you would use actual sector classifications
            sector_features = pd.DataFrame()
            
            # Create dummy sector features based on symbol characteristics
            for symbol in symbols:
                logger.info(f"SECTOR_DEBUG: Processing symbol {symbol}")
                
                if symbol in symbol_features:
                    features = symbol_features[symbol]
                    logger.info(f"SECTOR_DEBUG: Found features for {symbol}")
                    
                    if hasattr(features, 'technical_features') and len(features.technical_features) > 0:
                        logger.info(f"SECTOR_DEBUG: {symbol} has {len(features.technical_features)} technical features with {len(features.technical_features.columns)} columns")
                        
                        if sector_features.empty:
                            sector_features = pd.DataFrame(index=features.technical_features.index)
                            logger.info(f"SECTOR_DEBUG: Initialized sector_features with {len(sector_features)} rows from {symbol}")
                        
                        # Add sector-specific features based on actual data
                        close_prices = features.technical_features.get('close', pd.Series())
                        logger.info(f"SECTOR_DEBUG: {symbol} close_prices - empty: {close_prices.empty}, length: {len(close_prices)}")
                        
                        if not close_prices.empty:
                            logger.info(f"SECTOR_DEBUG: {symbol} close_prices range: {close_prices.min():.4f} to {close_prices.max():.4f}")
                            logger.info(f"SECTOR_DEBUG: {symbol} close_prices first 5 values: {close_prices.head().tolist()}")
                            logger.info(f"SECTOR_DEBUG: {symbol} close_prices last 5 values: {close_prices.tail().tolist()}")
                            
                            # Calculate momentum as 20-day price change percentage
                            logger.info(f"SECTOR_DEBUG: Calculating 20-day momentum for {symbol}")
                            momentum_raw = close_prices.pct_change(20)
                            momentum_nan_count = momentum_raw.isnull().sum()
                            logger.info(f"SECTOR_DEBUG: {symbol} momentum - raw NaN count: {momentum_nan_count} out of {len(momentum_raw)} values")
                            
                            if momentum_nan_count > 0:
                                logger.warning(f"SECTOR_DEBUG: {symbol} momentum has {momentum_nan_count} NaN values before fillna")
                                # Log first few NaN positions
                                nan_positions = momentum_raw.isnull()
                                first_nan_indices = nan_positions[nan_positions].head(10).index.tolist()
                                logger.warning(f"SECTOR_DEBUG: {symbol} first 10 NaN positions in momentum: {first_nan_indices}")
                                
                                # Check if we have enough data for 20-day calculation
                                if len(close_prices) < 20:
                                    logger.warning(f"SECTOR_DEBUG: {symbol} insufficient data for 20-day momentum: only {len(close_prices)} data points")
                                else:
                                    logger.info(f"SECTOR_DEBUG: {symbol} has sufficient data ({len(close_prices)} points) but still getting NaNs")
                            
                            momentum = momentum_raw.fillna(0)
                            sector_features[f'sector_{symbol}_momentum'] = momentum
                            logger.info(f"SECTOR_DEBUG: {symbol} momentum after fillna - NaN count: {momentum.isnull().sum()}")
                            
                            # Calculate volatility as 20-day rolling standard deviation of returns
                            logger.info(f"SECTOR_DEBUG: Calculating volatility for {symbol}")
                            returns = close_prices.pct_change()
                            returns_nan_count = returns.isnull().sum()
                            logger.info(f"SECTOR_DEBUG: {symbol} returns - NaN count: {returns_nan_count} out of {len(returns)} values")
                            
                            returns_filled = returns.fillna(0)
                            volatility_raw = returns_filled.rolling(20).std()
                            volatility_nan_count = volatility_raw.isnull().sum()
                            logger.info(f"SECTOR_DEBUG: {symbol} volatility - raw NaN count: {volatility_nan_count} out of {len(volatility_raw)} values")
                            
                            if volatility_nan_count > 0:
                                logger.warning(f"SECTOR_DEBUG: {symbol} volatility has {volatility_nan_count} NaN values before fillna")
                                # Log first few NaN positions
                                nan_positions = volatility_raw.isnull()
                                first_nan_indices = nan_positions[nan_positions].head(10).index.tolist()
                                logger.warning(f"SECTOR_DEBUG: {symbol} first 10 NaN positions in volatility: {first_nan_indices}")
                                
                                # Check if we have enough data for 20-day rolling calculation
                                if len(returns_filled) < 20:
                                    logger.warning(f"SECTOR_DEBUG: {symbol} insufficient data for 20-day volatility: only {len(returns_filled)} data points")
                                else:
                                    logger.info(f"SECTOR_DEBUG: {symbol} has sufficient data ({len(returns_filled)} points) but still getting NaNs")
                            
                            volatility = volatility_raw.fillna(0)
                            sector_features[f'sector_{symbol}_volatility'] = volatility
                            logger.info(f"SECTOR_DEBUG: {symbol} volatility after fillna - NaN count: {volatility.isnull().sum()}")
                            
                        else:
                            logger.warning(f"SECTOR_DEBUG: {symbol} has empty close_prices, using fallback zeros")
                            # Fallback to zeros if no price data available
                            sector_features[f'sector_{symbol}_momentum'] = 0
                            sector_features[f'sector_{symbol}_volatility'] = 0
                    else:
                        logger.warning(f"SECTOR_DEBUG: {symbol} has no technical_features or empty technical_features")
                else:
                    logger.warning(f"SECTOR_DEBUG: {symbol} not found in symbol_features")
            
            # Final validation of sector features
            logger.info(f"SECTOR_DEBUG: Final sector_features shape: {sector_features.shape}")
            for col in sector_features.columns:
                nan_count = sector_features[col].isnull().sum()
                if nan_count > 0:
                    logger.warning(f"SECTOR_DEBUG: Final check - {col} has {nan_count} NaN values")
                else:
                    logger.info(f"SECTOR_DEBUG: Final check - {col} has no NaN values")
            
            logger.info(f"Generated {len(sector_features.columns)} sector features")
            return sector_features
            
        except Exception as e:
            logger.error(f"Failed to engineer sector features: {e}")
            return pd.DataFrame()
    
    async def _create_universal_embeddings(self, 
                                         symbol_features: Dict[str, FeatureSet], 
                                         symbols: List[str],
                                         symbol_mappings: Dict[str, int]) -> pd.DataFrame:
        """Create universal symbol embeddings for neural networks"""
        try:
            logger.info(f"Creating universal embeddings for {len(symbols)} symbols")
            
            # Find common timestamps across all symbols
            common_timestamps = None
            for symbol, features in symbol_features.items():
                if hasattr(features, 'technical_features'):
                    timestamps = features.technical_features.index
                    if common_timestamps is None:
                        common_timestamps = set(timestamps)
                    else:
                        common_timestamps = common_timestamps.intersection(set(timestamps))
            
            if not common_timestamps:
                logger.warning("No common timestamps for universal embeddings")
                return pd.DataFrame()
            
            common_timestamps = sorted(list(common_timestamps))
            embeddings = pd.DataFrame(index=common_timestamps)
            
            # Add symbol ID embeddings
            for symbol in symbols:
                symbol_id = symbol_mappings[symbol]
                embeddings[f'symbol_id_{symbol}'] = symbol_id
                
                # Add one-hot encoding
                for other_symbol in symbols:
                    embeddings[f'symbol_{other_symbol}'] = 1 if other_symbol == symbol else 0
            
            logger.info(f"Generated universal embeddings with {len(embeddings.columns)} features")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create universal embeddings: {e}")
            return pd.DataFrame()
    
    async def _calculate_universal_feature_importance(self, 
                                                    symbol_features: Dict[str, FeatureSet],
                                                    cross_symbol_features: pd.DataFrame,
                                                    market_regime_features: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance across all symbols"""
        try:
            importance = {}
            
            # Aggregate individual symbol feature importance
            for symbol, features in symbol_features.items():
                if hasattr(features, 'feature_importance'):
                    for feature, imp in features.feature_importance.items():
                        key = f"{symbol}_{feature}"
                        importance[key] = imp
            
            # Add cross-symbol feature importance (simplified)
            for col in cross_symbol_features.columns:
                importance[f"cross_{col}"] = 0.5  # Placeholder
            
            # Add regime feature importance
            for col in market_regime_features.columns:
                importance[f"regime_{col}"] = 0.3  # Placeholder
            
            return importance
            
        except Exception as e:
            logger.error(f"Failed to calculate universal feature importance: {e}")
            return {}
    
    def _generate_universal_feature_metadata(self, 
                                           symbol_features: Dict[str, FeatureSet],
                                           cross_symbol_features: pd.DataFrame,
                                           market_regime_features: pd.DataFrame,
                                           symbols: List[str]) -> Dict[str, Any]:
        """Generate metadata for universal features"""
        try:
            metadata = {
                'symbols': symbols,
                'symbol_count': len(symbols),
                'individual_feature_count': sum(len(f.technical_features.columns) for f in symbol_features.values() if hasattr(f, 'technical_features')),
                'cross_symbol_feature_count': len(cross_symbol_features.columns),
                'regime_feature_count': len(market_regime_features.columns),
                'total_feature_count': 0,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            metadata['total_feature_count'] = (
                metadata['individual_feature_count'] + 
                metadata['cross_symbol_feature_count'] + 
                metadata['regime_feature_count']
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to generate universal feature metadata: {e}")
            return {}
    
    def _get_empty_universal_feature_set(self, symbols: List[str]) -> UniversalFeatureSet:
        """Return empty universal feature set"""
        return UniversalFeatureSet(
            symbol_features={},
            cross_symbol_features=pd.DataFrame(),
            market_regime_features=pd.DataFrame(),
            sector_features=pd.DataFrame(),
            universal_embeddings=pd.DataFrame(),
            feature_importance={},
            feature_metadata={},
            symbol_mappings={symbol: idx for idx, symbol in enumerate(symbols)}
        )
    
    async def prepare_universal_training_data(self, 
                                            universal_features: UniversalFeatureSet,
                                            target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare universal training data for ML models"""
        try:
            logger.info("Preparing universal training data")
            
            # Combine all features into a single DataFrame
            combined_features = []
            combined_targets = []
            
            for symbol, features in universal_features.symbol_features.items():
                # Combine all feature categories from FeatureSet
                feature_dfs = []
                feature_counts = {}
                
                # Add technical features
                if hasattr(features, 'technical_features') and not features.technical_features.empty:
                    feature_dfs.append(features.technical_features)
                    feature_counts['technical'] = len(features.technical_features.columns)
                    logger.info(f"[{symbol}] Technical features: {len(features.technical_features.columns)} columns")
                
                # Add market microstructure features
                if hasattr(features, 'market_microstructure') and not features.market_microstructure.empty:
                    feature_dfs.append(features.market_microstructure)
                    feature_counts['market_microstructure'] = len(features.market_microstructure.columns)
                    logger.info(f"[{symbol}] Market microstructure features: {len(features.market_microstructure.columns)} columns")
                
                # Add sentiment features
                if hasattr(features, 'sentiment_features') and not features.sentiment_features.empty:
                    feature_dfs.append(features.sentiment_features)
                    feature_counts['sentiment'] = len(features.sentiment_features.columns)
                    logger.info(f"[{symbol}] Sentiment features: {len(features.sentiment_features.columns)} columns")
                
                # Add macro features
                if hasattr(features, 'macro_features') and not features.macro_features.empty:
                    feature_dfs.append(features.macro_features)
                    feature_counts['macro'] = len(features.macro_features.columns)
                    logger.info(f"[{symbol}] Macro features: {len(features.macro_features.columns)} columns")
                
                # Add cross-asset features
                if hasattr(features, 'cross_asset_features') and not features.cross_asset_features.empty:
                    feature_dfs.append(features.cross_asset_features)
                    feature_counts['cross_asset'] = len(features.cross_asset_features.columns)
                    logger.info(f"[{symbol}] Cross-asset features: {len(features.cross_asset_features.columns)} columns")
                
                # Add engineered features
                if hasattr(features, 'engineered_features') and not features.engineered_features.empty:
                    feature_dfs.append(features.engineered_features)
                    feature_counts['engineered'] = len(features.engineered_features.columns)
                    logger.info(f"[{symbol}] Engineered features: {len(features.engineered_features.columns)} columns")
                
                if feature_dfs:
                    # Combine all feature categories for this symbol
                    symbol_df = pd.concat(feature_dfs, axis=1)
                    total_individual_features = sum(feature_counts.values())
                    logger.info(f"[{symbol}] Combined individual features: {total_individual_features} columns from {len(feature_dfs)} categories")
                    
                    # Add symbol embeddings
                    symbol_id = universal_features.symbol_mappings[symbol]
                    logger.info(f"DEBUG: Assigning symbol_id={symbol_id} for symbol={symbol}")
                    logger.info(f"DEBUG: symbol_mappings={universal_features.symbol_mappings}")
                    
                    # Validate symbol_id is in expected range
                    max_symbol_id = len(universal_features.symbol_mappings) - 1
                    if symbol_id < 0 or symbol_id > max_symbol_id:
                        logger.error(f"CRITICAL: Invalid symbol_id={symbol_id} for symbol={symbol}, expected range [0, {max_symbol_id}]")
                        symbol_id = 0  # Default to first symbol as fallback
                        logger.warning(f"Fallback: Using symbol_id={symbol_id} for symbol={symbol}")
                    
                    symbol_df['symbol_id'] = symbol_id
                    logger.info(f"DEBUG: Assigned symbol_id column with value {symbol_id} to {len(symbol_df)} rows for symbol {symbol}")
                    
                    # Verify the assignment worked correctly
                    unique_symbol_ids = symbol_df['symbol_id'].unique()
                    if len(unique_symbol_ids) != 1 or unique_symbol_ids[0] != symbol_id:
                        logger.error(f"CRITICAL: symbol_id assignment failed! Expected [{symbol_id}], got {unique_symbol_ids}")
                        # Force correct assignment
                        symbol_df['symbol_id'] = symbol_id
                        logger.warning(f"Forced symbol_id assignment to {symbol_id} for symbol {symbol}")
                    
                    # Note: Removed one-hot encoding to avoid duplicate symbol features
                    # symbol_id is sufficient for symbol identification
                    
                    # Add cross-symbol features if available
                    cross_symbol_count = 0
                    if not universal_features.cross_symbol_features.empty:
                        aligned_cross = universal_features.cross_symbol_features.reindex(symbol_df.index)
                        symbol_df = pd.concat([symbol_df, aligned_cross], axis=1)
                        cross_symbol_count = len(universal_features.cross_symbol_features.columns)
                        logger.info(f"[{symbol}] Added {cross_symbol_count} cross-symbol features")
                    
                    # Add regime features if available
                    regime_count = 0
                    if not universal_features.market_regime_features.empty:
                        aligned_regime = universal_features.market_regime_features.reindex(symbol_df.index)
                        symbol_df = pd.concat([symbol_df, aligned_regime], axis=1)
                        regime_count = len(universal_features.market_regime_features.columns)
                        logger.info(f"[{symbol}] Added {regime_count} market regime features")
                    
                    # Add sector features if available
                    sector_count = 0
                    if not universal_features.sector_features.empty:
                        aligned_sector = universal_features.sector_features.reindex(symbol_df.index)
                        symbol_df = pd.concat([symbol_df, aligned_sector], axis=1)
                        sector_count = len(universal_features.sector_features.columns)
                        logger.info(f"[{symbol}] Added {sector_count} sector features")
                    
                    # Add universal embeddings if available
                    embedding_count = 0
                    if not universal_features.universal_embeddings.empty:
                        aligned_embeddings = universal_features.universal_embeddings.reindex(symbol_df.index)
                        symbol_df = pd.concat([symbol_df, aligned_embeddings], axis=1)
                        embedding_count = len(universal_features.universal_embeddings.columns)
                        logger.info(f"[{symbol}] Added {embedding_count} universal embedding features")
                    
                    total_features = total_individual_features + 1 + cross_symbol_count + regime_count + sector_count + embedding_count
                    logger.info(f"[{symbol}] FEATURE_DEBUG: Total features for symbol: {total_features} (individual: {total_individual_features}, symbol_id: 1, cross_symbol: {cross_symbol_count}, regime: {regime_count}, sector: {sector_count}, embeddings: {embedding_count})")
                    
                    # Extract target if available
                    if target_column in symbol_df.columns:
                        target = symbol_df[target_column]
                        symbol_df = symbol_df.drop(columns=[target_column])
                        
                        combined_features.append(symbol_df)
                        combined_targets.append(target)
            
            if not combined_features:
                logger.error("No features available for universal training data")
                return pd.DataFrame(), pd.Series()
            
            # Combine all features and targets
            logger.info(f"DEBUG: About to concatenate {len(combined_features)} feature DataFrames")
            for i, df in enumerate(combined_features):
                if 'symbol_id' in df.columns:
                    unique_ids = df['symbol_id'].unique()
                    logger.info(f"DEBUG: DataFrame {i} has symbol_id values: {unique_ids} (shape: {df.shape})")
                else:
                    logger.warning(f"DEBUG: DataFrame {i} missing symbol_id column (shape: {df.shape})")
            
            X = pd.concat(combined_features, ignore_index=True)
            y = pd.concat(combined_targets, ignore_index=True) if combined_targets else pd.Series()
            
            logger.info(f"Before validation: X shape={X.shape}, y shape={y.shape}")
            
            # DEBUG: Check symbol_id values after concatenation
            if 'symbol_id' in X.columns:
                logger.info(f"DEBUG: After concatenation - symbol_id unique values: {X['symbol_id'].unique()}")
                logger.info(f"DEBUG: After concatenation - symbol_id min: {X['symbol_id'].min()}, max: {X['symbol_id'].max()}")
                logger.info(f"DEBUG: After concatenation - symbol_id dtype: {X['symbol_id'].dtype}")
            else:
                logger.error(f"CRITICAL: symbol_id column missing after concatenation!")
            
            # Comprehensive data validation and cleaning
            logger.info("Starting comprehensive data validation...")
            
            # Check for NaN values in features
            nan_counts = X.isnull().sum()
            total_nans = nan_counts.sum()
            if total_nans > 0:
                logger.warning(f"Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
                for col in nan_counts[nan_counts > 0].index:
                    logger.warning(f"  - {col}: {nan_counts[col]} NaN values")
            
            # Check for infinite values in features
            inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
            total_infs = inf_counts.sum()
            if total_infs > 0:
                logger.warning(f"Found {total_infs} infinite values across {(inf_counts > 0).sum()} columns")
                for col in inf_counts[inf_counts > 0].index:
                    logger.warning(f"  - {col}: {inf_counts[col]} infinite values")
            
            # Check for extremely large values that could cause numerical instability
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            large_value_threshold = 1e10
            for col in numeric_cols:
                if col in X.columns:
                    large_values = np.abs(X[col].values) > large_value_threshold
                    count = np.sum(large_values)
                    if count > 0:
                        max_val = np.max(np.abs(X[col].values))
                        logger.warning(f"Column {col} has {count} extremely large values (max: {max_val:.2e})")
            
            # Fill NaN values with forward fill, then backward fill, then zero
            X = X.ffill().bfill().fillna(0)
            
            # Replace infinite values with large but finite numbers
            X = X.replace([np.inf, -np.inf], [1e6, -1e6])
            
            # Clip extremely large values to prevent numerical instability
            for col in numeric_cols:
                if col in X.columns:
                    X[col] = np.clip(X[col], -1e6, 1e6)
            
            # Validate targets
            if len(y) > 0:
                y_nan_count = y.isnull().sum()
                y_inf_count = np.isinf(y).sum()
                
                if y_nan_count > 0:
                    logger.warning(f"Found {y_nan_count} NaN values in targets")
                    y = y.fillna(0)  # Fill target NaNs with 0 (neutral)
                
                if y_inf_count > 0:
                    logger.warning(f"Found {y_inf_count} infinite values in targets")
                    y = y.replace([np.inf, -np.inf], [1, -1])  # Replace with valid target values
                
                # Ensure targets are in valid range for binary classification
                y = np.clip(y, 0, 1)
            
            # Final validation
            final_nan_count = X.isnull().sum().sum()
            final_inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            
            if final_nan_count > 0:
                logger.error(f"CRITICAL: Still have {final_nan_count} NaN values after cleaning!")
                # Drop rows with remaining NaNs as last resort
                before_drop = len(X)
                X = X.dropna()
                y = y.iloc[X.index] if len(y) > 0 else y
                logger.warning(f"Dropped {before_drop - len(X)} rows with NaN values")
            
            if final_inf_count > 0:
                logger.error(f"CRITICAL: Still have {final_inf_count} infinite values after cleaning!")
            
            # Ensure data types are correct
            X = X.astype(np.float32)
            if len(y) > 0:
                y = y.astype(np.float32)
            
            logger.info(f"Data validation completed successfully")
            logger.info(f"Final data shapes: X={X.shape}, y={y.shape}")
            logger.info(f"Data ranges: X min={X.min().min():.6f}, X max={X.max().max():.6f}")
            if len(y) > 0:
                logger.info(f"Target range: y min={y.min():.6f}, y max={y.max():.6f}")
            
            # Validate feature dimensions for Phase 1 training
            validation_passed = self._validate_feature_dimensions(
                X, 
                "Phase 1 Base Training", 
                expected_total=184
            )
            
            if not validation_passed:
                logger.error("Feature validation failed for Phase 1 training - proceeding with caution")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to prepare universal training data: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame(), pd.Series()
    
    def _validate_feature_dimensions(self, features_df: pd.DataFrame, phase_name: str, expected_total: int = 184) -> bool:
        """
        Validate feature dimensions and provide detailed breakdown.
        
        Args:
            features_df: DataFrame containing features
            phase_name: Name of the training phase for logging
            expected_total: Expected total number of features
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            logger.info(f"=== Feature Dimension Validation - {phase_name} ===")
            
            total_features = len(features_df.columns)
            logger.info(f"Total features: {total_features}")
            
            # Count different feature types
            feature_counts = {
                'technical': 0,
                'symbol': 0,
                'cross_symbol': 0,
                'market_regime': 0,
                'sector': 0,
                'other': 0
            }
            
            # Check for duplicate columns
            duplicate_cols = features_df.columns[features_df.columns.duplicated()].tolist()
            if duplicate_cols:
                logger.warning(f"Found {len(duplicate_cols)} duplicate columns: {duplicate_cols}")
            
            # Check for near-duplicate columns (same name with slight variations)
            col_names = features_df.columns.tolist()
            potential_duplicates = []
            for i, col1 in enumerate(col_names):
                for j, col2 in enumerate(col_names[i+1:], i+1):
                    if col1.lower().replace('_', '').replace('-', '') == col2.lower().replace('_', '').replace('-', ''):
                        potential_duplicates.append((col1, col2))
            if potential_duplicates:
                logger.warning(f"Found {len(potential_duplicates)} potential duplicate column pairs: {potential_duplicates}")
            
            # Log all column names for debugging
            logger.info(f"All feature columns ({len(features_df.columns)}): {list(features_df.columns)}")
            
            for col in features_df.columns:
                col_lower = col.lower()
                if any(tech in col_lower for tech in ['rsi', 'macd', 'bb', 'sma', 'ema', 'stoch', 'atr', 'volume', 'price', 'return', 'momentum', 'volatility', 'trend', 'support', 'resistance', 'fibonacci', 'pivot', 'bollinger', 'williams', 'cci', 'mfi', 'adx', 'psar', 'ichimoku', 'keltner', 'donchian', 'vwap', 'obv', 'chaikin', 'accumulation', 'distribution', 'force_index', 'ease_of_movement', 'negative_volume', 'positive_volume', 'trix', 'ultimate_oscillator', 'commodity_channel', 'detrended_price', 'mass_index', 'coppock', 'know_sure_thing', 'schaff_trend', 'elder_ray', 'klinger', 'money_flow', 'price_volume_trend', 'on_balance_volume', 'accumulation_distribution', 'chaikin_money_flow', 'ease_of_movement', 'force_index', 'negative_volume_index', 'positive_volume_index', 'volume_price_trend', 'volume_weighted_average_price']):
                    feature_counts['technical'] += 1
                elif 'symbol' in col_lower and 'cross' not in col_lower:
                    feature_counts['symbol'] += 1
                elif any(cross in col_lower for cross in ['corr_', 'beta_', 'relative_strength', 'market_dispersion']):
                    feature_counts['cross_symbol'] += 1
                elif any(regime in col_lower for regime in ['market_volatility', 'vol_regime', 'vol_trend', 'vol_correlation', 'volatility_regime', 'composite_momentum', 'composite_volatility', 'composite_trend']):
                    feature_counts['market_regime'] += 1
                elif 'sector' in col_lower:
                    feature_counts['sector'] += 1
                elif any(time_feat in col_lower for time_feat in ['hour', 'minute', 'day_of_week', 'month', 'quarter', 'year', 'is_market_open', 'is_pre_market', 'is_after_hours', 'minutes_since_open', 'days_to_expiry', 'is_month_end', 'is_quarter_end', 'is_year_end', 'is_monday', 'is_friday']):
                    feature_counts['technical'] += 1  # Time features are considered technical
                elif any(price_feat in col_lower for price_feat in ['open_to_close', 'high_to_close', 'low_to_close', 'spread_ma', 'close_to_open', 'high_to_low', 'close_to_high', 'close_to_low', 'high_low_ratio']):
                    feature_counts['technical'] += 1  # Price-based features are technical
                elif any(flow_feat in col_lower for flow_feat in ['order_flow', 'market_depth', 'toxicity', 'imbalance', 'flow_imbalance']):
                    feature_counts['technical'] += 1  # Order flow features are technical
                elif col_lower.startswith('universal_embed_'):
                    feature_counts['symbol'] += 1  # Universal embeddings are symbol features
                # Additional technical feature patterns from the uncategorized list
                elif any(pattern in col_lower for pattern in ['transactions_', 'liquidity_', 'spread_', 'depth_', 'tick_', 'trade_', 'bid_', 'ask_', 'size_', 'count_', 'ratio_', '_ma_', '_std_', '_min_', '_max_', '_sum_', '_mean_', '_median_', '_q25_', '_q75_', '_skew_', '_kurt_', '_range_', '_iqr_', 'proxy', 'weighted', 'normalized', 'scaled', 'lag_', 'diff_', 'pct_', 'rolling_', 'ewm_', 'expanding_']):
                    feature_counts['technical'] += 1  # These are all technical indicators
                # Candlestick patterns and other technical indicators
                elif any(pattern in col_lower for pattern in ['doji', 'hammer', 'engulfing', 'consecutive_', 'frequency_ratio', 'roc_', 'gap_', 'breakout', 'reversal', 'continuation', 'pattern_', 'signal_', 'cross_', 'divergence']):
                    feature_counts['technical'] += 1  # Candlestick and pattern features
                # Basic OHLC and volatility features
                elif any(basic in col_lower for basic in ['open', 'high', 'low', 'close', 'realized_vol_', 'implied_vol_', 'historical_vol_', 'garch_vol_']):
                    feature_counts['technical'] += 1  # Basic price and volatility features
                else:
                    feature_counts['other'] += 1
                    logger.info(f"Uncategorized feature: {col}")
            
            # Log feature breakdown
            logger.info(f"Feature breakdown:")
            for feature_type, count in feature_counts.items():
                if count > 0:
                    logger.info(f"  - {feature_type}: {count}")
            
            # Check for expected total
            if total_features != expected_total:
                logger.warning(f"Feature count mismatch! Expected: {expected_total}, Actual: {total_features}")
                logger.warning(f"Difference: {total_features - expected_total}")
                
                # Provide suggestions based on the difference
                diff = total_features - expected_total
                if diff < 0:
                    logger.warning(f"Missing {abs(diff)} features - check for:")
                    logger.warning("  - Missing symbol embeddings")
                    logger.warning("  - Missing cross-symbol features")
                    logger.warning("  - Missing market regime features")
                else:
                    logger.warning(f"Extra {diff} features - check for:")
                    logger.warning("  - Duplicate features")
                    logger.warning("  - Unexpected feature additions")
                
                return False
            
            # Check for NaN or infinite values
            nan_count = features_df.isnull().sum().sum()
            inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
            
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in features")
            
            if inf_count > 0:
                logger.warning(f"Found {inf_count} infinite values in features")
            
            logger.info(f"Feature validation passed for {phase_name}")
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
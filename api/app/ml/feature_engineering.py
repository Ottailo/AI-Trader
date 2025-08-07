"""
Feature engineering for time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for financial time series data.
    
    This class calculates technical indicators and prepares features
    for machine learning models.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize the feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ("standard" or "minmax")
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = []
        self.is_fitted = False
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty:
            return df
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # Price-based indicators
        df_indicators = self._add_price_indicators(df_indicators)
        
        # Volume-based indicators
        df_indicators = self._add_volume_indicators(df_indicators)
        
        # Momentum indicators
        df_indicators = self._add_momentum_indicators(df_indicators)
        
        # Volatility indicators
        df_indicators = self._add_volatility_indicators(df_indicators)
        
        # Trend indicators
        df_indicators = self._add_trend_indicators(df_indicators)
        
        # Custom features
        df_indicators = self._add_custom_features(df_indicators)
        
        return df_indicators
    
    def _add_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # Moving averages
        df['sma_5'] = SMAIndicator(close, window=5).sma_indicator()
        df['sma_10'] = SMAIndicator(close, window=10).sma_indicator()
        df['sma_20'] = SMAIndicator(close, window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close, window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close, window=200).sma_indicator()
        
        df['ema_5'] = EMAIndicator(close, window=5).ema_indicator()
        df['ema_10'] = EMAIndicator(close, window=10).ema_indicator()
        df['ema_20'] = EMAIndicator(close, window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close, window=50).ema_indicator()
        
        # Price ratios
        df['price_to_sma_20'] = close / df['sma_20']
        df['price_to_sma_50'] = close / df['sma_50']
        df['price_to_ema_20'] = close / df['ema_20']
        
        # Price changes
        df['price_change'] = close.pct_change()
        df['price_change_5'] = close.pct_change(periods=5)
        df['price_change_10'] = close.pct_change(periods=10)
        df['price_change_20'] = close.pct_change(periods=20)
        
        # High-Low spread
        df['hl_spread'] = (high - low) / close
        df['hl_spread_5'] = df['hl_spread'].rolling(window=5).mean()
        
        # Open-Close spread
        df['oc_spread'] = (close - open_price) / close
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based technical indicators."""
        close = df['close']
        volume = df['volume']
        
        # Volume moving averages
        df['volume_sma_5'] = volume.rolling(window=5).mean()
        df['volume_sma_20'] = volume.rolling(window=20).mean()
        
        # Volume ratios
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # Volume Weighted Average Price
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=close,
            volume=volume
        ).volume_weighted_average_price()
        
        # Price-Volume trend
        df['pvt'] = (close.pct_change() * volume).cumsum()
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        df['rsi'] = RSIIndicator(close).rsi()
        df['rsi_5'] = RSIIndicator(close, window=5).rsi()
        df['rsi_14'] = RSIIndicator(close, window=14).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        
        # Rate of Change
        df['roc'] = ta.momentum.ROCIndicator(close).roc()
        df['roc_5'] = ta.momentum.ROCIndicator(close, window=5).roc()
        df['roc_10'] = ta.momentum.ROCIndicator(close, window=10).roc()
        
        # Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(
            high=high, low=low, close=close, volume=df['volume']
        ).money_flow_index()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Bollinger Bands
        bb = BollingerBands(close)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        df['atr_5'] = ta.volatility.AverageTrueRange(high, low, close, window=5).average_true_range()
        
        # Historical Volatility
        df['hist_volatility'] = close.pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high, low, close)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # MACD
        macd = MACD(close)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(high, low, close).psar()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(high, low)
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()
        df['adx_pos'] = ta.trend.ADXIndicator(high, low, close).adx_pos()
        df['adx_neg'] = ta.trend.ADXIndicator(high, low, close).adx_neg()
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.CCIIndicator(high, low, close).cci()
        
        return df
    
    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features."""
        close = df['close']
        
        # Price momentum features
        df['momentum_1d'] = close.pct_change(1)
        df['momentum_3d'] = close.pct_change(3)
        df['momentum_5d'] = close.pct_change(5)
        df['momentum_10d'] = close.pct_change(10)
        
        # Volatility features
        df['volatility_5d'] = close.pct_change().rolling(5).std()
        df['volatility_10d'] = close.pct_change().rolling(10).std()
        df['volatility_20d'] = close.pct_change().rolling(20).std()
        
        # Support and resistance levels (simplified)
        df['support_level'] = close.rolling(window=20).min()
        df['resistance_level'] = close.rolling(window=20).max()
        df['price_to_support'] = close / df['support_level']
        df['price_to_resistance'] = close / df['resistance_level']
        
        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter
        
        # Gap analysis
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning.
        
        Args:
            df: DataFrame with technical indicators
            target_column: Name of the target column
            
        Returns:
            Tuple of (features DataFrame, feature column names)
        """
        # Calculate technical indicators
        df_features = self.calculate_technical_indicators(df)
        
        # Remove rows with NaN values
        df_features = df_features.dropna()
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = [
            target_column, 'timestamp', 'symbol', 'interval',
            'open', 'high', 'low', 'close', 'volume'  # Original OHLCV
        ]
        
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        return df_features, feature_cols
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str,
        sequence_length: int = 60,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(sequence_length, len(df) - prediction_horizon + 1):
            # Input sequence
            X.append(df[feature_cols].iloc[i-sequence_length:i].values)
            
            # Target (future price)
            if prediction_horizon == 1:
                y.append(df[target_col].iloc[i])
            else:
                y.append(df[target_col].iloc[i:i+prediction_horizon].values)
        
        return np.array(X), np.array(y)
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit the scaler on training data.
        
        Args:
            X: Training data array
        """
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Reshape for scaler (flatten sequences)
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Input features array
            
        Returns:
            Scaled features array
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming features")
        
        # Reshape for scaler
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Transform
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back
        return X_scaled.reshape(original_shape)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values (if they were scaled).
        
        Args:
            y: Scaled target values
            
        Returns:
            Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        # For targets, we typically only scale the last feature (price)
        # This is a simplified approach - in practice, you might want separate scalers
        return y  # For now, return as-is since we typically don't scale targets
    
    def get_feature_importance_ranking(self) -> Dict[str, float]:
        """
        Get feature importance ranking (placeholder for now).
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        # This would typically be calculated during model training
        # For now, return equal importance
        return {col: 1.0 / len(self.feature_columns) for col in self.feature_columns}

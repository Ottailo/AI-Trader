"""
Trading strategies for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionType(str, Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class BaseStrategy(ABC):
    """
    Base class for trading strategies.
    
    All trading strategies should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.positions = {}  # Current positions
        self.trades = []  # Trade history
        self.signals = []  # Signal history
        
        logger.info("Strategy initialized", name=name, params=params)
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_price: float) -> SignalType:
        """
        Generate trading signal based on current market data.
        
        Args:
            data: Historical market data
            current_price: Current price
            
        Returns:
            Trading signal
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalType, current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate position size for the signal.
        
        Args:
            signal: Trading signal
            current_price: Current price
            available_capital: Available capital
            
        Returns:
            Position size (positive for long, negative for short)
        """
        pass
    
    def update_position(self, symbol: str, quantity: float, price: float, 
                       timestamp: pd.Timestamp) -> None:
        """
        Update current position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Execution price
            timestamp: Execution timestamp
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_price": 0,
                "total_cost": 0
            }
        
        pos = self.positions[symbol]
        
        if pos["quantity"] == 0:
            # New position
            pos["quantity"] = quantity
            pos["avg_price"] = price
            pos["total_cost"] = quantity * price
        else:
            # Existing position
            if (pos["quantity"] > 0 and quantity > 0) or (pos["quantity"] < 0 and quantity < 0):
                # Adding to position
                total_quantity = pos["quantity"] + quantity
                total_cost = pos["total_cost"] + (quantity * price)
                pos["avg_price"] = total_cost / total_quantity
                pos["quantity"] = total_quantity
                pos["total_cost"] = total_cost
            else:
                # Reducing or reversing position
                pos["quantity"] += quantity
                if pos["quantity"] == 0:
                    pos["avg_price"] = 0
                    pos["total_cost"] = 0
        
        # Record trade
        self.trades.append({
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "position_after": pos["quantity"]
        })
        
        logger.debug("Position updated",
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    new_position=pos["quantity"])
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """
        Get current position value.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            Position value
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        return pos["quantity"] * current_price
    
    def get_position_pnl(self, symbol: str, current_price: float) -> float:
        """
        Get current position P&L.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            Position P&L
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        if pos["quantity"] == 0:
            return 0.0
        
        return (current_price - pos["avg_price"]) * pos["quantity"]
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get strategy summary.
        
        Returns:
            Strategy summary dictionary
        """
        return {
            "name": self.name,
            "params": self.params,
            "current_positions": self.positions,
            "total_trades": len(self.trades),
            "signals_generated": len(self.signals)
        }


class MLStrategy(BaseStrategy):
    """
    Machine Learning-based trading strategy.
    
    Uses ML model predictions to generate trading signals.
    """
    
    def __init__(
        self,
        name: str,
        model_predictor,
        params: Optional[Dict] = None
    ):
        """
        Initialize ML strategy.
        
        Args:
            name: Strategy name
            model_predictor: ML model predictor function
            params: Strategy parameters
        """
        super().__init__(name, params)
        self.model_predictor = model_predictor
        
        # Default parameters
        self.confidence_threshold = self.params.get("confidence_threshold", 0.6)
        self.lookback_period = self.params.get("lookback_period", 60)
        self.position_size_pct = self.params.get("position_size_pct", 0.1)
        self.stop_loss_pct = self.params.get("stop_loss_pct", 0.02)
        self.take_profit_pct = self.params.get("take_profit_pct", 0.04)
        
        logger.info("ML Strategy initialized",
                   name=name,
                   confidence_threshold=self.confidence_threshold,
                   lookback_period=self.lookback_period)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float) -> SignalType:
        """
        Generate signal based on ML model prediction.
        
        Args:
            data: Historical market data
            current_price: Current price
            
        Returns:
            Trading signal
        """
        if len(data) < self.lookback_period:
            return SignalType.HOLD
        
        try:
            # Get prediction from ML model
            prediction = self.model_predictor(data)
            
            if prediction is None:
                return SignalType.HOLD
            
            # Calculate expected return
            expected_return = (prediction - current_price) / current_price
            
            # Generate signal based on expected return and confidence
            if abs(expected_return) < self.confidence_threshold:
                return SignalType.HOLD
            
            if expected_return > 0:
                signal = SignalType.BUY
            else:
                signal = SignalType.SELL
            
            # Record signal
            self.signals.append({
                "timestamp": data.index[-1],
                "signal": signal,
                "current_price": current_price,
                "predicted_price": prediction,
                "expected_return": expected_return
            })
            
            logger.debug("ML signal generated",
                        signal=signal,
                        current_price=current_price,
                        predicted_price=prediction,
                        expected_return=expected_return)
            
            return signal
            
        except Exception as e:
            logger.error("Error generating ML signal", error=str(e))
            return SignalType.HOLD
    
    def calculate_position_size(self, signal: SignalType, current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate position size based on ML confidence and risk management.
        
        Args:
            signal: Trading signal
            current_price: Current price
            available_capital: Available capital
            
        Returns:
            Position size
        """
        if signal == SignalType.HOLD:
            return 0.0
        
        # Base position size
        base_size = available_capital * self.position_size_pct
        
        # Adjust based on signal confidence (if available)
        if self.signals:
            last_signal = self.signals[-1]
            confidence = abs(last_signal.get("expected_return", 0))
            confidence_multiplier = min(confidence / self.confidence_threshold, 2.0)
            position_size = base_size * confidence_multiplier
        else:
            position_size = base_size
        
        # Convert to quantity
        quantity = position_size / current_price
        
        # Apply signal direction
        if signal == SignalType.SELL:
            quantity = -quantity
        
        return quantity


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Based on Bollinger Bands and RSI indicators.
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize mean reversion strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        super().__init__(name, params)
        
        # Default parameters
        self.bb_period = self.params.get("bb_period", 20)
        self.bb_std = self.params.get("bb_std", 2)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.position_size_pct = self.params.get("position_size_pct", 0.1)
        
        logger.info("Mean Reversion Strategy initialized",
                   name=name,
                   bb_period=self.bb_period,
                   rsi_period=self.rsi_period)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float) -> SignalType:
        """
        Generate signal based on mean reversion indicators.
        
        Args:
            data: Historical market data
            current_price: Current price
            
        Returns:
            Trading signal
        """
        if len(data) < max(self.bb_period, self.rsi_period):
            return SignalType.HOLD
        
        try:
            # Calculate Bollinger Bands
            bb_middle = data['close'].rolling(window=self.bb_period).mean()
            bb_std = data['close'].rolling(window=self.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.bb_std)
            bb_lower = bb_middle - (bb_std * self.bb_std)
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get current values
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Generate signals
            signal = SignalType.HOLD
            
            # Oversold condition (buy signal)
            if (current_price <= current_bb_lower and 
                current_rsi <= self.rsi_oversold):
                signal = SignalType.BUY
            
            # Overbought condition (sell signal)
            elif (current_price >= current_bb_upper and 
                  current_rsi >= self.rsi_overbought):
                signal = SignalType.SELL
            
            # Record signal
            self.signals.append({
                "timestamp": data.index[-1],
                "signal": signal,
                "current_price": current_price,
                "bb_upper": current_bb_upper,
                "bb_lower": current_bb_lower,
                "rsi": current_rsi
            })
            
            logger.debug("Mean reversion signal generated",
                        signal=signal,
                        current_price=current_price,
                        bb_upper=current_bb_upper,
                        bb_lower=current_bb_lower,
                        rsi=current_rsi)
            
            return signal
            
        except Exception as e:
            logger.error("Error generating mean reversion signal", error=str(e))
            return SignalType.HOLD
    
    def calculate_position_size(self, signal: SignalType, current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate position size for mean reversion strategy.
        
        Args:
            signal: Trading signal
            current_price: Current price
            available_capital: Available capital
            
        Returns:
            Position size
        """
        if signal == SignalType.HOLD:
            return 0.0
        
        # Calculate position size based on distance from mean
        if self.signals:
            last_signal = self.signals[-1]
            
            if signal == SignalType.BUY:
                # Distance from lower band
                distance = (last_signal["bb_lower"] - current_price) / current_price
            else:
                # Distance from upper band
                distance = (current_price - last_signal["bb_upper"]) / current_price
            
            # Adjust position size based on distance
            size_multiplier = min(abs(distance) * 10, 2.0)  # Cap at 2x
            position_size = available_capital * self.position_size_pct * size_multiplier
        else:
            position_size = available_capital * self.position_size_pct
        
        # Convert to quantity
        quantity = position_size / current_price
        
        # Apply signal direction
        if signal == SignalType.SELL:
            quantity = -quantity
        
        return quantity


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Based on moving averages and momentum indicators.
    """
    
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize momentum strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        super().__init__(name, params)
        
        # Default parameters
        self.short_ma = self.params.get("short_ma", 10)
        self.long_ma = self.params.get("long_ma", 30)
        self.momentum_period = self.params.get("momentum_period", 10)
        self.position_size_pct = self.params.get("position_size_pct", 0.1)
        
        logger.info("Momentum Strategy initialized",
                   name=name,
                   short_ma=self.short_ma,
                   long_ma=self.long_ma)
    
    def generate_signal(self, data: pd.DataFrame, current_price: float) -> SignalType:
        """
        Generate signal based on momentum indicators.
        
        Args:
            data: Historical market data
            current_price: Current price
            
        Returns:
            Trading signal
        """
        if len(data) < max(self.short_ma, self.long_ma, self.momentum_period):
            return SignalType.HOLD
        
        try:
            # Calculate moving averages
            short_ma = data['close'].rolling(window=self.short_ma).mean()
            long_ma = data['close'].rolling(window=self.long_ma).mean()
            
            # Calculate momentum
            momentum = data['close'].pct_change(periods=self.momentum_period)
            
            # Get current values
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            current_momentum = momentum.iloc[-1]
            
            # Generate signals
            signal = SignalType.HOLD
            
            # Bullish momentum
            if (current_short_ma > current_long_ma and 
                current_momentum > 0):
                signal = SignalType.BUY
            
            # Bearish momentum
            elif (current_short_ma < current_long_ma and 
                  current_momentum < 0):
                signal = SignalType.SELL
            
            # Record signal
            self.signals.append({
                "timestamp": data.index[-1],
                "signal": signal,
                "current_price": current_price,
                "short_ma": current_short_ma,
                "long_ma": current_long_ma,
                "momentum": current_momentum
            })
            
            logger.debug("Momentum signal generated",
                        signal=signal,
                        current_price=current_price,
                        short_ma=current_short_ma,
                        long_ma=current_long_ma,
                        momentum=current_momentum)
            
            return signal
            
        except Exception as e:
            logger.error("Error generating momentum signal", error=str(e))
            return SignalType.HOLD
    
    def calculate_position_size(self, signal: SignalType, current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate position size for momentum strategy.
        
        Args:
            signal: Trading signal
            current_price: Current price
            available_capital: Available capital
            
        Returns:
            Position size
        """
        if signal == SignalType.HOLD:
            return 0.0
        
        # Calculate position size based on momentum strength
        if self.signals:
            last_signal = self.signals[-1]
            momentum_strength = abs(last_signal.get("momentum", 0))
            
            # Adjust position size based on momentum strength
            size_multiplier = min(momentum_strength * 100, 2.0)  # Cap at 2x
            position_size = available_capital * self.position_size_pct * size_multiplier
        else:
            position_size = available_capital * self.position_size_pct
        
        # Convert to quantity
        quantity = position_size / current_price
        
        # Apply signal direction
        if signal == SignalType.SELL:
            quantity = -quantity
        
        return quantity

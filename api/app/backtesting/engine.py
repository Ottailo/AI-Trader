"""
Backtesting engine for simulating trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog

from .strategies import BaseStrategy, SignalType
from .performance import PerformanceAnalyzer

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_positions: int = 5
    position_size_pct: float = 0.2  # 20% per position
    stop_loss_pct: float = 0.02     # 2%
    take_profit_pct: float = 0.04   # 4%
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


class BacktestingEngine:
    """
    Backtesting engine for simulating trading strategies.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.current_capital = self.config.initial_capital
        self.positions = {}  # symbol -> position info
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        logger.info("Backtesting engine initialized",
                   initial_capital=self.config.initial_capital,
                   commission_rate=self.config.commission_rate)
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy
            data: Historical market data
            symbols: List of symbols to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Backtest results
        """
        logger.info("Starting backtest",
                   strategy_name=strategy.name,
                   symbols=symbols,
                   start_date=start_date,
                   end_date=end_date)
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Reset state
        self._reset_state()
        
        # Group data by symbol
        symbol_data = {}
        for symbol in symbols:
            symbol_data[symbol] = data[data['symbol'] == symbol].copy()
        
        # Get common date range
        common_dates = self._get_common_dates(symbol_data)
        
        # Run simulation
        for date in common_dates:
            self._process_date(strategy, symbol_data, date)
            self._update_equity_curve(date)
        
        # Calculate performance metrics
        results = self._calculate_results(strategy)
        
        logger.info("Backtest completed",
                   total_trades=len(self.trades),
                   final_capital=self.current_capital,
                   total_return=(self.current_capital - self.config.initial_capital) / self.config.initial_capital)
        
        return results
    
    def _reset_state(self) -> None:
        """Reset backtest state."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def _get_common_dates(self, symbol_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get common trading dates across all symbols."""
        common_dates = None
        for symbol, data in symbol_data.items():
            if common_dates is None:
                common_dates = set(data.index)
            else:
                common_dates = common_dates.intersection(set(data.index))
        
        return sorted(list(common_dates))
    
    def _process_date(
        self,
        strategy: BaseStrategy,
        symbol_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> None:
        """
        Process a single trading date.
        
        Args:
            strategy: Trading strategy
            symbol_data: Data for each symbol
            date: Current date
        """
        # Update positions with current prices
        self._update_positions(symbol_data, date)
        
        # Generate signals and execute trades for each symbol
        for symbol, data in symbol_data.items():
            current_data = data[data.index <= date]
            if len(current_data) == 0:
                continue
            
            current_price = current_data.iloc[-1]['close']
            
            # Generate signal
            signal = strategy.generate_signal(current_data, current_price)
            
            if signal != SignalType.HOLD:
                # Calculate position size
                available_capital = self._get_available_capital()
                quantity = strategy.calculate_position_size(
                    signal, current_price, available_capital
                )
                
                # Execute trade
                if abs(quantity) > 0:
                    self._execute_trade(symbol, quantity, current_price, date, signal)
    
    def _update_positions(self, symbol_data: Dict[str, pd.DataFrame], date: datetime) -> None:
        """
        Update position values with current prices.
        
        Args:
            symbol_data: Data for each symbol
            date: Current date
        """
        for symbol, data in symbol_data.items():
            if symbol in self.positions:
                current_data = data[data.index <= date]
                if len(current_data) > 0:
                    current_price = current_data.iloc[-1]['close']
                    self.positions[symbol]['current_price'] = current_price
                    
                    # Check stop loss and take profit
                    self._check_risk_management(symbol, current_price, date)
    
    def _check_risk_management(self, symbol: str, current_price: float, date: datetime) -> None:
        """
        Check and execute risk management rules.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            date: Current date
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        if quantity == 0:
            return
        
        # Calculate P&L
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Stop loss
        if quantity > 0 and pnl_pct <= -self.config.stop_loss_pct:
            self._execute_trade(symbol, -quantity, current_price, date, SignalType.SELL, "stop_loss")
        
        # Take profit
        elif quantity > 0 and pnl_pct >= self.config.take_profit_pct:
            self._execute_trade(symbol, -quantity, current_price, date, SignalType.SELL, "take_profit")
        
        # Short position stop loss
        elif quantity < 0 and pnl_pct >= self.config.stop_loss_pct:
            self._execute_trade(symbol, -quantity, current_price, date, SignalType.BUY, "stop_loss")
        
        # Short position take profit
        elif quantity < 0 and pnl_pct <= -self.config.take_profit_pct:
            self._execute_trade(symbol, -quantity, current_price, date, SignalType.BUY, "take_profit")
    
    def _execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: datetime,
        signal: SignalType,
        reason: str = "signal"
    ) -> None:
        """
        Execute a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            date: Trade date
            signal: Trading signal
            reason: Trade reason
        """
        # Apply slippage
        if quantity > 0:  # Buy
            execution_price = price * (1 + self.config.slippage_rate)
        else:  # Sell
            execution_price = price * (1 - self.config.slippage_rate)
        
        # Calculate commission
        trade_value = abs(quantity) * execution_price
        commission = trade_value * self.config.commission_rate
        
        # Update capital
        if quantity > 0:  # Buy
            self.current_capital -= (trade_value + commission)
        else:  # Sell
            self.current_capital += (trade_value - commission)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'entry_price': 0,
                'current_price': execution_price
            }
        
        position = self.positions[symbol]
        
        if position['quantity'] == 0:
            # New position
            position['quantity'] = quantity
            position['entry_price'] = execution_price
        else:
            # Existing position
            if (position['quantity'] > 0 and quantity > 0) or (position['quantity'] < 0 and quantity < 0):
                # Adding to position
                total_quantity = position['quantity'] + quantity
                total_cost = position['quantity'] * position['entry_price'] + quantity * execution_price
                position['entry_price'] = total_cost / total_quantity
                position['quantity'] = total_quantity
            else:
                # Reducing or reversing position
                position['quantity'] += quantity
                if position['quantity'] == 0:
                    position['entry_price'] = 0
        
        position['current_price'] = execution_price
        
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'price': execution_price,
            'date': date,
            'signal': signal,
            'reason': reason,
            'commission': commission,
            'capital_after': self.current_capital
        }
        
        self.trades.append(trade)
        
        logger.debug("Trade executed",
                    symbol=symbol,
                    quantity=quantity,
                    price=execution_price,
                    reason=reason,
                    capital=self.current_capital)
    
    def _get_available_capital(self) -> float:
        """
        Get available capital for new positions.
        
        Returns:
            Available capital
        """
        # Calculate current position values
        position_values = sum(
            abs(pos['quantity'] * pos['current_price'])
            for pos in self.positions.values()
        )
        
        # Available capital is current capital minus position values
        return max(0, self.current_capital - position_values)
    
    def _update_equity_curve(self, date: datetime) -> None:
        """
        Update equity curve with current portfolio value.
        
        Args:
            date: Current date
        """
        # Calculate total portfolio value
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                portfolio_value += position['quantity'] * position['current_price']
        
        self.equity_curve.append({
            'date': date,
            'equity': portfolio_value,
            'capital': self.current_capital,
            'positions_value': portfolio_value - self.current_capital
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (portfolio_value - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def _calculate_results(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Calculate backtest results and performance metrics.
        
        Args:
            strategy: Trading strategy
            
        Returns:
            Backtest results
        """
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(
            equity_df['equity'],
            pd.Series(self.daily_returns, index=equity_df.index[1:])
        )
        
        # Calculate trade statistics
        trade_stats = self._calculate_trade_statistics()
        
        # Prepare results
        results = {
            'strategy_name': strategy.name,
            'strategy_params': strategy.params,
            'backtest_config': self.config.__dict__,
            'performance_metrics': performance_metrics,
            'trade_statistics': trade_stats,
            'equity_curve': equity_df.to_dict('records'),
            'trades': self.trades,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.config.initial_capital) / self.config.initial_capital,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trades if t.get('pnl', 0) < 0])
        }
        
        return results
    
    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate trade statistics.
        
        Returns:
            Trade statistics
        """
        if not self.trades:
            return {}
        
        # Calculate P&L for each trade
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol in self.positions:
                position = self.positions[symbol]
                if position['quantity'] == 0:  # Position closed
                    entry_price = position.get('entry_price', 0)
                    if entry_price > 0:
                        pnl = (trade['price'] - entry_price) * trade['quantity']
                        trade['pnl'] = pnl
        
        # Calculate statistics
        pnls = [t.get('pnl', 0) for t in self.trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        }
        
        return stats

"""
Performance analysis for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import structlog

logger = structlog.get_logger()


class PerformanceAnalyzer:
    """
    Performance analyzer for trading strategies.
    
    Calculates various performance metrics including returns, risk metrics,
    and risk-adjusted performance measures.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252  # Assuming daily data
    
    def calculate_metrics(
        self, 
        equity_curve: pd.Series, 
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Portfolio equity values over time
            returns: Portfolio returns over time
            
        Returns:
            Dictionary of performance metrics
        """
        if len(equity_curve) < 2 or len(returns) < 1:
            return {}
        
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(equity_curve, returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(equity_curve))
        
        # Additional metrics
        metrics.update(self._calculate_additional_metrics(equity_curve, returns))
        
        return metrics
    
    def _calculate_return_metrics(
        self, 
        equity_curve: pd.Series, 
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate return-based metrics."""
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        # Total return
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = ((final_value / initial_value) ** (1 / years)) - 1 if years > 0 else 0
        
        # Average daily return
        avg_daily_return = returns.mean()
        
        # Positive and negative returns
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        avg_positive_return = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_negative_return = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_daily_return': avg_daily_return,
            'avg_positive_return': avg_positive_return,
            'avg_negative_return': avg_negative_return,
            'positive_days': len(positive_returns),
            'negative_days': len(negative_returns),
            'win_rate': len(positive_returns) / len(returns) if len(returns) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        # Volatility
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(self.trading_days_per_year)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Maximum loss
        max_loss = returns.min()
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            'max_loss': max_loss
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Sharpe ratio
        excess_return = avg_return - (self.risk_free_rate / self.trading_days_per_year)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        annualized_sharpe = sharpe_ratio * np.sqrt(self.trading_days_per_year)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        annualized_sortino = sortino_ratio * np.sqrt(self.trading_days_per_year)
        
        # Calmar ratio (annualized return / max drawdown)
        # This will be calculated in drawdown metrics
        
        # Information ratio (assuming benchmark return of 0 for simplicity)
        information_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Treynor ratio (assuming beta of 1 for simplicity)
        treynor_ratio = excess_return / 1.0  # Simplified
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': annualized_sharpe,
            'sortino_ratio': sortino_ratio,
            'annualized_sortino': annualized_sortino,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio
        }
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-based metrics."""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_duration(drawdown)
        
        # Calmar ratio
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years)) - 1 if years > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown_duration': drawdown_periods['avg_duration'],
            'max_drawdown_duration': drawdown_periods['max_duration'],
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> Dict[str, float]:
        """Calculate drawdown duration statistics."""
        durations = []
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Add final drawdown if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        if not durations:
            return {'avg_duration': 0, 'max_duration': 0}
        
        return {
            'avg_duration': np.mean(durations),
            'max_duration': max(durations)
        }
    
    def _calculate_additional_metrics(
        self, 
        equity_curve: pd.Series, 
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate additional performance metrics."""
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Best and worst periods
        best_day = returns.max()
        worst_day = returns.min()
        
        # Consecutive wins and losses
        consecutive_stats = self._calculate_consecutive_stats(returns)
        
        # Recovery factor
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        max_dd = abs((equity_curve - equity_curve.expanding().max()) / equity_curve.expanding().max()).min()
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0
        
        # Profit factor (if available from trade data)
        # This would typically come from trade-level analysis
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'best_day': best_day,
            'worst_day': worst_day,
            'max_consecutive_wins': consecutive_stats['max_wins'],
            'max_consecutive_losses': consecutive_stats['max_losses'],
            'recovery_factor': recovery_factor
        }
    
    def _calculate_consecutive_stats(self, returns: pd.Series) -> Dict[str, int]:
        """Calculate consecutive wins and losses."""
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return {
            'max_wins': max_consecutive_wins,
            'max_losses': max_consecutive_losses
        }
    
    def calculate_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Portfolio returns
            window: Rolling window size (default: 252 days)
            
        Returns:
            DataFrame with rolling metrics
        """
        rolling_metrics = pd.DataFrame()
        
        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window=window).mean() * self.trading_days_per_year
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(self.trading_days_per_year)
        
        # Rolling Sharpe ratio
        risk_free_daily = self.risk_free_rate / self.trading_days_per_year
        rolling_metrics['rolling_sharpe'] = (
            (rolling_metrics['rolling_return'] - self.risk_free_rate) / 
            rolling_metrics['rolling_volatility']
        )
        
        # Rolling maximum drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.rolling(window=window).max()
        rolling_metrics['rolling_drawdown'] = (equity_curve - rolling_max) / rolling_max
        
        return rolling_metrics
    
    def generate_performance_report(
        self, 
        equity_curve: pd.Series, 
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            equity_curve: Portfolio equity values
            returns: Portfolio returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Comprehensive performance report
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(equity_curve, returns)
        
        # Calculate rolling metrics
        rolling_metrics = self.calculate_rolling_metrics(returns)
        
        # Benchmark comparison (if provided)
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self.calculate_metrics(
                (1 + benchmark_returns).cumprod(), 
                benchmark_returns
            )
        
        # Performance attribution
        attribution = self._calculate_performance_attribution(returns, benchmark_returns)
        
        report = {
            'summary_metrics': metrics,
            'rolling_metrics': rolling_metrics.to_dict('records'),
            'benchmark_metrics': benchmark_metrics,
            'attribution': attribution,
            'period_analysis': self._calculate_period_analysis(returns),
            'risk_analysis': self._calculate_risk_analysis(returns)
        }
        
        return report
    
    def _calculate_performance_attribution(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate performance attribution vs benchmark."""
        if benchmark_returns is None:
            return {}
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        portfolio_returns = aligned_returns[0]
        benchmark_returns = aligned_returns[1]
        
        # Excess return
        excess_return = portfolio_returns - benchmark_returns
        
        # Information ratio
        information_ratio = excess_return.mean() / excess_return.std() if excess_return.std() > 0 else 0
        
        # Tracking error
        tracking_error = excess_return.std() * np.sqrt(self.trading_days_per_year)
        
        # Beta and Alpha
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        portfolio_return = portfolio_returns.mean() * self.trading_days_per_year
        benchmark_return = benchmark_returns.mean() * self.trading_days_per_year
        alpha = portfolio_return - (beta * benchmark_return)
        
        return {
            'excess_return': excess_return.mean() * self.trading_days_per_year,
            'information_ratio': information_ratio * np.sqrt(self.trading_days_per_year),
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha
        }
    
    def _calculate_period_analysis(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Calculate performance by different time periods."""
        if len(returns) == 0:
            return {}
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Quarterly returns
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        
        # Yearly returns
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'monthly': {
                'mean': monthly_returns.mean(),
                'std': monthly_returns.std(),
                'min': monthly_returns.min(),
                'max': monthly_returns.max(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'negative_months': len(monthly_returns[monthly_returns < 0])
            },
            'quarterly': {
                'mean': quarterly_returns.mean(),
                'std': quarterly_returns.std(),
                'min': quarterly_returns.min(),
                'max': quarterly_returns.max(),
                'positive_quarters': len(quarterly_returns[quarterly_returns > 0]),
                'negative_quarters': len(quarterly_returns[quarterly_returns < 0])
            },
            'yearly': {
                'mean': yearly_returns.mean(),
                'std': yearly_returns.std(),
                'min': yearly_returns.min(),
                'max': yearly_returns.max(),
                'positive_years': len(yearly_returns[yearly_returns > 0]),
                'negative_years': len(yearly_returns[yearly_returns < 0])
            }
        }
    
    def _calculate_risk_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate detailed risk analysis."""
        # VaR at different confidence levels
        var_levels = [0.01, 0.05, 0.10]
        var_metrics = {}
        
        for level in var_levels:
            var_metrics[f'var_{int((1-level)*100)}'] = np.percentile(returns, level * 100)
        
        # Expected shortfall at different confidence levels
        es_metrics = {}
        for level in var_levels:
            var_value = np.percentile(returns, level * 100)
            es_metrics[f'es_{int((1-level)*100)}'] = returns[returns <= var_value].mean()
        
        # Tail risk measures
        tail_risk = {
            'tail_dependence': returns[returns < returns.quantile(0.05)].std() / returns.std(),
            'tail_ratio': abs(returns[returns < returns.quantile(0.05)].mean()) / returns[returns > returns.quantile(0.95)].mean() if len(returns[returns > returns.quantile(0.95)]) > 0 else 0
        }
        
        return {
            'var_metrics': var_metrics,
            'es_metrics': es_metrics,
            'tail_risk': tail_risk
        }

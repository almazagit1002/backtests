import pandas as pd
import numpy as np


class TradingBacktester:
    """
    A class for backtesting trading strategies with support for both long and short positions,
    take profit and stop loss levels, and performance metrics calculation.
    """
    
    def __init__(self, initial_budget=1000, tp_level=2, sl_level=1, fee_rate=0.005, max_positions=4):
        """
        Initialize the backtester with strategy parameters.
        
        Parameters:
        -----------
        initial_budget : float
            Initial trading budget
        tp_level : float
            Take profit level as a multiple of ATR
        sl_level : float
            Stop loss level as a multiple of ATR
        fee_rate : float
            Trading fee as a percentage (0.005 = 0.5%)
        max_positions : int
            Maximum number of simultaneous positions
        """
        self.initial_budget = initial_budget
        self.tp_level = tp_level
        self.sl_level = sl_level
        self.fee_rate = fee_rate
        self.max_positions = max_positions
        
        # Initialize result containers
        self.trades_df = None
        self.portfolio_df = None
        self.metrics = None
        
    def backtest(self, signals_df):
        """
        Run a backtest on the provided signals dataframe.
        
        Parameters:
        -----------
        signals_df : pandas DataFrame
            DataFrame with columns: 'timestamp', 'close', 'ATR', 'LongSignal', 'ShortSignal'
            
        Returns:
        --------
        self : TradingBacktester
            Returns self for method chaining
        """
        # Run the backtest
        self.trades_df, self.portfolio_df = self._backtest_strategy(signals_df)
        
        # Calculate performance metrics
        self.metrics = self._calculate_performance_metrics()
        
        return self
    
    def _backtest_strategy(self, signals_df):
        """
        Internal method to backtest a trading strategy based on signal data.
        
        Parameters:
        -----------
        signals_df : pandas DataFrame
            DataFrame with columns: 'timestamp', 'close', 'ATR', 'LongSignal', 'ShortSignal'
            
        Returns:
        --------
        trades_df : pandas DataFrame
            DataFrame with trade results
        portfolio_df : pandas DataFrame
            DataFrame with portfolio value over time
        """
        # Clone the dataframe to avoid modifying the original
        df = signals_df.copy()
        
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by date
        df = df.sort_values('timestamp')
        
        # Initialize tracking variables
        budget = self.initial_budget
        portfolio_value = self.initial_budget
        position_size = budget / self.max_positions
        active_positions = []
        
        # Lists to track trades and portfolio value
        trades = []
        portfolio_history = []
        
        # Process each day in the dataframe
        for i in range(len(df)):
            current_date = df.iloc[i]['timestamp']
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['ATR']
            
            # Record portfolio value
            portfolio_history.append({
                'timestamp': current_date,
                'Portfolio_Value': portfolio_value,
                'Budget': budget,
                'Active_Positions': len(active_positions)
            })
            
            # Check for closed positions (take profit or stop loss hit)
            closed_positions = []
            for pos_idx, pos in enumerate(active_positions):
                if pos['type'] == 'long':
                    # Check if take profit hit
                    if current_price >= pos['take_profit']:
                        profit = pos['shares'] * pos['take_profit'] * (1 - self.fee_rate) - pos['cost']
                        budget += (pos['cost'] + profit)
                        trades.append({
                            'Entry_Date': pos['entry_date'],
                            'Exit_Date': current_date,
                            'Type': 'long',
                            'Entry_Price': pos['entry_price'],
                            'Exit_Price': pos['take_profit'],
                            'Shares': pos['shares'],
                            'Cost': pos['cost'],
                            'Result': 'take_profit',
                            'Profit': profit,
                            'Return_Pct': (profit / pos['cost']) * 100
                        })
                        closed_positions.append(pos_idx)
                    # Check if stop loss hit
                    elif current_price <= pos['stop_loss']:
                        loss = pos['shares'] * pos['stop_loss'] * (1 - self.fee_rate) - pos['cost']
                        budget += (pos['cost'] + loss)
                        trades.append({
                            'Entry_Date': pos['entry_date'],
                            'Exit_Date': current_date,
                            'Type': 'long',
                            'Entry_Price': pos['entry_price'],
                            'Exit_Price': pos['stop_loss'],
                            'Shares': pos['shares'],
                            'Cost': pos['cost'],
                            'Result': 'stop_loss',
                            'Profit': loss,
                            'Return_Pct': (loss / pos['cost']) * 100
                        })
                        closed_positions.append(pos_idx)
                
                elif pos['type'] == 'short':
                    # Check if take profit hit (price goes down)
                    if current_price <= pos['take_profit']:
                        profit = pos['cost'] - pos['shares'] * pos['take_profit'] * (1 + self.fee_rate)
                        budget += (pos['cost'] + profit)
                        trades.append({
                            'Entry_Date': pos['entry_date'],
                            'Exit_Date': current_date,
                            'Type': 'short',
                            'Entry_Price': pos['entry_price'],
                            'Exit_Price': pos['take_profit'],
                            'Shares': pos['shares'],
                            'Cost': pos['cost'],
                            'Result': 'take_profit',
                            'Profit': profit,
                            'Return_Pct': (profit / pos['cost']) * 100
                        })
                        closed_positions.append(pos_idx)
                    # Check if stop loss hit (price goes up)
                    elif current_price >= pos['stop_loss']:
                        loss = pos['cost'] - pos['shares'] * pos['stop_loss'] * (1 + self.fee_rate)
                        budget += (pos['cost'] + loss)
                        trades.append({
                            'Entry_Date': pos['entry_date'],
                            'Exit_Date': current_date,
                            'Type': 'short',
                            'Entry_Price': pos['entry_price'],
                            'Exit_Price': pos['stop_loss'],
                            'Shares': pos['shares'],
                            'Cost': pos['cost'],
                            'Result': 'stop_loss',
                            'Profit': loss,
                            'Return_Pct': (loss / pos['cost']) * 100
                        })
                        closed_positions.append(pos_idx)
            
            # Remove closed positions (in reverse order to maintain correct indices)
            for pos_idx in sorted(closed_positions, reverse=True):
                active_positions.pop(pos_idx)
            
            # Recalculate position size based on current budget
            position_size = budget / self.max_positions if len(active_positions) < self.max_positions else 0
            
            # Check for new long signals
            if df.iloc[i]['LongSignal'] and position_size > 0:
                entry_price = current_price
                take_profit = entry_price + (self.tp_level * current_atr)
                stop_loss = entry_price - (self.sl_level * current_atr)
                
                # Calculate shares and cost (including fee)
                shares = position_size / (entry_price * (1 + self.fee_rate))
                cost = shares * entry_price * (1 + self.fee_rate)
                
                if cost <= position_size:  # Ensure we have enough budget
                    budget -= cost
                    active_positions.append({
                        'type': 'long',
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'shares': shares,
                        'cost': cost
                    })
            
            # Check for new short signals
            if df.iloc[i]['ShortSignal'] and position_size > 0:
                entry_price = current_price
                take_profit = entry_price - (self.tp_level * current_atr)  # Short target is lower
                stop_loss = entry_price + (self.sl_level * current_atr)    # Short stop is higher
                
                # Calculate shares and cost (including fee)
                shares = position_size / (entry_price * (1 + self.fee_rate))
                cost = shares * entry_price * (1 + self.fee_rate)
                
                if cost <= position_size:  # Ensure we have enough budget
                    budget -= cost
                    active_positions.append({
                        'type': 'short',
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'take_profit': take_profit,
                        'stop_loss': stop_loss,
                        'shares': shares,
                        'cost': cost
                    })
            
            # Update portfolio value (budget + value of open positions)
            open_positions_value = 0
            for pos in active_positions:
                if pos['type'] == 'long':
                    open_positions_value += pos['shares'] * current_price
                elif pos['type'] == 'short':
                    open_positions_value += pos['cost'] - (pos['shares'] * (current_price - pos['entry_price']))
            
            portfolio_value = budget + open_positions_value
        
        # Close any remaining positions at the last price
        last_date = df.iloc[-1]['timestamp']
        last_price = df.iloc[-1]['close']
        
        for pos in active_positions:
            if pos['type'] == 'long':
                profit = pos['shares'] * last_price * (1 - self.fee_rate) - pos['cost']
                trades.append({
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': last_date,
                    'Type': 'long',
                    'Entry_Price': pos['entry_price'],
                    'Exit_Price': last_price,
                    'Shares': pos['shares'],
                    'Cost': pos['cost'],
                    'Result': 'end_of_period',
                    'Profit': profit,
                    'Return_Pct': (profit / pos['cost']) * 100
                })
            elif pos['type'] == 'short':
                profit = pos['cost'] - pos['shares'] * last_price * (1 + self.fee_rate)
                trades.append({
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': last_date,
                    'Type': 'short',
                    'Entry_Price': pos['entry_price'],
                    'Exit_Price': last_price,
                    'Shares': pos['shares'],
                    'Cost': pos['cost'],
                    'Result': 'end_of_period',
                    'Profit': profit,
                    'Return_Pct': (profit / pos['cost']) * 100
                })
        
        
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
        return trades_df, portfolio_df
    
    def _calculate_performance_metrics(self):
        """
        Internal method to calculate performance metrics for overall, long, and short trades.
        
        Returns:
        --------
        metrics : dict
            Dictionary with performance metrics for overall, long, and short trades
        """
        def get_trade_metrics(trades):
            """Helper function to calculate metrics for a given subset of trades."""
            metrics = {}
            metrics['total_trades'] = len(trades)
            metrics['winning_trades'] = len(trades[trades['Profit'] > 0])
            metrics['losing_trades'] = len(trades[trades['Profit'] <= 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            metrics['total_profit'] = trades['Profit'].sum()
            metrics['avg_profit_per_trade'] = trades['Profit'].mean() if metrics['total_trades'] > 0 else 0
            metrics['avg_profit_winning'] = trades[trades['Profit'] > 0]['Profit'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_loss_losing'] = trades[trades['Profit'] <= 0]['Profit'].mean() if metrics['losing_trades'] > 0 else 0
            metrics['profit_factor'] = abs(metrics['avg_profit_winning'] / metrics['avg_loss_losing']) if metrics['avg_loss_losing'] != 0 else float('inf') if metrics['avg_profit_winning'] > 0 else 0
            return metrics

        # Overall metrics
        overall_metrics = get_trade_metrics(self.trades_df)

        # Separate long and short metrics
        long_trades = self.trades_df[self.trades_df['Type'] == 'long']
        short_trades = self.trades_df[self.trades_df['Type'] == 'short']
        
        long_metrics = get_trade_metrics(long_trades)
        short_metrics = get_trade_metrics(short_trades)

        # Return results
        return {
            'overall': overall_metrics,
            'long': long_metrics,
            'short': short_metrics,
            'total_return_pct': (self.portfolio_df['Portfolio_Value'].iloc[-1] / self.initial_budget - 1) * 100,
            'max_drawdown': self.portfolio_df['Portfolio_Value'].cummax().sub(self.portfolio_df['Portfolio_Value']).div(self.portfolio_df['Portfolio_Value'].cummax()).max() * 100
        }
    
    def print_results(self):
        """
        Print the backtest results in a formatted way.
        
        Returns:
        --------
        self : TradingBacktester
            Returns self for method chaining
        """
        if self.metrics is None:
            print("No backtest results to display. Run backtest() first.")
            return self
            
        print("\n----- BACKTEST -----")
        print(f"Initial Budget: ${self.initial_budget:.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_df['Portfolio_Value'].iloc[-1]:.2f}")
        
        print("\n----- OVERALL RESULTS -----")
        print(f"Total Return: {self.metrics['total_return_pct']:.2f}%")
        print(f"Total Overall Trades: {self.metrics['overall']['total_trades']}")
        print(f"Win Rate: {self.metrics['overall']['win_rate'] * 100:.2f}%")
        print(f"Profit Factor: {self.metrics['overall']['profit_factor']:.2f}")
        print(f"Average Profit per Trade: ${self.metrics['overall']['avg_profit_per_trade']:.2f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")

        print("\n----- LONG RESULTS -----")
        print(f"Total Long Trades: {self.metrics['long']['total_trades']}")
        print(f"Win Rate: {self.metrics['long']['win_rate'] * 100:.2f}%")
        print(f"Winning Trades: {self.metrics['long']['winning_trades']}")
        print(f"Losing Trades: {self.metrics['long']['losing_trades']}")
        print(f"Total Profit: ${self.metrics['long']['total_profit']:.2f}")
        print(f"Profit Factor: {self.metrics['long']['profit_factor']:.2f}")
        print(f"Average Profit per Trade: ${self.metrics['long']['avg_profit_per_trade']:.2f}")

        print("\n----- SHORT RESULTS -----")
        print(f"Total Short Trades: {self.metrics['short']['total_trades']}")
        print(f"Win Rate: {self.metrics['short']['win_rate'] * 100:.2f}%")
        print(f"Winning Trades: {self.metrics['short']['winning_trades']}")
        print(f"Losing Trades: {self.metrics['short']['losing_trades']}")
        print(f"Total Profit: ${self.metrics['short']['total_profit']:.2f}")
        print(f"Profit Factor: {self.metrics['short']['profit_factor']:.2f}")
        print(f"Average Profit per Trade: ${self.metrics['short']['avg_profit_per_trade']:.2f}")
        print("\n-------------------")
        
        return self
    
    def get_trades(self):
        """
        Get the trades dataframe from the backtest.
        
        Returns:
        --------
        trades_df : pandas DataFrame
            DataFrame with trade results
        """
        return self.trades_df
    
    def get_portfolio(self):
        """
        Get the portfolio dataframe from the backtest.
        
        Returns:
        --------
        portfolio_df : pandas DataFrame
            DataFrame with portfolio value over time
        """
        return self.portfolio_df
    
    def get_metrics(self):
        """
        Get the performance metrics from the backtest.
        
        Returns:
        --------
        metrics : dict
            Dictionary with performance metrics
        """
        return self.metrics


class StrategyOptimizer:
    """
    A class for optimizing trading strategy parameters by running multiple backtests
    with different parameter combinations.
    """
    
    def __init__(self, initial_budget=1000, fee_rate=0.005, max_positions=4):
        """
        Initialize the optimizer with fixed strategy parameters.
        
        Parameters:
        -----------
        initial_budget : float
            Initial trading budget
        fee_rate : float
            Trading fee as a percentage (0.005 = 0.5%)
        max_positions : int
            Maximum number of simultaneous positions
        """
        self.initial_budget = initial_budget
        self.fee_rate = fee_rate
        self.max_positions = max_positions
        
    def optimize(self, signals_df, tp_levels=[1, 2, 3], sl_levels=[0.5, 1, 1.5]):
        """
        Optimize strategy parameters by running multiple backtests with different parameters.
        
        Parameters:
        -----------
        signals_df : pandas DataFrame
            DataFrame with columns: 'timestamp', 'close', 'ATR', 'LongSignal', 'ShortSignal'
        tp_levels : list
            List of take profit levels to test
        sl_levels : list
            List of stop loss levels to test
            
        Returns:
        --------
        results : pandas DataFrame
            DataFrame with optimization results
        """
        results = []
        
        for tp in tp_levels:
            for sl in sl_levels:
                # Create a new backtester with these parameters
                backtester = TradingBacktester(
                    initial_budget=self.initial_budget,
                    tp_level=tp,
                    sl_level=sl,
                    fee_rate=self.fee_rate,
                    max_positions=self.max_positions
                )
                
                # Run backtest
                backtester.backtest(signals_df)
                metrics = backtester.get_metrics()
                
                # Store results
                results.append({
                    'TP_Level': tp,
                    'SL_Level': sl,
                    'Total_Return_Pct': metrics['total_return_pct'],
                    'Win_Rate': metrics['overall']['win_rate'] * 100,
                    'Profit_Factor': metrics['overall']['profit_factor'],
                    'Max_Drawdown': metrics['max_drawdown'],
                    'Total_Trades': metrics['overall']['total_trades']
                })
        
        # Convert to DataFrame and sort by Total Return
        results_df = pd.DataFrame(results).sort_values('Total_Return_Pct', ascending=False)
        return results_df
    
    def get_best_parameters(self, results_df, metric='Total_Return_Pct'):
        """
        Get the best parameters based on a specific metric.
        
        Parameters:
        -----------
        results_df : pandas DataFrame
            DataFrame with optimization results
        metric : str
            Metric to use for determining the best parameters
            
        Returns:
        --------
        best_params : dict
            Dictionary with the best parameters
        """
        best_row = results_df.sort_values(metric, ascending=False).iloc[0]
        return {
            'TP_Level': best_row['TP_Level'],
            'SL_Level': best_row['SL_Level'],
            metric: best_row[metric]
        }
    
    
    
    def print_optimization_results(self, results_df, top_n=5):
        """
        Print the top optimization results in a formatted way.
        
        Parameters:
        -----------
        results_df : pandas DataFrame
            DataFrame with optimization results
        top_n : int
            Number of top results to display
            
        Returns:
        --------
        self : StrategyOptimizer
            Returns self for method chaining
        """
        print("\n----- OPTIMIZATION RESULTS -----")
        print(f"Top {top_n} Parameter Combinations:")
        for i, row in results_df.head(top_n).iterrows():
            print(f"\nRank {i+1}:")
            print(f"  TP Level: {row['TP_Level']}")
            print(f"  SL Level: {row['SL_Level']}")
            print(f"  Total Return: {row['Total_Return_Pct']:.2f}%")
            print(f"  Win Rate: {row['Win_Rate']:.2f}%")
            print(f"  Profit Factor: {row['Profit_Factor']:.2f}")
            print(f"  Max Drawdown: {row['Max_Drawdown']:.2f}%")
            print(f"  Total Trades: {row['Total_Trades']}")
        
        print("\n-------------------")
        return self
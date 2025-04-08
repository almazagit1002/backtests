import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TradingBacktester:
    """
    A class for backtesting trading strategies with support for both long and short positions,
    take profit and stop loss levels, and performance metrics calculation.
    Includes detailed fee analysis.
    """
    
    def __init__(self, initial_budget, tp_level, sl_level, fee_rate, max_positions):
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
        buy_fee_rate : float
            Buy/entry trading fee as a percentage (0.005 = 0.5%)
        sell_fee_rate : float
            Sell/exit trading fee as a percentage (0.005 = 0.5%)
        max_positions : int
            Maximum number of simultaneous positions
        """
        self.initial_budget = initial_budget
        self.tp_level = tp_level
        self.sl_level = sl_level
        self.buy_fee_rate = fee_rate
        self.sell_fee_rate = fee_rate
        self.max_positions = max_positions
        
        # Initialize result containers
        self.trades_df = None
        self.portfolio_df = None
        self.metrics = None
        self.fee_analysis_df = None
        
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
        
        # Calculate fee analysis
        self.fee_analysis_df = self._analyze_fee_impact()
        
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
        # Prepare data
        df = self._prepare_dataframe(signals_df)
        
        # Initialize tracking variables
        budget = self.initial_budget
        portfolio_value = self.initial_budget
        position_size = budget / self.max_positions
        active_positions = []
        trades = []
        portfolio_history = []
        
        # Process each day in the dataframe
        for i in range(len(df)):
            current_date = df.iloc[i]['timestamp']
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['ATR']
            
            # Track portfolio state
            portfolio_history.append(self._create_portfolio_snapshot(
                current_date, portfolio_value, budget, len(active_positions)
            ))
            
            # Process existing positions (check for take profit/stop loss)
            budget, closed_positions = self._process_existing_positions(
                active_positions, current_date, current_price, trades, budget
            )
            
            # Remove closed positions
            active_positions = [p for i, p in enumerate(active_positions) if i not in closed_positions]
            
            # Recalculate position size for new trades
            position_size = budget / self.max_positions if len(active_positions) < self.max_positions else 0
            
            # Open new positions based on signals
            budget, new_positions = self._open_new_positions(
                df.iloc[i], current_date, current_price, current_atr, position_size, budget
            )
            active_positions.extend(new_positions)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(budget, active_positions, current_price)
        
        # Close any remaining positions at the last price
        self._close_remaining_positions(active_positions, df.iloc[-1], trades)
        
        # Create final dataframes
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
    
        return trades_df, portfolio_df

    def _prepare_dataframe(self, signals_df):
        """Prepare the dataframe for backtesting."""
        df = signals_df.copy()
        
        # Ensure date is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by date
        return df.sort_values('timestamp')

    def _create_portfolio_snapshot(self, date, portfolio_value, budget, active_positions_count):
        """Create a snapshot of portfolio state for the given date."""
        return {
            'timestamp': date,
            'Portfolio_Value': portfolio_value,
            'Budget': budget,
            'Active_Positions': active_positions_count
        }

    def _process_trade(self, pos, exit_price, result, current_date):
        """Process a trade when a position is closed with detailed fee tracking."""
        # Calculate both entry and exit fees separately for analysis
        entry_fee = pos['entry_fee']
        
        if pos['type'] == 'long':
            # For long positions, exit fee is on selling
            exit_shares_value = pos['shares'] * exit_price
            exit_fee = exit_shares_value * self.sell_fee_rate
            gross_profit = pos['shares'] * (exit_price - pos['entry_price'])
            net_proceeds = exit_shares_value - exit_fee
            net_profit = net_proceeds - pos['cost']
        elif pos['type'] == 'short':
            # For short positions, exit fee is on buying back
            exit_shares_value = pos['shares'] * exit_price
            exit_fee = exit_shares_value * self.buy_fee_rate
            gross_profit = pos['shares'] * (pos['entry_price'] - exit_price)
            buy_back_cost = exit_shares_value + exit_fee
            net_profit = pos['cost'] - buy_back_cost
        
        total_fees = entry_fee + exit_fee
        
        # Calculate fee impact metrics
        fee_percentage_of_trade_value = total_fees / (pos['shares'] * exit_price) * 100
        fee_percentage_of_gross_profit = (total_fees / abs(gross_profit) * 100) if gross_profit != 0 else float('inf')
        
        trade_record = {
            'Entry_Date': pos['entry_date'],
            'Exit_Date': current_date,
            'Type': pos['type'],
            'Entry_Price': pos['entry_price'],
            'Exit_Price': exit_price,
            'Shares': pos['shares'],
            'Cost': pos['cost'],
            'Result': result,
            'Gross_Profit': gross_profit,
            'Entry_Fee': entry_fee,
            'Exit_Fee': exit_fee,
            'Total_Fees': total_fees,
            'Net_Profit': net_profit,
            'Fee_Pct_of_Value': fee_percentage_of_trade_value,
            'Fee_Pct_of_Gross_Profit': fee_percentage_of_gross_profit,
            'Return_Pct': (net_profit / pos['cost']) * 100
        }
        
        return trade_record, net_profit

    def _process_existing_positions(self, active_positions, current_date, current_price, trades, budget):
        """Check and process existing positions for potential exit conditions."""
        closed_positions = []
        
        for pos_idx, pos in enumerate(active_positions):
            exit_triggered = False
            exit_price = None
            result = None
            
            # Check take profit and stop loss conditions
            if pos['type'] == 'long':
                if current_price >= pos['take_profit']:
                    exit_triggered = True
                    exit_price = pos['take_profit']
                    result = 'take_profit'
                elif current_price <= pos['stop_loss']:
                    exit_triggered = True
                    exit_price = pos['stop_loss']
                    result = 'stop_loss'
            elif pos['type'] == 'short':
                if current_price <= pos['take_profit']:
                    exit_triggered = True
                    exit_price = pos['take_profit']
                    result = 'take_profit'
                elif current_price >= pos['stop_loss']:
                    exit_triggered = True
                    exit_price = pos['stop_loss']
                    result = 'stop_loss'
                    
            if exit_triggered:
                trade, profit = self._process_trade(pos, exit_price, result, current_date)
                budget += (pos['cost'] + profit)
                trades.append(trade)
                closed_positions.append(pos_idx)
        
        return budget, closed_positions
    
    def _open_new_positions(self, current_row, current_date, current_price, current_atr, position_size, budget):
        """Open new positions based on trading signals."""
        new_positions = []
        
        # Check for long signal
        if current_row['LongSignal'] and position_size > 0:
            pos = self._create_position(
                'long', current_date, current_price, current_atr, position_size
            )
            if pos['cost'] <= position_size:
                budget -= pos['cost']
                new_positions.append(pos)
        
        # Check for short signal
        if current_row['ShortSignal'] and position_size > 0:
            pos = self._create_position(
                'short', current_date, current_price, current_atr, position_size
            )
            if pos['cost'] <= position_size:
                budget -= pos['cost']
                new_positions.append(pos)
        
        return budget, new_positions

    def _create_position(self, position_type, date, price, atr, position_size):
        """Create a new position dictionary with separate tracking of entry fee."""
        if position_type == 'long':
            take_profit = price + (self.tp_level * atr)
            stop_loss = price - (self.sl_level * atr)
            # For long positions, entry fee is on buying
            shares = position_size / (price * (1 + self.buy_fee_rate))
            entry_fee = shares * price * self.buy_fee_rate
            cost = shares * price + entry_fee
        else:  # short
            take_profit = price - (self.tp_level * atr)
            stop_loss = price + (self.sl_level * atr)
            # For short positions, entry fee is on selling
            shares = position_size / (price * (1 - self.sell_fee_rate))
            entry_fee = shares * price * self.sell_fee_rate
            cost = shares * price - entry_fee
        
        return {
            'type': position_type,
            'entry_date': date,
            'entry_price': price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'shares': shares,
            'cost': cost,
            'entry_fee': entry_fee
        }

    def _calculate_portfolio_value(self, budget, active_positions, current_price):
        """Calculate the current portfolio value including open positions."""
        open_positions_value = 0
        for pos in active_positions:
            if pos['type'] == 'long':
                # For longs, current value is just shares * current price
                open_positions_value += pos['shares'] * current_price
            elif pos['type'] == 'short':
                # For shorts, we need to calculate the value differently
                # The initial position value was the cost (already in our accounting)
                # The current liability is what it would cost to buy back
                initial_value = pos['shares'] * pos['entry_price']
                current_value = pos['shares'] * current_price
                # Short positions gain value as the price goes down
                open_positions_value += pos['cost'] - (current_value - initial_value)
        
        return budget + open_positions_value

    def _close_remaining_positions(self, active_positions, last_row, trades):
        """Close any remaining open positions at the end of the backtest period."""
        last_date = last_row['timestamp']
        last_price = last_row['close']
        
        for pos in active_positions:
            trade, profit = self._process_trade(pos, last_price, 'end_of_period', last_date)
            trades.append(trade)
            
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
            metrics['winning_trades'] = len(trades[trades['Net_Profit'] > 0])
            metrics['losing_trades'] = len(trades[trades['Net_Profit'] <= 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            metrics['total_profit'] = trades['Net_Profit'].sum()
            metrics['avg_profit_per_trade'] = trades['Net_Profit'].mean() if metrics['total_trades'] > 0 else 0
            metrics['avg_profit_winning'] = trades[trades['Net_Profit'] > 0]['Net_Profit'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_loss_losing'] = trades[trades['Net_Profit'] <= 0]['Net_Profit'].mean() if metrics['losing_trades'] > 0 else 0
            metrics['profit_factor'] = abs(metrics['avg_profit_winning'] / metrics['avg_loss_losing']) if metrics['avg_loss_losing'] != 0 else float('inf') if metrics['avg_profit_winning'] > 0 else 0
            
            # Fee-specific metrics
            metrics['total_fees'] = trades['Total_Fees'].sum()
            metrics['total_entry_fees'] = trades['Entry_Fee'].sum()
            metrics['total_exit_fees'] = trades['Exit_Fee'].sum()
            metrics['avg_fee_per_trade'] = trades['Total_Fees'].mean()
            metrics['fee_as_pct_of_profit'] = (metrics['total_fees'] / abs(metrics['total_profit'])) * 100 if metrics['total_profit'] != 0 else float('inf')
            
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
    
    def _analyze_fee_impact(self):
        """Analyze the impact of fees on trading performance."""
        # Average fee percentage
        fee_percentage = self.trades_df.copy()
        fee_percentage['Entry_Fee_Pct'] = (fee_percentage['Entry_Fee'] / fee_percentage['Cost']) * 100

        # Exit Value = Exit_Price * Shares
        fee_percentage['Exit_Value'] = fee_percentage['Exit_Price'] * fee_percentage['Shares']

        # Exit Fee % = Exit_Fee / Exit_Value * 100
        fee_percentage['Exit_Fee_Pct'] = (fee_percentage['Exit_Fee'] / fee_percentage['Exit_Value']) * 100

        # Calculate the averages
        avg_entry_fee_pct = fee_percentage['Entry_Fee_Pct'].mean()
        avg_exit_fee_pct = fee_percentage['Exit_Fee_Pct'].mean()

        # Create summary statistics
        fee_analysis = {
            'Total_Gross_Profit': self.trades_df['Gross_Profit'].sum(),
            'Total_Net_Profit': self.trades_df['Net_Profit'].sum(),
            'Total_Fees': self.trades_df['Total_Fees'].sum(),
            'Total_Entry_Fees': self.trades_df['Entry_Fee'].sum(),
            'Total_Exit_Fees': self.trades_df['Exit_Fee'].sum(),
            'Fee_Percentage_of_Gross_Profit': (self.trades_df['Total_Fees'].sum() / abs(self.trades_df['Gross_Profit'].sum()) * 100) 
                                            if self.trades_df['Gross_Profit'].sum() != 0 else float('inf'),
            'Average_Fee_Per_Trade': self.trades_df['Total_Fees'].mean(),
            'Average_Entry_Fee': self.trades_df['Entry_Fee'].mean(),
            'Average_Exit_Fee': self.trades_df['Exit_Fee'].mean(),
            'Median_Fee_Per_Trade': self.trades_df['Total_Fees'].median(),
            'Average_Fee_Pct_of_Value': self.trades_df['Fee_Pct_of_Value'].mean(),
            'Trades_Where_Fees_Exceeded_Profit': ((self.trades_df['Total_Fees'] > self.trades_df['Gross_Profit']) & 
                                                (self.trades_df['Gross_Profit'] > 0)).sum(),
            'Profitable_Trades_Before_Fees': (self.trades_df['Gross_Profit'] > 0).sum(),
            'Profitable_Trades_After_Fees': (self.trades_df['Net_Profit'] > 0).sum(),
            'Lost_Profits_Due_To_Fees': ((self.trades_df['Gross_Profit'] > 0) & 
                                        (self.trades_df['Net_Profit'] <= 0)).sum(),
            'Avg_Entry_Fee_Pct': avg_entry_fee_pct,  # Changed to match naming convention
            'Avg_Exit_Fee_Pct': avg_exit_fee_pct     # Changed to match naming convention
        }
        
        # Create a DataFrame for the fee analysis
        fee_analysis_df = pd.DataFrame([fee_analysis])
        
        return fee_analysis_df
        
    def print_results(self, signal_type='mix', include_fee_analysis=True):
        """
        Print the backtest results in a formatted way based on the signal type.
        
        Parameters:
        -----------
        signal_type : str, optional
            Type of signals to display - 'long', 'short', or 'mix' (default)
        include_fee_analysis : bool, optional
            Whether to include detailed fee analysis in the output
            
        Returns:
        --------
        self : TradingBacktester
            Returns self for method chaining
        """
        if self.metrics is None:
            print("No backtest results to display. Run backtest() first.")
            return self
        
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            print(f"Warning: Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
            
        print("\n----- BACKTEST -----")
        print(f"Initial Budget: ${self.initial_budget:.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_df['Portfolio_Value'].iloc[-1]:.2f}")
        
        # Always print overall results for all signal types
        print("\n----- OVERALL RESULTS -----")
        print(f"Total Return: {self.metrics['total_return_pct']:.2f}%")
        
        if signal_type == 'mix':
            # Print overall trade metrics
            print(f"Total Overall Trades: {self.metrics['overall']['total_trades']}")
            print(f"Win Rate: {self.metrics['overall']['win_rate'] * 100:.2f}%")
            print(f"Profit Factor: {self.metrics['overall']['profit_factor']:.2f}")
            print(f"Average Profit per Trade: ${self.metrics['overall']['avg_profit_per_trade']:.2f}")
            print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")

            # Print long results
            print("\n----- LONG RESULTS -----")
            print(f"Total Long Trades: {self.metrics['long']['total_trades']}")
            print(f"Win Rate: {self.metrics['long']['win_rate'] * 100:.2f}%")
            print(f"Winning Trades: {self.metrics['long']['winning_trades']}")
            print(f"Losing Trades: {self.metrics['long']['losing_trades']}")
            print(f"Total Profit: ${self.metrics['long']['total_profit']:.2f}")
            print(f"Profit Factor: {self.metrics['long']['profit_factor']:.2f}")
            print(f"Average Profit per Trade: ${self.metrics['long']['avg_profit_per_trade']:.2f}")

            # Print short results
            print("\n----- SHORT RESULTS -----")
            print(f"Total Short Trades: {self.metrics['short']['total_trades']}")
            print(f"Win Rate: {self.metrics['short']['win_rate'] * 100:.2f}%")
            print(f"Winning Trades: {self.metrics['short']['winning_trades']}")
            print(f"Losing Trades: {self.metrics['short']['losing_trades']}")
            print(f"Total Profit: ${self.metrics['short']['total_profit']:.2f}")
            print(f"Profit Factor: {self.metrics['short']['profit_factor']:.2f}")
            print(f"Average Profit per Trade: ${self.metrics['short']['avg_profit_per_trade']:.2f}")
        
        elif signal_type == 'long':
            # For long only, print just maximum drawdown from overall and then long details
            print(f"Total Trades: {self.metrics['long']['total_trades']}")
            print(f"Win Rate: {self.metrics['long']['win_rate'] * 100:.2f}%")
            print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")
            
            print("\n----- LONG RESULTS -----")
            print(f"Winning Trades: {self.metrics['long']['winning_trades']}")
            print(f"Losing Trades: {self.metrics['long']['losing_trades']}")
            print(f"Total Profit: ${self.metrics['long']['total_profit']:.2f}")
            print(f"Profit Factor: {self.metrics['long']['profit_factor']:.2f}")
            print(f"Average Profit per Trade: ${self.metrics['long']['avg_profit_per_trade']:.2f}")
        
        elif signal_type == 'short':
            # For short only, print just maximum drawdown from overall and then short details
            print(f"Total Trades: {self.metrics['short']['total_trades']}")
            print(f"Win Rate: {self.metrics['short']['win_rate'] * 100:.2f}%")
            print(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2f}%")
            
            print("\n----- SHORT RESULTS -----")
            print(f"Winning Trades: {self.metrics['short']['winning_trades']}")
            print(f"Losing Trades: {self.metrics['short']['losing_trades']}")
            print(f"Total Profit: ${self.metrics['short']['total_profit']:.2f}")
            print(f"Profit Factor: {self.metrics['short']['profit_factor']:.2f}")
            print(f"Average Profit per Trade: ${self.metrics['short']['avg_profit_per_trade']:.2f}")
        
        # Print fee analysis if requested
        if include_fee_analysis and self.fee_analysis_df is not None:
            print("\n----- FEE ANALYSIS -----")
            fa = self.fee_analysis_df.iloc[0]
            print(f"Total Gross Profit: ${fa['Total_Gross_Profit']:.2f}")
            print(f"Total Net Profit: ${fa['Total_Net_Profit']:.2f}")
            print(f"Total Fees: ${fa['Total_Fees']:.2f}")
            print(f"  - Entry Fees: ${fa['Total_Entry_Fees']:.2f}")
            print(f"  - Exit Fees: ${fa['Total_Exit_Fees']:.2f}")
            print(f"Fee Percentage of Gross Profit: {fa['Fee_Percentage_of_Gross_Profit']:.2f}%")
            print(f"Average Fee Per Trade: ${fa['Average_Fee_Per_Trade']:.2f}")
            print(f"  - Average Entry Fee: ${fa['Average_Entry_Fee']:.2f}")
            print(f"  - Average Exit Fee: ${fa['Average_Exit_Fee']:.2f}")
            print(f"Trades Where Fees Exceeded Profit: {fa['Trades_Where_Fees_Exceeded_Profit']}")
            print(f"Profitable Trades Before Fees: {fa['Profitable_Trades_Before_Fees']}")
            print(f"Profitable Trades After Fees: {fa['Profitable_Trades_After_Fees']}")
            print(f"Profitable Trades Lost Due To Fees: {fa['Lost_Profits_Due_To_Fees']}")
        
        print("\n-------------------")
        with open("mean_reversion_volatility_bands/data/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
        
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
    
    def get_fee_analysis(self):
        """
        Get the fee analysis dataframe from the backtest.
        
        Returns:
        --------
        fee_analysis_df : pandas DataFrame
            DataFrame with fee analysis metrics
        """
        return self.fee_analysis_df
    
    
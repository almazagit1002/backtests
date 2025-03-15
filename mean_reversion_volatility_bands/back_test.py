import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to simulate the backtest
def backtest_strategy(signals_df, initial_budget=1000, tp_level=2, sl_level=1, fee_rate=0.005, max_positions=4):
    """
    Backtest a trading strategy based on signal data.
    
    Parameters:
    -----------
    signals_df : pandas DataFrame
        DataFrame with columns: 'Date', 'Close', 'ATR', 'Long_Signal', 'Short_Signal'
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
        
    Returns:
    --------
    results_df : pandas DataFrame
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
    budget = initial_budget
    portfolio_value = initial_budget
    position_size = budget / max_positions
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
                    profit = pos['shares'] * pos['take_profit'] * (1 - fee_rate) - pos['cost']
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
                    loss = pos['shares'] * pos['stop_loss'] * (1 - fee_rate) - pos['cost']
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
                    profit = pos['cost'] - pos['shares'] * pos['take_profit'] * (1 + fee_rate)
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
                    loss = pos['cost'] - pos['shares'] * pos['stop_loss'] * (1 + fee_rate)
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
        position_size = budget / max_positions if len(active_positions) < max_positions else 0
        
        # Check for new long signals
        if df.iloc[i]['LongSignal'] and position_size > 0:
            entry_price = current_price
            take_profit = entry_price + (tp_level * current_atr)
            stop_loss = entry_price - (sl_level * current_atr)
            
            # Calculate shares and cost (including fee)
            shares = position_size / (entry_price * (1 + fee_rate))
            cost = shares * entry_price * (1 + fee_rate)
            
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
            take_profit = entry_price - (tp_level * current_atr)  # Short target is lower
            stop_loss = entry_price + (sl_level * current_atr)    # Short stop is higher
            
            # Calculate shares and cost (including fee)
            shares = position_size / (entry_price * (1 + fee_rate))
            cost = shares * entry_price * (1 + fee_rate)
            
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
            profit = pos['shares'] * last_price * (1 - fee_rate) - pos['cost']
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
            profit = pos['cost'] - pos['shares'] * last_price * (1 + fee_rate)
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
    
    # Create DataFrames for results
    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_history)
    
    return trades_df, portfolio_df

# Function to generate performance metrics
def calculate_performance_metrics(trades_df, portfolio_df, initial_budget):
    """
    Calculate performance metrics based on trade results.
    
    Parameters:
    -----------
    trades_df : pandas DataFrame
        DataFrame with trade results
    portfolio_df : pandas DataFrame
        DataFrame with portfolio value over time
    initial_budget : float
        Initial trading budget
        
    Returns:
    --------
    metrics : dict
        Dictionary with performance metrics
    """
    metrics = {}
    
    # Basic trade statistics
    metrics['total_trades'] = len(trades_df)
    metrics['winning_trades'] = len(trades_df[trades_df['Profit'] > 0])
    metrics['losing_trades'] = len(trades_df[trades_df['Profit'] <= 0])
    
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
    else:
        metrics['win_rate'] = 0
    
    # Profit metrics
    metrics['total_profit'] = trades_df['Profit'].sum()
    metrics['avg_profit_per_trade'] = trades_df['Profit'].mean()
    metrics['avg_profit_winning'] = trades_df[trades_df['Profit'] > 0]['Profit'].mean() if metrics['winning_trades'] > 0 else 0
    metrics['avg_loss_losing'] = trades_df[trades_df['Profit'] <= 0]['Profit'].mean() if metrics['losing_trades'] > 0 else 0
    
    # Risk metrics
    if metrics['avg_loss_losing'] != 0:
        metrics['profit_factor'] = abs(metrics['avg_profit_winning'] / metrics['avg_loss_losing']) if metrics['avg_loss_losing'] != 0 else float('inf')
    else:
        metrics['profit_factor'] = float('inf') if metrics['avg_profit_winning'] > 0 else 0
    
    # Return metrics
    metrics['total_return_pct'] = (portfolio_df['Portfolio_Value'].iloc[-1] / initial_budget - 1) * 100
    
    # Drawdown analysis
    portfolio_df['Peak'] = portfolio_df['Portfolio_Value'].cummax()
    portfolio_df['Drawdown'] = (portfolio_df['Portfolio_Value'] / portfolio_df['Peak'] - 1) * 100
    metrics['max_drawdown'] = portfolio_df['Drawdown'].min()
    
    return metrics


# Function to run the complete backtest with a sample dataset
def run_backtest(signals_df, tp_level=2, sl_level=1, initial_budget=1000, fee_rate=0.005):
    """
    Run the complete backtest and display results.
    
    Parameters:
    -----------
    signals_df : pandas DataFrame
        DataFrame with columns: 'Date', 'Close', 'ATR', 'Long_Signal', 'Short_Signal'
    tp_level : float
        Take profit level as a multiple of ATR
    sl_level : float
        Stop loss level as a multiple of ATR
    initial_budget : float
        Initial trading budget
    fee_rate : float
        Trading fee rate (e.g., 0.005 = 0.5%)
    """
    # Run the backtest
    trades_df, portfolio_df = backtest_strategy(
        signals_df=signals_df,
        initial_budget=initial_budget,
        tp_level=tp_level,
        sl_level=sl_level,
        fee_rate=fee_rate,
        max_positions=4
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trades_df, portfolio_df, initial_budget)
    
    # Display performance metrics
    print("\n----- BACKTEST RESULTS -----")
    print(f"Initial Budget: ${initial_budget:.2f}")
    print(f"Final Portfolio Value: ${portfolio_df['Portfolio_Value'].iloc[-1]:.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")

  
    
    return trades_df, portfolio_df




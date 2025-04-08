import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
import logging

# Configure logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class TradingVisualizer:
    """
    A class for visualizing trading data, indicators, signals and performance metrics.
    """
    
    def __init__(self, default_figsize=(16, 12)):
        """
        Initialize the TradingVisualizer with default figure size.
        
        Parameters:
        -----------
        default_figsize : tuple, optional
            Default figure size (width, height) to use for visualizations
        """
        self.default_figsize = default_figsize
        
    def visualize_indicators(self, df, figsize=None):
        """
        Visualize the trading indicators without signals or trades.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data and indicators
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with indicators visualization
        """
        figsize = figsize or self.default_figsize
        
        # Check if DataFrame is valid
        if df is None or len(df) == 0:
            logging.error("No data available for visualization.")
            return None
            
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Price with Keltner Channels and Bollinger Bands
        ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        
        # Add Bollinger Bands using your column names
        if 'BollingerUpper' in df.columns and 'BollingerLower' in df.columns:
            # Drop NaN values for plotting
            valid_bb = df.dropna(subset=['BollingerUpper', 'BollingerLower'])
            if not valid_bb.empty:
                ax1.plot(valid_bb.index, valid_bb['BollingerUpper'], 'r--', label='Bollinger Upper', linewidth=1)
                ax1.plot(valid_bb.index, valid_bb['BollingerLower'], 'r--', label='Bollinger Lower', linewidth=1)
                ax1.fill_between(valid_bb.index, valid_bb['BollingerUpper'], valid_bb['BollingerLower'], color='red', alpha=0.1)
        
        # Add Keltner Channels using your column names
        if 'KeltnerUpper' in df.columns and 'KeltnerLower' in df.columns:
            # Drop NaN values for plotting
            valid_kc = df.dropna(subset=['KeltnerUpper', 'KeltnerLower'])
            if not valid_kc.empty:
                ax1.plot(valid_kc.index, valid_kc['KeltnerUpper'], 'g--', label='Keltner Upper', linewidth=1)
                ax1.plot(valid_kc.index, valid_kc['KeltnerLower'], 'g--', label='Keltner Lower', linewidth=1)
                ax1.fill_between(valid_kc.index, valid_kc['KeltnerUpper'], valid_kc['KeltnerLower'], color='green', alpha=0.1)
        
        # Add EMA if available
        if 'EMA' in df.columns:
            # Drop NaN values for plotting
            valid_ema = df.dropna(subset=['EMA'])
            if not valid_ema.empty:
                ax1.plot(valid_ema.index, valid_ema['EMA'], 'g-', label='EMA', linewidth=1)
        
        ax1.set_title('Price with Volatility Bands')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.set_xticklabels([])
        
        # Plot 2: CCI using your column name
        if 'CCI' in df.columns:
            # Drop NaN values for plotting
            valid_cci = df.dropna(subset=['CCI'])
            if not valid_cci.empty:
                ax2.plot(valid_cci.index, valid_cci['CCI'], label='CCI', color='purple', linewidth=1.5)
                ax2.axhline(y=-30, color='r', linestyle='--', alpha=0.3)
                ax2.axhline(y=70, color='g', linestyle='--', alpha=0.3)
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax2.set_ylabel('CCI')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper left')
                ax2.set_xticklabels([])
        
        # Plot 3: ATR using your column name
        if 'ATR' in df.columns:
            # Drop NaN values for plotting
            valid_atr = df.dropna(subset=['ATR'])
            if not valid_atr.empty:
                ax3.plot(valid_atr.index, valid_atr['ATR'], label='ATR', color='orange', linewidth=1.5)
                ax3.set_ylabel('ATR')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper left')
        
        # Format x-axis dates
        for ax in [ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        return fig
    
    def visualize_indicators_splited(self, df, figsize=None):
        """
        Visualize the trading indicators using a 2x2 grid layout:
        - Top Left: All indicators mixed
        - Top Right: Bollinger Bands with price
        - Bottom Left: Keltner Channels with price
        - Bottom Right: EMA with price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data and indicators
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with grid visualization
        """
        figsize = figsize or self.default_figsize
        
        # Check if DataFrame is valid
        if df is None or len(df) == 0:
            logging.error("No data available for visualization.")
            return None
        
        # Create a figure with 2x2 grid
        fig = plt.figure(figsize=figsize)
        
        # Create grid of subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
        
        # 1. Top Left: All indicators mixed (same as before)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot price
        ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        
        # Add Bollinger Bands
        if 'BollingerUpper' in df.columns and 'BollingerLower' in df.columns:
            valid_bb = df.dropna(subset=['BollingerUpper', 'BollingerLower'])
            if not valid_bb.empty:
                ax1.plot(valid_bb.index, valid_bb['BollingerUpper'], 'r--', label='Bollinger Upper', linewidth=1)
                ax1.plot(valid_bb.index, valid_bb['BollingerLower'], 'r--', label='Bollinger Lower', linewidth=1)
                ax1.fill_between(valid_bb.index, valid_bb['BollingerUpper'], valid_bb['BollingerLower'], color='red', alpha=0.1)
        
        # Add Keltner Channels
        if 'KeltnerUpper' in df.columns and 'KeltnerLower' in df.columns:
            valid_kc = df.dropna(subset=['KeltnerUpper', 'KeltnerLower'])
            if not valid_kc.empty:
                ax1.plot(valid_kc.index, valid_kc['KeltnerUpper'], 'g--', label='Keltner Upper', linewidth=1)
                ax1.plot(valid_kc.index, valid_kc['KeltnerLower'], 'g--', label='Keltner Lower', linewidth=1)
                ax1.fill_between(valid_kc.index, valid_kc['KeltnerUpper'], valid_kc['KeltnerLower'], color='green', alpha=0.1)
        
        # Add EMA
        if 'EMA' in df.columns:
            valid_ema = df.dropna(subset=['EMA'])
            if not valid_ema.empty:
                ax1.plot(valid_ema.index, valid_ema['EMA'], 'g-', label='EMA', linewidth=1)
        
        ax1.set_title('All Indicators Mixed')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_xticklabels([])
        
        # 2. Top Right: Bollinger Bands with price
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        
        if 'BollingerUpper' in df.columns and 'BollingerLower' in df.columns:
            valid_bb = df.dropna(subset=['BollingerUpper', 'BollingerLower'])
            if not valid_bb.empty:
                ax2.plot(valid_bb.index, valid_bb['BollingerUpper'], 'r--', label='Bollinger Upper', linewidth=1)
                ax2.plot(valid_bb.index, valid_bb['BollingerLower'], 'r--', label='Bollinger Lower', linewidth=1)
                ax2.fill_between(valid_bb.index, valid_bb['BollingerUpper'], valid_bb['BollingerLower'], color='red', alpha=0.1)
                
                if 'SMA' in df.columns:  # SMA is often the midline for Bollinger Bands
                    valid_sma = df.dropna(subset=['SMA'])
                    if not valid_sma.empty:
                        ax2.plot(valid_sma.index, valid_sma['SMA'], 'r-', label='SMA', linewidth=1)
        
        ax2.set_title('Bollinger Bands with Price')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        # 3. Bottom Left: Keltner Channels with price
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        
        if 'KeltnerUpper' in df.columns and 'KeltnerLower' in df.columns:
            valid_kc = df.dropna(subset=['KeltnerUpper', 'KeltnerLower'])
            if not valid_kc.empty:
                ax3.plot(valid_kc.index, valid_kc['KeltnerUpper'], 'g--', label='Keltner Upper', linewidth=1)
                ax3.plot(valid_kc.index, valid_kc['KeltnerLower'], 'g--', label='Keltner Lower', linewidth=1)
                ax3.fill_between(valid_kc.index, valid_kc['KeltnerUpper'], valid_kc['KeltnerLower'], color='green', alpha=0.1)
        
        ax3.set_title('Keltner Channels with Price')
        ax3.set_ylabel('Price')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left', fontsize=8)
        
        # 4. Bottom Right: EMA with price
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
        
        if 'EMA' in df.columns:
            valid_ema = df.dropna(subset=['EMA'])
            if not valid_ema.empty:
                ax4.plot(valid_ema.index, valid_ema['EMA'], 'g-', label='EMA', linewidth=1)
        
        ax4.set_title('EMA with Price')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.set_yticklabels([])
        
        # Format x-axis dates for all subplots
        for ax in [ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig

    def plot_trading_signals(self, df, signal_type='mix', figsize=None):
        """
        Create a subplot showing price, EMA, and trading signals based on signal_type:
        - 'mix': Two subplots (Long signals, Short signals)
        - 'long': One subplot (Long signals only)
        - 'short': One subplot (Short signals only)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing trading data and signals
        signal_type : str, optional
            Type of signals to display - 'long', 'short', or 'mix' (default)
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with trading signals
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            print(f"Warning: Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Determine number of subplots based on signal_type
        if signal_type == 'mix':
            n_plots = 2
            figsize = figsize or (15, 10)
        else:
            n_plots = 1
            figsize = figsize or (15, 6)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a figure with appropriate number of subplots
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # For single subplot, make axs indexable
        if n_plots == 1:
            axs = [axs]
        
        fig.suptitle('Trading Signals Analysis', fontsize=16)
        
        # Common time axis formatting
        date_format = mdates.DateFormatter('%Y-%m-%d')
        
        # Plot based on signal_type
        if signal_type in ['long', 'mix']:
            # Long signals subplot
            long_idx = 0
            ax_long = axs[long_idx]
            
            # Plot price and EMA
            ax_long.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
            ax_long.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
            
            # Highlight long signal areas
            long_signals = df[df['LongSignal'] == True]
            if not long_signals.empty:
                ax_long.scatter(long_signals['timestamp'], long_signals['close'], 
                              color='green', marker='^', s=30, label='Long Signal')
    
            # Set titles and labels
            ax_long.set_title('Long Signals', fontsize=14)
            ax_long.set_ylabel('Price', fontsize=12)
            ax_long.grid(True, alpha=0.3)
            ax_long.legend(loc='upper left')
            
            # If only long signals, add x-axis label and formatting
            if signal_type == 'long':
                ax_long.set_xlabel('Date', fontsize=12)
                ax_long.xaxis.set_major_formatter(date_format)
        
        if signal_type in ['short', 'mix']:
            # Short signals subplot
            short_idx = 0 if signal_type == 'short' else 1
            ax_short = axs[short_idx]
            
            # Plot price and EMA
            ax_short.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
            ax_short.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
            
            # Highlight short signal areas
            short_signals = df[df['ShortSignal'] == True]
            if not short_signals.empty:
                ax_short.scatter(short_signals['timestamp'], short_signals['close'], 
                               color='red', marker='v', s=30, label='Short Signal')
            
            # Set titles and labels
            ax_short.set_title('Short Signals', fontsize=14)
            ax_short.set_xlabel('Date', fontsize=12)
            ax_short.set_ylabel('Price', fontsize=12)
            ax_short.grid(True, alpha=0.3)
            ax_short.legend(loc='upper left')
            
            # Format x-axis dates
            ax_short.xaxis.set_major_formatter(date_format)
        
        # Auto-format date labels
        fig.autofmt_xdate()
        
        # Tight layout and adjust for the suptitle
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        return fig

    def plot_backtest_results(self, signals_df, portfolio_df, trades_df, signal_type='mix', figsize=None):
        """
        Create a figure with appropriate subplots based on signal_type:
        - 'mix': Three subplots (Long trades, Short trades, Portfolio value)
        - 'long': Two subplots (Long trades, Portfolio value)
        - 'short': Two subplots (Short trades, Portfolio value)
        
        Parameters:
        -----------
        signals_df : pandas DataFrame
            DataFrame with price and signal data
        portfolio_df : pandas DataFrame
            DataFrame with portfolio value over time
        trades_df : pandas DataFrame
            DataFrame with trade details
        signal_type : str, optional
            Type of signals to display - 'long', 'short', or 'mix' (default)
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with appropriate subplots
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            print(f"Warning: Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Filter trades by type
        long_trades = trades_df[trades_df['Type'] == 'long']
        short_trades = trades_df[trades_df['Type'] == 'short']
        
        # Determine number of subplots based on signal_type
        if signal_type == 'mix':
            n_plots = 3
            figsize = figsize or (14, 16)
            height_ratios = [1, 1, 1]
        else:
            n_plots = 2
            figsize = figsize or (14, 12)
            height_ratios = [1.5, 1]
        
        # Create a figure with appropriate subplots
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})
        
        # Index for portfolio subplot (varies based on signal_type)
        portfolio_idx = n_plots - 1
        
        # Plot long trades if signal_type is 'long' or 'mix'
        if signal_type in ['long', 'mix']:
            long_idx = 0
            axs[long_idx].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
            
            # Mark long entries and exits
            for _, trade in long_trades.iterrows():
                entry_date = trade['Entry_Date']
                entry_price = trade['Entry_Price']
                result = trade['Result']
                
                # Different colors based on result
                exit_color = 'green' if trade['Net_Profit'] > 0 else 'red'
                
                axs[long_idx].scatter(entry_date, entry_price, color='green', marker='^', s=100, 
                                    label='Long Entry' if _ == long_trades.index[0] else "")
                
                # Draw line to exit
                exit_date = trade['Exit_Date']
                exit_price = trade['Exit_Price']
                axs[long_idx].scatter(exit_date, exit_price, color=exit_color, marker='o', s=100, 
                                     label=f'Exit ({result})' if _ == long_trades.index[0] else "")
                axs[long_idx].plot([entry_date, exit_date], [entry_price, exit_price], 'g--', alpha=0.3)
                
                # Annotate profit percentage
                profit_pct = trade['Return_Pct']
                axs[long_idx].annotate(f"{profit_pct:.1f}%", 
                                      xy=(exit_date, exit_price),
                                      xytext=(5, 0),
                                      textcoords="offset points",
                                      fontsize=8,
                                      color=exit_color)
            
            axs[long_idx].set_title('Long Trades Analysis')
            axs[long_idx].set_ylabel('Price')
            axs[long_idx].grid(True)
            
            # Create legend without duplicates
            handles, labels = axs[long_idx].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[long_idx].legend(by_label.values(), by_label.keys())
        
        # Plot short trades if signal_type is 'short' or 'mix'
        if signal_type in ['short', 'mix']:
            short_idx = 0 if signal_type == 'short' else 1
            axs[short_idx].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
            
            # Mark short entries and exits
            for _, trade in short_trades.iterrows():
                entry_date = trade['Entry_Date']
                entry_price = trade['Entry_Price']
                result = trade['Result']
                
                # Different colors based on result
                exit_color = 'green' if trade['Net_Profit'] > 0 else 'red'
                
                axs[short_idx].scatter(entry_date, entry_price, color='red', marker='v', s=100, 
                                      label='Short Entry' if _ == short_trades.index[0] else "")
                
                # Draw line to exit
                exit_date = trade['Exit_Date']
                exit_price = trade['Exit_Price']
                axs[short_idx].scatter(exit_date, exit_price, color=exit_color, marker='o', s=100, 
                                      label=f'Exit ({result})' if _ == short_trades.index[0] else "")
                axs[short_idx].plot([entry_date, exit_date], [entry_price, exit_price], 'r--', alpha=0.3)
                
                # Annotate profit percentage
                profit_pct = trade['Return_Pct']
                axs[short_idx].annotate(f"{profit_pct:.1f}%", 
                                       xy=(exit_date, exit_price),
                                       xytext=(5, 0),
                                       textcoords="offset points",
                                       fontsize=8,
                                       color=exit_color)
            
            axs[short_idx].set_title('Short Trades Analysis')
            axs[short_idx].set_ylabel('Price')
            axs[short_idx].grid(True)
            
            # Create legend without duplicates
            handles, labels = axs[short_idx].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[short_idx].legend(by_label.values(), by_label.keys())
        
        # Portfolio value plot (always shown)
        axs[portfolio_idx].plot(portfolio_df['timestamp'], portfolio_df['Portfolio_Value'], 
                               label='Portfolio Value', color='blue')
        
        # Add peak line for drawdown visualization
        axs[portfolio_idx].plot(portfolio_df['timestamp'], portfolio_df['Peak'], 
                               label='Peak Value', color='green', linestyle='--', alpha=0.5)
        
        # Calculate drawdown percentage for shading
        dd = (portfolio_df['Portfolio_Value'] / portfolio_df['Peak'] - 1) * -100  # Convert to positive percentage
        
        # Shade drawdown areas
        for i in range(len(portfolio_df) - 1):
            if dd.iloc[i] > 0:  # If there's a drawdown
                axs[portfolio_idx].fill_between([portfolio_df['timestamp'].iloc[i], portfolio_df['timestamp'].iloc[i+1]], 
                                              [portfolio_df['Portfolio_Value'].iloc[i], portfolio_df['Portfolio_Value'].iloc[i+1]],
                                              [portfolio_df['Peak'].iloc[i], portfolio_df['Peak'].iloc[i+1]],
                                              color='red', alpha=0.3)
        
        axs[portfolio_idx].set_title('Portfolio Value Over Time')
        axs[portfolio_idx].set_ylabel('Value')
        axs[portfolio_idx].set_xlabel('Date')
        axs[portfolio_idx].grid(True)
        axs[portfolio_idx].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig

    def plot_performance_comparison(self, metrics, signal_type='mix', figsize=None):
        """
        Create a figure comparing performance metrics based on signal_type:
        - 'mix': Shows overall, long, and short trade metrics
        - 'long': Shows only overall and long trade metrics
        - 'short': Shows only overall and short trade metrics

        Parameters:
        -----------
        metrics : dict
            Dictionary containing performance metrics for overall, long, and short trades.
        signal_type : str, optional
            Type of signals to display - 'long', 'short', or 'mix' (default)
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with performance metrics comparison.
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            print(f"Warning: Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        figsize = figsize or (14, 10)
        
        categories = ['Win Rate (%)', 'Profit Factor', 'Total Profit', 'Avg Profit per Trade']
        
        # Determine which metrics to include based on signal_type
        labels = []
        data = []
        colors = {}
        
        # Always include overall metrics
        if signal_type == 'long':
            # For long-only strategy, overall = long
            overall_values = [
                metrics['long']['win_rate'] * 100,
                metrics['long']['profit_factor'],
                metrics['long']['total_profit'],
                metrics['long']['avg_profit_per_trade']
            ]
            labels = ['Long']
            data = [overall_values]
            colors = {'Long': 'green'}
            
        elif signal_type == 'short':
            # For short-only strategy, overall = short
            overall_values = [
                metrics['short']['win_rate'] * 100,
                metrics['short']['profit_factor'],
                metrics['short']['total_profit'],
                metrics['short']['avg_profit_per_trade']
            ]
            labels = ['Short']
            data = [overall_values]
            colors = {'Short': 'red'}
            
        else:  # 'mix'
            # Extract values for each category
            long_values = [
                metrics['long']['win_rate'] * 100,
                metrics['long']['profit_factor'],
                metrics['long']['total_profit'],
                metrics['long']['avg_profit_per_trade']
            ]
            
            short_values = [
                metrics['short']['win_rate'] * 100,
                metrics['short']['profit_factor'],
                metrics['short']['total_profit'],
                metrics['short']['avg_profit_per_trade']
            ]
            
            overall_values = [
                metrics['overall']['win_rate'] * 100,
                metrics['overall']['profit_factor'],
                metrics['overall']['total_profit'],
                metrics['overall']['avg_profit_per_trade']
            ]
            
            labels = ['Long', 'Short', 'Overall']
            data = [long_values, short_values, overall_values]
            colors = {'Long': 'green', 'Short': 'red', 'Overall': 'blue'}

        x = np.arange(len(categories))  # Label positions
        width = 0.3  # Width of bars

        fig, axes = plt.subplots(2, 2, figsize=figsize)  # 2x2 subplots
        axes = axes.flatten()  # Flatten to 1D array for easier iteration

        for i, ax in enumerate(axes):
            for j, (label, values) in enumerate(zip(labels, data)):
                ax.bar(j, values[i], width, label=label, color=colors[label])
            
            ax.set_title(categories[i])
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a single legend for all plots
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        fig.legend(handles, labels, loc='upper right', fontsize=12)

        # Set appropriate title based on signal_type
        if signal_type == 'mix':
            fig.suptitle('Performance Comparison: Overall vs Long vs Short', fontsize=14)
        elif signal_type == 'long':
            fig.suptitle('Long-Only Performance Metrics', fontsize=14)
        else:  # 'short'
            fig.suptitle('Short-Only Performance Metrics', fontsize=14)
            
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust for the title

        return fig
    def plot_profit_histograms(self, trades_df, signal_type='mix', figsize=None):
        """
        Create histograms for trade profits based on signal_type:
        - 'mix': Three histograms (Overall, Long trades, Short trades)
        - 'long': One histogram (Long trades)
        - 'short': One histogram (Short trades)

        Parameters:
        -----------
        trades_df : pandas DataFrame
            DataFrame containing 'Type' (long/short) and 'Profit' columns.
        signal_type : str, optional
            Type of signals to display - 'long', 'short', or 'mix' (default)
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure containing the histograms.
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            print(f"Warning: Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        if signal_type == 'mix':
            n_plots = 3
            figsize = figsize or (10, 12)
        else:
            n_plots = 1
            figsize = figsize or (10, 5)

        long_profits = trades_df[trades_df['Type'] == 'long']['Net_Profit']
        short_profits = trades_df[trades_df['Type'] == 'short']['Net_Profit']

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = np.array([axes])

        def add_vertical_line(ax):
            ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Profit Threshold')
            ax.legend()

        if signal_type == 'mix':
            axes[0].hist(trades_df['Net_Profit'], bins=max(10, len(trades_df) // 10), 
                        color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_title("Overall Profit Distribution")
            axes[0].set_xlabel("Profit")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            add_vertical_line(axes[0])

            axes[1].hist(long_profits, bins=max(10, len(long_profits) // 10), 
                        color='green', alpha=0.7, edgecolor='black')
            axes[1].set_title("Long Trades Profit Distribution")
            axes[1].set_xlabel("Profit")
            axes[1].set_ylabel("Frequency")
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)
            add_vertical_line(axes[1])

            axes[2].hist(short_profits, bins=max(10, len(short_profits) // 10), 
                        color='red', alpha=0.7, edgecolor='black')
            axes[2].set_title("Short Trades Profit Distribution")
            axes[2].set_xlabel("Profit")
            axes[2].set_ylabel("Frequency")
            axes[2].grid(axis='y', linestyle='--', alpha=0.7)
            add_vertical_line(axes[2])

        elif signal_type == 'long':
            axes[0].hist(long_profits, bins=max(10, len(long_profits) // 10), 
                        color='green', alpha=0.7, edgecolor='black')
            axes[0].set_title("Long Trades Profit Distribution")
            axes[0].set_xlabel("Profit")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            add_vertical_line(axes[0])

        else:  # signal_type == 'short'
            axes[0].hist(short_profits, bins=max(10, len(short_profits) // 10), 
                        color='red', alpha=0.7, edgecolor='black')
            axes[0].set_title("Short Trades Profit Distribution")
            axes[0].set_xlabel("Profit")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)
            add_vertical_line(axes[0])

        plt.tight_layout()
        return fig
        
    def save_figure(self, fig, filename, dpi=300):
        """
        Save a figure to a file.
        
        Parameters:
        -----------
        fig : matplotlib Figure
            The figure to save
        filename : str
            The filename to save to
        dpi : int, optional
            The resolution in dots per inch
        """
        if fig is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            logging.info(f"Figure saved to {filename}")
        else:
            logging.warning(f"Unable to save figure: figure object is None")

    def plot_heatmaps(self, optimization_results):
        """
        Create and return a figure with 4 heatmaps showing different parameter combinations.

        Parameters:
        -----------
        optimization_results : pandas DataFrame
            DataFrame containing optimization results.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing the heatmaps.
        """

        # Define the most interesting parameter combinations
        param_combinations = [
            ('cci_period', 'bollinger_period'),
            ('tp_level', 'sl_level'),
            ('max_positions', 'window_size'),
            ('CCI_up_threshold', 'CCI_low_threshold')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 grid of plots

        for ax, (x_param, y_param) in zip(axes.flat, param_combinations):
            try:
                # Pivot the data for heatmap
                heatmap_data = optimization_results.pivot_table(index=y_param, columns=x_param, values='total_return_pct')

                # Plot heatmap
                sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f", ax=ax)
                ax.set_title(f'Heatmap of {y_param} vs {x_param} (Total Return %)')
                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)

            except Exception as e:
                ax.set_title(f"Error: {x_param} vs {y_param}")
                print(f"Error during pivot operation for {x_param} vs {y_param}: {e}")

        plt.tight_layout()
        return fig
    def visualize_fee_impact(self, trades_df, fee_analysis_df):
        """
        Visualize the impact of fees on trading performance.

        Returns:
        --------
        fig : matplotlib Figure
            Figure with fee impact visualizations
        """
        if trades_df is None or len(trades_df) == 0:
            print("No trade data available. Run backtest() first.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Gross Profit (bars) with Total Fees (line) per trade
        trades_sample = trades_df.head(min(50, len(trades_df)))
        trade_indices = range(len(trades_sample))

        axes[0, 0].bar(trade_indices, trades_sample['Gross_Profit'], label='Gross Profit', color='#1f77b4')
        axes[0, 0].plot(trade_indices, trades_sample['Total_Fees'], label='Fees', color='#ff7f0e', linewidth=2, marker='o')
        axes[0, 0].set_title('Gross Profit vs Total Fees by Trade')
        axes[0, 0].set_xlabel('Trade Index')
        axes[0, 0].set_ylabel('Amount ($)')
        axes[0, 0].legend()

        # Plot 2: Total Gross vs Fees vs Net Profit (with default colors)
        fa = fee_analysis_df.iloc[0]
        profit_labels = ['Gross Profit', 'Total Fees', 'Net Profit']
        profit_values = [
            fa['Total_Gross_Profit'],
            fa['Total_Fees'],
            fa['Total_Net_Profit']
        ]
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = axes[0, 1].bar(profit_labels, profit_values, color=bar_colors)
        axes[0, 1].set_title('Total Gross vs Fees vs Net Profit')
        axes[0, 1].set_ylabel('Amount ($)')

        for bar, value in zip(bars, profit_values):
            pct = (value / profit_values[0]) * 100 if profit_values[0] != 0 else 0
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, color='black')

        # Plot 3: Distribution of fee percentage of trade value
        axes[1, 0].hist(trades_df['Fee_Pct_of_Value'], bins=20, color='#ff7f0e', edgecolor='black')
        axes[1, 0].axvline(1.0, color='black', linestyle='--', linewidth=2, label='1.0 Threshold')
        axes[1, 0].set_title('Distribution of Fee % of Trade Value')
        axes[1, 0].set_xlabel('Fee % of Trade Value')
        axes[1, 0].set_ylabel('Number of Trades')
        axes[1, 0].legend()

        # Plot 4: Impact of fees on profitability
        labels = ['Profitable After Fees', 'Would be Profitable Without Fees', 'Unprofitable Regardless']
        sizes = [
            fa['Profitable_Trades_After_Fees'],
            fa['Trades_Where_Fees_Exceeded_Profit'],
            len(trades_df) - fa['Profitable_Trades_After_Fees'] -fa['Trades_Where_Fees_Exceeded_Profit']
        ]
        axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Impact of Fees on Profitability')

        plt.tight_layout()
        return fig

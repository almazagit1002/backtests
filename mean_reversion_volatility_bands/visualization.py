import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
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

    def plot_trading_signals(self, df, figsize=None):
        """
        Create a subplot with 2 figures showing price, EMA, and trading signals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing trading data and signals
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with trading signals
        """
        figsize = figsize or (15, 10)
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a figure with 2 subplots stacked vertically
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle('Trading Signals Analysis', fontsize=16)
        
        # Common time axis formatting
        date_format = mdates.DateFormatter('%Y-%m-%d')
        
        # -------------------------------------------------------------------------
        # Top subplot: Price, EMA, and Long Signals
        # -------------------------------------------------------------------------
        ax1 = axs[0]
        
        # Plot price and EMA
        ax1.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
        ax1.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
        
        # Highlight long signal areas
        long_signals = df[df['LongSignal'] == True]
        if not long_signals.empty:
            ax1.scatter(long_signals['timestamp'], long_signals['close'], 
                       color='green', marker='^', s=30, label='Long Signal')

        # Set titles and labels
        ax1.set_title('Long Signals', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # -------------------------------------------------------------------------
        # Bottom subplot: Price, EMA, and Short Signals
        # -------------------------------------------------------------------------
        ax2 = axs[1]
        
        # Plot price and EMA
        ax2.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
        ax2.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
        
        # Highlight short signal areas
        short_signals = df[df['ShortSignal'] == True]
        if not short_signals.empty:
            ax2.scatter(short_signals['timestamp'], short_signals['close'], 
                       color='red', marker='v', s=30, label='Short Signal')
        
        # Set titles and labels
        ax2.set_title('Short Signals', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Price', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # Auto-format date labels
        
        # Tight layout and show
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
        
        return fig

    def plot_backtest_results(self, signals_df, portfolio_df, trades_df, figsize=None):
        """
        Create a single figure with three subplots:
        1. Long trades
        2. Short trades
        3. Portfolio value
        
        Parameters:
        -----------
        signals_df : pandas DataFrame
            DataFrame with price and signal data
        portfolio_df : pandas DataFrame
            DataFrame with portfolio value over time
        trades_df : pandas DataFrame
            DataFrame with trade details
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with three subplots
        """
        figsize = figsize or (14, 16)
        
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1, 1]})
        
        # Filter trades by type
        long_trades = trades_df[trades_df['Type'] == 'long']
        short_trades = trades_df[trades_df['Type'] == 'short']
        
        # Subplot 1: Long Trades
        axs[0].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
        
        # Mark long entries and exits
        for _, trade in long_trades.iterrows():
            entry_date = trade['Entry_Date']
            entry_price = trade['Entry_Price']
            result = trade['Result']
            
            # Different colors based on result
            exit_color = 'green' if trade['Profit'] > 0 else 'red'
            
            axs[0].scatter(entry_date, entry_price, color='green', marker='^', s=100, label='Long Entry' if _ == long_trades.index[0] else "")
            
            # Draw line to exit
            exit_date = trade['Exit_Date']
            exit_price = trade['Exit_Price']
            axs[0].scatter(exit_date, exit_price, color=exit_color, marker='o', s=100, label=f'Exit ({result})' if _ == long_trades.index[0] else "")
            axs[0].plot([entry_date, exit_date], [entry_price, exit_price], 'g--', alpha=0.3)
            
            # Annotate profit percentage
            profit_pct = trade['Return_Pct']
            axs[0].annotate(f"{profit_pct:.1f}%", 
                           xy=(exit_date, exit_price),
                           xytext=(5, 0),
                           textcoords="offset points",
                           fontsize=8,
                           color=exit_color)
        
        axs[0].set_title('Long Trades Analysis')
        axs[0].set_ylabel('Price')
        axs[0].grid(True)
        
        # Create legend without duplicates
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys())
        
        # Subplot 2: Short Trades
        axs[1].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
        
        # Mark short entries and exits
        for _, trade in short_trades.iterrows():
            entry_date = trade['Entry_Date']
            entry_price = trade['Entry_Price']
            result = trade['Result']
            
            # Different colors based on result
            exit_color = 'green' if trade['Profit'] > 0 else 'red'
            
            axs[1].scatter(entry_date, entry_price, color='red', marker='v', s=100, label='Short Entry' if _ == short_trades.index[0] else "")
            
            # Draw line to exit
            exit_date = trade['Exit_Date']
            exit_price = trade['Exit_Price']
            axs[1].scatter(exit_date, exit_price, color=exit_color, marker='o', s=100, label=f'Exit ({result})' if _ == short_trades.index[0] else "")
            axs[1].plot([entry_date, exit_date], [entry_price, exit_price], 'r--', alpha=0.3)
            
            # Annotate profit percentage
            profit_pct = trade['Return_Pct']
            axs[1].annotate(f"{profit_pct:.1f}%", 
                           xy=(exit_date, exit_price),
                           xytext=(5, 0),
                           textcoords="offset points",
                           fontsize=8,
                           color=exit_color)
        
        axs[1].set_title('Short Trades Analysis')
        axs[1].set_ylabel('Price')
        axs[1].grid(True)
        
        # Create legend without duplicates
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())
        
        # Subplot 3: Portfolio value
        axs[2].plot(portfolio_df['timestamp'], portfolio_df['Portfolio_Value'], label='Portfolio Value', color='blue')
        
        # Add peak line for drawdown visualization
        axs[2].plot(portfolio_df['timestamp'], portfolio_df['Peak'], label='Peak Value', color='green', linestyle='--', alpha=0.5)
        
        # Calculate drawdown percentage for shading
        dd = (portfolio_df['Portfolio_Value'] / portfolio_df['Peak'] - 1) * -100  # Convert to positive percentage
        
        # Shade drawdown areas
        for i in range(len(portfolio_df) - 1):
            if dd[i] > 0:  # If there's a drawdown
                axs[2].fill_between([portfolio_df['timestamp'].iloc[i], portfolio_df['timestamp'].iloc[i+1]], 
                                   [portfolio_df['Portfolio_Value'].iloc[i], portfolio_df['Portfolio_Value'].iloc[i+1]],
                                   [portfolio_df['Peak'].iloc[i], portfolio_df['Peak'].iloc[i+1]],
                                   color='red', alpha=0.3)
        
        axs[2].set_title('Portfolio Value Over Time')
        axs[2].set_ylabel('Value')
        axs[2].set_xlabel('Date')
        axs[2].grid(True)
        axs[2].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig

    def plot_performance_comparison(self, metrics, figsize=None):
        """
        Create a figure comparing overall, long, and short trade performance metrics.

        Parameters:
        -----------
        metrics : dict
            Dictionary containing performance metrics for overall, long, and short trades.
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure with performance metrics comparison.
        """
        figsize = figsize or (14, 10)
        
        categories = ['Win Rate (%)', 'Profit Factor', 'Total Profit', 'Avg Profit per Trade']
        
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
        
        x = np.arange(len(categories))  # Label positions
        width = 0.3  # Width of bars

        fig, axes = plt.subplots(2, 2, figsize=figsize)  # 2x2 subplots
        axes = axes.flatten()  # Flatten to 1D array for easier iteration

        colors = {'Long': 'green', 'Short': 'red', 'Overall': 'blue'}
        labels = ['Long', 'Short', 'Overall']
        data = [long_values, short_values, overall_values]

        for i, ax in enumerate(axes):
            for j, (label, values) in enumerate(zip(labels, data)):
                ax.bar(j, values[i], width, label=label, color=colors[label])
            
            ax.set_title(categories[i])
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a single legend for all plots
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        fig.legend(handles, labels, loc='upper right', fontsize=12)

        fig.suptitle('Performance Comparison: Overall vs Long vs Short', fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust for the title

        return fig

    def plot_profit_histograms(self, trades_df, figsize=None):
        """
        Create histograms for overall, long, and short trade profits.

        Parameters:
        -----------
        trades_df : pandas DataFrame
            DataFrame containing 'Type' (long/short) and 'Profit' columns.
        figsize : tuple, optional
            Figure size (width, height)
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure containing the histograms.
        """
        figsize = figsize or (10, 12)
        
        # Define number of bins based on data size
        num_bins = max(10, len(trades_df) // 10)  # At least 10 bins, roughly 10 per bin group
        
        # Separate data by type
        long_profits = trades_df[trades_df['Type'] == 'long']['Profit']
        short_profits = trades_df[trades_df['Type'] == 'short']['Profit']
        overall_profits = trades_df['Profit']

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Overall Profit Histogram
        axes[0].hist(overall_profits, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_title("Overall Profit Distribution")
        axes[0].set_xlabel("Profit")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Long Trades Profit Histogram
        axes[1].hist(long_profits, bins=num_bins, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title("Long Trades Profit Distribution")
        axes[1].set_xlabel("Profit")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Short Trades Profit Histogram
        axes[2].hist(short_profits, bins=num_bins, color='red', alpha=0.7, edgecolor='black')
        axes[2].set_title("Short Trades Profit Distribution")
        axes[2].set_xlabel("Profit")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)

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
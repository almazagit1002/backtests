
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
import logging
import traceback

from utils.error_handler import ErrorHandler

# Configure logging - reduce log noise from visualization libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class TradingVisualizer:
    """
    A class for visualizing trading data, indicators, signals and performance metrics.
    Enhanced with robust error handling.
    """
    
    def __init__(self, default_figsize=(16, 12), debug_mode=False):
        """
        Initialize the TradingVisualizer with default figure size.
        
        Parameters:
        -----------
        default_figsize : tuple, optional
            Default figure size (width, height) to use for visualizations
        debug_mode : bool, optional
            Whether to enable debug level logging
        """
        self.default_figsize = default_figsize
        
        # Initialize error handler
        self.error_handler = ErrorHandler(logger_name="TradingVisualizer", debug_mode=debug_mode)
        self.error_handler.logger.info("TradingVisualizer initialized")
        
    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode for more verbose output.
        
        Parameters:
        -----------
        enabled : bool
            Whether to enable debug mode
        """
        self.error_handler.set_debug_mode(enabled)
        self.error_handler.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        return self
        
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
        
        # Check if DataFrame is valid using error handler
        required_columns = ['close']
        is_valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot visualize indicators: {message}")
            return None
        
        # Validate presence of indicator columns
        indicator_columns = ['timestamp', 'open', 'high', 'low', 'close', 'num_data_points', 'EMA',
                            'ATR', 'KeltnerUpper', 'KeltnerLower', 'SMA', 'mean_deviation', 'CCI',
                            'StdDev', 'BollingerUpper', 'BollingerLower']
                             
        # Get a list of available indicator columns
        available_indicators = [col for col in indicator_columns if col in df.columns]
        
        if not available_indicators:
            self.error_handler.logger.warning("No indicator columns found in DataFrame. Basic price chart will be displayed.")
        else:
            self.error_handler.logger.info(f"Available indicators: {available_indicators}")
        
        try:
            # Create subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot 1: Price with Keltner Channels and Bollinger Bands
            ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
            
            # Add Bollinger Bands if available
            bollinger_cols = ['bollinger_upper', 'bollinger_lower', 'bollinger_middle']
            if all(col in df.columns for col in bollinger_cols):
                # Drop NaN values for plotting
                valid_bb = df.dropna(subset=bollinger_cols)
                if not valid_bb.empty:
                    ax1.plot(valid_bb.index, valid_bb['bollinger_upper'], 'r--', label='Bollinger Upper', linewidth=1)
                    ax1.plot(valid_bb.index, valid_bb['bollinger_lower'], 'r--', label='Bollinger Lower', linewidth=1)
                    ax1.fill_between(valid_bb.index, valid_bb['bollinger_upper'], valid_bb['bollinger_lower'], color='red', alpha=0.1)
                    self.error_handler.logger.debug("Added Bollinger Bands to visualization")
                else:
                    self.error_handler.logger.warning("Bollinger Bands columns contain only NaN values")
            
            # Add Keltner Channels if available
            keltner_cols = ['keltner_upper', 'keltner_lower', 'keltner_middle']
            if all(col in df.columns for col in keltner_cols):
                # Drop NaN values for plotting
                valid_kc = df.dropna(subset=keltner_cols)
                if not valid_kc.empty:
                    ax1.plot(valid_kc.index, valid_kc['keltner_upper'], 'g--', label='Keltner Upper', linewidth=1)
                    ax1.plot(valid_kc.index, valid_kc['keltner_lower'], 'g--', label='Keltner Lower', linewidth=1)
                    ax1.fill_between(valid_kc.index, valid_kc['keltner_upper'], valid_kc['keltner_lower'], color='green', alpha=0.1)
                    self.error_handler.logger.debug("Added Keltner Channels to visualization")
                else:
                    self.error_handler.logger.warning("Keltner Channels columns contain only NaN values")
            
            # Add EMA if available
            if 'ema' in df.columns.str.lower():
                ema_col = next(col for col in df.columns if col.lower() == 'ema')
                valid_ema = df.dropna(subset=[ema_col])
                if not valid_ema.empty:
                    ax1.plot(valid_ema.index, valid_ema[ema_col], 'g-', label='EMA', linewidth=1)
                    self.error_handler.logger.debug("Added EMA to visualization")
            
            ax1.set_title('Price with Volatility Bands')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.set_xticklabels([])
            
            # Plot 2: CCI if available
            if 'cci' in df.columns:
                # Drop NaN values for plotting
                valid_cci = df.dropna(subset=['cci'])
                if not valid_cci.empty:
                    ax2.plot(valid_cci.index, valid_cci['cci'], label='CCI', color='purple', linewidth=1.5)
                    ax2.axhline(y=-100, color='r', linestyle='--', alpha=0.3)
                    ax2.axhline(y=100, color='g', linestyle='--', alpha=0.3)
                    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    ax2.set_ylabel('CCI')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='upper left')
                    ax2.set_xticklabels([])
                    self.error_handler.logger.debug("Added CCI to visualization")
                else:
                    self.error_handler.logger.warning("CCI column contains only NaN values")
                    ax2.text(0.5, 0.5, 'No valid CCI data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'CCI indicator not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                self.error_handler.logger.debug("CCI indicator not available")
            
            # Plot 3: ATR if available
            if 'atr' in df.columns:
                # Drop NaN values for plotting
                valid_atr = df.dropna(subset=['atr'])
                if not valid_atr.empty:
                    ax3.plot(valid_atr.index, valid_atr['atr'], label='ATR', color='orange', linewidth=1.5)
                    ax3.set_ylabel('ATR')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend(loc='upper left')
                    self.error_handler.logger.debug("Added ATR to visualization")
                else:
                    self.error_handler.logger.warning("ATR column contains only NaN values")
                    ax3.text(0.5, 0.5, 'No valid ATR data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'ATR indicator not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes)
                self.error_handler.logger.debug("ATR indicator not available")
            
            # Format x-axis dates
            for ax in [ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.tight_layout()
            plt.xticks(rotation=45)
            
            self.error_handler.logger.info("Indicators visualization created successfully")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating indicators visualization: {str(e)}")
            self.error_handler.logger.debug(traceback.format_exc())
            return None
    
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
        
        # Check if DataFrame is valid using error handler
        required_columns = ['close']
        is_valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot visualize split indicators: {message}")
            return None
        
        # Map original column names to lowercase for case-insensitive matching
        col_map = {col.lower(): col for col in df.columns}
        
        # Check if we have enough indicators to make the visualization useful
        indicator_types = []
        if any(col.startswith('bollinger_') for col in col_map.keys()):
            indicator_types.append('bollinger')
        if any(col.startswith('keltner_') for col in col_map.keys()):
            indicator_types.append('keltner')
        if 'ema' in col_map:
            indicator_types.append('ema')
            
        if not indicator_types:
            self.error_handler.logger.warning("No indicators found for split visualization")
            
        try:
            # Create a figure with 2x2 grid
            fig = plt.figure(figsize=figsize)
            
            # Create grid of subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
            
            # 1. Top Left: All indicators mixed
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Plot price
            ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
            
            # Add Bollinger Bands if available
            bollinger_columns = {
                'upper': col_map.get('bollinger_upper'),
                'lower': col_map.get('bollinger_lower'),
                'middle': col_map.get('bollinger_middle')
            }
            
            if all(bollinger_columns.values()):
                valid_bb = df.dropna(subset=list(bollinger_columns.values()))
                if not valid_bb.empty:
                    upper_col = bollinger_columns['upper']
                    lower_col = bollinger_columns['lower']
                    
                    ax1.plot(valid_bb.index, valid_bb[upper_col], 'r--', label='Bollinger Upper', linewidth=1)
                    ax1.plot(valid_bb.index, valid_bb[lower_col], 'r--', label='Bollinger Lower', linewidth=1)
                    ax1.fill_between(valid_bb.index, valid_bb[upper_col], valid_bb[lower_col], color='red', alpha=0.1)
            
            # Add Keltner Channels if available
            keltner_columns = {
                'upper': col_map.get('keltner_upper'),
                'lower': col_map.get('keltner_lower'),
                'middle': col_map.get('keltner_middle')
            }
            
            if all(keltner_columns.values()):
                valid_kc = df.dropna(subset=list(keltner_columns.values()))
                if not valid_kc.empty:
                    upper_col = keltner_columns['upper']
                    lower_col = keltner_columns['lower']
                    
                    ax1.plot(valid_kc.index, valid_kc[upper_col], 'g--', label='Keltner Upper', linewidth=1)
                    ax1.plot(valid_kc.index, valid_kc[lower_col], 'g--', label='Keltner Lower', linewidth=1)
                    ax1.fill_between(valid_kc.index, valid_kc[upper_col], valid_kc[lower_col], color='green', alpha=0.1)
            
            # Add EMA if available
            ema_col = col_map.get('ema')
            if ema_col:
                valid_ema = df.dropna(subset=[ema_col])
                if not valid_ema.empty:
                    ax1.plot(valid_ema.index, valid_ema[ema_col], 'g-', label='EMA', linewidth=1)
            
            ax1.set_title('All Indicators Mixed')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.set_xticklabels([])
            
            # 2. Top Right: Bollinger Bands with price
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
            
            if all(bollinger_columns.values()):
                valid_bb = df.dropna(subset=list(bollinger_columns.values()))
                if not valid_bb.empty:
                    upper_col = bollinger_columns['upper']
                    lower_col = bollinger_columns['lower']
                    middle_col = bollinger_columns['middle']
                    
                    ax2.plot(valid_bb.index, valid_bb[upper_col], 'r--', label='Bollinger Upper', linewidth=1)
                    ax2.plot(valid_bb.index, valid_bb[lower_col], 'r--', label='Bollinger Lower', linewidth=1)
                    ax2.plot(valid_bb.index, valid_bb[middle_col], 'r-', label='Bollinger Middle', linewidth=1)
                    ax2.fill_between(valid_bb.index, valid_bb[upper_col], valid_bb[lower_col], color='red', alpha=0.1)
                    self.error_handler.logger.debug("Added Bollinger Bands to split visualization")
                else:
                    ax2.text(0.5, 0.5, 'No valid Bollinger Bands data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes)
                    self.error_handler.logger.warning("Bollinger Bands columns contain only NaN values")
            else:
                ax2.text(0.5, 0.5, 'Bollinger Bands not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                self.error_handler.logger.debug("Bollinger Bands not available")
            
            ax2.set_title('Bollinger Bands with Price')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left', fontsize=8)
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            
            # 3. Bottom Left: Keltner Channels with price
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
            
            if all(keltner_columns.values()):
                valid_kc = df.dropna(subset=list(keltner_columns.values()))
                if not valid_kc.empty:
                    upper_col = keltner_columns['upper']
                    lower_col = keltner_columns['lower']
                    middle_col = keltner_columns['middle']
                    
                    ax3.plot(valid_kc.index, valid_kc[upper_col], 'g--', label='Keltner Upper', linewidth=1)
                    ax3.plot(valid_kc.index, valid_kc[lower_col], 'g--', label='Keltner Lower', linewidth=1)
                    ax3.plot(valid_kc.index, valid_kc[middle_col], 'g-', label='Keltner Middle', linewidth=1)
                    ax3.fill_between(valid_kc.index, valid_kc[upper_col], valid_kc[lower_col], color='green', alpha=0.1)
                    self.error_handler.logger.debug("Added Keltner Channels to split visualization")
                else:
                    ax3.text(0.5, 0.5, 'No valid Keltner Channels data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes)
                    self.error_handler.logger.warning("Keltner Channels columns contain only NaN values")
            else:
                ax3.text(0.5, 0.5, 'Keltner Channels not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes)
                self.error_handler.logger.debug("Keltner Channels not available")
            
            ax3.set_title('Keltner Channels with Price')
            ax3.set_ylabel('Price')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left', fontsize=8)
            
            # 4. Bottom Right: EMA with price
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1.5)
            
            if ema_col:
                valid_ema = df.dropna(subset=[ema_col])
                if not valid_ema.empty:
                    ax4.plot(valid_ema.index, valid_ema[ema_col], 'g-', label='EMA', linewidth=1)
                    self.error_handler.logger.debug("Added EMA to split visualization")
                else:
                    ax4.text(0.5, 0.5, 'No valid EMA data', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes)
                    self.error_handler.logger.warning("EMA column contains only NaN values")
            else:
                ax4.text(0.5, 0.5, 'EMA not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
                self.error_handler.logger.debug("EMA not available")
            
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
            
            self.error_handler.logger.info("Split indicators visualization created successfully")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating split indicators visualization: {str(e)}")
            self.error_handler.logger.debug(traceback.format_exc())
            return None

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
            self.error_handler.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Check required columns based on signal_type
        required_columns = ['close', 'timestamp']
        
        if signal_type in ['long', 'mix']:
            required_columns.append('LongSignal')
        if signal_type in ['short', 'mix']:
            required_columns.append('ShortSignal')
            
        # Check if DataFrame is valid
        is_valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot plot trading signals: {message}")
            return None
            
        # Check if 'EMA' column exists (optional)
        ema_available = 'EMA' in df.columns
        if not ema_available:
            self.error_handler.logger.warning("EMA column not found, will plot without EMA")
        
        try:
            # Determine number of subplots based on signal_type
            if signal_type == 'mix':
                n_plots = 2
                figsize = figsize or (15, 10)
            else:
                n_plots = 1
                figsize = figsize or (15, 6)
            
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check if we have signals to plot
            if signal_type in ['long', 'mix'] and df['LongSignal'].sum() == 0:
                self.error_handler.logger.warning("No long signals found in data")
                
            if signal_type in ['short', 'mix'] and df['ShortSignal'].sum() == 0:
                self.error_handler.logger.warning("No short signals found in data")
            
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
                
                # Plot price
                ax_long.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
                
                # Plot EMA if available
                if ema_available:
                    ax_long.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
                
                # Highlight long signal areas
                long_signals = df[df['LongSignal'] == True]
                if not long_signals.empty:
                    ax_long.scatter(long_signals['timestamp'], long_signals['close'], 
                                  color='green', marker='^', s=50, label='Long Signal')
                    self.error_handler.logger.debug(f"Plotted {len(long_signals)} long signals")
                else:
                    self.error_handler.logger.warning("No long signals to plot")
        
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
                
                # Plot price
                ax_short.plot(df['timestamp'], df['close'], label='Close Price', color='black', linewidth=1.2)
                
                # Plot EMA if available
                if ema_available:
                    ax_short.plot(df['timestamp'], df['EMA'], label='EMA', color='blue', linewidth=1, alpha=0.8)
                
                # Highlight short signal areas
                short_signals = df[df['ShortSignal'] == True]
                if not short_signals.empty:
                    ax_short.scatter(short_signals['timestamp'], short_signals['close'], 
                                   color='red', marker='v', s=50, label='Short Signal')
                    self.error_handler.logger.debug(f"Plotted {len(short_signals)} short signals")
                else:
                    self.error_handler.logger.warning("No short signals to plot")
                
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
            
            self.error_handler.logger.info(f"Trading signals visualization created successfully for {signal_type} strategy")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating trading signals visualization: {str(e)}")
            self.error_handler.logger.debug(traceback.format_exc())
            return None

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
       
        # Validate input DataFrames
        signal_valid, signal_msg = self.error_handler.check_dataframe_validity(
            signals_df, required_columns=['timestamp', 'close'], min_rows=2
        )
        
        portfolio_valid, portfolio_msg = self.error_handler.check_dataframe_validity(
            portfolio_df, required_columns=['timestamp', 'Portfolio_Value', 'Peak'], min_rows=2
        )
        
        trades_valid, trades_msg = self.error_handler.check_dataframe_validity(
            trades_df, required_columns=['Type', 'Entry_Date', 'Exit_Date', 'Entry_Price', 
                                       'Exit_Price', 'Net_Profit', 'Return_Pct', 'Result'], min_rows=1
        )
        
        # If any validations fail, log and return None
        if not signal_valid:
            self.error_handler.logger.error(f"Invalid signals DataFrame: {signal_msg}")
            return None
        
        if not portfolio_valid:
            self.error_handler.logger.error(f"Invalid portfolio DataFrame: {portfolio_msg}")
            return None
        
        if not trades_valid:
            self.error_handler.logger.error(f"Invalid trades DataFrame: {trades_msg}")
            return None
        
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            self.error_handler.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Safely filter trades by type
        try:
            long_trades = trades_df[trades_df['Type'] == 'long']
            short_trades = trades_df[trades_df['Type'] == 'short']
        except Exception as e:
            self.error_handler.logger.error(f"Error filtering trades: {str(e)}")
            return None
        
        # Determine number of subplots based on signal_type
        if signal_type == 'mix':
            n_plots = 3
            figsize = figsize or (14, 16)
            height_ratios = [1, 1, 1]
        else:
            n_plots = 2
            figsize = figsize or (14, 12)
            height_ratios = [1.5, 1]
        
        # Create a figure with appropriate subplots - wrapped in try-except
        try:
          
            fig, axs = plt.subplots(n_plots, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})
            
            # Ensure axs is always a list/array for consistent indexing
            if n_plots == 1:
                axs = np.array([axs])
                
            # Index for portfolio subplot (varies based on signal_type)
            portfolio_idx = n_plots - 1
            
            # Plot long trades if signal_type is 'long' or 'mix'
            if signal_type in ['long', 'mix']:
                long_idx = 0
                axs[long_idx].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
                
                # Mark long entries and exits - wrapped in try-except
                try:
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
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting long trades: {str(e)}")
                
                axs[long_idx].set_title('Long Trades Analysis')
                axs[long_idx].set_ylabel('Price')
                axs[long_idx].grid(True)
                
                # Create legend without duplicates - wrapped in try-except
                try:
                    handles, labels = axs[long_idx].get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    axs[long_idx].legend(by_label.values(), by_label.keys())
                except Exception as e:
                    self.error_handler.logger.warning(f"Error creating legend for long trades: {str(e)}")
            
            # Plot short trades if signal_type is 'short' or 'mix'
            if signal_type in ['short', 'mix']:
                short_idx = 0 if signal_type == 'short' else 1
                axs[short_idx].plot(signals_df['timestamp'], signals_df['close'], label='Close Price', color='black')
                
                # Mark short entries and exits - wrapped in try-except
                try:
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
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting short trades: {str(e)}")
                
                axs[short_idx].set_title('Short Trades Analysis')
                axs[short_idx].set_ylabel('Price')
                axs[short_idx].grid(True)
                
                # Create legend without duplicates - wrapped in try-except
                try:
                    handles, labels = axs[short_idx].get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    axs[short_idx].legend(by_label.values(), by_label.keys())
                except Exception as e:
                    self.error_handler.logger.warning(f"Error creating legend for short trades: {str(e)}")
            
            # Portfolio value plot (always shown) - wrapped in try-except
            try:
                axs[portfolio_idx].plot(portfolio_df['timestamp'], portfolio_df['Portfolio_Value'], 
                                       label='Portfolio Value', color='blue')
                
                # Add peak line for drawdown visualization
                axs[portfolio_idx].plot(portfolio_df['timestamp'], portfolio_df['Peak'], 
                                       label='Peak Value', color='green', linestyle='--', alpha=0.5)
                
                # Calculate drawdown percentage for shading
                # Use error handler's safe division method
                dd = self.error_handler.handle_division(
                    portfolio_df['Portfolio_Value'] - portfolio_df['Peak'],
                    portfolio_df['Peak'],
                    replace_value=0.0
                ) * -100  # Convert to positive percentage
                
                # Shade drawdown areas
                if len(portfolio_df) > 1:
                    for i in range(len(portfolio_df) - 1):
                        if dd[i] > 0:  # If there's a drawdown
                            axs[portfolio_idx].fill_between([portfolio_df['timestamp'].iloc[i], portfolio_df['timestamp'].iloc[i+1]], 
                                                          [portfolio_df['Portfolio_Value'].iloc[i], portfolio_df['Portfolio_Value'].iloc[i+1]],
                                                          [portfolio_df['Peak'].iloc[i], portfolio_df['Peak'].iloc[i+1]],
                                                          color='red', alpha=0.3)
            except Exception as e:
                self.error_handler.logger.error(f"Error plotting portfolio value: {str(e)}")
            
            axs[portfolio_idx].set_title('Portfolio Value Over Time')
            axs[portfolio_idx].set_ylabel('Value')
            axs[portfolio_idx].set_xlabel('Date')
            axs[portfolio_idx].grid(True)
            axs[portfolio_idx].legend()
            
            # Adjust layout
            plt.tight_layout()
            
            self.error_handler.logger.info("Successfully created backtest results visualization")
            return fig
        
        except Exception as e:
            self.error_handler.logger.error(f"Error creating visualization: {str(e)}")
            return None

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
        
        
        # Validate inputs
        if not isinstance(metrics, dict):
            self.error_handler.logger.error(f"Invalid metrics input: expected dict, got {type(metrics)}")
            return None
            
        # Check if metrics contain required keys
        if signal_type == 'mix':
            required_keys = ['overall', 'long', 'short']
        elif signal_type == 'long':
            required_keys = ['long']
        else:  # 'short'
            required_keys = ['short']
            
        for key in required_keys:
            if key not in metrics:
                self.error_handler.logger.error(f"Missing required metrics key: {key}")
                return None
                
            # Check for required metrics in each category
            required_metrics = ['win_rate', 'profit_factor', 'total_profit', 'avg_profit_per_trade']
            missing_metrics = [m for m in required_metrics if m not in metrics[key]]
            
            if missing_metrics:
                self.error_handler.logger.error(f"Missing required metrics in {key}: {missing_metrics}")
                return None
        
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            self.error_handler.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        figsize = figsize or (14, 10)
        
        categories = ['Win Rate (%)', 'Profit Factor', 'Total Profit', 'Avg Profit per Trade']
        
        # Determine which metrics to include based on signal_type
        labels = []
        data = []
        colors = {}
        
        # Safety wrapper for extracting metrics values
        def safe_extract_metrics(metric_dict, metrics_list):
            try:
                return [
                    metric_dict['win_rate'] * 100,
                    metric_dict['profit_factor'],
                    metric_dict['total_profit'],
                    metric_dict['avg_profit_per_trade']
                ]
            except Exception as e:
                self.error_handler.logger.error(f"Failed to extract metrics: {str(e)}")
                return [0, 0, 0, 0]  # Return zeros as fallback
        
        # Always include overall metrics
        try:
            if signal_type == 'long':
                # For long-only strategy, overall = long
                overall_values = safe_extract_metrics(metrics['long'], categories)
                labels = ['Long']
                data = [overall_values]
                colors = {'Long': 'green'}
                
            elif signal_type == 'short':
                # For short-only strategy, overall = short
                overall_values = safe_extract_metrics(metrics['short'], categories)
                labels = ['Short']
                data = [overall_values]
                colors = {'Short': 'red'}
                
            else:  # 'mix'
                # Extract values for each category
                long_values = safe_extract_metrics(metrics['long'], categories)
                short_values = safe_extract_metrics(metrics['short'], categories)
                overall_values = safe_extract_metrics(metrics['overall'], categories)
                
                labels = ['Long', 'Short', 'Overall']
                data = [long_values, short_values, overall_values]
                colors = {'Long': 'green', 'Short': 'red', 'Overall': 'blue'}
        except Exception as e:
            self.error_handler.logger.error(f"Error preparing metric data: {str(e)}")
            return None

        # Initialize plotting
        try:
         
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
            
            self.error_handler.logger.info("Successfully created performance comparison visualization")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating performance visualization: {str(e)}")
            return None

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
        
        # Validate trades DataFrame
        trades_valid, trades_msg = self.error_handler.check_dataframe_validity(
            trades_df, required_columns=['Type', 'Net_Profit'], min_rows=1
        )
        
        if not trades_valid:
            self.error_handler.logger.error(f"Invalid trades DataFrame: {trades_msg}")
            return None
        
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            self.error_handler.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Set up plot parameters based on signal_type
        if signal_type == 'mix':
            n_plots = 3
            figsize = figsize or (10, 12)
        else:
            n_plots = 1
            figsize = figsize or (10, 5)

        # Safely filter data
        try:
            long_profits = trades_df[trades_df['Type'] == 'long']['Net_Profit']
            short_profits = trades_df[trades_df['Type'] == 'short']['Net_Profit']
        except Exception as e:
            self.error_handler.logger.error(f"Error filtering profit data: {str(e)}")
            return None

        # Create plots
        try:
           
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
            if n_plots == 1:
                axes = np.array([axes])

            def add_vertical_line(ax):
                ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Profit Threshold')
                ax.legend()

            # Safely calculate bin count
            def safe_bins(data_series):
                if data_series.empty:
                    return 10  # Default if empty
                return max(10, len(data_series) // 10)

            if signal_type == 'mix':
                # Overall profit distribution
                try:
                    axes[0].hist(trades_df['Net_Profit'], bins=safe_bins(trades_df['Net_Profit']), 
                                color='blue', alpha=0.7, edgecolor='black')
                    axes[0].set_title("Overall Profit Distribution")
                    axes[0].set_xlabel("Profit")
                    axes[0].set_ylabel("Frequency")
                    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
                    add_vertical_line(axes[0])
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting overall profit histogram: {str(e)}")

                # Long trades profit distribution
                try:
                    axes[1].hist(long_profits, bins=safe_bins(long_profits), 
                                color='green', alpha=0.7, edgecolor='black')
                    axes[1].set_title("Long Trades Profit Distribution")
                    axes[1].set_xlabel("Profit")
                    axes[1].set_ylabel("Frequency")
                    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
                    add_vertical_line(axes[1])
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting long profit histogram: {str(e)}")

                # Short trades profit distribution
                try:
                    axes[2].hist(short_profits, bins=safe_bins(short_profits), 
                                color='red', alpha=0.7, edgecolor='black')
                    axes[2].set_title("Short Trades Profit Distribution")
                    axes[2].set_xlabel("Profit")
                    axes[2].set_ylabel("Frequency")
                    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
                    add_vertical_line(axes[2])
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting short profit histogram: {str(e)}")

            elif signal_type == 'long':
                try:
                    axes[0].hist(long_profits, bins=safe_bins(long_profits), 
                                color='green', alpha=0.7, edgecolor='black')
                    axes[0].set_title("Long Trades Profit Distribution")
                    axes[0].set_xlabel("Profit")
                    axes[0].set_ylabel("Frequency")
                    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
                    add_vertical_line(axes[0])
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting long profit histogram: {str(e)}")

            else:  # signal_type == 'short'
                try:
                    axes[0].hist(short_profits, bins=safe_bins(short_profits), 
                                color='red', alpha=0.7, edgecolor='black')
                    axes[0].set_title("Short Trades Profit Distribution")
                    axes[0].set_xlabel("Profit")
                    axes[0].set_ylabel("Frequency")
                    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
                    add_vertical_line(axes[0])
                except Exception as e:
                    self.error_handler.logger.error(f"Error plotting short profit histogram: {str(e)}")

            plt.tight_layout()
            self.error_handler.logger.info("Successfully created profit histograms")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating profit histograms: {str(e)}")
            return None
        
   

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
       
        # Validate input DataFrame
        opt_valid, opt_msg = self.error_handler.check_dataframe_validity(
            optimization_results, 
            required_columns=['total_return_pct'], 
            min_rows=4  # Need at least some combinations for meaningful heatmaps
        )
        
        if not opt_valid:
            self.error_handler.logger.error(f"Invalid optimization results: {opt_msg}")
            return None
            
        # Define the most interesting parameter combinations
        param_combinations = [
            ('cci_period', 'bollinger_period'),
            ('tp_level', 'sl_level'),
            ('max_positions', 'window_size'),
            ('CCI_up_threshold', 'CCI_low_threshold')
        ]
        
        # Check for required parameters
        for x_param, y_param in param_combinations:
            if x_param not in optimization_results.columns:
                self.error_handler.logger.warning(f"Missing parameter column: {x_param}")
            if y_param not in optimization_results.columns:
                self.error_handler.logger.warning(f"Missing parameter column: {y_param}")
        
        # Create the figure with error handling
        try:

            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 grid of plots

            for ax, (x_param, y_param) in zip(axes.flat, param_combinations):
                try:
                    # Check if both parameters exist
                    if x_param not in optimization_results.columns or y_param not in optimization_results.columns:
                        ax.set_title(f"Missing parameter: {x_param} or {y_param}")
                        self.error_handler.logger.warning(f"Skipping heatmap for {x_param} vs {y_param}: parameter missing")
                        continue
                        
                    # Check if we have multiple values for each parameter
                    x_unique = optimization_results[x_param].nunique()
                    y_unique = optimization_results[y_param].nunique()
                    
                    if x_unique <= 1 or y_unique <= 1:
                        ax.set_title(f"Not enough unique values: {x_param}({x_unique}) vs {y_param}({y_unique})")
                        self.error_handler.logger.warning(
                            f"Skipping heatmap: {x_param}({x_unique}) vs {y_param}({y_unique}) needs multiple values"
                        )
                        continue
                    
                    # Pivot the data for heatmap
                    heatmap_data = self.error_handler.safe_calculation(
                        func=lambda: optimization_results.pivot_table(
                            index=y_param, 
                            columns=x_param, 
                            values='total_return_pct'
                        ),
                        default_value=None
                    )
                    
                    if heatmap_data is None:
                        ax.set_title(f"Error creating pivot table: {x_param} vs {y_param}")
                        self.error_handler.logger.error(f"Failed to create pivot table for {x_param} vs {y_param}")
                        continue

                    # Plot heatmap
                    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f", ax=ax)
                    ax.set_title(f'Heatmap of {y_param} vs {x_param} (Total Return %)')
                    ax.set_xlabel(x_param)
                    ax.set_ylabel(y_param)

                except Exception as e:
                    ax.set_title(f"Error: {x_param} vs {y_param}")
                    self.error_handler.logger.error(f"Error during heatmap creation for {x_param} vs {y_param}: {str(e)}")

            plt.tight_layout()
            self.error_handler.logger.info("Successfully created parameter heatmaps")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating heatmap figure: {str(e)}")
            return None

    def visualize_fee_impact(self, trades_df, fee_analysis_df):
        """
        Visualize the impact of fees on trading performance.

        Parameters:
        -----------
        trades_df : pandas DataFrame
            DataFrame containing trade details with fees
        fee_analysis_df : pandas DataFrame
            DataFrame containing fee impact analysis summary

        Returns:
        --------
        fig : matplotlib Figure
            Figure with fee impact visualizations
        """
       
        # Validate input DataFrames
        trades_valid, trades_msg = self.error_handler.check_dataframe_validity(
            trades_df, 
            required_columns=['Gross_Profit', 'Total_Fees', 'Fee_Pct_of_Value'], 
            min_rows=1
        )
        
        if not trades_valid:
            self.error_handler.logger.error(f"Invalid trades DataFrame: {trades_msg}")
            return None
        
        fee_valid, fee_msg = self.error_handler.check_dataframe_validity(
            fee_analysis_df,
            required_columns=['Total_Gross_Profit', 'Total_Fees', 'Total_Net_Profit',
                            'Profitable_Trades_After_Fees', 'Trades_Where_Fees_Exceeded_Profit'],
            min_rows=1
        )
        
        if not fee_valid:
            self.error_handler.logger.error(f"Invalid fee analysis DataFrame: {fee_msg}")
            return None

        # Create the visualization with error handling
        try:
        
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Gross Profit (bars) with Total Fees (line) per trade
            try:
                trades_sample = trades_df.head(min(50, len(trades_df)))
                trade_indices = range(len(trades_sample))

                axes[0, 0].bar(trade_indices, trades_sample['Gross_Profit'], label='Gross Profit', color='#1f77b4')
                axes[0, 0].plot(trade_indices, trades_sample['Total_Fees'], label='Fees', color='#ff7f0e', 
                            linewidth=2, marker='o')
                axes[0, 0].set_title('Gross Profit vs Total Fees by Trade')
                axes[0, 0].set_xlabel('Trade Index')
                axes[0, 0].set_ylabel('Amount ($)')
                axes[0, 0].legend()
            except Exception as e:
                self.error_handler.logger.error(f"Error creating profit vs fees plot: {str(e)}")
                axes[0, 0].text(0.5, 0.5, 'Error creating plot', ha='center', va='center')

            # Plot 2: Total Gross vs Fees vs Net Profit (with default colors)
            try:
                fa = fee_analysis_df.iloc[0]
                profit_labels = ['Gross Profit', 'Total Fees', 'Net Profit']
                profit_values = [
                    fa['Total_Gross_Profit'],
                    fa['Total_Fees'],
                    fa['Total_Net_Profit']
                ]
                
                # Safety check for nan or inf values
                for i, value in enumerate(profit_values):
                    if pd.isna(value) or np.isinf(value):
                        self.error_handler.logger.warning(f"Invalid value for {profit_labels[i]}: {value}, replacing with 0")
                        profit_values[i] = 0
                        
                bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                bars = axes[0, 1].bar(profit_labels, profit_values, color=bar_colors)
                axes[0, 1].set_title('Total Gross vs Fees vs Net Profit')
                axes[0, 1].set_ylabel('Amount ($)')

                # Calculate percentages safely using error handler's safe division
                for bar, value in zip(bars, profit_values):
                    pct = self.error_handler.handle_division(
                        value * 100, 
                        profit_values[0], 
                        replace_value=0.0
                    )
                    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, color='black')
            except Exception as e:
                self.error_handler.logger.error(f"Error creating totals comparison plot: {str(e)}")
                axes[0, 1].text(0.5, 0.5, 'Error creating plot', ha='center', va='center')

            # Plot 3: Distribution of fee percentage of trade value
            try:
                # Check for extreme values and clip if necessary
                fee_pct = trades_df['Fee_Pct_of_Value']
                q99 = fee_pct.quantile(0.99)
                if fee_pct.max() > q99 * 10:  # If max is more than 10x the 99th percentile
                    self.error_handler.logger.warning(
                        f"Extreme fee percentage values detected (max={fee_pct.max()}), clipping to {q99 * 5}"
                    )
                    fee_pct = fee_pct.clip(upper=q99 * 5)
                
                axes[1, 0].hist(fee_pct, bins=20, color='#ff7f0e', edgecolor='black')
                axes[1, 0].axvline(1.0, color='black', linestyle='--', linewidth=2, label='1.0 Threshold')
                axes[1, 0].set_title('Distribution of Fee % of Trade Value')
                axes[1, 0].set_xlabel('Fee % of Trade Value')
                axes[1, 0].set_ylabel('Number of Trades')
                axes[1, 0].legend()
            except Exception as e:
                self.error_handler.logger.error(f"Error creating fee distribution plot: {str(e)}")
                axes[1, 0].text(0.5, 0.5, 'Error creating plot', ha='center', va='center')

            # Plot 4: Impact of fees on profitability
            try:
                labels = ['Profitable After Fees', 'Would be Profitable Without Fees', 'Unprofitable Regardless']
                
                # Safely extract data with validations
                profitable_after_fees = int(fa['Profitable_Trades_After_Fees'])
                fees_exceeded_profit = int(fa['Trades_Where_Fees_Exceeded_Profit'])
                total_trades = len(trades_df)
                
                # Validation
                if profitable_after_fees < 0 or fees_exceeded_profit < 0:
                    self.error_handler.logger.error("Invalid negative trade counts detected")
                    raise ValueError("Invalid negative trade counts")
                    
                if profitable_after_fees + fees_exceeded_profit > total_trades:
                    self.error_handler.logger.error(
                        f"Trade counts inconsistent: {profitable_after_fees} + {fees_exceeded_profit} > {total_trades}"
                    )
                    # Fix the data for visualization
                    fees_exceeded_profit = min(fees_exceeded_profit, total_trades - profitable_after_fees)
                
                sizes = [
                    profitable_after_fees,
                    fees_exceeded_profit,
                    total_trades - profitable_after_fees - fees_exceeded_profit
                ]
                
                # Make sure there are no negative values
                sizes = [max(0, s) for s in sizes]
                
                # Only create pie chart if we have non-zero values
                if sum(sizes) > 0:
                    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
                    axes[1, 1].set_title('Impact of Fees on Profitability')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No valid trade counts available', ha='center', va='center')
                    self.error_handler.logger.warning("No valid trade counts for fee impact pie chart")
                    
            except Exception as e:
                self.error_handler.logger.error(f"Error creating fee impact pie chart: {str(e)}")
                axes[1, 1].text(0.5, 0.5, 'Error creating plot', ha='center', va='center')

            plt.tight_layout()
            self.error_handler.logger.info("Successfully created fee impact visualization")
            return fig
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating fee impact visualization: {str(e)}")
            return None
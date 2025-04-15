import traceback
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Optional, Tuple

from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator
from mean_reversion_volatility_bands.visualization import TradingVisualizer
from mean_reversion_volatility_bands.back_test import TradingBacktester

from utils.utils import load_config
from utils.error_handler import ErrorHandler

# Configure logging - consider removing this since ErrorHandler has its own logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = 'mean_reversion_volatility_bands/config.yaml'


class Backtester:
    """
    Backtester for the mean reversion volatility bands trading strategy.
    This class handles the entire backtesting workflow including:
    - Loading data
    - Calculating technical indicators
    - Generating trading signals
    - Running the backtest simulation
    - Visualizing and exporting results
    """

    def __init__(self, config, df=None):
        """
        Initialize the backtester with data and configuration.
        
        Args:
            config (dict): Configuration parameters
            df (pd.DataFrame, optional): Historical price data. Defaults to None.
        """
        self.config = config
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.indicators_df = None
        self.signals_df = None
        self.trades_df = None
        self.portfolio_df = None
        self.metrics = None
        self.fee_analysis = None
        # Default modes
        self.debug_mode = False
        self.plot_enabled = False
        self.csv_export_enabled = False

        # Signal types (long, short, mix)
        self.signal_type = self.config['signal_type']
        
        # Initialize the error handler
        self.error_handler = ErrorHandler(logger_name="Backtester", debug_mode=self.debug_mode)
        
        self.error_handler.logger.info("Backtester initialized")

    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled (bool): Whether to enable debug mode.
        """
        self.debug_mode = enabled
        # Update error handler's debug mode as well
        self.error_handler.set_debug_mode(enabled)
        self.error_handler.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        return self

    def set_plot_mode(self, enabled):
        """
        Enable or disable plotting of results.
        
        Args:
            enabled (bool): Whether to enable plotting.
        """
        self.plot_enabled = enabled
        self.error_handler.logger.info(f"Plot mode {'enabled' if enabled else 'disabled'}")
        return self
    
    def set_csv_export(self, enabled):
        """
        Enable or disable CSV export of results.
        
        Args:
            enabled (bool): Whether to enable CSV export.
        """
        self.csv_export_enabled = enabled
        self.error_handler.logger.info(f"CSV export {'enabled' if enabled else 'disabled'}")
        return self

    def load_data(self, file_path):
        """
        Load historical price data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing historical data.
            
        Returns:
            Backtester: Self for method chaining.
        """
        try:
            self.df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col=0)
            self.error_handler.logger.info(f"Data loaded successfully from {file_path}. Data points: {len(self.df)}")
            
            # Validate loaded data
            required_columns = ['open', 'high', 'low', 'close']  # Adjust as needed
            is_valid, message = self.error_handler.check_dataframe_validity(
                self.df, 
                required_columns=required_columns, 
                min_rows=50,  # Minimum rows needed for strategy indicators
                check_full=True
            )
            
            if not is_valid:
                self.error_handler.logger.error(f"Invalid data loaded: {message}")
                return self
                
            # Log statistics about the loaded data
            self.error_handler.log_dataframe_stats(self.df, ['open', 'high', 'low', 'close'])
            
        except Exception as e:
            self.error_handler.logger.error(f"Error loading data from {file_path}: {e}")
        
        return self

    def prepare_indicators(self):
        """
        Compute all necessary technical indicators for the strategy.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators or False on error.
        """
        indicators_config = self.config["indicators"]
        
        self.error_handler.logger.info("Calculating indicators for backtest...")
        
        # First check if we have valid data to work with
        required_columns = ['open', 'high', 'low', 'close']
        is_valid, message = self.error_handler.check_dataframe_validity(
            self.df, 
            required_columns=required_columns,
            min_rows=max(
                indicators_config["keltner_period"],
                indicators_config["cci_period"],
                indicators_config["bollinger_period"],
                20  # Minimum default
            )
        )
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot calculate indicators: {message}")
            return False
        
        try:
            # Make a copy to avoid modifying the original
            indicators_df = self.df.copy()
            indicators = TradingIndicators(debug_mode=self.debug_mode)  # Instantiate the class

            # Add Keltner Channels - using safe calculation
            def add_keltner():
                return indicators.add_keltner_channels(
                    indicators_df, 
                    period=indicators_config["keltner_period"], 
                    atr_multiplier=indicators_config["atr_multiplier"])
            
            indicators_df = self.error_handler.safe_calculation(add_keltner, default_value=indicators_df)
            
            self.error_handler.logger.info(f"Keltner Channels added with period={indicators_config['keltner_period']}, "
                         f"multiplier={indicators_config['atr_multiplier']}")
            
            # Add Commodity Channel Index (CCI)
            def add_cci():
                return indicators.add_cci(
                    indicators_df,
                    period=indicators_config["cci_period"])
            
            indicators_df = self.error_handler.safe_calculation(add_cci, default_value=indicators_df)
                
            self.error_handler.logger.info(f"CCI added with period={indicators_config['cci_period']}")
            
            # Add Bollinger Bands
            def add_bollinger():
                return indicators.add_bollinger_bands(
                    indicators_df, period=indicators_config["bollinger_period"],
                    std_multiplier=indicators_config["std_multiplier"])
            
            indicators_df = self.error_handler.safe_calculation(add_bollinger, default_value=indicators_df)
            
            self.error_handler.logger.info(f"Bollinger Bands added with period={indicators_config['bollinger_period']}, "
                         f"multiplier={indicators_config['std_multiplier']}")
         
            
            # Verify that indicator columns were actually added
            expected_indicator_columns = [
                 'EMA', 'ATR', 'KeltnerUpper', 'KeltnerLower', 'SMA', 'mean_deviation', 'CCI',
                 'StdDev', 'BollingerUpper', 'BollingerLower']
            
            is_valid, message = self.error_handler.check_dataframe_validity(
                indicators_df, 
                required_columns=expected_indicator_columns,
                check_full=False  # Only check the last row for complete data
            )
            
            if not is_valid:
                self.error_handler.logger.error(f"Indicator calculation failed: {message}")
                return False
                
            self.error_handler.logger.info("All indicators calculated successfully.")
            self.indicators_df = indicators_df
            
            # If debug mode is enabled, inspect the indicators
            if self.debug_mode:
                self.inspect_indicators()
                
        except Exception as e:
            self.error_handler.logger.error(f"Error in indicator calculation: {e}")
            self.error_handler.logger.debug(traceback.format_exc())
            return False
            
        return self.indicators_df
        
    def inspect_indicators(self):
        """
        Print diagnostic information about the calculated indicators.
        Useful for debugging and verifying data integrity.
        
        Returns:
            pd.DataFrame: DataFrame with indicators.
        """

        if self.indicators_df is None:
            self.error_handler.logger.warning("No indicators to inspect. Call prepare_indicators() first.")
            return None
        
        # Using error handler to log dataframe stats
        key_columns = ['close',  'EMA',
       'ATR', 'KeltnerUpper', 'KeltnerLower', 'SMA', 'mean_deviation', 'CCI',
       'StdDev', 'BollingerUpper', 'BollingerLower']
        
        self.error_handler.logger.info(f"DataFrame columns: {list(self.indicators_df.columns)}")
        self.error_handler.logger.info(f"DataFrame shape: {self.indicators_df.shape}")
        self.error_handler.logger.debug(f"First 5 rows of data with indicators:\n{self.indicators_df.head()}")
        
        # Log statistics about indicator columns
        self.error_handler.log_dataframe_stats(self.indicators_df, key_columns)
 
        # Check for missing values
        missing = self.indicators_df.isna().sum()
        if missing.sum() > 0:
            self.error_handler.logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            self.error_handler.logger.info("No missing values detected in the indicators.")
        
        return self.indicators_df

    def generate_signals(self):
        """
        Generate trading signals based on the calculated indicators.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals or None on error.
        """
        if self.indicators_df is None:
            self.error_handler.logger.warning("No indicators available. Call prepare_indicators() first.")
            return None
        
        # Verify indicator data before proceeding
        required_indicator_columns = [
            'EMA','ATR', 'KeltnerUpper', 'KeltnerLower', 'SMA', 'mean_deviation', 'CCI',
            'StdDev', 'BollingerUpper', 'BollingerLower'
        ]
        
        is_valid, message = self.error_handler.check_dataframe_validity(
            self.indicators_df, 
            required_columns=required_indicator_columns
        )
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot generate signals: {message}")
            return None
        
        try:
            self.error_handler.logger.info("Generating trading signals...")
            signal_generator = SignalGenerator()
            # Get signal config
            signals_config = self.config["signals"]
            min_required_rows = signals_config["min_required_rows"]
            CCI_up_threshold = signals_config["CCI_up_threshold"]
            CCI_low_threshold = signals_config["CCI_low_threshold"]
            Bollinger_Keltner_alignment = signals_config["Bollinger_Keltner_alignment"]
            window_size = signals_config["window_size"]
            
            # Safely generate signals
            def generate_signals_func():
                return signal_generator.generate_signals(
                    self.indicators_df,
                    CCI_up_threshold, 
                    CCI_low_threshold, 
                    Bollinger_Keltner_alignment, 
                    window_size,
                    min_required_rows,
                    self.signal_type
                )
            
            self.signals_df = self.error_handler.safe_calculation(
                generate_signals_func, 
                default_value=None
            )
            
            if self.signals_df is None:
                self.error_handler.logger.error("Signal generation failed.")
                return None
                
            # Verify required signal columns exist
            signal_columns = ['LongSignal', 'ShortSignal']
            is_valid, message = self.error_handler.check_dataframe_validity(
                self.signals_df, 
                required_columns=signal_columns
            )
            
            if not is_valid:
                self.error_handler.logger.error(f"Signal generation failed: {message}")
                return None
            
            signal_count = ((self.signals_df['LongSignal']) | (self.signals_df['ShortSignal'])).sum()
            self.error_handler.logger.info(f"Generated {signal_count} trading signals")
            
            # Additional validation for signal quality
            if signal_count == 0:
                self.error_handler.logger.warning("No trading signals were generated. Consider adjusting parameters.")
            
            # Log signal distribution statistics
            long_signals = self.signals_df['LongSignal'].sum()
            short_signals = self.signals_df['ShortSignal'].sum()
            
            if self.signal_type != 'short':
                self.error_handler.logger.info(f"Long signals: {long_signals} ({long_signals/signal_count*100:.2f}% of total)")
            
            if self.signal_type != 'long':
                self.error_handler.logger.info(f"Short signals: {short_signals} ({short_signals/signal_count*100:.2f}% of total)")
            
        except Exception as e:
            self.error_handler.logger.error(f"Error generating signals: {e}")
            self.error_handler.logger.debug(traceback.format_exc())
            return None
            
        return self.signals_df

    def run_backtest(self):
        """
        Execute the backtest simulation using the generated signals.
        
        Standard mode: Runs a single backtest with predefined parameters
        
        Returns:
            tuple: (trades_df, portfolio_df) containing trade details and portfolio performance.
        """
        # Check if signals are available
        if self.signals_df is None:
            self.error_handler.logger.warning("No signals available. Call generate_signals() first.")
            return None, None
        
        # Validate signal data before proceeding
        required_signal_columns = ['LongSignal', 'ShortSignal', 'close']
        is_valid, message = self.error_handler.check_dataframe_validity(
            self.signals_df, 
            required_columns=required_signal_columns
        )
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot run backtest: {message}")
            return None, None
            
        backtest_config = self.config['backtest']
        
        # Define default parameters
        initial_budget = backtest_config['initial_budget']
        fee_rate = backtest_config['fee_rate']
        max_positions = backtest_config['max_positions']
        
        try:
            # Standard mode with default parameters
            self.error_handler.logger.info("Running standard backtest simulation...")
            backtester = TradingBacktester(
                initial_budget=initial_budget,
                tp_level=backtest_config['tp_level'],
                sl_level=backtest_config['sl_level'],
                fee_rate=fee_rate,
                max_positions=max_positions
            )
            
            # Run the backtest
            backtester.backtest(self.signals_df)
            
            # Store results
            self.trades_df = backtester.get_trades()
            self.portfolio_df = backtester.get_portfolio()
            self.metrics = backtester.get_metrics()
            self.fee_analysis = backtester.get_fee_analysis()
            
            # Validate backtest results
            if self.trades_df is None or self.trades_df.empty:
                self.error_handler.logger.warning("No trades were executed in the backtest.")
            else:
                # Log trade statistics
                num_trades = len(self.trades_df)
                profitable_trades = (self.trades_df['Net_Profit'] > 0).sum()
                profit_rate = (profitable_trades / num_trades) * 100 if num_trades > 0 else 0
                
                self.error_handler.logger.info(f"Backtest completed: {num_trades} trades executed")
                self.error_handler.logger.info(f"Profitable trades: {profitable_trades}/{num_trades} ({profit_rate:.2f}%)")
                
                print(self.portfolio_df.columns)
                # Validate portfolio data
                if self.portfolio_df is not None and not self.portfolio_df.empty:
                    initial_value = self.portfolio_df['Portfolio_Value'].iloc[0]
                    final_value = self.portfolio_df['Portfolio_Value'].iloc[-1]
                    total_return = ((final_value / initial_value) - 1) * 100
                    
                    self.error_handler.logger.info(f"Initial portfolio value: {initial_value:.2f}")
                    self.error_handler.logger.info(f"Final portfolio value: {final_value:.2f}")
                    self.error_handler.logger.info(f"Total return: {total_return:.2f}%")
                else:
                    self.error_handler.logger.warning("Portfolio data is empty or invalid.")
            
            # Print detailed results
            backtester.print_results(self.signal_type)
            
            return self.trades_df, self.portfolio_df
            
        except Exception as e:
            self.error_handler.logger.error(f"Error running backtest: {str(e)}")
            self.error_handler.logger.debug(traceback.format_exc())
            return None, None

    def visualize_results(self):
        """
        Generate and save visualizations of indicators and backtest results.
        
        Returns:
            list: List of figure objects created.
        """
        if not self.plot_enabled:
            self.error_handler.logger.info("Plotting is disabled. Set plot_mode=True to enable.")
            return []
        
        # Check if necessary dataframes are available
        if self.indicators_df is None:
            self.error_handler.logger.warning("No indicators to visualize. Call prepare_indicators() first.")
            return []
            
        if self.signals_df is None:
            self.error_handler.logger.warning("No signals to visualize. Call generate_signals() first.")
            return []
            
        if self.portfolio_df is None or self.trades_df is None:
            self.error_handler.logger.warning("No backtest results to visualize. Call run_backtest() first.")
            return []
            
        plot_path_config = self.config['plot_paths']
        save_path_prefix = plot_path_config['prefix']
        figures = []
        
        try:
            self.error_handler.logger.info("Generating visualizations...")
            
            # Create an instance of the visualizer
            visualizer = TradingVisualizer(default_figsize=(16, 12))
            
            # Wrap each visualization in safe_calculation to prevent failure of the entire visualization process
            
            # 1 Create a chart with indicators
            def create_indicators_visualization():
                return visualizer.visualize_indicators(self.indicators_df)
                
            fig_indicators = self.error_handler.safe_calculation(
                create_indicators_visualization, 
                default_value=None
            )
            
            if fig_indicators is not None:
                indicators_path = f"{save_path_prefix}{plot_path_config['indicators']}"
                plt.savefig(indicators_path, dpi=150, bbox_inches='tight')
                figures.append(fig_indicators)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved combined indicators plot to {indicators_path}")
            else:
                self.error_handler.logger.warning("Failed to create indicators visualization.")
            
            # 2 Create and save multi-panel split indicator visualization
            def create_split_indicators_visualization():
                return visualizer.visualize_indicators_splited(self.indicators_df)
                
            fig_splited = self.error_handler.safe_calculation(
                create_split_indicators_visualization, 
                default_value=None
            )
            
            if fig_splited is not None:
                indicators_splited_path = f"{save_path_prefix}{plot_path_config['indicators_splited']}"
                plt.savefig(indicators_splited_path, dpi=150, bbox_inches='tight')
                figures.append(fig_splited)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved split indicators plot to {indicators_splited_path}")
            else:
                self.error_handler.logger.warning("Failed to create split indicators visualization.")
            
            # 3 Plot backtest
            def create_backtest_visualization():
                return visualizer.plot_backtest_results(self.signals_df, self.portfolio_df, self.trades_df, self.signal_type)
                
            fig_backtest = self.error_handler.safe_calculation(
                create_backtest_visualization, 
                default_value=None
            )
            
            if fig_backtest is not None:
                backtest_path = f"{save_path_prefix}{plot_path_config['back_test']}"
                plt.savefig(backtest_path, dpi=150, bbox_inches='tight')
                figures.append(fig_backtest)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved backtest results plot to {backtest_path}")
            else:
                self.error_handler.logger.warning("Failed to create backtest visualization.")
            
            # 4 Plot profit hist
            def create_profit_histogram():
                return visualizer.plot_profit_histograms(self.trades_df, self.signal_type)
                
            fig_profit_hist = self.error_handler.safe_calculation(
                create_profit_histogram, 
                default_value=None
            )
            
            if fig_profit_hist is not None:
                profit_hist_path = f"{save_path_prefix}{plot_path_config['profit_distribution']}"
                plt.savefig(profit_hist_path, dpi=150, bbox_inches='tight')
                figures.append(fig_profit_hist)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved profit histogram plot to {profit_hist_path}")
            else:
                self.error_handler.logger.warning("Failed to create profit histogram visualization.")
            
            # 5 Plot signals
            def create_signals_visualization():
                return visualizer.plot_trading_signals(self.signals_df, self.signal_type)
                
            fig_signals = self.error_handler.safe_calculation(
                create_signals_visualization, 
                default_value=None
            )
            
            if fig_signals is not None:
                signals_path = f"{save_path_prefix}{plot_path_config['signals']}"
                plt.savefig(signals_path, dpi=150, bbox_inches='tight')
                figures.append(fig_signals)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved signals plot to {signals_path}")
            else:
                self.error_handler.logger.warning("Failed to create signals visualization.")
            
            # 6 Plot metrics
            def create_metrics_visualization():
                return visualizer.plot_performance_comparison(self.metrics, self.signal_type)
                
            fig_metrics = self.error_handler.safe_calculation(
                create_metrics_visualization, 
                default_value=None
            )
            
            if fig_metrics is not None:
                metrics_path = f"{save_path_prefix}{plot_path_config['metrics']}"
                plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
                figures.append(fig_metrics)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved metrics plot to {metrics_path}")
            else:
                self.error_handler.logger.warning("Failed to create metrics visualization.")
            
            # 7 Plot fees
            def create_fees_visualization():
                return visualizer.visualize_fee_impact(self.trades_df, self.fee_analysis)
                
            fig_fees = self.error_handler.safe_calculation(
                create_fees_visualization, 
                default_value=None
            )
            
            if fig_fees is not None:
                fee_path = f"{save_path_prefix}{plot_path_config['fees']}"
                plt.savefig(fee_path, dpi=150, bbox_inches='tight')
                figures.append(fig_fees)
                self.error_handler.logger.info(f"Plot {len(figures)}...Saved fee impact plot to {fee_path}")
            else:
                self.error_handler.logger.warning("Failed to create fee visualization.")
            
            self.error_handler.logger.info(f"Successfully generated {len(figures)} visualizations")
            
        except Exception as e:
            self.error_handler.logger.error(f"Error generating visualizations: {e}")
            self.error_handler.logger.debug(traceback.format_exc())
            
        return figures

    def export_results(self):
        """
        Export the calculated indicators, signals, trades and portfolio results to CSV files.
        
        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not self.csv_export_enabled:
            self.error_handler.logger.info("CSV export is disabled. Set csv_export=True to enable.")
            return False
            
        data_path_config = self.config['data_paths']
        save_path_prefix = data_path_config['prefix']
        
        success = True
            
        try:
            self.error_handler.logger.info("Exporting results to CSV files...")
            
            # Define a helper function for safer CSV export
            def export_dataframe(df, name, path):
                if df is None or df.empty:
                    self.error_handler.logger.warning(f"No {name} data to export.")
                    return False
                    
                try:
                    df.to_csv(path)
                    self.error_handler.logger.info(f"Saved {name} to {path}")
                    return True
                except Exception as e:
                    self.error_handler.logger.error(f"Error exporting {name} to {path}: {e}")
                    return False
            
            # Export each dataframe
            if self.indicators_df is not None:
                indicators_path = f"{save_path_prefix}{data_path_config['indicators']}"
                success &= export_dataframe(self.indicators_df, "indicators", indicators_path)
                
            if self.signals_df is not None:
                signals_path = f"{save_path_prefix}{data_path_config['signals']}"
                success &= export_dataframe(self.signals_df, "signals", signals_path)
                
            if self.trades_df is not None:
                trades_path = f"{save_path_prefix}{data_path_config['trades']}"
                success &= export_dataframe(self.trades_df, "trades", trades_path)
                
            if self.portfolio_df is not None:
                portfolio_path = f"{save_path_prefix}{data_path_config['portafolio']}"
                success &= export_dataframe(self.portfolio_df, "portfolio performance", portfolio_path)

            if self.fee_analysis is not None:
                fees_path = f"{save_path_prefix}{data_path_config['fees']}"
                success &= export_dataframe(self.fee_analysis, "fee analysis", fees_path)
                
            if success:
                self.error_handler.logger.info("All results exported successfully.")
            else:
                self.error_handler.logger.warning("Some results could not be exported. Check previous warnings.")
                
            return success
            
        except Exception as e:
            self.error_handler.logger.error(f"Error in export process: {e}")
            self.error_handler.logger.debug(traceback.format_exc())
            return False

    def run_complete_backtest(self):
        """
        Execute the complete backtesting workflow from data loading to visualization.
        This is a convenience method that chains all the individual steps.
        
        Returns:
            Backtester: Self for method chaining.
        """
        data_path = self.config['data_path']
        
        # Load data if path is provided
        if data_path:
            self.load_data(data_path)
        else:
            self.error_handler.logger.warning("No data path provided in config. Using pre-loaded data if available.")
            
            # Check if we have data
            if self.df.empty:
                self.error_handler.logger.error("No data loaded and no data path provided. Cannot run backtest.")
                return self
        
        # Execute backtesting workflow
        self.error_handler.logger.info("Starting complete backtesting workflow...")

        # Execute each step and check for failures
        indicators_success = self.prepare_indicators()
        if indicators_success is False:
            self.error_handler.logger.error("Indicator calculation failed. Stopping backtest.")
            return self
        
        signals_success = self.generate_signals()
        if signals_success is None:
            self.error_handler.logger.error("Signal generation failed. Stopping backtest.")
            return self
        
        
        backtest_success = self.run_backtest()
       
        if backtest_success[0] is None and backtest_success[1] is None:
            self.error_handler.logger.error("Backtest execution failed. Stopping visualization and export.")
            return self
        
        self.visualize_results()
        self.export_results()
        
        self.error_handler.logger.info("Backtesting workflow completed")
        return self


if __name__ == "__main__":
    # Load configuration
    config = load_config(CONFIG_PATH)
    config_modes = config['modes']
    
    # Create backtester with configuration flags
    backtester = Backtester(config)
    backtester.set_plot_mode(config_modes['plot_mode'])
    backtester.set_csv_export(config_modes['csv_export'])
    backtester.set_debug_mode(config_modes['debug_mode'])
    
    # Run the complete backtesting workflow
    backtester.run_complete_backtest()
import traceback
import logging
import pandas as pd
import matplotlib.pyplot as plt

from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator
from mean_reversion_volatility_bands.visualization import TradingVisualizer
from mean_reversion_volatility_bands.back_test import TradingBacktester

from utils.utils import load_config

# Configure logging
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

        #signal types (long, short, mix)
        self.signal_type = self.config['signal_type']
     
        logging.info("Backtester initialized")

    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled (bool): Whether to enable debug mode.
        """
        self.debug_mode = enabled
        logging.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        return self

    def set_plot_mode(self, enabled):
        """
        Enable or disable plotting of results.
        
        Args:
            enabled (bool): Whether to enable plotting.
        """
        self.plot_enabled = enabled
        logging.info(f"Plot mode {'enabled' if enabled else 'disabled'}")
        return self
    

    def set_csv_export(self, enabled):
        """
        Enable or disable CSV export of results.
        
        Args:
            enabled (bool): Whether to enable CSV export.
        """
        self.csv_export_enabled = enabled
        logging.info(f"CSV export {'enabled' if enabled else 'disabled'}")
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
            logging.info(f"Data loaded successfully from {file_path}. Data points: {len(self.df)}")
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
        return self


    def prepare_indicators(self):
        """
        Compute all necessary technical indicators for the strategy.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators or False on error.
        """
        indicators_config = self.config["indicators"]
        
        logging.info("Calculating indicators for backtest...")
        try:
            # Make a copy to avoid modifying the original
            indicators_df = self.df.copy()
            indicators = TradingIndicators(debug_mode=self.debug_mode, logging_mode=True)  # Instantiate the class

            # Add Keltner Channels
            indicators_df = indicators.add_keltner_channels(
                indicators_df, 
                period=indicators_config["keltner_period"], 
                atr_multiplier=indicators_config["atr_multiplier"])
            
            logging.info(f"Keltner Channels added with period={indicators_config['keltner_period']}, "
                         f"multiplier={indicators_config['atr_multiplier']}")
            
            # Add Commodity Channel Index (CCI)
            indicators_df = indicators.add_cci(
                indicators_df,
                period=indicators_config["cci_period"])

            logging.info(f"CCI added with period={indicators_config['cci_period']}")
            
            # Add Bollinger Bands
            indicators_df = indicators.add_bollinger_bands(
                indicators_df, period=indicators_config["bollinger_period"],
                std_multiplier=indicators_config["std_multiplier"])
            
            logging.info(f"Bollinger Bands added with period={indicators_config['bollinger_period']}, "
                         f"multiplier={indicators_config['std_multiplier']}")
            
            logging.info("All indicators calculated successfully.")
            self.indicators_df = indicators_df
            
            # If debug mode is enabled, inspect the indicators
            if self.debug_mode:
                self.inspect_indicators()
                
        except Exception as e:
            logging.error(f"Error in indicator calculation: {e}")
            return False
        
        
    
    def inspect_indicators(self):
        """
        Print diagnostic information about the calculated indicators.
        Useful for debugging and verifying data integrity.
        
        Returns:
            pd.DataFrame: DataFrame with indicators.
        """

        if self.indicators_df is None:
            logging.warning("No indicators to inspect. Call prepare_indicators() first.")
            return None
        
        logging.info(f"DataFrame columns: {list(self.indicators_df.columns)}")
        logging.info(f"DataFrame shape: {self.indicators_df.shape}")
        logging.info(f"First 5 rows of data with indicators:\n{self.indicators_df.head()}")
 
        # Check for missing values
        missing = self.indicators_df.isna().sum()
        if missing.sum() > 0:
            logging.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logging.info("No missing values detected in the indicators.")
        
        return self.indicators_df

    def generate_signals(self):
        """
        Generate trading signals based on the calculated indicators.
        
        Returns:
            pd.DataFrame: DataFrame with trading signals or None on error.
        """
        if self.indicators_df is None:
            logging.warning("No indicators available. Call prepare_indicators() first.")
            return None
        
        
        try:
            logging.info("Generating trading signals...")
            signal_generator = SignalGenerator()
            #get signal config
            signals_config = self.config["signals"]
            min_required_rows= signals_config["min_required_rows"]
            CCI_up_threshold = signals_config["CCI_up_threshold"]
            CCI_low_threshold = signals_config["CCI_low_threshold"]
            Bollinger_Keltner_alignment = signals_config["Bollinger_Keltner_alignment"]
            window_size = signals_config["window_size"]
            

            self.signals_df = signal_generator.generate_signals(self.indicators_df,
                                                                 CCI_up_threshold, 
                                                                 CCI_low_threshold, 
                                                                 Bollinger_Keltner_alignment, 
                                                                 window_size,
                                                                 min_required_rows,
                                                                 self.signal_type)

            logging.info(f"Generated {((self.signals_df['LongSignal']) | (self.signals_df['ShortSignal'])).sum()} trading signals")

            
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return None


    def run_backtest(self):
        """
        Execute the backtest simulation using the generated signals.
        
        
        Standard mode: Runs a single backtest with predefined parameters
        
        Returns:
            tuple: (trades_df, portfolio_df) containing trade details and portfolio performance.
        """

    
        # Check if signals are available
        if self.signals_df is None:
            logging.warning("No signals available. Call generate_signals() first.")
            return None, None
            
        backtest_config = self.config['backtest']

        
        # Define default parameters
        initial_budget = backtest_config['initial_budget']
        fee_rate = backtest_config['fee_rate']
        max_positions = backtest_config['max_positions']
        
        try:
            # Standard mode with default parameters
            logging.info("Running standard backtest simulation...")
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
            
            # Print detailed results
            backtester.print_results(self.signal_type)
            
            return self.trades_df, self.portfolio_df
            
        except Exception as e:
            logging.error(f"Error running backtest: {str(e)}")
            logging.debug(traceback.format_exc())
            return None, None
           

    def visualize_results(self):
        """
        Generate and save visualizations of indicators and backtest results.
        
        Returns:
            list: List of figure objects created.
        """
        if not self.plot_enabled:
            logging.info("Plotting is disabled. Set plot_mode=True to enable.")
            return []
            
        plot_path_config = self.config['plot_paths']
        save_path_prefix = plot_path_config['prefix']
        figures = []
        
        try:
            logging.info("Generating visualizations...")
            
            # Create an instance of the visualizer
            visualizer = TradingVisualizer(default_figsize=(16, 12))
            
            # 1 Create a chart with indicators
            fig_indicators = visualizer.visualize_indicators(self.indicators_df)
            indicators_path = f"{save_path_prefix}{plot_path_config['indicators']}"
            plt.savefig(indicators_path, dpi=300, bbox_inches='tight')
            figures.append(fig_indicators)
            logging.info(f"Plot {len(figures)}...Saved combined indicators plot to {indicators_path}")
            

            #2 Create and save multi-panel split indicator visualization
            fig_splited = visualizer.visualize_indicators_splited(self.indicators_df)
            indicators_splited_path = f"{save_path_prefix}{plot_path_config['indicators_splited']}"
            plt.savefig(indicators_splited_path, dpi=300, bbox_inches='tight')
            figures.append(fig_splited)
            logging.info(f"Plot {len(figures)}...Saved split indicators plot to {indicators_splited_path}")
            

            # 3Plot backtest
            fig_backtest = visualizer.plot_backtest_results(self.signals_df, self.portfolio_df, self.trades_df,self.signal_type)
            backtest_path = f"{save_path_prefix}{plot_path_config['back_test']}"
            plt.savefig(backtest_path, dpi=300, bbox_inches='tight')
            figures.append(fig_backtest)
            logging.info(f"Plot {len(figures)}...Saved backtest results plot to {backtest_path}")
            
                
            # 4 Plot profit hist
            fig_profit_hist = visualizer.plot_profit_histograms(self.trades_df,self.signal_type)
            profit_hist_path = f"{save_path_prefix}{plot_path_config['profit_distribution']}"
            plt.savefig(profit_hist_path, dpi=300, bbox_inches='tight')
            figures.append(fig_profit_hist)
            logging.info(f"Plot {len(figures)}...Saved profit histogram plot to {profit_hist_path}")
            
            
            
            # 5 Plot signals
            fig_signals = visualizer.plot_trading_signals(self.signals_df,self.signal_type)
            signals_path = f"{save_path_prefix}{plot_path_config['signals']}"
            plt.savefig(signals_path, dpi=300, bbox_inches='tight')
            figures.append(fig_signals)
            logging.info(f"Plot {len(figures)}...Saved signals plot to {signals_path}")
            

            # 6Plot metrics
            fig_metrics = visualizer.plot_performance_comparison(self.metrics,self.signal_type)
            metrics_path = f"{save_path_prefix}{plot_path_config['metrics']}"
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            figures.append(fig_metrics)
            logging.info(f"Plot {len(figures)}...Saved metrics plot to {metrics_path}")

            # 7 plot fees
            
            fig_fees = visualizer.visualize_fee_impact(self.trades_df,self.fee_analysis)
            fee_path = f"{save_path_prefix}{plot_path_config['fees']}"
            plt.savefig(fee_path, dpi=300, bbox_inches='tight')
            figures.append(fig_fees)
            logging.info(f"Plot {len(figures)}...Saved metrics plot to {fee_path}")
            


          
            logging.info(f"Successfully generated {len(figures)} visualizations")
            
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            
        return figures

    def export_results(self):
        """
        Export the calculated indicators, signals, trades and portfolio results to CSV files.
        
        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not self.csv_export_enabled:
            logging.info("CSV export is disabled. Set csv_export=True to enable.")
            return False
            
        data_path_config = self.config['data_paths']
        save_path_prefix = data_path_config['prefix']
            
        try:
            logging.info("Exporting results to CSV files...")
            
            if self.indicators_df is not None:
                indicators_path = f"{save_path_prefix}{data_path_config['indicators']}"
                self.indicators_df.to_csv(indicators_path)
                logging.info(f"Saved indicators to {indicators_path}")
                
            if self.signals_df is not None:
                signals_path = f"{save_path_prefix}{data_path_config['signals']}"
                self.signals_df.to_csv(signals_path)
                logging.info(f"Saved signals to {signals_path}")
                
            if self.trades_df is not None:
                trades_path = f"{save_path_prefix}{data_path_config['trades']}"
                self.trades_df.to_csv(trades_path)
                logging.info(f"Saved trades to {trades_path}")
                
            if self.portfolio_df is not None:
                portfolio_path = f"{save_path_prefix}{data_path_config['portafolio']}"
                self.portfolio_df.to_csv(portfolio_path)
                logging.info(f"Saved portfolio performance to {portfolio_path}")

            if self.fee_analysis is not None:
                fees_path = f"{save_path_prefix}{data_path_config['fees']}"
                self.fee_analysis.to_csv(fees_path)
                logging.info(f"Saved fee analysis to {fees_path}")
            
                
            return True
            
        except Exception as e:
            logging.error(f"Error exporting results to CSV: {e}")
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
        
        # Execute backtesting workflow
        logging.info("Starting complete backtesting workflow...")

        
        self.prepare_indicators()
        self.generate_signals()
        self.run_backtest()
        self.visualize_results()
        self.export_results()
        
        # logging.info("Backtesting workflow completed")
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
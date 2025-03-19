import traceback
import logging
import pandas as pd
import matplotlib.pyplot as plt

from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator
from mean_reversion_volatility_bands.visualization import TradingVisualizer
from mean_reversion_volatility_bands.back_test import TradingBacktester, StrategyOptimizer
from utils.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        
        # Default modes
        self.debug_mode = False
        self.plot_enabled = False
        self.optimize_enabled = False
        self.csv_export_enabled = False
     
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
    
    def set_optimize_mode(self, enabled):
        """
        Enable or disable optimization.
        
        Args:
            enabled (bool): Whether to enable optimization.
        """
        self.optimize_enabled = enabled
        logging.info(f"Optimization mode {'enabled' if enabled else 'disabled'}")
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
        print(indicators_config)
        logging.info("Calculating indicators for backtest...")
        try:
            # Make a copy to avoid modifying the original
            self.indicators_df = self.df.copy()
            
            # Add Keltner Channels
            self.indicators_df = TradingIndicators.add_keltner_channels(
                self.indicators_df, 
                period=indicators_config["keltner_period"], 
                atr_multiplier=indicators_config["atr_multiplier"]
            )
            logging.info(f"Keltner Channels added with period={indicators_config['keltner_period']}, "
                         f"multiplier={indicators_config['atr_multiplier']}")
            
            # Add Commodity Channel Index (CCI)
            self.indicators_df = TradingIndicators.add_cci(
                self.indicators_df, 
                period=indicators_config["cci_period"]
            )
            logging.info(f"CCI added with period={indicators_config['cci_period']}")
            
            # Add Bollinger Bands
            self.indicators_df = TradingIndicators.add_bollinger_bands(
                self.indicators_df, 
                period=indicators_config["bollinger_period"], 
                std_multiplier=indicators_config["std_multiplier"]
            )
            logging.info(f"Bollinger Bands added with period={indicators_config['bollinger_period']}, "
                         f"multiplier={indicators_config['std_multiplier']}")
            
            logging.info("All indicators calculated successfully.")
            
            # If debug mode is enabled, inspect the indicators
            if self.debug_mode:
                self.inspect_indicators()
                
        except Exception as e:
            logging.error(f"Error in indicator calculation: {e}")
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
            signals_config = self.config["signals"]
            self.signals_df = signal_generator.generate_signals(self.indicators_df, signals_config)

            logging.info(f"Generated {((self.signals_df['LongSignal']) | (self.signals_df['ShortSignal'])).sum()} trading signals")

            return self.signals_df
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return None

    def run_backtest(self):
        """
        Execute the backtest simulation using the generated signals.
        
        The function supports two modes:
        1. Optimization mode: Tests multiple TP/SL combinations to find optimal parameters
        2. Standard mode: Runs a single backtest with predefined parameters
        
        Returns:
            tuple: (trades_df, portfolio_df) containing trade details and portfolio performance.
        """
        # Check if signals are available
        if self.signals_df is None:
            logging.warning("No signals available. Call generate_signals() first.")
            return None, None
            
        backtest_config = self.config['backtest']
        optimization_config = self.config['optimization']
        
        # Define default parameters
        initial_budget = backtest_config['initial_budget']
        fee_rate = backtest_config['fee_rate']
        max_positions = backtest_config['max_positions']
        
        try:
            if self.optimize_enabled:
                logging.info("Running backtest simulation with parameter optimization...")
                
                # Create StrategyOptimizer instance for optimization
                optimizer = StrategyOptimizer(
                    initial_budget=initial_budget,
                    fee_rate=fee_rate,
                    max_positions=max_positions
                )
                
                # Run optimization
                self.optimization_results = optimizer.optimize(
                    self.signals_df,
                    tp_levels=optimization_config['tp_levels'],
                    sl_levels=optimization_config['sl_levels']
                )
               
                # Process optimization results
                if self.optimization_results.empty:
                    logging.error("Optimization returned no valid results")
                    return None, None
                    
                # Get best parameters
                best_params = optimizer.get_best_parameters(self.optimization_results)
                best_tp = best_params['TP_Level']
                best_sl = best_params['SL_Level']
                
                # Log optimization results
                optimizer.print_optimization_results(self.optimization_results, top_n=3)
                
                logging.info(f"Using optimal parameters: TP={best_tp}, SL={best_sl}")
                
                # Create backtester with optimal parameters
                backtester = TradingBacktester(
                    initial_budget=initial_budget,
                    tp_level=best_tp,
                    sl_level=best_sl,
                    fee_rate=fee_rate,
                    max_positions=max_positions
                )
            else:
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
            
            # Print detailed results
            backtester.print_results()
            
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
            
            # Create a chart with indicators
            fig_indicators = visualizer.visualize_indicators(self.indicators_df)
            indicators_path = f"{save_path_prefix}{plot_path_config['indicators']}"
            plt.savefig(indicators_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved combined indicators plot to {indicators_path}")
            figures.append(fig_indicators)

            # Create and save multi-panel split indicator visualization
            fig_splited = visualizer.visualize_indicators_splited(self.indicators_df)
            indicators_splited_path = f"{save_path_prefix}{plot_path_config['indicators_splited']}"
            plt.savefig(indicators_splited_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved split indicators plot to {indicators_splited_path}")
            figures.append(fig_splited)

            # Plot backtest
            fig_backtest = visualizer.plot_backtest_results(self.signals_df, self.portfolio_df, self.trades_df)
            backtest_path = f"{save_path_prefix}{plot_path_config['back_test']}"
            plt.savefig(backtest_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved backtest results plot to {backtest_path}")
            figures.append(fig_backtest)
                
            # Plot profit hist
            fig_profit_hist = visualizer.plot_profit_histograms(self.trades_df)
            profit_hist_path = f"{save_path_prefix}{plot_path_config['profit_distribution']}"
            plt.savefig(profit_hist_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved profit histogram plot to {profit_hist_path}")
            figures.append(fig_profit_hist)
            
            # Plot signals
            fig_signals = visualizer.plot_trading_signals(self.signals_df)
            signals_path = f"{save_path_prefix}{plot_path_config['signals']}"
            plt.savefig(signals_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved signals plot to {signals_path}")
            figures.append(fig_signals)

            # Plot metrics
            fig_metrics = visualizer.plot_performance_comparison(self.metrics)
            metrics_path = f"{save_path_prefix}{plot_path_config['metrics']}"
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved metrics plot to {metrics_path}")
            figures.append(fig_metrics)

            #plot optimization heatmap
            if self.optimize_enabled:
                fig_optimization = visualizer.plot_heatmap(self.optimization_results)
                optimization_path = f"{save_path_prefix}{plot_path_config['optimization']}"
                plt.savefig(optimization_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved metrics plot to {optimization_path}")
                figures.append(fig_optimization)

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
        
        logging.info("Backtesting workflow completed")
        return self


if __name__ == "__main__":
    # Load configuration
    config_path = 'mean_reversion_volatility_bands/config.yaml'
    config = load_config(config_path)
    config_modes = config['modes']
    
    # Create backtester with configuration flags
    backtester = Backtester(config)
    backtester.set_plot_mode(config_modes['plot_mode'])
    backtester.set_csv_export(config_modes['csv_export'])
    backtester.set_debug_mode(config_modes['debug_mode'])
    backtester.set_optimize_mode(config_modes['optimize_mode'])
    
    # Run the complete backtesting workflow
    backtester.run_complete_backtest()
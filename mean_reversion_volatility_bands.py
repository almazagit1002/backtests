import logging
import pandas as pd
import matplotlib.pyplot as plt

from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator
from mean_reversion_volatility_bands.visualization import (
    visualize_indicators_multi,
    visualize_indicators,
    plot_backtest_results
)
from mean_reversion_volatility_bands.back_test import run_backtest

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

    def __init__(self, df=None, trading_config=None):
        """
        Initialize the backtester with data and configuration.
        
        Args:
            df (pd.DataFrame, optional): Historical price data. Defaults to None.
            trading_config (dict, optional): Trading strategy parameters. Defaults to None.
        """
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.config = trading_config if trading_config is not None else {}
        self.indicators_df = None
        self.signals_df = None
        self.trades_df = None
        self.portfolio_df = None
        self.plot_enabled = True
        self.csv_export_enabled = True
        self.debug_mode = False
        logging.info("Backtester initialized")

    def set_debug_mode(self, enabled=True):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled (bool, optional): Whether to enable debug mode. Defaults to True.
        """
        self.debug_mode = enabled
        logging.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        return self

    def set_plot_mode(self, enabled=True):
        """
        Enable or disable plotting of results.
        
        Args:
            enabled (bool, optional): Whether to enable plotting. Defaults to True.
        """
        self.plot_enabled = enabled
        logging.info(f"Plot mode {'enabled' if enabled else 'disabled'}")
        return self

    def set_csv_export(self, enabled=True):
        """
        Enable or disable CSV export of results.
        
        Args:
            enabled (bool, optional): Whether to enable CSV export. Defaults to True.
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

    def set_trading_config(self, config):
        """
        Set or update the trading strategy configuration.
        
        Args:
            config (dict): Dictionary containing trading parameters.
            
        Returns:
            Backtester: Self for method chaining.
        """
        self.config = config
        logging.info(f"Trading configuration set: {config}")
        return self

    def prepare_indicators(self):
        """
        Compute all necessary technical indicators for the strategy.
        
        Returns:
            pd.DataFrame: DataFrame with calculated indicators or False on error.
        """
        logging.info("Calculating indicators for backtest...")
        try:
            # Make a copy to avoid modifying the original
            self.indicators_df = self.df.copy()
            
            # Add Keltner Channels
            self.indicators_df = TradingIndicators.add_keltner_channels(
                self.indicators_df, 
                period=self.config["keltner_period"], 
                atr_multiplier=self.config["atr_multiplier"]
            )
            logging.info(f"Keltner Channels added with period={self.config['keltner_period']}, "
                         f"multiplier={self.config['atr_multiplier']}")
            
            # Add Commodity Channel Index (CCI)
            self.indicators_df = TradingIndicators.add_cci(
                self.indicators_df, 
                period=self.config["cci_period"]
            )
            logging.info(f"CCI added with period={self.config['cci_period']}")
            
            # Add Bollinger Bands
            self.indicators_df = TradingIndicators.add_bollinger_bands(
                self.indicators_df, 
                period=self.config["bollinger_period"], 
                std_multiplier=self.config["std_multiplier"]
            )
            logging.info(f"Bollinger Bands added with period={self.config['bollinger_period']}, "
                         f"multiplier={self.config['std_multiplier']}")
            
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
            self.signals_df = signal_generator.generate_signals(self.indicators_df, self.config)

            logging.info(f"Generated {((self.signals_df['LongSignal']) | (self.signals_df['ShortSignal'])).sum()} trading signals")

            return self.signals_df
        except Exception as e:
            logging.error(f"Error generating signals: {e}")
            return None

    def run_backtest(self):
        """
        Execute the backtest simulation using the generated signals.
        
        Returns:
            tuple: (trades_df, portfolio_df) containing trade details and portfolio performance.
        """
        if self.signals_df is None:
            logging.warning("No signals available. Call generate_signals() first.")
            return None, None
            
        try:
            logging.info("Running backtest simulation...")
            self.trades_df, self.portfolio_df = run_backtest(self.signals_df)
            
            # Log summary of backtest results
            initial_capital = self.portfolio_df['Portfolio_Value'].iloc[0]
            final_capital = self.portfolio_df['Portfolio_Value'].iloc[-1]
            profit_loss = final_capital - initial_capital
            profit_percentage = (profit_loss / initial_capital) * 100
            num_trades = len(self.trades_df)
            
            logging.info(f"Backtest completed with {num_trades} trades")
            logging.info(f"Initial capital: ${initial_capital:.2f}")
            logging.info(f"Final capital: ${final_capital:.2f}")
            logging.info(f"Profit/Loss: ${profit_loss:.2f} ({profit_percentage:.2f}%)")
            
            return self.trades_df, self.portfolio_df
        except Exception as e:
            logging.error(f"Error running backtest: {e}")
            return None, None

    def visualize_results(self, save_path_prefix=""):
        """
        Generate and save visualizations of indicators and backtest results.
        
        Args:
            save_path_prefix (str, optional): Prefix for saved files. Defaults to "".
            
        Returns:
            list: List of figure objects created.
        """
        if not self.plot_enabled:
            logging.info("Plotting is disabled. Set plot_mode=True to enable.")
            return []
            
        figures = []
        
        try:
            logging.info("Generating visualizations...")
            
            if self.indicators_df is not None:
                # Create and save indicator visualization
                fig1 = visualize_indicators(self.indicators_df)
                save_path1 = f"{save_path_prefix}mean_reversion_volatility_bands_all.pdf"
                plt.savefig(save_path1, dpi=300, bbox_inches='tight')
                logging.info(f"Saved combined indicators plot to {save_path1}")
                figures.append(fig1)

                # Create and save multi-panel indicator visualization
                fig2 = visualize_indicators_multi(self.indicators_df)
                save_path2 = f"{save_path_prefix}mean_reversion_volatility_bands_splited.pdf"
                plt.savefig(save_path2, dpi=300, bbox_inches='tight')
                logging.info(f"Saved split indicators plot to {save_path2}")
                figures.append(fig2)
            
            if self.signals_df is not None and self.portfolio_df is not None and self.trades_df is not None:
                # Create and save backtest results visualization
                fig3 = plot_backtest_results(self.signals_df, self.portfolio_df, self.trades_df)
                save_path3 = f"{save_path_prefix}back_test.pdf"
                plt.savefig(save_path3, dpi=300, bbox_inches='tight')
                logging.info(f"Saved backtest results plot to {save_path3}")
                figures.append(fig3)
                
            logging.info(f"Successfully generated {len(figures)} visualizations")
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
            
        return figures

    def export_results(self, save_path_prefix=""):
        """
        Export the calculated indicators, signals, trades and portfolio results to CSV files.
        
        Args:
            save_path_prefix (str, optional): Prefix for saved files. Defaults to "".
            
        Returns:
            bool: True if export was successful, False otherwise.
        """
        if not self.csv_export_enabled:
            logging.info("CSV export is disabled. Set csv_export=True to enable.")
            return False
            
        try:
            logging.info("Exporting results to CSV files...")
            
            if self.indicators_df is not None:
                indicators_path = f"{save_path_prefix}indicators.csv"
                self.indicators_df.to_csv(indicators_path)
                logging.info(f"Saved indicators to {indicators_path}")
                
            if self.signals_df is not None:
                signals_path = f"{save_path_prefix}signals.csv"
                self.signals_df.to_csv(signals_path)
                logging.info(f"Saved signals to {signals_path}")
                
            if self.trades_df is not None:
                trades_path = f"{save_path_prefix}trades.csv"
                self.trades_df.to_csv(trades_path)
                logging.info(f"Saved trades to {trades_path}")
                
            if self.portfolio_df is not None:
                portfolio_path = f"{save_path_prefix}portfolio.csv"
                self.portfolio_df.to_csv(portfolio_path)
                logging.info(f"Saved portfolio performance to {portfolio_path}")
                
            return True
            
        except Exception as e:
            logging.error(f"Error exporting results to CSV: {e}")
            return False

    def run_complete_backtest(self, data_path=None, config=None):
        """
        Execute the complete backtesting workflow from data loading to visualization.
        This is a convenience method that chains all the individual steps.
        
        Args:
            data_path (str, optional): Path to input data CSV. Defaults to None.
            config (dict, optional): Trading strategy configuration. Defaults to None.
            
        Returns:
            Backtester: Self for method chaining.
        """
        # Load data if path is provided
        if data_path:
            self.load_data(data_path)
            
        # Update config if provided
        if config:
            self.set_trading_config(config)
            
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
    # Example trading configuration
    trading_config = {
        "budget": 1000, 
        "keltner_period": 20,
        "cci_period": 20,
        "bollinger_period": 20,
        "atr_multiplier": 2.0,
        "std_multiplier": 2.0,
        "tp_level": 2.0, 
        "sl_level": 1.0,
        "CCI_up_threshold": 70,
        "CCI_low_threshold": -30,
        "Bollinger_Keltner_alignment": 0.005,
        "window_size": 3
    }

    # Create backtester with configuration flags
    backtester = Backtester()
    backtester.set_plot_mode(True)
    backtester.set_csv_export(True)
    backtester.set_debug_mode(False)
    
    # Run the complete backtesting workflow
    backtester.run_complete_backtest(
        data_path="sol_price.csv",
        config=trading_config
    )
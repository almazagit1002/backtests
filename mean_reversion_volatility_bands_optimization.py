import logging
import traceback
import pandas as pd

from mean_reversion_volatility_bands.optimization import StrategyOptimizer
from mean_reversion_volatility_bands.optimization_analysis import OptimizationAnalyzer
from utils.utils import load_config, save_config_yaml

# Import the ErrorHandler class
from utils.error_handler import ErrorHandler

# Configure logging - consider removing this since ErrorHandler has its own logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Reduce log noise from third-party libraries
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

CONFIG_PATH = 'mean_reversion_volatility_bands/optimization_config.yaml'

class BacktesterOptimization:
    """
    Class for optimizing backtesting parameters for the mean reversion volatility bands strategy.
    This class handles the parameter optimization workflow including:
    - Loading data
    - Running the Bayesian optimization process
    - Analyzing optimization results
    - Saving optimized parameters to configuration
    """
    
    def __init__(self, config):
        """
        Initialize the optimization backtester with configuration.
        
        Args:
            config (dict): Configuration parameters including optimization settings
        """
        self.config = config
        self.df = None
        self.optimization_results = None
        self.optimization_analysis_enabled = False
        
        # Initialize the error handler
        self.error_handler = ErrorHandler(logger_name="BacktesterOptimization", debug_mode=False)
        
        self.error_handler.logger.info("BacktesterOptimization initialized")
    
    def set_debug_mode(self, enabled):
        """
        Enable or disable debug mode for more verbose output.
        
        Args:
            enabled (bool): Whether to enable debug mode.
        """
        # Update error handler's debug mode
        self.error_handler.set_debug_mode(enabled)
        self.error_handler.logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
        return self

    def set_optimization_analysis_mode(self, enabled):
        """
        Enable or disable optimization analysis.
        
        Args:
            enabled (bool): Whether to enable optimization analysis.
        """
        self.optimization_analysis_enabled = enabled
        self.error_handler.logger.info(f"Analyzing Optimization Results mode {'enabled' if enabled else 'disabled'}")
        return self
        
    def load_data(self, file_path):
        """
        Load historical price data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing historical data.
            
        Returns:
            BacktesterOptimization: Self for method chaining.
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
            self.error_handler.logger.debug(traceback.format_exc())
        
        return self
    
    def analyise_optimization(self):
        """
        Analyze optimization results and generate visualizations.
        
        Returns:
            BacktesterOptimization: Self for method chaining.
        """
        if self.optimization_results is None:
            self.error_handler.logger.warning("No optimization results to analyze. Run optimization first.")
            return self
            
        try:
            prefix = self.config['save_data_paths']['prefix']
            _dir = self.config['save_data_paths']['optimization_analysis']
            optimization_analysis_path = prefix + _dir
            
            self.error_handler.logger.info(f"Analyzing optimization results and saving to {optimization_analysis_path}")
            
            # Create optimizer analyzer
            optimization_analyser = OptimizationAnalyzer(self.optimization_results, optimization_analysis_path)
            
            # Wrap analysis in safe calculation
            def run_analyses():
                return optimization_analyser.run_all_analyses()
                
            result = self.error_handler.safe_calculation(
                run_analyses,
                default_value=False
            )
            
            if result:
                self.error_handler.logger.info(f"Optimization analysis completed successfully")
            else:
                self.error_handler.logger.warning("Optimization analysis may not have completed successfully")
                
        except Exception as e:
            self.error_handler.logger.error(f"Error analyzing optimization results: {e}")
            self.error_handler.logger.debug(traceback.format_exc())
            
        return self
    
    def run_optimiziation(self):
        """
        Run Bayesian optimization to find optimal strategy parameters.
        
        Returns:
            BacktesterOptimization: Self for method chaining.
        """
        # Check if we have valid data to work with
        if self.df is None or self.df.empty:
            self.error_handler.logger.error("No data loaded. Call load_data() first.")
            return self
            
        # Validate data before proceeding
        required_columns = ['open', 'high', 'low', 'close']
        is_valid, message = self.error_handler.check_dataframe_validity(
            self.df, 
            required_columns=required_columns,
            min_rows=50  # Minimum default
        )
        
        if not is_valid:
            self.error_handler.logger.error(f"Cannot run optimization: {message}")
            return self
        
        backtest_config = self.config['backtest']
        optimize_config = self.config['optimization']
        pbounds = self.config['pbounds']
        signal_type = self.config['signal_type']

        # Define default parameters
        initial_budget = backtest_config['initial_budget']
        fee_rate = backtest_config['fee_rate']
        init_points = optimize_config['init_points']
        n_iter = optimize_config['n_iter']
    
        try:
            self.error_handler.logger.info("Running backtest simulation with parameter optimization...")
            
            # Validate optimization configuration
            if not isinstance(pbounds, dict) or len(pbounds) == 0:
                self.error_handler.logger.error("Invalid parameter bounds configuration")
                return self
                
            if init_points < 1 or n_iter < 1:
                self.error_handler.logger.error(f"Invalid optimization parameters: init_points={init_points}, n_iter={n_iter}")
                return self
                
            # Log optimization parameters
            self.error_handler.logger.info(f"Optimization parameters: init_points={init_points}, n_iter={n_iter}")
            self.error_handler.logger.info(f"Parameter bounds: {pbounds}")
            self.error_handler.logger.info(f"Signal type: {signal_type}")
            
            # Create StrategyOptimizer instance
            optimizer = StrategyOptimizer(
                initial_budget=initial_budget,
                fee_rate=fee_rate,
                init_points=init_points,
                n_iter=n_iter,
                df=self.df,
                signal_type=signal_type
            )
            
            # Run optimization safely
            def run_optimize():
                return optimizer.optimize(pbounds)
                
            self.optimization_results = self.error_handler.safe_calculation(
                run_optimize,
                default_value=None
            )
            
            if self.optimization_results is None:
                self.error_handler.logger.error("Optimization failed or produced no results")
                return self
                
            # Get the best parameters safely
            def get_best_params():
                return optimizer.get_best_parameters(self.optimization_results)
                
            best_optimization_results = self.error_handler.safe_calculation(
                get_best_params,
                default_value=None
            )
            
            if best_optimization_results is None:
                self.error_handler.logger.error("Failed to extract best parameters from optimization results")
                return self
            
            # Log the best parameters found
            self.error_handler.logger.info(f"Best parameters found: {best_optimization_results}")
            
            # Update configuration with best parameters
            self.error_handler.logger.info("Updating configuration parameters")
            
            try:
                save_config_yaml(best_optimization_results, self.config['save_data_paths']['update_config'])
                self.error_handler.logger.info(f"Configuration saved to {self.config['save_data_paths']['update_config']}")
            except Exception as e:
                self.error_handler.logger.error(f"Error saving configuration: {e}")
            
            return self
                
        except Exception as e:
            self.error_handler.logger.error(f"Error running backtest optimization: {str(e)}")
            self.error_handler.logger.debug(traceback.format_exc())
            return self
        
    def run_optimization_backtest(self):
        """
        Execute the complete optimization workflow from data loading to analysis.
        This is a convenience method that chains all the individual steps.
        
        Returns:
            BacktesterOptimization: Self for method chaining.
        """
        data_path = self.config['load_data_path']
        
        # Load data if path is provided
        if data_path:
            self.load_data(data_path)
        else:
            self.error_handler.logger.error("No data path provided in config. Cannot run optimization.")
            return self
            
        if self.df is None or self.df.empty:
            self.error_handler.logger.error("Failed to load data. Cannot run optimization.")
            return self
        
        self.error_handler.logger.info("Starting Optimization workflow...")
        
        # Run optimization process
        optimization_success = self.run_optimiziation()
        
        if self.optimization_results is None:
            self.error_handler.logger.error("Optimization failed. Stopping workflow.")
            return self
            
        self.error_handler.logger.info("Optimization completed successfully")
        
        # Run analysis if enabled
        if self.optimization_analysis_enabled:
            self.error_handler.logger.info("Running optimization analysis...")
            self.analyise_optimization()
            
        self.error_handler.logger.info("Optimization workflow completed")
        return self

if __name__ == "__main__":
    # Load configuration
    config = load_config(CONFIG_PATH)
    config_modes = config['modes']
    
    # Create optimization backtester
    backtest_optimizer = BacktesterOptimization(config)
    
    # Set modes from configuration
    backtest_optimizer.set_optimization_analysis_mode(config_modes['analise_optimization_mode'])
    if 'debug_mode' in config_modes:
        backtest_optimizer.set_debug_mode(config_modes['debug_mode'])

    # Run optimization workflow
    backtest_optimizer.run_optimization_backtest()
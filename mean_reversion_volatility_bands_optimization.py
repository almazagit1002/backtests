import logging
import traceback

import pandas as pd


from mean_reversion_volatility_bands.optimization import StrategyOptimizer
from mean_reversion_volatility_bands.optimization_analysis import OptimizationAnalyzer
from utils.utils import load_config, save_config_yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)



CONFIG_PATH = 'mean_reversion_volatility_bands/optimization_config.yaml'


class BacktesterOptimization:
    def __init__(self,config):
        self.config = config

    def set_optimization_analysis_mode(self, enabled):
        """
        Enable or disable optimization.
        
        Args:
            enabled (bool): Whether to enable optimization.
        """
        self.optimization_analysis_enabled = enabled
        logging.info(f"Analysing Optimization Results mode {'enabled' if enabled else 'disabled'}")
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
    
    def analyise_optimization(self):
        prefix = self.config['save_data_paths']['prefix']
        _dir = self.config['save_data_paths']['optimization_analysis']
        optimization_analysis_path = prefix + _dir
        optimization_analyser = OptimizationAnalyzer(self.optimization_results,optimization_analysis_path)
        # Run all analyses
        optimization_analyser.run_all_analyses()
                
        logging.info(f"Saved metrics plot to {optimization_analysis_path}")
          
    
    def run_optimiziation(self):
        backtest_config = self.config['backtest']
        optimize_config = self.config['optimization']
        pbounds = self.config['pbounds']

        # Define default parameters
        initial_budget = backtest_config['initial_budget']
        fee_rate = backtest_config['fee_rate']
        init_points = optimize_config['init_points']
        n_iter = optimize_config['n_iter']
        #if optimization enambeled optimize first 
        try:
            logging.info("Running backtest simulation with parameter optimization...")
            # Create StrategyOptimizer instance for optimization
            optimizer = StrategyOptimizer(
                    initial_budget=initial_budget,
                    fee_rate=fee_rate,
                    init_points = init_points,
                    n_iter=n_iter,
                    df =self.df
                )
                
               
            self.optimization_results = optimizer.optimize(pbounds)
            
            
            best_optimization_results = optimizer.get_best_parameters(self.optimization_results)
            
            
           
            logging.info("Updating configuration parameters")
            
            save_config_yaml(best_optimization_results, self.config['save_data_paths']['update_config'])
            
            
            return self
                
        except Exception as e:
            logging.error(f"Error running backtest optimization: {str(e)}")
            logging.error(traceback.format_exc())  # Make sure traceback is visible
            return None, None
        
    def run_optimization_backtest(self):
        """
        Execute the complete backtesting workflow from data loading to visualization.
        This is a convenience method that chains all the individual steps.
        
        Returns:
            Backtester: Self for method chaining.
        """
        data_path = self.config['load_data_path']
        
        # Load data if path is provided
        if data_path:
            self.load_data(data_path)
            
        
        logging.info("Starting Optimization workflow...")
        self.run_optimiziation()
        logging.info("Optimization Workflow Finished Succesfully")

        if self.optimization_analysis_enabled:
            self.analyise_optimization()

if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    config_modes = config['modes']
    backtest_optimizer = BacktesterOptimization(config)
    backtest_optimizer.set_optimization_analysis_mode(config_modes['analise_optimization_mode'])

    #run otpimiztion
    backtest_optimizer.run_optimization_backtest()
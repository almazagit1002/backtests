from bayes_opt import BayesianOptimization
import pandas as pd
import logging

from mean_reversion_volatility_bands.back_test import TradingBacktester
from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator

class StrategyOptimizer:
    """
    A class for optimizing trading strategy parameters using Bayesian Optimization.
    """

    def __init__(self, initial_budget, fee_rate, init_points, n_iter, df, log_level=logging.INFO):
        """
        Initialize the optimizer with fixed strategy parameters.

        Parameters:
        -----------
        initial_budget : float
            Initial trading budget
        fee_rate : float
            Trading fee as a percentage (0.005 = 0.5%)
        init_points : int
            Number of initial random points for Bayesian optimization
        n_iter : int
            Number of iterations for Bayesian optimization
        df : pandas.DataFrame
            DataFrame containing price data
        log_level : logging level
            Level of logging detail (default: logging.INFO)
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        
        # Create handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.initial_budget = initial_budget
        self.fee_rate = fee_rate
        self.init_points = init_points
        self.n_iter = n_iter
        self.df = df.copy()
        self.optimization_results = []  # Store all optimization trials
        
        self.logger.info(f"StrategyOptimizer initialized with budget: {initial_budget}, fee rate: {fee_rate}, data shape: {df.shape}")
    def _backtest_objective(self, tp_level, sl_level, max_positions, 
                        atr_multiplier,keltner_period, cci_period, bollinger_period, std_multiplier,
                        CCI_up_threshold, CCI_low_threshold, Bollinger_Keltner_alignment, window_size):
        """
        Objective function for Bayesian optimization.
        
        This function configures trading indicators with the given parameters,
        generates trading signals, runs a backtest, and returns the total return percentage.
        
        Parameters:
        -----------
        tp_level : float
            Take profit level as a percentage of entry price
        sl_level : float
            Stop loss level as a percentage of entry price
        max_positions : float (will be rounded to int)
            Maximum number of concurrent positions
        atr_multiplier : float
            Multiplier for Average True Range (Keltner Channels)
        cci_period : float (will be rounded to int)
            Period for Commodity Channel Index calculation
        bollinger_period : float (will be rounded to int)
            Period for Bollinger Bands calculation
        std_multiplier : float
            Standard deviation multiplier for Bollinger Bands
        CCI_up_threshold : float (will be rounded to int)
            Upper threshold for CCI indicator
        CCI_low_threshold : float (will be rounded to int)
            Lower threshold for CCI indicator
        Bollinger_Keltner_alignment : float
            Alignment threshold between Bollinger Bands and Keltner Channels
        window_size : float (will be rounded to int)
            Window size for signal smoothing
            
        Returns:
        --------
        float : Total return percentage from backtest
        """
        try:
            self.logger.debug(f"Starting backtest objective with parameters: tp={tp_level}, sl={sl_level}, max_pos={max_positions}")
        
            # Apply rounding and type conversion
            max_positions = int(round(max_positions))
            cci_period = int(round(cci_period))
            keltner_period = int(round(keltner_period))
            bollinger_period = int(round(bollinger_period))
            window_size = int(round(window_size))
            
            std_multiplier = round(std_multiplier, 2)
            atr_multiplier = round(atr_multiplier, 2)
            tp_level = round(tp_level, 2)
            sl_level = round(sl_level, 2)
            CCI_up_threshold = int(round(CCI_up_threshold))
            CCI_low_threshold = int(round(CCI_low_threshold))
            Bollinger_Keltner_alignment = round(Bollinger_Keltner_alignment, 4)
            min_required_rows = 30  # Minimum rows needed for indicator calculation

            self.logger.debug(f"Rounded parameters: cci_period={cci_period}, bollinger_period={bollinger_period}, window_size={window_size}")
            
            # Apply Indicators
            indicators_df = self.df.copy()  # Ensure we're working with a copy
            
            if indicators_df is None or indicators_df.empty:
                self.logger.error("DataFrame is empty or None")
                return -100.0  # Return a poor score for invalid data
                
            self.logger.debug(f"DataFrame shape before indicators: {indicators_df.shape}")
            
            indicators = TradingIndicators(debug_mode=False, logging_mode=False)  # Instantiate the class
            
            self.logger.debug("Applying trading indicators...")
            
            try:
                # Add Keltner Channels
                indicators_df = indicators.add_keltner_channels(indicators_df, period=keltner_period, atr_multiplier=atr_multiplier)
                self.logger.debug(f"Added Keltner Channels with atr_multiplier={atr_multiplier}")
                
                # Add CCI
                indicators_df = indicators.add_cci(indicators_df, period=cci_period)
                self.logger.debug(f"Added CCI with period={cci_period}")
                
                # Add Bollinger Bands
                indicators_df = indicators.add_bollinger_bands(indicators_df, period=bollinger_period, std_multiplier=std_multiplier)
                self.logger.debug(f"Added Bollinger Bands with period={bollinger_period}, std_multiplier={std_multiplier}")
                
                self.logger.debug(f"DataFrame shape after indicators: {indicators_df.shape}")
                
                # Log column names for debugging
                self.logger.debug(f"DataFrame columns after indicators: {indicators_df.columns.tolist()}")

                # Generate trading signals
                signals_df = SignalGenerator.generate_signals(
                    indicators_df, 
                    CCI_up_threshold, 
                    CCI_low_threshold, 
                    Bollinger_Keltner_alignment, 
                    window_size, 
                    min_required_rows
                )
                self.logger.debug(f"Generated signals with thresholds: CCI_up={CCI_up_threshold}, CCI_low={CCI_low_threshold}, alignment={Bollinger_Keltner_alignment}")
                self.logger.debug(f"Signals DataFrame shape: {signals_df.shape}")

                # Run backtest
                backtester = TradingBacktester(
                    initial_budget=self.initial_budget,
                    tp_level=tp_level,
                    sl_level=sl_level,
                    fee_rate=self.fee_rate,
                    max_positions=max_positions
                )
                self.logger.debug(f"Initialized backtester with tp={tp_level}, sl={sl_level}, max_positions={max_positions}")

                backtester.backtest(signals_df)
                metrics = backtester.get_metrics()
                total_return_pct = metrics['total_return_pct']
                
                
                
                # Store the parameters and results
                result_item = {
                    'tp_level': tp_level,
                    'sl_level': sl_level,
                    'max_positions': max_positions,
                    'keltner_period': keltner_period,
                    'cci_period': cci_period,
                    'bollinger_period': bollinger_period,
                    'atr_multiplier': atr_multiplier,
                    'std_multiplier': std_multiplier,
                    'CCI_up_threshold': CCI_up_threshold,
                    'CCI_low_threshold': CCI_low_threshold,
                    'Bollinger_Keltner_alignment': Bollinger_Keltner_alignment,
                    'window_size': window_size,
                    'total_return_pct': total_return_pct
                }
                
                self.optimization_results.append(result_item)
                self.logger.debug(f"Added result to optimization_results. Current length: {len(self.optimization_results)}")
                
                return total_return_pct
                
            except Exception as e:
                self.logger.error(f"Error during indicator calculation or backtest: {str(e)}", exc_info=True)
                return -100.0  # Return a poor score for failed iterations
                
        except Exception as e:
            self.logger.error(f"Unexpected error in backtest objective: {str(e)}", exc_info=True)
            return -100.0  # Return a poor score for failed iterations


    def optimize(self,pbounds):
        """
        Perform Bayesian Optimization to find the best trading strategy parameters.
        
        This method defines parameter bounds and uses Bayesian Optimization to
        search for the best combination of parameters that maximize the total return.
        
        Returns:
        --------
        pandas.DataFrame : A DataFrame containing all optimization results
        """
        self.logger.info("Starting Bayesian optimization")
        
        # # Define parameter bounds
        # pbounds = {
        #     'tp_level': (1.0, 3.0),            # Take profit level
        #     'sl_level': (0.5, 2.0),            # Stop loss level
        #     'max_positions': (1, 10),          # Maximum concurrent positions
        #     'atr_multiplier': (1.5, 3.5),      # ATR multiplier for Keltner Channels
        #     'keltner_period': (15,30),
        #     'cci_period': (14, 50),            # CCI indicator period
        #     'bollinger_period': (10, 50),      # Bollinger Bands period
        #     'std_multiplier': (1.5, 3.0),      # Standard deviation multiplier for Bollinger Bands
        #     'CCI_up_threshold': (50, 100),      # CCI upper threshold
        #     'CCI_low_threshold': (-100, -10),   # CCI lower threshold
        #     'Bollinger_Keltner_alignment': (0.001, 0.01),  # Alignment threshold
        #     'window_size': (3, 7)              # Signal smoothing window size
        # }
        
        self.logger.debug(f"Parameter bounds: {pbounds}")

        try:
            # Create the Bayesian optimizer
            optimizer = BayesianOptimization(
                f=lambda tp_level, sl_level, max_positions, atr_multiplier,keltner_period, cci_period, 
                        bollinger_period, std_multiplier, CCI_up_threshold, CCI_low_threshold, 
                        Bollinger_Keltner_alignment, window_size: 
                        self._backtest_objective(tp_level, sl_level, max_positions, atr_multiplier, 
                                                keltner_period,cci_period, bollinger_period, std_multiplier, 
                                                CCI_up_threshold, CCI_low_threshold, 
                                                Bollinger_Keltner_alignment, window_size),
                pbounds=pbounds,
                random_state=42
            )
            
            self.logger.info(f"Running optimization with {self.init_points} initial points and {self.n_iter} iterations")
            
            # Add progress tracking logs
            self.logger.info("Starting maximize method")
            
            # Run the optimization
            optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)  # Adjust iterations as needed
            
            # Log immediate confirmation that maximize completed
            self.logger.info("Maximize method completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            # Create an empty DataFrame if optimization fails
            return pd.DataFrame(self.optimization_results)
            
        self.logger.info(f"Optimization completed. Total trials: {len(self.optimization_results)}")
        
        # Log confirmation before creating DataFrame
        self.logger.info("Creating results DataFrame")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(self.optimization_results)
        self.logger.debug(f"Results shape: {df_results.shape}")
        
        return df_results


    def get_best_parameters(self, optimization_result):
        """
        Extracts the best parameters from the optimization result.
        
        Parameters:
        -----------
        optimization_result : pandas.DataFrame
            DataFrame containing optimization results
            
        Returns:
        --------
        pandas.Series : The row with the highest total return percentage
        """
        self.logger.info("Finding best parameters from optimization results")
        
        # Check if optimization_result is None
        if optimization_result is None:
            self.logger.warning("Optimization result is None")
            return None
            
        # Check if the dataframe is empty
        if optimization_result.empty:
            self.logger.warning("Optimization result is empty")
            
            # Check if we have results in our internal list
            if self.optimization_results:
                self.logger.info("Using internal optimization_results instead")
                # Convert internal results to DataFrame
                optimization_result = pd.DataFrame(self.optimization_results)
            else:
                self.logger.warning("No optimization results available")
                return None
                
        try:
            # Log all results for debugging
            self.logger.debug("All optimization results:")
            for i, row in optimization_result.iterrows():
                self.logger.debug(f"Trial {i}: return={row['total_return_pct']:.2f}%, tp={row['tp_level']}, sl={row['sl_level']}")
                
            # Find the parameters with the highest return
            best_index = optimization_result['total_return_pct'].idxmax()
            self.logger.info(f"Best result at index {best_index}")
            
            best_result = optimization_result.loc[best_index]

            
            print(best_result)
            return best_result
            
        except Exception as e:
            self.logger.error(f"Error finding best parameters: {str(e)}", exc_info=True)
            
            # Try a different approach if there was an error
            self.logger.info("Attempting alternative method to find best parameters")
            
            try:
                # Sort by total_return_pct and take the first row
                best_row = optimization_result.sort_values('total_return_pct', ascending=False).iloc[0]
                self.logger.info(f"Alternative method - Best return: {best_row['total_return_pct']:.2f}%")
                print(best_row)
                return best_row
                
            except Exception as e2:
                self.logger.error(f"Alternative method also failed: {str(e2)}", exc_info=True)
                return None
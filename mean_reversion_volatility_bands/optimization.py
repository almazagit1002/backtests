from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
import logging

from mean_reversion_volatility_bands.back_test import TradingBacktester
from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator

from utils.error_handler import ErrorHandler

class StrategyOptimizer:
    """
    A class for optimizing trading strategy parameters using Bayesian Optimization.
    """

    def __init__(self, initial_budget, fee_rate, init_points, n_iter, df, signal_type, log_level=logging.INFO, debug_mode=False):
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
        signal_type : str
            Type of signals to generate ('long', 'short', or 'mix')
        log_level : logging level
            Level of logging detail (default: logging.INFO)
        debug_mode : bool
            Whether to enable debug mode for error handler (default: False)
        """
        # Initialize the error handler
        self.error_handler = ErrorHandler(logger_name="StrategyOptimizer", debug_mode=debug_mode)
        
        # Set log level based on parameter
        if debug_mode:
            self.error_handler.set_debug_mode(True)
        elif log_level != logging.INFO:
            self.error_handler.logger.setLevel(log_level)
        
        # Validate initial parameters
        if initial_budget <= 0:
            self.error_handler.logger.warning(f"Invalid initial_budget: {initial_budget}. Using default value of 10000.")
            initial_budget = 10000
            
        if fee_rate < 0 or fee_rate > 0.1:  # Sanity check for fee rate (0-10%)
            self.error_handler.logger.warning(f"Unusual fee_rate: {fee_rate}. Fees are typically between 0 and 0.1 (0-10%)")
            
        if init_points <= 0 or n_iter <= 0:
            self.error_handler.logger.error(f"Invalid optimization parameters: init_points={init_points}, n_iter={n_iter}")
            self.error_handler.logger.warning("Using default values: init_points=5, n_iter=10")
            init_points = max(5, init_points)
            n_iter = max(10, n_iter)
            
        # Validate signal_type
        valid_signal_types = ['long', 'short', 'mix']
        if signal_type not in valid_signal_types:
            self.error_handler.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Validate input DataFrame
        df_valid, df_msg = self.error_handler.check_dataframe_validity(
            df, 
            required_columns=['timestamp', 'open', 'high', 'low', 'close'], 
            min_rows=50,  # Need sufficient data for backtesting
            check_full=True  # Check all rows for price data
        )
        
        if not df_valid:
            self.error_handler.logger.error(f"Invalid input DataFrame: {df_msg}")
            # If there's an issue with the DataFrame, we'll create a minimal valid DataFrame
            # to allow initialization to continue, but optimization will fail gracefully
            if df is None or len(df) == 0:
                self.error_handler.logger.warning("Creating minimal DataFrame for initialization")
                df = pd.DataFrame({
                    'timestamp': pd.date_range(start='2020-01-01', periods=100),
                    'open': np.ones(100),
                    'high': np.ones(100),
                    'low': np.ones(100),
                    'close': np.ones(100)
                })
        else:
            # Log data statistics for the valid DataFrame
            self.error_handler.log_dataframe_stats(df, ['open', 'high', 'low', 'close'])
        
        self.initial_budget = initial_budget
        self.fee_rate = fee_rate
        self.init_points = init_points
        self.n_iter = n_iter
        self.df = df.copy()
        self.signal_type = signal_type
        self.optimization_results = []  # Store all optimization trials
        
        self.error_handler.logger.info(f"StrategyOptimizer initialized with budget: {initial_budget}, fee rate: {fee_rate}, data shape: {df.shape}")

    def _backtest_objective(self, tp_level, sl_level, max_positions, 
                        atr_multiplier, keltner_period, cci_period, bollinger_period, std_multiplier,
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
        keltner_period : float (will be rounded to int)
            Period for Keltner Channels calculation
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
        # Define a default return for failed backtests - using a function for consistency
        def fail_with_value(reason, value=-100.0):
            self.error_handler.logger.error(f"Backtest failed: {reason}")
            return value
            
        try:
            self.error_handler.logger.debug(f"Starting backtest objective with parameters: tp={tp_level}, sl={sl_level}, max_pos={max_positions}")
        
            # Validate input parameters before rounding
            if not all(isinstance(param, (int, float)) for param in [
                tp_level, sl_level, max_positions, atr_multiplier, keltner_period, 
                cci_period, bollinger_period, std_multiplier, CCI_up_threshold, 
                CCI_low_threshold, Bollinger_Keltner_alignment, window_size
            ]):
                return fail_with_value("Non-numeric parameters provided")
                
            # Apply rounding and type conversion with error checking
            try:
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
            except Exception as e:
                return fail_with_value(f"Parameter conversion error: {str(e)}")
            
            # Additional parameter validation
            if max_positions <= 0:
                return fail_with_value("max_positions must be greater than 0")
                
            if cci_period <= 1 or bollinger_period <= 1 or keltner_period <= 1:
                return fail_with_value("Indicator periods must be greater than 1")
                
            if window_size < 1:
                return fail_with_value("window_size must be at least 1")
                
            if std_multiplier <= 0 or atr_multiplier <= 0:
                return fail_with_value("Multipliers must be greater than 0")
                
            min_required_rows = max(30, cci_period * 3, bollinger_period * 3, keltner_period * 3)
            
            self.error_handler.logger.debug(f"Rounded parameters: cci_period={cci_period}, bollinger_period={bollinger_period}, window_size={window_size}")
            
            # Validate DataFrame
            df_valid, df_msg = self.error_handler.check_dataframe_validity(
                self.df,
                required_columns=['timestamp', 'open', 'high', 'low', 'close'],
                min_rows=min_required_rows
            )
            
            if not df_valid:
                return fail_with_value(f"Invalid DataFrame: {df_msg}")
            
            # Apply Indicators
            indicators_df = self.df.copy()  # Ensure we're working with a copy
            self.error_handler.logger.debug(f"DataFrame shape before indicators: {indicators_df.shape}")
            
            indicators = TradingIndicators(debug_mode=False)  # Instantiate the class
            
            self.error_handler.logger.debug("Applying trading indicators...")
            
            # Apply each indicator with error handling
            try:
                # Add Keltner Channels
                indicators_df = self.error_handler.safe_calculation(
                    func=lambda: indicators.add_keltner_channels(indicators_df, period=keltner_period, atr_multiplier=atr_multiplier),
                    default_value=None
                )
                
                if indicators_df is None:
                    
                    return fail_with_value("Failed to add Keltner Channels")
                    
                self.error_handler.logger.debug(f"Added Keltner Channels with atr_multiplier={atr_multiplier}")
                
                # Add CCI
                indicators_df = self.error_handler.safe_calculation(
                    func=lambda: indicators.add_cci(indicators_df, period=cci_period),
                    default_value=None
                )
                
                if indicators_df is None:
                    return fail_with_value("Failed to add CCI")
                    
                self.error_handler.logger.debug(f"Added CCI with period={cci_period}")
                
                # Add Bollinger Bands
                indicators_df = self.error_handler.safe_calculation(
                    func=lambda: indicators.add_bollinger_bands(indicators_df, period=bollinger_period, std_multiplier=std_multiplier),
                    default_value=None
                )
                
                if indicators_df is None:
                    return fail_with_value("Failed to add Bollinger Bands")
                    
                self.error_handler.logger.debug(f"Added Bollinger Bands with period={bollinger_period}, std_multiplier={std_multiplier}")
                
                # Check for expected columns after adding indicators
                required_indicator_columns = [
                    'KeltnerUpper', 'KeltnerLower',
                    'CCI', 'BollingerUpper', 'BollingerLower'
                ]
                
                missing_columns = [col for col in required_indicator_columns if col not in indicators_df.columns]
                if missing_columns:
                    
                    self.error_handler.logger.warning(f"Missing indicator columns: {missing_columns}")
                    # Continue anyway, the signal generator will handle this
        
                # Generate trading signals
                # Create an instance of SignalGenerator first
                signal_generator = SignalGenerator() 
                signals_df = self.error_handler.safe_calculation(
                    func=lambda: signal_generator.generate_signals(
                        indicators_df, 
                        CCI_up_threshold, 
                        CCI_low_threshold, 
                        Bollinger_Keltner_alignment, 
                        window_size, 
                        min_required_rows,
                        self.signal_type
                    ),
                    default_value=None
                )
        
                if signals_df is None:
                    return fail_with_value("Failed to generate signals")
                    
                self.error_handler.logger.debug(f"Generated signals with thresholds: CCI_up={CCI_up_threshold}, CCI_low={CCI_low_threshold}, alignment={Bollinger_Keltner_alignment}")
                
                    
               # Check if there are any signals
                long_signal_count = signals_df['LongSignal'].sum()  # Sum of True values
                short_signal_count = signals_df['ShortSignal'].sum()  # Sum of True values
                total_signal_count = long_signal_count + short_signal_count

                if total_signal_count == 0:
                    self.error_handler.logger.warning("No trading signals generated with current parameters")
                    return -50.0  # Return a moderate penalty for no signals
                    
                self.error_handler.logger.debug(f"Signals count: {total_signal_count} out of {len(signals_df)} rows " 
                                            f"({long_signal_count} long, {short_signal_count} short)")

                # Run backtest
                backtester = TradingBacktester(
                    initial_budget=self.initial_budget,
                    tp_level=tp_level,
                    sl_level=sl_level,
                    fee_rate=self.fee_rate,
                    max_positions=max_positions
                )
                self.error_handler.logger.debug(f"Initialized backtester with tp={tp_level}, sl={sl_level}, max_positions={max_positions}")

                # Run the backtest with error handling
                backtest_success = self.error_handler.safe_calculation(
                    func=lambda: backtester.backtest(signals_df),
                    default_value=False
                )
                if not backtest_success:
                    return fail_with_value("Backtest execution failed")
                    
                # Get metrics with error handling
                metrics = self.error_handler.safe_calculation(
                    func=lambda: backtester.get_metrics(),
                    default_value=None
                )
                if metrics is None or 'total_return_pct' not in metrics:
                    return fail_with_value("Failed to retrieve backtest metrics")
                    
                total_return_pct = metrics['total_return_pct']
                
                # Sanity check for extreme outlier results
                if total_return_pct > 1000:
                    self.error_handler.logger.warning(f"Suspiciously high return: {total_return_pct}%. Capping at 1000%")
                    total_return_pct = 1000
                elif total_return_pct < -100:
                    self.error_handler.logger.warning(f"Extremely negative return: {total_return_pct}%. Floor at -100%")
                    total_return_pct = -100
                
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
                
                # Additional metrics to store if available
                metrics_to_store = ['win_rate', 'profit_factor', 'max_drawdown_pct', 'total_trades']
                for metric in metrics_to_store:
                    if metric in metrics:
                        result_item[metric] = metrics[metric]
                
                self.optimization_results.append(result_item)
                self.error_handler.logger.debug(f"Added result {len(self.optimization_results)}: return={total_return_pct:.2f}%")
                
                return total_return_pct
                
            except Exception as e:
                return fail_with_value(f"Error during indicator calculation or backtest: {str(e)}")
                
        except Exception as e:
            return fail_with_value(f"Unexpected error in backtest objective: {str(e)}")


    def optimize(self, pbounds):
        """
        Perform Bayesian Optimization to find the best trading strategy parameters.
        
        Parameters:
        -----------
        pbounds : dict, optional
            Dictionary of parameter bounds for optimization.
            If None, default bounds will be used.
        
        Returns:
        --------
        pandas.DataFrame : A DataFrame containing all optimization results
        """
        self.error_handler.logger.info("Starting Bayesian optimization")

        try:
            # Create the Bayesian optimizer
            optimizer = BayesianOptimization(
                f=lambda tp_level, sl_level, max_positions, atr_multiplier, keltner_period, cci_period, 
                        bollinger_period, std_multiplier, CCI_up_threshold, CCI_low_threshold, 
                        Bollinger_Keltner_alignment, window_size: 
                        self._backtest_objective(tp_level, sl_level, max_positions, atr_multiplier, 
                                                keltner_period, cci_period, bollinger_period, std_multiplier, 
                                                CCI_up_threshold, CCI_low_threshold, 
                                                Bollinger_Keltner_alignment, window_size),
                pbounds=pbounds,
                random_state=42
            )
            
            self.error_handler.logger.info(f"Running optimization with {self.init_points} initial points and {self.n_iter} iterations")
            
            # Add progress tracking logs
            self.error_handler.logger.info("Starting maximize method")
            
            # Run the optimization with progress tracking
            try:
                for i in range(self.init_points + self.n_iter):
                    if i == 0:
                        self.error_handler.logger.info("Starting random initialization phase...")
                    elif i == self.init_points:
                        self.error_handler.logger.info("Starting Bayesian optimization phase...")
                    
                    # We can't directly track progress inside maximize, so we'll log before and after
                    if i > 0 and i % 5 == 0:
                        self.error_handler.logger.info(f"Progress: {i}/{self.init_points + self.n_iter} iterations completed")
                
                # Run the optimization
                optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
                
                # Log immediate confirmation that maximize completed
                self.error_handler.logger.info("Maximize method completed successfully")
                
            except Exception as e:
                self.error_handler.logger.error(f"Error during optimization maximize: {str(e)}")
                # Continue to results processing even if maximize fails
            
        except Exception as e:
            self.error_handler.logger.error(f"Error during optimization setup: {str(e)}")
            # Create an empty DataFrame if optimization fails completely
            return pd.DataFrame(self.optimization_results)
            
        self.error_handler.logger.info(f"Optimization completed. Total trials: {len(self.optimization_results)}")
        
        # Convert results to DataFrame with error handling
        try:
            # Log confirmation before creating DataFrame
            self.error_handler.logger.info("Creating results DataFrame")
            
            if not self.optimization_results:
                self.error_handler.logger.warning("No optimization results collected")
                return pd.DataFrame()
                
            # Convert results to DataFrame
            df_results = pd.DataFrame(self.optimization_results)
            
            # Log basic statistics about the results
            if 'total_return_pct' in df_results.columns:
                self.error_handler.logger.info(f"Results summary - Min return: {df_results['total_return_pct'].min():.2f}%, " + 
                                            f"Max return: {df_results['total_return_pct'].max():.2f}%, " +
                                            f"Mean return: {df_results['total_return_pct'].mean():.2f}%")
            
            self.error_handler.logger.debug(f"Results shape: {df_results.shape}")
            return df_results
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating results DataFrame: {str(e)}")
            # Try a more direct approach if DataFrame creation fails
            try:
                return pd.DataFrame(self.optimization_results)
            except:
                self.error_handler.logger.error("Failed to create results DataFrame even with direct approach")
                return pd.DataFrame()


    def get_best_parameters(self, optimization_result=None):
        """
        Extracts the best parameters from the optimization result.
        
        Parameters:
        -----------
        optimization_result : pandas.DataFrame, optional
            DataFrame containing optimization results. If None, uses internal results.
            
        Returns:
        --------
        pandas.Series : The row with the highest total return percentage
        dict : Best parameters in dictionary format if DataFrame conversion fails
        """
        self.error_handler.logger.info("Finding best parameters from optimization results")
        
        # Use internal results if no DataFrame is provided
        if optimization_result is None:
            self.error_handler.logger.debug("No optimization_result provided, using internal results")
            
            if not self.optimization_results:
                self.error_handler.logger.warning("No optimization results available")
                return None
                
            try:
                optimization_result = pd.DataFrame(self.optimization_results)
            except Exception as e:
                self.error_handler.logger.error(f"Error creating DataFrame from internal results: {str(e)}")
                
                # If we can't create a DataFrame, find the best result directly from the list
                if self.optimization_results:
                    best_result = max(self.optimization_results, key=lambda x: x.get('total_return_pct', -float('inf')))
                    self.error_handler.logger.info(f"Best result found (dict format): {best_result['total_return_pct']:.2f}%")
                    return best_result
                return None
            
        # Check if optimization_result is None after attempted conversion
        if optimization_result is None:
            self.error_handler.logger.warning("Failed to obtain valid optimization results")
            return None
            
        # Check if the dataframe is empty
        if isinstance(optimization_result, pd.DataFrame) and optimization_result.empty:
            self.error_handler.logger.warning("Optimization result is empty")
            return None
                
        try:
            # Validate the DataFrame has the expected column
            if 'total_return_pct' not in optimization_result.columns:
                self.error_handler.logger.error("DataFrame missing 'total_return_pct' column")
                return None
                
            # Log statistics about returns
            self.error_handler.logger.debug(f"Return statistics: min={optimization_result['total_return_pct'].min():.2f}%, " +
                                        f"max={optimization_result['total_return_pct'].max():.2f}%, " +
                                        f"mean={optimization_result['total_return_pct'].mean():.2f}%")
                
            # Find the parameters with the highest return
            best_index = optimization_result['total_return_pct'].idxmax()
            self.error_handler.logger.info(f"Best result at index {best_index}")
            
            best_result = optimization_result.loc[best_index]
            
            # Log the best parameters
            param_str = ", ".join([f"{param}={best_result[param]}" for param in [
                'tp_level', 'sl_level', 'max_positions', 'cci_period', 'bollinger_period'
            ]])
            self.error_handler.logger.info(f"Best parameters: {param_str} with return: {best_result['total_return_pct']:.2f}%")
            
            print(best_result)
            return best_result
            
        except Exception as e:
            self.error_handler.logger.error(f"Error finding best parameters: {str(e)}")
            
            # Try a different approach if there was an error
            self.error_handler.logger.info("Attempting alternative method to find best parameters")
            
            try:
                # Sort by total_return_pct and take the first row
                best_row = optimization_result.sort_values('total_return_pct', ascending=False).iloc[0]
                self.error_handler.logger.info(f"Alternative method - Best return: {best_row['total_return_pct']:.2f}%")
                print(best_row)
                return best_row
                
            except Exception as e2:
                self.error_handler.logger.error(f"Alternative method also failed: {str(e2)}")
                
                # Last resort: manual search through the list
                if isinstance(self.optimization_results, list) and self.optimization_results:
                    try:
                        best_result = max(self.optimization_results, key=lambda x: x.get('total_return_pct', -float('inf')))
                        self.error_handler.logger.info(f"Manual search found best result: {best_result['total_return_pct']:.2f}%")
                        return best_result
                    except Exception as e3:
                        self.error_handler.logger.error(f"Manual search also failed: {str(e3)}")
                
                return None
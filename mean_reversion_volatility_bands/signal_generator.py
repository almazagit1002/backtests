import pandas as pd
from  utils.error_handler import ErrorHandler

class SignalGenerator:
    """
    Generates trading signals based on technical indicators.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the signal generator with error handling.
        
        Parameters:
        -----------
        debug_mode : bool, optional
            Whether to enable debug level logging (default: False)
        """
        self.error_handler = ErrorHandler("SignalGenerator", debug_mode)
        self.logger = self.error_handler.logger
    
    def generate_signals(self, df: pd.DataFrame, CCI_up_threshold, CCI_low_threshold, 
                         Bollinger_Keltner_alignment, window_size, min_required_rows, 
                         signal_type='mix') -> pd.DataFrame:
        """
        Generate signals based on trading indicators.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and indicator data
        CCI_up_threshold : float
            Upper threshold for CCI indicator
        CCI_low_threshold : float
            Lower threshold for CCI indicator
        Bollinger_Keltner_alignment : float
            Alignment threshold for Bollinger and Keltner bands
        window_size : int
            Window size for checking conditions
        min_required_rows : int
            Minimum required rows for signal generation
        signal_type : str, optional
            Type of signals to generate - 'long', 'short', or 'mix' (default: 'mix')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with generated signals
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            self.logger.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
        
        # Define required columns
        required_columns = ['close', 'KeltnerLower', 'KeltnerUpper', 'CCI', 'BollingerLower', 'BollingerUpper']
        
        # Check dataframe validity
        valid, message = self.error_handler.check_dataframe_validity(
            df, required_columns, min_required_rows, check_full=False
        )
        
        if not valid:
            self.logger.warning(f"Invalid DataFrame: {message}. Returning original DataFrame.")
            return df
        
        # Log statistics about key columns
        self.error_handler.log_dataframe_stats(df, required_columns)
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize signal columns
        result['LongSignal'] = False
        result['ShortSignal'] = False
        
        # Initialize condition tracking columns
        result['price_below_keltner'] = False
        result['cci_below_threshold'] = False
        result['bands_aligned_long'] = False
        result['price_above_keltner'] = False
        result['cci_above_threshold'] = False
        result['bands_aligned_short'] = False
        
        # Generate signals
        for i in range(min_required_rows, len(result)):
            current_index = result.index[i]
            window_data = result.iloc[max(0, i - window_size + 1) : i + 1]
            
            # Process signal generation based on type
            if signal_type in ['long', 'mix']:
                self._generate_long_signals(result, current_index, window_data, 
                                           CCI_low_threshold, Bollinger_Keltner_alignment)
            
            if signal_type in ['short', 'mix']:
                self._generate_short_signals(result, current_index, window_data,
                                            CCI_up_threshold, Bollinger_Keltner_alignment)
        
        # Log signal generation results
        long_signals = result['LongSignal'].sum()
        short_signals = result['ShortSignal'].sum()
        self.logger.info(f"Generated {long_signals} long signals and {short_signals} short signals")
        
        return result
    
    def _generate_long_signals(self, result, current_index, window_data, 
                              CCI_low_threshold, Bollinger_Keltner_alignment):
        """Generate long signals based on conditions."""
        try:
            # Long signal conditions
            price_below_keltner = any(window_data['close'] < window_data['KeltnerLower'])
            cci_below_threshold = any(window_data['CCI'] < CCI_low_threshold)
            
            # Safe calculation of bands alignment with proper division handling
            keltner_lower = window_data['KeltnerLower'].replace(0, float("inf"))
            bollinger_diff = abs(window_data['BollingerLower'] - window_data['KeltnerLower'])
            alignment_values = self.error_handler.handle_division(bollinger_diff, keltner_lower)
            bands_aligned_long = any(alignment_values < Bollinger_Keltner_alignment)
            
            # Store condition results in DataFrame
            result.at[current_index, 'price_below_keltner'] = price_below_keltner
            result.at[current_index, 'cci_below_threshold'] = cci_below_threshold
            result.at[current_index, 'bands_aligned_long'] = bands_aligned_long
            
            # Set long signals
            if price_below_keltner and cci_below_threshold and bands_aligned_long:
                result.at[current_index, 'LongSignal'] = True
        except Exception as e:
            self.logger.error(f"Error generating long signal at index {current_index}: {str(e)}")
    
    def _generate_short_signals(self, result, current_index, window_data,
                               CCI_up_threshold, Bollinger_Keltner_alignment):
        """Generate short signals based on conditions."""
        try:
            # Short signal conditions
            price_above_keltner = any(window_data['close'] > window_data['KeltnerUpper'])
            cci_above_threshold = any(window_data['CCI'] > CCI_up_threshold)
            
            # Safe calculation of bands alignment with proper division handling
            keltner_upper = window_data['KeltnerUpper'].replace(0, float("inf"))
            bollinger_diff = abs(window_data['BollingerUpper'] - window_data['KeltnerUpper'])
            alignment_values = self.error_handler.handle_division(bollinger_diff, keltner_upper)
            bands_aligned_short = any(alignment_values < Bollinger_Keltner_alignment)
            
            # Store condition results in DataFrame
            result.at[current_index, 'price_above_keltner'] = price_above_keltner
            result.at[current_index, 'cci_above_threshold'] = cci_above_threshold
            result.at[current_index, 'bands_aligned_short'] = bands_aligned_short
            
            # Set short signals
            if price_above_keltner and cci_above_threshold and bands_aligned_short:
                result.at[current_index, 'ShortSignal'] = True
        except Exception as e:
            self.logger.error(f"Error generating short signal at index {current_index}: {str(e)}")
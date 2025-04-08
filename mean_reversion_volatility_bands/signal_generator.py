import logging

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalGenerator:
    """
    Generates trading signals based on technical indicators.
    """
    
    @staticmethod
    def check_data_validity(df: pd.DataFrame, required_columns: list) -> bool:
        """
        Check if DataFrame has required columns and sufficient data.
        """
        if len(df) == 0:
            logging.warning("Empty DataFrame, cannot generate signals")
            return False
        
        last_index = df.index[-1]
        
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Missing column {col}, cannot generate signals")
                return False
            
            if pd.isna(df.at[last_index, col]):
                logging.warning(f"NaN value in column {col}, cannot generate signals")
                return False
        
        return True
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, CCI_up_threshold, CCI_low_threshold, 
                         Bollinger_Keltner_alignment, window_size, min_required_rows, 
                         signal_type='mix') -> pd.DataFrame:
        """
        Generate signals based on trading indicators.
        
        Parameters:
        df (pd.DataFrame): DataFrame with price and indicator data
        CCI_up_threshold: Upper threshold for CCI indicator
        CCI_low_threshold: Lower threshold for CCI indicator
        Bollinger_Keltner_alignment: Alignment threshold for Bollinger and Keltner bands
        window_size: Window size for checking conditions
        min_required_rows: Minimum required rows for signal generation
        signal_type (str): Type of signals to generate - 'long', 'short', or 'mix' (default)
        
        Returns:
        pd.DataFrame: DataFrame with generated signals
        """
        # Validate signal_type parameter
        if signal_type not in ['long', 'short', 'mix']:
            logging.warning(f"Invalid signal_type: {signal_type}. Using default 'mix'.")
            signal_type = 'mix'
            
        if len(df) < min_required_rows:
            logging.warning(f"Not enough data points ({len(df)}). Need at least {min_required_rows}.")
            return df  # Return original DataFrame instead of False
        
        required_columns = ['close', 'KeltnerLower', 'KeltnerUpper', 'CCI', 'BollingerLower', 'BollingerUpper']
        if not SignalGenerator.check_data_validity(df, required_columns):
            return df  # Return original DataFrame instead of False

        result = df.copy()
        result['LongSignal'] = False
        result['ShortSignal'] = False
        result['price_below_keltner'] = False
        result['cci_below_threshold'] = False
        result['bands_aligned_long'] = False
        result['price_above_keltner'] = False
        result['cci_above_threshold'] = False
        result['bands_aligned_short'] = False

        for i in range(min_required_rows, len(result)):
            current_index = result.index[i]
            window_data = result.iloc[max(0, i - window_size + 1) : i + 1]

            # Check conditions based on signal_type
            if signal_type in ['long', 'mix']:
                # Long signal conditions
                price_below_keltner = any(window_data['close'] < window_data['KeltnerLower'])
                cci_below_threshold = any(window_data['CCI'] < CCI_low_threshold)
                bands_aligned_long = any(
                    abs(window_data['BollingerLower'] - window_data['KeltnerLower']) / window_data['KeltnerLower'].replace(0, float("inf")) < Bollinger_Keltner_alignment
                )
                
                # Store condition results in DataFrame
                result.at[current_index, 'price_below_keltner'] = price_below_keltner
                result.at[current_index, 'cci_below_threshold'] = cci_below_threshold
                result.at[current_index, 'bands_aligned_long'] = bands_aligned_long
                
                # Set long signals
                if price_below_keltner and cci_below_threshold and bands_aligned_long:
                    result.at[current_index, 'LongSignal'] = True

            if signal_type in ['short', 'mix']:
                # Short signal conditions
                price_above_keltner = any(window_data['close'] > window_data['KeltnerUpper'])
                cci_above_threshold = any(window_data['CCI'] > CCI_up_threshold)
                bands_aligned_short = any(
                    abs(window_data['BollingerUpper'] - window_data['KeltnerUpper']) / window_data['KeltnerUpper'].replace(0, float("inf")) < Bollinger_Keltner_alignment
                )
                
                # Store condition results in DataFrame
                result.at[current_index, 'price_above_keltner'] = price_above_keltner
                result.at[current_index, 'cci_above_threshold'] = cci_above_threshold
                result.at[current_index, 'bands_aligned_short'] = bands_aligned_short
                
                # Set short signals
                if price_above_keltner and cci_above_threshold and bands_aligned_short:
                    result.at[current_index, 'ShortSignal'] = True

        return result
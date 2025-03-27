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
    def generate_signals(df: pd.DataFrame, CCI_up_threshold, CCI_low_threshold, Bollinger_Keltner_alignment, window_size, min_required_rows) -> pd.DataFrame:

        """
        Generate buy (long) and sell (short) signals based on trading indicators.
        """
        # min_required_rows= thresholds["min_required_rows"]
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

        # CCI_up_threshold = thresholds["CCI_up_threshold"]
        # CCI_low_threshold = thresholds["CCI_low_threshold"]
        # Bollinger_Keltner_alignment = thresholds["Bollinger_Keltner_alignment"]
        # window_size = thresholds["window_size"]

        for i in range(min_required_rows, len(result)):
            current_index = result.index[i]
            window_data = result.iloc[max(0, i - window_size + 1) : i + 1]

            # Check conditions for Long and Short in a single iteration
            price_below_keltner = any(window_data['close'] < window_data['KeltnerLower'])
            cci_below_threshold = any(window_data['CCI'] < CCI_low_threshold)
            bands_aligned_long = any(
                abs(window_data['BollingerLower'] - window_data['KeltnerLower']) / window_data['KeltnerLower'].replace(0, float("inf")) < Bollinger_Keltner_alignment
            )

            price_above_keltner = any(window_data['close'] > window_data['KeltnerUpper'])
            cci_above_threshold = any(window_data['CCI'] > CCI_up_threshold)
            bands_aligned_short = any(
                abs(window_data['BollingerUpper'] - window_data['KeltnerUpper']) / window_data['KeltnerUpper'].replace(0, float("inf")) < Bollinger_Keltner_alignment
            )

            # Store condition results in DataFrame
            result.at[current_index, 'price_below_keltner'] = price_below_keltner
            result.at[current_index, 'cci_below_threshold'] = cci_below_threshold
            result.at[current_index, 'bands_aligned_long'] = bands_aligned_long
            result.at[current_index, 'price_above_keltner'] = price_above_keltner
            result.at[current_index, 'cci_above_threshold'] = cci_above_threshold
            result.at[current_index, 'bands_aligned_short'] = bands_aligned_short

            # Set signals
            if price_below_keltner and cci_below_threshold and bands_aligned_long:
                result.at[current_index, 'LongSignal'] = True

            if price_above_keltner and cci_above_threshold and bands_aligned_short:
                result.at[current_index, 'ShortSignal'] = True

        return result
   
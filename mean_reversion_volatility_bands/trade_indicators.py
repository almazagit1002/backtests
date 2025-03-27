import logging

import pandas as pd
import numpy as np


class TradingIndicators:
    """
    Calculates technical indicators used in trading strategies.
    """
    def __init__(self, debug_mode: bool = False, logging_mode: bool = True):
        self.debug_mode = debug_mode
        self.logging_mode = logging_mode
        self.logger = logging.getLogger(__name__)  # Create a logger for this class only

        if not self.logging_mode:
            self.logger.setLevel(logging.CRITICAL)  # Disable only this logger
        elif self.debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent affecting other loggers


    
    def calculate_average_true_range(self,df: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR), an indicator of market volatility.
        ATR is the average of true ranges over the specified period.
        True Range is the greatest of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        
        Args:
            df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns
            
        Returns:
            pd.Series: Series containing ATR values
        """
        # Calculate the three components of True Range
        high_low = df['high'] - df['low']  # Current High - Current Low
        high_close = abs(df['high'] - df['close'].shift())  # |Current High - Previous Close|
        low_close = abs(df['low'] - df['close'].shift())  # |Current Low - Previous Close|
        
        if self.debug_mode:
            logging.debug(f"Calculated high-low range: {high_low.head()}")
            logging.debug(f"Calculated high-close range: {high_close.head()}")
            logging.debug(f"Calculated low-close range: {low_close.head()}")
        
        true_range = pd.Series(np.maximum(np.maximum(high_low, high_close), low_close), index=df.index)
        
        if self.debug_mode:
            logging.debug(f"True Range calculated: {true_range.head()}")
        
        return true_range
    
    
    def add_keltner_channels(self, df: pd.DataFrame, period, atr_multiplier) -> pd.DataFrame:
        """
        Calculate Keltner Channels using Exponential Moving Average (EMA) and Average True Range (ATR).
        Keltner Channels are volatility-based envelopes set above and below an EMA.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            period (int): The period for the EMA calculation
            atr_multiplier (float): Multiplier for the ATR to determine channel width
            
        Returns:
            pd.DataFrame: DataFrame with added Keltner Channel columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate Exponential Moving Average
        result['EMA'] = result['close'].ewm(span=period, adjust=False).mean()
        if self.debug_mode:
            logging.debug(f"EMA calculated: {result['EMA'].head()}")
        
        # Calculate Average True Range
        result['ATR'] = self.calculate_average_true_range(result)
        
        # Calculate upper and lower bands
        result['KeltnerUpper'] = result['EMA'] + (atr_multiplier * result['ATR'])
        result['KeltnerLower'] = result['EMA'] - (atr_multiplier * result['ATR'])
        
        if self.logging_mode:
            nan_upper = result['KeltnerUpper'].isna().sum()
            nan_lower = result['KeltnerLower'].isna().sum()
            total_rows = len(result)
            logging.info(f"Keltner Channels NaN values: Upper={nan_upper}/{total_rows} ({nan_upper/total_rows:.2%}), Lower={nan_lower}/{total_rows} ({nan_lower/total_rows:.2%})")
        return result
    
    
    def add_cci(self,df: pd.DataFrame, period) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index (CCI) to measure market trend strength.
        CCI measures the current price level relative to an average price level over a period of time.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            period (int): The lookback period for calculation
            
        Returns:
            pd.DataFrame: DataFrame with added CCI columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # First ensure we have enough data
        if len(result) < period:
            logging.warning(f"Not enough data points for CCI calculation. Have {len(result)}, need at least {period}.")
            # Initialize columns with default values
            result['SMA'] = result['close'].rolling(window=period, min_periods=1).mean()
            result['mean_deviation'] = 0.0
            result['CCI'] = 0.0
            return result
            
        # Calculate Simple Moving Average
        result['SMA'] = result['close'].rolling(window=period).mean()
        
        # Calculate mean deviation (average distance of price from SMA)
        mean_deviation = (result['close'] - result['SMA']).abs().rolling(window=period).mean()
        result['mean_deviation'] = mean_deviation
        
        # Calculate CCI with proper NaN and division by zero handling
        # Formula: CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress numpy warnings
            raw_cci = (result['close'] - result['SMA']) / (0.015 * mean_deviation)
        
        # Replace infinity and NaN with 0
        result['CCI'] = np.where(
            np.isfinite(raw_cci),  # Only use values that are finite (not inf or NaN)
            raw_cci,
            0.0  # Default for invalid calculations
        )
        
        # Log the percentage of zero values in CCI
        if self.logging_mode:
            zero_count = (result['CCI'] == 0).sum()
            total_rows = len(result)
            logging.info(f"CCI zeros: {zero_count}/{total_rows} ({zero_count/total_rows:.2%})")
        
        return result
    

    def add_bollinger_bands(self,df: pd.DataFrame, period, std_multiplier) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using Simple Moving Average (SMA) and standard deviation.
        Bollinger Bands consist of a middle band (SMA) with upper and lower bands
        at standard deviation levels above and below the middle band.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            period (int): The period for SMA calculation
            std_multiplier (float): Number of standard deviations for the bands
            
        Returns:
            pd.DataFrame: DataFrame with added Bollinger Band columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate Simple Moving Average
        result['SMA'] = result['close'].rolling(window=period).mean()
        
        # Calculate Standard Deviation
        result['StdDev'] = result['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        result['BollingerUpper'] = result['SMA'] + (std_multiplier * result['StdDev'])
        result['BollingerLower'] = result['SMA'] - (std_multiplier * result['StdDev'])
        
        # Log the percentage of NaN values in Bollinger Bands (due to warm-up period)
        if self.logging_mode:
            nan_upper = result['BollingerUpper'].isna().sum()
            nan_lower = result['BollingerLower'].isna().sum()
            total_rows = len(result)
            logging.info(f"Bollinger Bands NaN values: Upper={nan_upper}/{total_rows} ({nan_upper/total_rows:.2%}), Lower={nan_lower}/{total_rows} ({nan_lower/total_rows:.2%})")
            
        return result

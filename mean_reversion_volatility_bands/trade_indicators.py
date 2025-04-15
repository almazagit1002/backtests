import pandas as pd
import numpy as np
from utils.error_handler import ErrorHandler

class TradingIndicators:
    """
    Calculates technical indicators used in trading strategies.
    """
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the trading indicators calculator with error handling.
        
        Parameters:
        -----------
        debug_mode : bool, optional
            Whether to enable debug level logging (default: False)
        """
        self.error_handler = ErrorHandler("TradingIndicators", debug_mode)
        self.logger = self.error_handler.logger
    
    def calculate_average_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the Average True Range (ATR), an indicator of market volatility.
        ATR is the average of true ranges over the specified period.
        True Range is the greatest of:
        1. Current High - Current Low
        2. |Current High - Previous Close|
        3. |Current Low - Previous Close|
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'high', 'low', and 'close' columns
            
        Returns:
        --------
        pd.Series
            Series containing ATR values
        """
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not valid:
            self.logger.error(f"Cannot calculate ATR: {message}")
            return pd.Series(index=df.index)
        
        try:
            # Calculate the three components of True Range
            high_low = df['high'] - df['low']  # Current High - Current Low
            high_close = abs(df['high'] - df['close'].shift())  # |Current High - Previous Close|
            low_close = abs(df['low'] - df['close'].shift())  # |Current Low - Previous Close|
            
            self.logger.debug(f"Calculated ranges - High-Low: {high_low.head()}, High-Close: {high_close.head()}, Low-Close: {low_close.head()}")
            
            # Use numpy to safely compute the maximum
            components = np.vstack([high_low.values, high_close.values, low_close.values])
            true_range = pd.Series(np.nanmax(components, axis=0), index=df.index)
            
            # Log statistics about the calculated ATR
            nan_count = true_range.isna().sum()
            if nan_count > 0:
                self.logger.warning(f"ATR calculation produced {nan_count} NaN values ({nan_count/len(true_range):.2%})")
            
            return true_range
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=df.index)
    
    def add_keltner_channels(self, df: pd.DataFrame, period: int, atr_multiplier: float) -> pd.DataFrame:
        """
        Calculate Keltner Channels using Exponential Moving Average (EMA) and Average True Range (ATR).
        Keltner Channels are volatility-based envelopes set above and below an EMA.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        period : int
            The period for the EMA calculation
        atr_multiplier : float
            Multiplier for the ATR to determine channel width
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added Keltner Channel columns
        """
        # Check if required columns exist
        required_columns = ['high', 'low', 'close']
        valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not valid:
            self.logger.error(f"Cannot calculate Keltner Channels: {message}")
            return df.copy()
        
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Calculate Exponential Moving Average
            result['EMA'] = result['close'].ewm(span=period, adjust=False).mean()
            self.logger.debug(f"EMA calculated: {result['EMA'].head()}")
            
            # Calculate Average True Range
            result['ATR'] = self.calculate_average_true_range(result)
            
            # Log ATR statistics
            self.error_handler.log_dataframe_stats(result, ['ATR'])
            
            # Calculate upper and lower bands
            result['KeltnerUpper'] = result['EMA'] + (atr_multiplier * result['ATR'])
            result['KeltnerLower'] = result['EMA'] - (atr_multiplier * result['ATR'])
            
            # Log channel statistics
            self.error_handler.log_dataframe_stats(result, ['KeltnerUpper', 'KeltnerLower'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return df.copy()
    
    def add_cci(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Commodity Channel Index (CCI) to measure market trend strength.
        CCI measures the current price level relative to an average price level over a period of time.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        period : int
            The lookback period for calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added CCI columns
        """
        # Check if required columns exist
        required_columns = ['close']
        valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not valid:
            self.logger.error(f"Cannot calculate CCI: {message}")
            return df.copy()
        
        # First ensure we have enough data
        if len(df) < period:
            self.logger.warning(f"Not enough data points for CCI calculation. Have {len(df)}, need at least {period}.")
            # Initialize columns with default values
            result = df.copy()
            result['SMA'] = result['close'].rolling(window=period, min_periods=1).mean()
            result['mean_deviation'] = 0.0
            result['CCI'] = 0.0
            return result
        
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Calculate Simple Moving Average
            result['SMA'] = result['close'].rolling(window=period).mean()
            
            # Calculate mean deviation (average distance of price from SMA)
            mean_deviation = (result['close'] - result['SMA']).abs().rolling(window=period).mean()
            result['mean_deviation'] = mean_deviation
            
            # Use safe division to calculate CCI
            # Formula: CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
            price_minus_sma = result['close'] - result['SMA']
            denominator = 0.015 * mean_deviation
            
            # Handle division safely
            cci_values = self.error_handler.handle_division(
                price_minus_sma, denominator, replace_value=0.0
            )
            
            # Convert to Series and assign to DataFrame
            result['CCI'] = pd.Series(cci_values, index=result.index)
            
            # Log CCI statistics
            self.error_handler.log_dataframe_stats(result, ['CCI'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return df.copy()
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int, std_multiplier: float) -> pd.DataFrame:
        """
        Calculate Bollinger Bands using Simple Moving Average (SMA) and standard deviation.
        Bollinger Bands consist of a middle band (SMA) with upper and lower bands
        at standard deviation levels above and below the middle band.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        period : int
            The period for SMA calculation
        std_multiplier : float
            Number of standard deviations for the bands
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added Bollinger Band columns
        """
        # Check if required columns exist
        required_columns = ['close']
        valid, message = self.error_handler.check_dataframe_validity(df, required_columns)
        
        if not valid:
            self.logger.error(f"Cannot calculate Bollinger Bands: {message}")
            return df.copy()
        
        try:
            # Make a copy to avoid modifying the original
            result = df.copy()
            
            # Calculate Simple Moving Average
            result['SMA'] = result['close'].rolling(window=period).mean()
            
            # Calculate Standard Deviation with min_periods to handle initial NaN values
            result['StdDev'] = result['close'].rolling(window=period, min_periods=1).std()
            
            # Replace any remaining NaN values in StdDev with zeros to avoid propagation
            result['StdDev'].fillna(0, inplace=True)
            
            # Calculate upper and lower bands
            result['BollingerUpper'] = result['SMA'] + (std_multiplier * result['StdDev'])
            result['BollingerLower'] = result['SMA'] - (std_multiplier * result['StdDev'])
            
            # Log bands statistics
            self.error_handler.log_dataframe_stats(result, ['BollingerUpper', 'BollingerLower'])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return df.copy()
    
    def calculate_all_indicators(self, df: pd.DataFrame, 
                                  ema_period: int = 20, 
                                  atr_period: int = 14,
                                  keltner_multiplier: float = 2.0,
                                  cci_period: int = 20,
                                  bollinger_period: int = 20,
                                  bollinger_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate all technical indicators used in the strategy.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC price data
        ema_period : int, optional
            Period for EMA calculation (default: 20)
        atr_period : int, optional
            Period for ATR calculation (default: 14)
        keltner_multiplier : float, optional
            Multiplier for Keltner Channels (default: 2.0)
        cci_period : int, optional
            Period for CCI calculation (default: 20)
        bollinger_period : int, optional
            Period for Bollinger Bands calculation (default: 20)
        bollinger_std : float, optional
            Standard deviation multiplier for Bollinger Bands (default: 2.0)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all calculated indicators
        """
        self.logger.info(f"Calculating all indicators on DataFrame with {len(df)} rows")
        
        try:
            # Add Keltner Channels
            result = self.add_keltner_channels(df, ema_period, keltner_multiplier)
            
            # Add CCI
            result = self.add_cci(result, cci_period)
            
            # Add Bollinger Bands
            result = self.add_bollinger_bands(result, bollinger_period, bollinger_std)
            
            # Check if all required columns for signal generation are present
            required_columns = ['close', 'KeltnerLower', 'KeltnerUpper', 'CCI', 'BollingerLower', 'BollingerUpper']
            valid, message = self.error_handler.check_dataframe_validity(result, required_columns, check_full=True)
            
            if not valid:
                self.logger.warning(f"Indicator calculation incomplete: {message}")
            else:
                self.logger.info("All indicators calculated successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return df.copy()
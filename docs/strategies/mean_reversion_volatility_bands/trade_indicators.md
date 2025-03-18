# Technical Indicators

The Mean Reversion Volatility Bands strategy uses several technical indicators to identify potential trading opportunities. These indicators are implemented in the `TradingIndicators` class and are essential for strategy operation.

## Average True Range (ATR)

The Average True Range is a measure of market volatility calculated by determining the average range between high and low prices over a specified period.

### Calculation

ATR is calculated by taking the greatest of:
1. Current High - Current Low
2. |Current High - Previous Close|
3. |Current Low - Previous Close|

```python
def calculate_average_true_range(df):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.Series(
        np.maximum(np.maximum(high_low, high_close), low_close), 
        index=df.index
    )
    
    return true_range
```

ATR is particularly important in this strategy as it is used to:
- Determine the width of Keltner Channels
- Calculate position exit points (take profit and stop loss levels)

## Keltner Channels

Keltner Channels are volatility-based envelopes set above and below an Exponential Moving Average (EMA). They help identify potential overbought and oversold conditions.

### Components
- **Middle Line**: EMA of the price (typically 20-period)
- **Upper Band**: EMA + (ATR × Multiplier)
- **Lower Band**: EMA - (ATR × Multiplier)

### Implementation

```python
def add_keltner_channels(df, period=20, atr_multiplier=2.0):
    result = df.copy()
    
    # Exponential Moving Average
    result['EMA'] = result['close'].ewm(span=period, adjust=False).mean()
    
    # Average True Range
    result['ATR'] = calculate_average_true_range(result)
    
    # Upper and lower bands
    result['KeltnerUpper'] = result['EMA'] + (atr_multiplier * result['ATR'])
    result['KeltnerLower'] = result['EMA'] - (atr_multiplier * result['ATR'])
    
    return result
```

### Usage in Strategy
- Price breaking below the lower Keltner Channel contributes to long signal generation
- Price breaking above the upper Keltner Channel contributes to short signal generation

## Commodity Channel Index (CCI)

The Commodity Channel Index measures the current price level relative to an average price level over a period of time. It helps identify cyclical turns in markets.

### Calculation

CCI is calculated using the formula:
```
CCI = (Price - SMA) / (0.015 × Mean Deviation)
```
Where:
- SMA is the Simple Moving Average of price
- Mean Deviation is the average distance of price from SMA

### Implementation

```python
def add_cci(df, period=20):
    result = df.copy()
    
    # Simple Moving Average
    result['SMA'] = result['close'].rolling(window=period).mean()
    
    # Mean deviation
    mean_deviation = (result['close'] - result['SMA']).abs().rolling(window=period).mean()
    
    # CCI calculation with handling for division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_cci = (result['close'] - result['SMA']) / (0.015 * mean_deviation)
    
    result['CCI'] = np.where(np.isfinite(raw_cci), raw_cci, 0.0)
    
    return result
```

### Usage in Strategy
- CCI below the oversold threshold (default: -30) contributes to long signal generation
- CCI above the overbought threshold (default: 70) contributes to short signal generation

## Bollinger Bands

Bollinger Bands are volatility bands placed above and below a moving average. They expand when volatility increases and contract when volatility decreases.

### Components
- **Middle Band**: SMA of the price (typically 20-period)
- **Upper Band**: SMA + (Standard Deviation × Multiplier)
- **Lower Band**: SMA - (Standard Deviation × Multiplier)

### Implementation

```python
def add_bollinger_bands(df, period=20, std_multiplier=2.0):
    result = df.copy()
    
    # Simple Moving Average
    result['SMA'] = result['close'].rolling(window=period).mean()
    
    # Standard Deviation
    result['StdDev'] = result['close'].rolling(window=period).std()
    
    # Upper and lower bands
    result['BollingerUpper'] = result['SMA'] + (std_multiplier * result['StdDev'])
    result['BollingerLower'] = result['SMA'] - (std_multiplier * result['StdDev'])
    
    return result
```

### Usage in Strategy
- The alignment of Bollinger Bands with Keltner Channels is a key signal confirmation factor
- When Bollinger and Keltner bands are closely aligned, it increases signal reliability

## Indicator Combination

The power of the Mean Reversion Volatility Bands strategy comes from combining these indicators:

1. **Volatility Measurement**: ATR provides the base volatility measurement
2. **Envelope Comparison**: Bollinger Bands and Keltner Channels together identify potential mean reversion points
3. **Oscillator Confirmation**: CCI provides additional confirmation of overbought/oversold conditions

By requiring alignment between these different indicator types, the strategy aims to generate higher-quality signals with reduced false positives.
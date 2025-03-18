# Signal Generation

The Mean Reversion Volatility Bands strategy uses a combination of technical indicators to generate trading signals. The main component responsible for signal generation is the `SignalGenerator` class.

## Signal Logic 

The strategy generates two types of entry signals:

1. **Long Signals**: Generated when the price potentially indicates an oversold condition with a likely upward reversion
2. **Short Signals**: Generated when the price potentially indicates an overbought condition with a likely downward reversion

Both signal types indicate buying opportunities but with different directional expectations. The actual exit points (selling) are determined by the backtest strategy using take profit and stop loss levels based on the Average True Range (ATR).

## Signal Conditions

### Long Signal Conditions

A long signal is generated when all of the following conditions are met within the specified window:

1. **Price Below Keltner Channel Lower Band**: Indicates price is potentially oversold relative to the Keltner volatility measure
2. **CCI Below Threshold**: The Commodity Channel Index is below the oversold threshold (default: -30)
3. **Bands Aligned for Long Entry**: The Bollinger and Keltner lower bands are closely aligned (within a specified percentage)

### Short Signal Conditions

A short signal (for shorting the market) is generated when all of the following conditions are met within the specified window:

1. **Price Above Keltner Channel Upper Band**: Indicates price is potentially overbought relative to the Keltner volatility measure
2. **CCI Above Threshold**: The Commodity Channel Index is above the overbought threshold (default: 70)
3. **Bands Aligned for Short Entry**: The Bollinger and Keltner upper bands are closely aligned (within a specified percentage)

Note that "short" here refers to the position direction (selling with expectation to buy back lower), not to the exit/sell signal.

## Window-Based Confirmation

The signal generator uses a sliding window approach to confirm signals:

- The window size is configurable (default: 3 periods)
- Signal conditions must be met at least once within the window for a signal to be generated
- This helps filter out noise and reduce false signals

## Implementation Details

The `SignalGenerator` class implements the following methods:

### `check_data_validity()`

Ensures the data meets required conditions:

- Presence of all required technical indicator columns
- No missing values in critical data points
- Sufficient historical data for reliable signal generation

### `generate_signals()`

The main method that processes price and indicator data to produce trading signals:

1. Verifies data has sufficient length (minimum required rows)
2. Checks for presence of required columns
3. Creates a window of data for each analysis point
4. Evaluates long and short signal conditions
5. Records both the final signals and the intermediate condition results

## Code Example

```python
# Generate signals with default parameters
signal_generator = SignalGenerator()
signals_df = signal_generator.generate_signals(
    df=indicators_df,
    thresholds={
        "CCI_up_threshold": 70,
        "CCI_low_threshold": -30,
        "Bollinger_Keltner_alignment": 0.005,
        "window_size": 3
    },
    min_required_rows=21
)
```

## Signal Output

The signal generation process adds several columns to the input DataFrame:

- `LongSignal`: Boolean indicating a buy signal
- `ShortSignal`: Boolean indicating a sell signal
- Intermediate condition columns:
  - `price_below_keltner`: Price is below Keltner lower band
  - `cci_below_threshold`: CCI is below the oversold threshold
  - `bands_aligned_long`: Bollinger and Keltner lower bands are aligned
  - `price_above_keltner`: Price is above Keltner upper band
  - `cci_above_threshold`: CCI is above the overbought threshold
  - `bands_aligned_short`: Bollinger and Keltner upper bands are aligned

These intermediate columns are helpful for debugging and understanding why specific signals were or were not generated at certain points.
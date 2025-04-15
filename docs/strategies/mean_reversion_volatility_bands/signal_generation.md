# Signal Generation

The Mean Reversion Volatility Bands strategy uses a combination of technical indicators to generate trading signals. The main component responsible for signal generation is the `SignalGenerator` class.

## Signal Logic 

The strategy generates two types of entry signals:

1. **Long Signals**: Generated when the price potentially indicates an oversold condition with a likely upward reversion
2. **Short Signals**: Generated when the price potentially indicates an overbought condition with a likely downward reversion

Both signal types indicate trading opportunities with different directional expectations. The actual exit points are determined by the backtest strategy using take profit and stop loss levels based on the Average True Range (ATR).

## Signal Types

The `SignalGenerator` supports three types of signal generation:

- **'long'**: Generates only long entry signals
- **'short'**: Generates only short entry signals
- **'mix'** (default): Generates both long and short entry signals

## Signal Conditions

### Long Signal Conditions

A long signal is generated when all of the following conditions are met within the specified window:

1. **Price Below Keltner Channel Lower Band**: Indicates price is potentially oversold relative to the Keltner volatility measure
2. **CCI Below Threshold**: The Commodity Channel Index is below the oversold threshold
3. **Bands Aligned for Long Entry**: The Bollinger and Keltner lower bands are closely aligned (within a specified percentage)

### Short Signal Conditions

A short signal is generated when all of the following conditions are met within the specified window:

1. **Price Above Keltner Channel Upper Band**: Indicates price is potentially overbought relative to the Keltner volatility measure
2. **CCI Above Threshold**: The Commodity Channel Index is above the overbought threshold
3. **Bands Aligned for Short Entry**: The Bollinger and Keltner upper bands are closely aligned (within a specified percentage)

## Window-Based Confirmation

The signal generator uses a sliding window approach to confirm signals:

- The window size is configurable
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

```python
@staticmethod
def generate_signals(df: pd.DataFrame, CCI_up_threshold, CCI_low_threshold, 
                     Bollinger_Keltner_alignment, window_size, min_required_rows, 
                     signal_type='mix') -> pd.DataFrame:
```

Parameters:
- `df` (pd.DataFrame): DataFrame with price and indicator data
- `CCI_up_threshold`: Upper threshold for CCI indicator (typically positive, e.g., 70)
- `CCI_low_threshold`: Lower threshold for CCI indicator (typically negative, e.g., -30)
- `Bollinger_Keltner_alignment`: Alignment threshold for Bollinger and Keltner bands (e.g., 0.005)
- `window_size`: Window size for checking conditions (e.g., 3)
- `min_required_rows`: Minimum required rows for signal generation
- `signal_type` (str): Type of signals to generate - 'long', 'short', or 'mix' (default)

The method:
1. Validates the `signal_type` parameter
2. Verifies data has sufficient length (minimum required rows)
3. Checks for presence of required columns
4. Creates a window of data for each analysis point
5. Evaluates long and/or short signal conditions based on the `signal_type`
6. Records both the final signals and the intermediate condition results

## Error Handling

The `SignalGenerator` class includes comprehensive error handling:

- Logging of warning messages for invalid data or configuration
- Validation of input parameters
- Returning the original DataFrame instead of failing when conditions aren't met
- Safety checks to prevent division by zero when calculating band alignments

## Signal Output

The signal generation process adds several columns to the input DataFrame:

- `LongSignal`: Boolean indicating a long entry signal
- `ShortSignal`: Boolean indicating a short entry signal
- Intermediate condition columns:
  - `price_below_keltner`: Price is below Keltner lower band
  - `cci_below_threshold`: CCI is below the oversold threshold
  - `bands_aligned_long`: Bollinger and Keltner lower bands are aligned
  - `price_above_keltner`: Price is above Keltner upper band
  - `cci_above_threshold`: CCI is above the overbought threshold
  - `bands_aligned_short`: Bollinger and Keltner upper bands are aligned

These intermediate columns are helpful for debugging and understanding why specific signals were or were not generated at certain points.

## Code Example

```python
# Generate signals with explicit parameters
signals_df = SignalGenerator.generate_signals(
    df=indicators_df,
    CCI_up_threshold=70,
    CCI_low_threshold=-30,
    Bollinger_Keltner_alignment=0.005,
    window_size=3,
    min_required_rows=21,
    signal_type='mix'  # Can be 'long', 'short', or 'mix'
)
```

## Required Columns

The signal generator requires the following columns in the input DataFrame:
- `close`: Close price data
- `KeltnerLower`: Lower Keltner Channel band
- `KeltnerUpper`: Upper Keltner Channel band
- `CCI`: Commodity Channel Index values
- `BollingerLower`: Lower Bollinger Band
- `BollingerUpper`: Upper Bollinger Band
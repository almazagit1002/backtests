# Mean Reversion Volatility Bands Strategy

## Strategy Overview

The Mean Reversion Volatility Bands strategy identifies potential mean reversion opportunities by combining multiple technical indicators:

- **Bollinger Bands**: Measures price volatility using standard deviations
- **Keltner Channels**: Measures price volatility using Average True Range (ATR)
- **Commodity Channel Index (CCI)**: Identifies potential overbought/oversold conditions

The strategy enters positions when price action indicates a potential reversion to the mean, with additional confirmation from oscillator indicators.

## Strategy Logic

1. **Entry Signals**:
   - CCI moves below the oversold threshold (default: -30) for long positions
   - CCI moves above the overbought threshold (default: 70) for short positions
   - Price is near or beyond the Bollinger Bands
   - Bollinger Bands and Keltner Channels have proper alignment

2. **Exit Signals**:
   - Take profit: When price reaches the target profit level (default: 2 times entry distance)
   - Stop loss: When price moves against the position by a defined threshold (default: 1 times entry distance)
   - Time-based exit: After a specified maximum holding period

## Performance Characteristics

The strategy is designed to:

- Capitalize on short-term price overextensions
- Profit from price normalization after volatility events
- Maintain controlled risk through predefined stop loss and take profit levels
- Diversify risk across multiple positions (default: maximum 4 concurrent positions)

## Sample Results

The strategy has been tested on various assets with the following general observations:

- Higher win rate in range-bound market conditions
- Reduced effectiveness during strong trend phases
- Better performance on assets with well-defined volatility characteristics
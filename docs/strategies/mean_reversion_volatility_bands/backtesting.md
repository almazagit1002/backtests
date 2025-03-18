# Backtesting

The Mean Reversion Volatility Bands strategy includes a comprehensive backtesting framework implemented in the `TradingBacktester` class. This system allows for thorough evaluation of strategy performance across different market conditions and parameter settings.

## Backtester Features

The backtesting framework provides several key features:

- Support for both long and short positions
- Dynamic take profit and stop loss levels based on ATR
- Transaction cost modeling
- Position sizing and portfolio management
- Comprehensive performance metrics
- Parameter optimization capabilities

## Position Management

### Entry Logic

The backtester opens new positions when entry signals occur if:

1. There is sufficient budget available
2. The number of open positions is below the configured maximum

Position size is calculated by dividing the available budget by the maximum number of positions, ensuring risk diversification.

### Exit Logic

Each position has two primary exit conditions:

1. **Take Profit (TP)**: For long positions, TP is set at entry price + (tp_level × ATR). For short positions, TP is set at entry price - (tp_level × ATR).

2. **Stop Loss (SL)**: For long positions, SL is set at entry price - (sl_level × ATR). For short positions, SL is set at entry price + (sl_level × ATR).

The TP and SL levels are dynamic, adapting to the market's volatility at the time of entry through ATR.

## Implementation Details

```python
# Example of exit level calculation
take_profit = entry_price + (tp_level * current_atr)  # For long positions
stop_loss = entry_price - (sl_level * current_atr)    # For long positions

# For short positions
take_profit = entry_price - (tp_level * current_atr)
stop_loss = entry_price + (sl_level * current_atr)
```

## Performance Metrics

The backtester calculates a comprehensive set of performance metrics:

### Overall Performance

- **Total Return**: Percentage gain/loss over the backtest period
- **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value
- **Final Portfolio Value**: Value at the end of the backtest period

### Trade Analysis

- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Average Profit per Trade**: Mean profit/loss across all trades

### Position Type Analysis

The metrics are calculated separately for:

- **Overall**: All trades combined
- **Long Positions**: Trades opened with a long signal
- **Short Positions**: Trades opened with a short signal

This separation allows for analysis of how the strategy performs in different market conditions.

## Parameter Optimization

The backtesting framework includes an optimization feature that tests multiple combinations of parameters to find optimal settings:

```python
def optimize_parameters(self, signals_df, tp_levels=[1, 2, 3], sl_levels=[0.5, 1, 1.5]):
    # Runs backtests with all combinations of tp_levels and sl_levels
    # Returns results sorted by total return
```

The optimization process evaluates different combinations of:

- Take profit levels
- Stop loss levels

Results are sorted by total return, but other metrics like drawdown and profit factor can be considered for selecting the best parameter combination.

## Usage Example

```python
# Initialize backtester with strategy parameters
backtester = TradingBacktester(
    initial_budget=10000,
    tp_level=2.0,
    sl_level=1.0,
    fee_rate=0.005,
    max_positions=4
)

# Run backtest on signal data
backtester.backtest(signals_df)

# Print results
backtester.print_results()

# Optimize parameters
optimization_results = backtester.optimize_parameters(
    signals_df,
    tp_levels=[1.5, 2.0, 2.5, 3.0],
    sl_levels=[0.5, 1.0, 1.5]
)
```

## Output Data

The backtesting process generates two main DataFrames:

1. **Trades DataFrame**: Contains detailed information about each trade:
   - Entry and exit dates/prices
   - Position type (long/short)
   - Result (take profit, stop loss, end of period)
   - Profit/loss amount and percentage

2. **Portfolio DataFrame**: Tracks the portfolio value over time:
   - Portfolio value at each time step
   - Available budget
   - Number of active positions
   - Peak portfolio value (for drawdown calculation)

These outputs can be accessed via the `get_trades()` and `get_portfolio()` methods for further analysis or visualization.
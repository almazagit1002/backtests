# Backtesting

The Mean Reversion Volatility Bands strategy includes a comprehensive backtesting framework implemented in the `TradingBacktester` class. This system allows for thorough evaluation of strategy performance across different market conditions and parameter settings.

## Backtester Features

The backtesting framework provides several key features:

- Support for both long and short positions
- Dynamic take profit and stop loss levels based on ATR
- Detailed transaction cost modeling with separate entry/exit fee tracking
- Position sizing and portfolio management
- Comprehensive performance metrics with separate long/short analysis
- Detailed fee impact analysis
- Flexible reporting options by signal type (long, short, or mix)

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

### Fee Handling

The backtester tracks and analyzes trading fees with high detail:

- Entry fees (buying for long positions, selling for short positions)
- Exit fees (selling for long positions, buying for short positions)
- Fee impact on gross profit and overall performance
- Trades that would have been profitable without fees

Fees are calculated as a percentage of the trade value:
```python
# For long positions
entry_fee = shares * price * buy_fee_rate
exit_fee = (shares * exit_price) * sell_fee_rate

# For short positions
entry_fee = shares * price * sell_fee_rate
exit_fee = (shares * exit_price) * buy_fee_rate
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
- **Winning/Losing Trades**: Count of profitable and unprofitable trades

### Fee Analysis

- **Total Fees**: Sum of all trading fees (entry + exit)
- **Fee Percentage of Gross Profit**: How much of the gross profit was consumed by fees
- **Average Fee Per Trade**: Mean fee across all trades
- **Trades Lost Due to Fees**: Count of trades that would have been profitable without fees
- **Entry/Exit Fee Analysis**: Separate metrics for fees at entry and exit points

### Position Type Analysis

The metrics are calculated separately for:

- **Overall**: All trades combined
- **Long Positions**: Trades opened with a long signal
- **Short Positions**: Trades opened with a short signal

This separation allows for analysis of how the strategy performs in different market conditions.

## Reporting Options

The backtester provides flexible reporting options through the `print_results` method:

```python
backtester.print_results(
    signal_type='mix',  # Options: 'long', 'short', or 'mix'
    include_fee_analysis=True  # Whether to include fee impact details
)
```

- **Signal Type Filtering**: Focus analysis on long positions, short positions, or both
- **Fee Analysis Toggle**: Include or exclude detailed fee impact reporting
- **JSON Export**: Performance metrics are automatically exported to a JSON file

## Usage Example

```python
# Initialize backtester with strategy parameters
backtester = TradingBacktester(
    initial_budget=10000,
    tp_level=2.0,
    sl_level=1.0,
    fee_rate=0.005,  # 0.5% fee for both entry and exit
    max_positions=4
)

# Run backtest on signal data
backtester.backtest(signals_df)

# Print results (default: includes both long and short signals)
backtester.print_results()

# Print results for long signals only
backtester.print_results(signal_type='long')

# Get detailed fee analysis
fee_analysis = backtester.get_fee_analysis()
```

## Output Data

The backtesting process generates several data structures:

1. **Trades DataFrame**: Contains detailed information about each trade:
   - Entry and exit dates/prices
   - Position type (long/short)
   - Result (take profit, stop loss, end of period)
   - Gross profit/loss amount
   - Entry and exit fees
   - Net profit/loss amount and percentage
   - Fee impact metrics

2. **Portfolio DataFrame**: Tracks the portfolio value over time:
   - Portfolio value at each time step
   - Available budget
   - Number of active positions
   - Peak portfolio value (for drawdown calculation)

3. **Fee Analysis DataFrame**: Provides detailed statistics about fee impact:
   - Total fees and breakdown between entry/exit
   - Fee percentage of total trade value
   - Fee percentage of gross profit
   - Trades affected by fees

4. **Performance Metrics**: A dictionary with comprehensive statistics:
   - Overall performance metrics
   - Separate metrics for long and short positions
   - Return percentages and drawdown information

These outputs can be accessed via the following methods:
- `get_trades()`: Returns the trades DataFrame
- `get_portfolio()`: Returns the portfolio value history
- `get_metrics()`: Returns the performance metrics dictionary
- `get_fee_analysis()`: Returns the fee analysis DataFrame
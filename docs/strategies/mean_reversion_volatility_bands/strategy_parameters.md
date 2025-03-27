# Mean Reversion Volatility Bands Trading Strategy Parameters

This document provides a comprehensive explanation of the parameters used in the Mean Reversion Volatility Bands trading strategy. The strategy uses a combination of technical indicators, specifically Bollinger Bands, Keltner Channels, and CCI (Commodity Channel Index), to identify potential mean reversion opportunities in the market.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Data Management Parameters](#data-management-parameters)
3. [Execution Mode Parameters](#execution-mode-parameters)
4. [Backtesting Parameters](#backtesting-parameters)
5. [Optimization Parameters](#optimization-parameters)
6. [Trading Strategy Parameters](#trading-strategy-parameters)
   - [Entry and Exit Parameters](#entry-and-exit-parameters)
   - [Position Management Parameters](#position-management-parameters)
   - [Technical Indicator Parameters](#technical-indicator-parameters)
   - [Signal Generation Parameters](#signal-generation-parameters)
7. [Parameter Optimization](#parameter-optimization)

## Configuration File Structure

The configuration file uses YAML format to organize parameters in a hierarchical structure. The main sections include:

- Data paths for input and output
- Mode settings
- Backtesting parameters
- Optimization settings and bounds
- Strategy-specific parameters

## Data Management Parameters

```yaml
load_data_path: sol_price.csv

save_data_paths:
  prefix: mean_reversion_volatility_bands/data/
  optimization: optimization.csv
  optimization_analysis: optimization_analysis
  update_config: mean_reversion_volatility_bands/config.yaml
```

| Parameter | Description |
|-----------|-------------|
| `load_data_path` | The CSV file containing historical price data for the asset (SOL in this case) |
| `save_data_paths.prefix` | Base directory for all output files |
| `save_data_paths.optimization` | File to save optimization results |
| `save_data_paths.optimization_analysis` | Directory for optimization analysis outputs |
| `save_data_paths.update_config` | Path to save the optimized configuration |

These parameters define where the strategy reads historical price data from and where it saves various outputs. Price data is essential for backtesting and optimization.

## Execution Mode Parameters

```yaml
modes:
  analise_optimization_mode: True
```

| Parameter | Description |
|-----------|-------------|
| `modes.analise_optimization_mode` | Whether to run analysis on the optimization results |

This setting controls whether the framework runs the optimization analysis process after completing optimization. When enabled, it generates visualizations and metrics to understand how different parameters affect strategy performance.

## Backtesting Parameters

```yaml
backtest:
  initial_budget: 1000
  fee_rate: 0.005
```

| Parameter | Description | Impact on Strategy |
|-----------|-------------|-------------------|
| `backtest.initial_budget` | Starting capital for the backtest (1000 units) | Determines the scale of the backtest and influences position sizing |
| `backtest.fee_rate` | Trading fee as a percentage (0.5%) | Accounts for transaction costs that reduce profitability |

These parameters define the basic economic conditions for the backtest. The initial budget represents the starting capital, while the fee rate accounts for transaction costs when executing trades.

## Optimization Parameters

```yaml
optimization:
  init_points: 5
  n_iter: 10
```

| Parameter | Description | Impact on Optimization |
|-----------|-------------|------------------------|
| `optimization.init_points` | Number of initial random exploration points | Higher values provide more thorough initial exploration but increase runtime |
| `optimization.n_iter` | Number of Bayesian optimization iterations | Higher values provide more refined optimization but increase runtime |

These parameters control the Bayesian optimization process. The algorithm starts with `init_points` random parameter combinations to explore the parameter space, then runs `n_iter` iterations of guided optimization to find the best parameters.

## Trading Strategy Parameters

The trading strategy uses multiple parameters that control different aspects of its behavior. These parameters are the ones being optimized.

### Entry and Exit Parameters

```yaml
pbounds:
  tp_level: [1.0, 3.5]
  sl_level: [0.5, 2.5]
```

| Parameter | Description | Impact on Strategy |
|-----------|-------------|-------------------|
| `tp_level` | Take profit level as percentage of entry price | Higher values lead to larger profit targets but potentially fewer winning trades |
| `sl_level` | Stop loss level as percentage of entry price | Higher values reduce the number of stopped-out trades but increase potential losses per trade |

These parameters define when the strategy exits positions:

- **Take Profit (TP)**: When price moves favorably by the specified percentage, the position is closed for a profit. For example, a `tp_level` of 2.0 means the strategy will exit when price moves 2% in the favorable direction.
- **Stop Loss (SL)**: When price moves unfavorably by the specified percentage, the position is closed to limit losses. For example, a `sl_level` of 1.0 means the strategy will exit when price moves 1% against the position.

In mean reversion strategies, these levels are typically asymmetric, with tighter profit targets than stop losses, based on the expectation that prices revert to their mean more often than they continue trending.

### Position Management Parameters

```yaml
pbounds:
  max_positions: [1, 10]
```

| Parameter | Description | Impact on Strategy |
|-----------|-------------|-------------------|
| `max_positions` | Maximum number of concurrent positions | Higher values increase diversification but may reduce allocation per trade |

This parameter limits how many open positions the strategy can have simultaneously:

- Lower values (1-3) focus capital on the highest-conviction trades
- Higher values (4+) spread risk across more trades but may diminish returns if capital is limited

### Technical Indicator Parameters

```yaml
pbounds:
  keltner_period: [15, 30]
  atr_multiplier: [1.2, 3.5]
  bollinger_period: [10, 50]
  std_multiplier: [1.2, 3.0]
  cci_period: [14, 50]
```

| Parameter | Description | Impact on Strategy |
|-----------|-------------|-------------------|
| `keltner_period` | Lookback period for Keltner Channels | Shorter periods are more responsive to recent price action; longer periods capture longer-term trends |
| `atr_multiplier` | Multiplier for Average True Range in Keltner Channels | Higher values create wider channels, reducing false signals but potentially missing opportunities |
| `bollinger_period` | Lookback period for Bollinger Bands | Shorter periods create more signals but increase false positives; longer periods are more stable |
| `std_multiplier` | Standard deviation multiplier for Bollinger Bands | Higher values create wider bands, reducing false signals but potentially missing opportunities |
| `cci_period` | Lookback period for Commodity Channel Index | Shorter periods create more responsive CCI; longer periods create more stable readings |

These parameters control how the technical indicators are calculated:

- **Keltner Channels**: A volatility-based indicator that creates bands around an exponential moving average. The bands are `atr_multiplier` times the Average True Range (ATR) away from the central line. The `keltner_period` determines how many bars are used to calculate both the EMA and the ATR.

- **Bollinger Bands**: A volatility-based indicator that creates bands around a simple moving average. The bands are `std_multiplier` standard deviations away from the central line. The `bollinger_period` determines how many bars are used to calculate both the moving average and the standard deviation.

- **Commodity Channel Index (CCI)**: An oscillator that measures the current price level relative to an average price level over a given period. The `cci_period` determines how many bars are used to calculate this average.

### Signal Generation Parameters

```yaml
pbounds:
  CCI_up_threshold: [50, 110]
  CCI_low_threshold: [-110, -10]
  Bollinger_Keltner_alignment: [0.001, 0.01]
  window_size: [3, 10]
```

| Parameter | Description | Impact on Strategy |
|-----------|-------------|-------------------|
| `CCI_up_threshold` | Upper threshold for CCI indicator | Higher values require stronger overbought conditions before generating sell signals |
| `CCI_low_threshold` | Lower threshold for CCI indicator | Lower (more negative) values require stronger oversold conditions before generating buy signals |
| `Bollinger_Keltner_alignment` | Alignment threshold between Bollinger Bands and Keltner Channels | Smaller values require more precise alignment of the bands, reducing signals but increasing quality |
| `window_size` | Signal smoothing window size | Larger values create more stable signals but may introduce lag |

These parameters control how trading signals are generated:

- **CCI Thresholds**: The CCI is an oscillator that indicates when a market is potentially overbought (high positive values) or oversold (high negative values). The `CCI_up_threshold` and `CCI_low_threshold` define when the indicator suggests overbought or oversold conditions, respectively.

- **Bollinger-Keltner Alignment**: This parameter measures how closely the Bollinger Bands need to align with the Keltner Channels to generate a signal. When Bollinger Bands move outside Keltner Channels, it often indicates a potential mean reversion opportunity. The alignment threshold controls how significant this divergence needs to be.

- **Window Size**: This parameter controls signal smoothing by looking at multiple periods to confirm a signal. Larger window sizes reduce false signals but may delay entries and exits.

## Parameter Optimization

The `pbounds` section defines the search space for the Bayesian optimization process:

```yaml
pbounds:
  tp_level: [1.0, 3.5]
  sl_level: [0.5, 2.5]
  max_positions: [1, 10]
  atr_multiplier: [1.2, 3.5]
  keltner_period: [15,30]
  cci_period: [14, 50]
  bollinger_period: [10, 50]
  std_multiplier: [1.2, 3.0]
  CCI_up_threshold: [50, 110]
  CCI_low_threshold: [-110, -10]
  Bollinger_Keltner_alignment: [0.001, 0.01]
  window_size: [3, 10]
```

Each parameter is given a lower and upper bound for the optimization to explore. The Bayesian optimization algorithm will:

1. Start with `init_points` random combinations within these bounds
2. Use the performance of these initial points to build a probabilistic model
3. Run `n_iter` iterations, each time choosing parameter combinations that are likely to improve performance
4. Return the best parameter combination found during the process

The optimization seeks to maximize the total return percentage of the strategy during backtesting.

## Strategy Logic Overview

The Mean Reversion Volatility Bands strategy operates on the principle that prices tend to revert to their mean after significant deviations. The strategy uses three key technical indicators:

1. **Bollinger Bands**: Identifies when prices are statistically high or low
2. **Keltner Channels**: Provides volatility-based bands that typically encompass most price action
3. **CCI (Commodity Channel Index)**: Identifies overbought and oversold conditions

### Entry Conditions:

- **Buy Signal**: When:
  - Price is below the lower Bollinger Band
  - Bollinger Band is below the lower Keltner Channel (with alignment threshold)
  - CCI is below the `CCI_low_threshold` (oversold)
  - These conditions persist for at least `window_size` periods

- **Sell Signal**: When:
  - Price is above the upper Bollinger Band
  - Bollinger Band is above the upper Keltner Channel (with alignment threshold)
  - CCI is above the `CCI_up_threshold` (overbought)
  - These conditions persist for at least `window_size` periods

### Exit Conditions:

- **Take Profit**: When price moves favorably by `tp_level` percentage
- **Stop Loss**: When price moves unfavorably by `sl_level` percentage

The strategy manages risk by limiting the number of concurrent positions to `max_positions` and sizing each position based on the available capital.

By optimizing these parameters, the strategy can be fine-tuned to specific market conditions and trading instruments, potentially improving its performance and robustness.
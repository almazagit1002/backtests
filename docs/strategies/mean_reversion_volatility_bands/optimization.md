# StrategyOptimizer

The `StrategyOptimizer` class provides a framework for optimizing trading strategy parameters using Bayesian Optimization. It systematically searches for the optimal combination of parameters to maximize trading strategy performance.

## Overview

Trading strategies often involve multiple parameters that need fine-tuning to achieve optimal performance. This class utilizes Bayesian Optimization, a powerful technique for efficiently searching complex parameter spaces, to find the best combination of parameters for mean reversion strategies using volatility bands.

The optimizer integrates with trading indicators, signal generation, and backtesting components to evaluate parameter combinations and identify those yielding the highest returns.

## Installation

The `StrategyOptimizer` class relies on several Python packages:

```python
from bayes_opt import BayesianOptimization
import pandas as pd
import logging
```

It also requires the following modules from the mean reversion volatility bands package:

```python
from mean_reversion_volatility_bands.back_test import TradingBacktester
from mean_reversion_volatility_bands.trade_indicators import TradingIndicators
from mean_reversion_volatility_bands.signal_generator import SignalGenerator
```

Ensure these packages and modules are installed in your environment before using this class.

## Basic Usage

```python
# Import required libraries
import pandas as pd
import logging
from strategy_optimizer import StrategyOptimizer

# Load your price data
price_data = pd.read_csv('price_data.csv')

# Define optimization parameters
initial_budget = 10000  # Starting capital
fee_rate = 0.001        # Trading fee (0.1%)
init_points = 5         # Initial random exploration points
n_iter = 20             # Number of Bayesian optimization iterations

# Initialize the optimizer
optimizer = StrategyOptimizer(
    initial_budget=initial_budget,
    fee_rate=fee_rate,
    init_points=init_points,
    n_iter=n_iter,
    df=price_data,
    log_level=logging.INFO
)

# Define parameter bounds for optimization
pbounds = {
    'tp_level': (1.0, 3.0),            # Take profit level
    'sl_level': (0.5, 2.0),            # Stop loss level
    'max_positions': (1, 10),          # Maximum concurrent positions
    'atr_multiplier': (1.5, 3.5),      # ATR multiplier for Keltner Channels
    'keltner_period': (15, 30),        # Keltner Channels period
    'cci_period': (14, 50),            # CCI indicator period
    'bollinger_period': (10, 50),      # Bollinger Bands period
    'std_multiplier': (1.5, 3.0),      # Standard deviation multiplier for Bollinger Bands
    'CCI_up_threshold': (50, 100),     # CCI upper threshold
    'CCI_low_threshold': (-100, -10),  # CCI lower threshold
    'Bollinger_Keltner_alignment': (0.001, 0.01),  # Alignment threshold
    'window_size': (3, 7)              # Signal smoothing window size
}

# Run the optimization
results_df = optimizer.optimize(pbounds)

# Extract the best parameters
best_params = optimizer.get_best_parameters(results_df)
print("Best Parameters:", best_params)
```

## Key Features

- **Bayesian Optimization**: Efficiently explores parameter space to find optimal settings
- **Comprehensive Parameter Tuning**: Optimizes multiple trading parameters simultaneously
- **Integration**: Works with indicator calculation, signal generation, and backtesting components
- **Logging**: Detailed logging at multiple levels for debugging and progress tracking
- **Result Storage**: Tracks all optimization trials for later analysis
- **Performance Metrics**: Evaluates strategy performance using total return percentage

## Constructor

### `__init__(initial_budget, fee_rate, init_points, n_iter, df, log_level=logging.INFO)`

Initializes the StrategyOptimizer with fixed strategy parameters.

**Parameters:**

- `initial_budget` (float): Initial trading budget
- `fee_rate` (float): Trading fee as a percentage (0.005 = 0.5%)
- `init_points` (int): Number of initial random points for Bayesian optimization
- `n_iter` (int): Number of iterations for Bayesian optimization
- `df` (pandas.DataFrame): DataFrame containing price data
- `log_level` (int, optional): Level of logging detail (default: logging.INFO)

**Example:**

```python
optimizer = StrategyOptimizer(
    initial_budget=10000,
    fee_rate=0.001,
    init_points=5,
    n_iter=20,
    df=price_data,
    log_level=logging.DEBUG
)
```

## Core Methods

### `optimize(pbounds)`

Performs Bayesian Optimization to find the best trading strategy parameters.

**Parameters:**

- `pbounds` (dict): Dictionary defining the parameter bounds for optimization

**Returns:**

- `pandas.DataFrame`: A DataFrame containing all optimization results

**Example:**

```python
pbounds = {
    'tp_level': (1.0, 3.0),
    'sl_level': (0.5, 2.0),
    # Other parameters...
}

results_df = optimizer.optimize(pbounds)
```

### `get_best_parameters(optimization_result)`

Extracts the best parameters from the optimization result.

**Parameters:**

- `optimization_result` (pandas.DataFrame): DataFrame containing optimization results

**Returns:**

- `pandas.Series`: The row with the highest total return percentage

**Example:**

```python
best_params = optimizer.get_best_parameters(results_df)
print(f"Best parameters: {best_params}")
```

## Internal Methods

### `_backtest_objective(tp_level, sl_level, max_positions, atr_multiplier, keltner_period, cci_period, bollinger_period, std_multiplier, CCI_up_threshold, CCI_low_threshold, Bollinger_Keltner_alignment, window_size)`

Objective function for Bayesian optimization. This internal method:
1. Applies trading indicators to the price data
2. Generates trading signals based on indicators
3. Runs a backtest with the specified parameters
4. Returns the total return percentage

**Parameters:**

- `tp_level` (float): Take profit level as a percentage of entry price
- `sl_level` (float): Stop loss level as a percentage of entry price
- `max_positions` (float): Maximum number of concurrent positions (rounded to int)
- `atr_multiplier` (float): Multiplier for Average True Range (Keltner Channels)
- `keltner_period` (float): Period for Keltner Channels calculation (rounded to int)
- `cci_period` (float): Period for Commodity Channel Index calculation (rounded to int)
- `bollinger_period` (float): Period for Bollinger Bands calculation (rounded to int)
- `std_multiplier` (float): Standard deviation multiplier for Bollinger Bands
- `CCI_up_threshold` (float): Upper threshold for CCI indicator (rounded to int)
- `CCI_low_threshold` (float): Lower threshold for CCI indicator (rounded to int)
- `Bollinger_Keltner_alignment` (float): Alignment threshold between Bollinger Bands and Keltner Channels
- `window_size` (float): Window size for signal smoothing (rounded to int)

**Returns:**

- `float`: Total return percentage from backtest

## Optimization Parameters

The StrategyOptimizer can optimize the following parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| tp_level | Take profit level as percentage | 1.0 to 3.0 |
| sl_level | Stop loss level as percentage | 0.5 to 2.0 |
| max_positions | Maximum concurrent positions | 1 to 10 |
| atr_multiplier | Multiplier for ATR in Keltner Channels | 1.5 to 3.5 |
| keltner_period | Period for Keltner Channels | 15 to 30 |
| cci_period | Period for CCI calculation | 14 to 50 |
| bollinger_period | Period for Bollinger Bands | 10 to 50 |
| std_multiplier | Standard deviation multiplier for Bollinger Bands | 1.5 to 3.0 |
| CCI_up_threshold | Upper threshold for CCI indicator | 50 to 100 |
| CCI_low_threshold | Lower threshold for CCI indicator | -100 to -10 |
| Bollinger_Keltner_alignment | Alignment threshold | 0.001 to 0.01 |
| window_size | Signal smoothing window size | 3 to 7 |

## Error Handling

The `StrategyOptimizer` class includes comprehensive error handling:

- Errors during indicator calculation are caught and logged
- Optimization continues even if individual iterations fail
- Failed iterations return a poor score (-100.0) to guide the optimizer away from problematic parameter regions
- Alternative methods are attempted when extracting best parameters if the primary method fails

## Logging

The class includes detailed logging at multiple levels:

- ERROR: Critical issues that prevent optimization
- INFO: Progress updates on optimization process
- DEBUG: Detailed information about parameter values, DataFrame shapes, and intermediate steps

## Advanced Usage

### Custom Parameter Bounds

You can customize the parameter bounds based on your trading strategy requirements:

```python
# Example of custom parameter bounds for a faster-moving strategy
custom_bounds = {
    'tp_level': (0.5, 1.5),         # Lower take profit levels
    'sl_level': (0.3, 1.0),         # Lower stop loss levels
    'max_positions': (5, 20),       # More concurrent positions
    'cci_period': (5, 20),          # Shorter CCI periods
    'bollinger_period': (5, 20),    # Shorter Bollinger periods
    # Other parameters...
}

results_df = optimizer.optimize(custom_bounds)
```

### High-Resolution Optimization

For more precise results, you can increase the number of iterations:

```python
# Initialize with more iterations for higher resolution
optimizer = StrategyOptimizer(
    initial_budget=10000,
    fee_rate=0.001,
    init_points=10,  # More initial exploration
    n_iter=50,       # More Bayesian optimization iterations
    df=price_data
)
```

### Analyzing Optimization Results

You can use the optimization results dataframe for further analysis:

```python
# Get optimization results
results_df = optimizer.optimize(pbounds)

# Analyze parameter correlations with returns
correlation_with_returns = results_df.corr()['total_return_pct'].sort_values(ascending=False)
print("Parameter correlations with returns:", correlation_with_returns)

# Visualize parameter distributions for top performers
top_performers = results_df.sort_values('total_return_pct', ascending=False).head(10)
print("Top performing parameter sets:", top_performers)
```

## Notes

- The class performs integer rounding for parameters that require integer values
- Failed optimization trials are assigned a return of -100.0% to guide the optimizer away from those regions
- The optimization results are stored both as a DataFrame and in an internal list for redundancy
- Price data should contain Open, High, Low, Close prices for indicator calculation

## Integration with Other Components

The StrategyOptimizer integrates with several other components:

1. **TradingIndicators**: Calculates technical indicators like Keltner Channels, CCI, and Bollinger Bands
2. **SignalGenerator**: Creates trading signals based on indicator values
3. **TradingBacktester**: Evaluates strategy performance with the given parameters

These components must be available in the `mean_reversion_volatility_bands` package for the optimizer to function correctly.
# BacktesterOptimization

The `BacktesterOptimization` class provides a high-level workflow for optimizing trading strategy parameters and analyzing optimization results. It acts as a coordinator between the optimization process and results analysis.

## Overview

Trading strategy optimization involves finding the best combination of parameters that maximize trading returns. The `BacktesterOptimization` class orchestrates this process by:

1. Loading historical price data
2. Running the optimization process using the `StrategyOptimizer`
3. Analyzing optimization results with `OptimizationAnalyzer`
4. Saving optimized parameters for future use

This class is designed to make the optimization workflow simple and automated, requiring minimal user intervention once configured properly.

## Installation

The `BacktesterOptimization` class relies on several Python packages and modules:

```python
import logging
import traceback
import pandas as pd

from mean_reversion_volatility_bands.optimization import StrategyOptimizer
from mean_reversion_volatility_bands.optimization_analysis import OptimizationAnalyzer
from utils.utils import load_config, save_config_yaml
```

Ensure these packages and modules are installed in your environment before using this class.

## Basic Usage

```python
# Import required modules
from backtester_optimization import BacktesterOptimization
from utils.utils import load_config

# Path to your configuration file
CONFIG_PATH = 'mean_reversion_volatility_bands/optimization_config.yaml'

# Load configuration
config = load_config(CONFIG_PATH)

# Initialize the backtester optimizer
backtest_optimizer = BacktesterOptimization(config)

# Enable optimization analysis
backtest_optimizer.set_optimization_analysis_mode(True)

# Run the full optimization workflow
backtest_optimizer.run_optimization_backtest()
```

## Configuration File

The `BacktesterOptimization` class relies on a YAML configuration file with the following structure:

```yaml
# Sample optimization_config.yaml
load_data_path: "data/historical_prices.csv"

backtest:
  initial_budget: 10000
  fee_rate: 0.001

optimization:
  init_points: 5
  n_iter: 20

pbounds:
  tp_level: [1.0, 3.0]
  sl_level: [0.5, 2.0]
  max_positions: [1, 10]
  atr_multiplier: [1.5, 3.5]
  keltner_period: [15, 30]
  cci_period: [14, 50]
  bollinger_period: [10, 50]
  std_multiplier: [1.5, 3.0]
  CCI_up_threshold: [50, 100]
  CCI_low_threshold: [-100, -10]
  Bollinger_Keltner_alignment: [0.001, 0.01]
  window_size: [3, 7]

save_data_paths:
  prefix: "output/"
  optimization_analysis: "optimization_analysis"
  update_config: "optimized_config.yaml"

modes:
  analise_optimization_mode: true
```

## Key Features

- **Workflow Automation**: Coordinates the entire optimization process from data loading to result analysis
- **Configuration-Driven**: Uses a YAML configuration file for flexibility and reproducibility
- **Analysis Integration**: Seamlessly connects optimization with result analysis
- **Parameter Persistence**: Saves optimized parameters to a new configuration file
- **Logging**: Comprehensive logging to track progress and diagnose issues

## Constructor

### `__init__(config)`

Initializes the BacktesterOptimization with a configuration dictionary.

**Parameters:**

- `config` (dict): Configuration dictionary loaded from a YAML file

**Example:**

```python
config = load_config('optimization_config.yaml')
backtest_optimizer = BacktesterOptimization(config)
```

## Core Methods

### `set_optimization_analysis_mode(enabled)`

Enables or disables the optimization analysis step.

**Parameters:**

- `enabled` (bool): Whether to enable optimization analysis

**Returns:**

- `self`: Returns self for method chaining

**Example:**

```python
backtest_optimizer.set_optimization_analysis_mode(True)
```

### `load_data(file_path)`

Loads historical price data from a CSV file.

**Parameters:**

- `file_path` (str): Path to the CSV file containing historical data

**Returns:**

- `self`: Returns self for method chaining

**Example:**

```python
backtest_optimizer.load_data('data/historical_prices.csv')
```

### `run_optimiziation()`

Runs the optimization process using the StrategyOptimizer and saves the best parameters.

**Returns:**

- `self`: Returns self for method chaining

**Example:**

```python
backtest_optimizer.run_optimiziation()
```

### `analyise_optimization()`

Analyzes the optimization results using the OptimizationAnalyzer.

**Example:**

```python
backtest_optimizer.analyise_optimization()
```

### `run_optimization_backtest()`

Executes the complete backtesting workflow from data loading to visualization.

**Returns:**

- `self`: Returns self for method chaining

**Example:**

```python
backtest_optimizer.run_optimization_backtest()
```

## Workflow Process

The `BacktesterOptimization` class follows these steps when running the complete workflow:

1. **Data Loading**: Loads historical price data from the path specified in the configuration
2. **Optimization**: Creates a `StrategyOptimizer` instance and runs the optimization process
3. **Parameter Extraction**: Extracts the best parameters from the optimization results
4. **Configuration Update**: Saves the optimized parameters to a new configuration file
5. **Analysis** (if enabled): Analyzes the optimization results using `OptimizationAnalyzer`

## Error Handling

The class implements error handling to ensure that exceptions are properly caught and logged:

- Data loading errors are caught and logged
- Optimization errors are caught with full traceback for debugging
- Each step can continue independently if previous steps fail (when possible)

## Logging

The class uses Python's logging module to provide informative messages about the optimization process:

- INFO level: Progress updates and successful operations
- ERROR level: Errors during data loading or optimization
- WARNING level: Reduced log noise from third-party libraries (PIL, matplotlib)

## Advanced Usage

### Running Only Optimization Without Analysis

```python
# Initialize with configuration
backtest_optimizer = BacktesterOptimization(config)

# Disable analysis
backtest_optimizer.set_optimization_analysis_mode(False)

# Run only the optimization part
backtest_optimizer.load_data(data_path)
backtest_optimizer.run_optimiziation()
```

### Custom Analysis of Optimization Results

```python
# Initialize with configuration
backtest_optimizer = BacktesterOptimization(config)

# Run the optimization
backtest_optimizer.run_optimization_backtest()

# Access the optimization results for custom analysis
optimization_results = backtest_optimizer.optimization_results

# Custom analysis
import pandas as pd
# Filter for only positive returns
positive_returns = optimization_results[optimization_results['total_return_pct'] > 0]
print(f"Percentage of positive returns: {len(positive_returns) / len(optimization_results) * 100:.2f}%")
```

## Integration with Other Components

The BacktesterOptimization class integrates with:

1. **StrategyOptimizer**: Performs the actual optimization process
2. **OptimizationAnalyzer**: Analyzes and visualizes optimization results
3. **Configuration Utilities**: Loads and saves configuration files

## Command Line Usage

The class can be run directly from the command line:

```bash
python -m mean_reversion_volatility_bands.backtester_optimization
```

This will:
1. Load the configuration from the default path
2. Set up the backtester optimizer
3. Run the complete optimization workflow

## Notes

- The configuration file must be properly formatted for the class to work correctly
- The `modes` section in the configuration controls whether optimization analysis is performed
- The `pbounds` section defines the parameter bounds for optimization
- The class saves optimized parameters to the path specified in `save_data_paths.update_config`
- Analysis results are saved to the directory specified in `save_data_paths.prefix` + `save_data_paths.optimization_analysis`
# Main Implementation

The main implementation of the Mean Reversion Volatility Bands strategy is provided by the `Backtester` class, which serves as the central orchestrator for the entire backtesting workflow. This class integrates all components of the strategy into a cohesive system.

## Backtester Class Overview

The `Backtester` class handles the entire backtesting process from start to finish, including:

- Loading historical price data
- Calculating technical indicators
- Generating trading signals
- Running backtest simulations
- Visualizing and exporting results

## Execution Flow

The backtesting process follows a logical sequence of steps:

1. **Initialization**: Set up the backtester with configuration parameters
2. **Data Loading**: Import historical price data from CSV files
3. **Indicator Calculation**: Compute technical indicators (Bollinger Bands, Keltner Channels, CCI)
4. **Signal Generation**: Identify trading signals based on indicator values
5. **Backtesting**: Simulate trades and track portfolio performance
6. **Visualization**: Generate charts for analysis
7. **Results Export**: Save data and performance metrics to files

## Configuration Modes

The backtester supports several operational modes that can be enabled or disabled:

```python
backtester.set_plot_mode(True)       # Enable visualization generation
backtester.set_csv_export(True)      # Enable CSV export of results
backtester.set_debug_mode(True)      # Enable detailed logging and diagnostics
backtester.set_optimize_mode(True)   # Enable parameter optimization
```

These modes allow for flexible execution depending on the user's needs.

## Complete Workflow Method

For convenience, the `run_complete_backtest()` method executes the entire workflow in sequence:

```python
backtester.run_complete_backtest()
```

This method chains together all the individual steps in the correct order, providing a streamlined way to execute the complete backtesting process.

## Usage Example

Here's an example of how to use the backtester from scratch:

```python
from mean_reversion_volatility_bands.back_test import Backtester
from utils.utils import load_config

# Load configuration from YAML file
config_path = 'mean_reversion_volatility_bands/config.yaml'
config = load_config(config_path)

# Configure operation modes
config_modes = config['modes']

# Initialize backtester
backtester = Backtester(config)
backtester.set_plot_mode(config_modes['plot_mode'])
backtester.set_csv_export(config_modes['csv_export'])
backtester.set_debug_mode(config_modes['debug_mode'])
backtester.set_optimize_mode(config_modes['optimize_mode'])

# Run complete backtesting workflow
backtester.run_complete_backtest()
```

## Individual Steps

For more granular control, each step can be executed separately:

```python
# Initialize and configure
backtester = Backtester(config)
backtester.set_debug_mode(True)

# Step 1: Load data
backtester.load_data('path/to/data.csv')

# Step 2: Calculate technical indicators
backtester.prepare_indicators()

# Step 3: Generate trading signals
backtester.generate_signals()

# Step 4: Run backtest simulation
backtester.run_backtest()

# Step 5: Visualize results
figures = backtester.visualize_results()

# Step 6: Export results to CSV
backtester.export_results()
```

This approach allows for inspection and validation between steps, which can be useful for debugging or detailed analysis.

## Integration with Components

The `Backtester` integrates with all other components of the framework:

- **TradingIndicators**: Calculates technical indicators 
- **SignalGenerator**: Identifies entry signals
- **TradingBacktester**: Simulates trades and tracks performance
- **TradingVisualizer**: Creates charts and visualizations

This modular design allows for easy maintenance and extension of the system.

## Debugging and Diagnostics

The backtester includes built-in diagnostics through the `inspect_indicators()` method, which provides detailed information about the calculated indicators:

```python
backtester.inspect_indicators()
```

This helps identify issues such as missing values or unexpected results in the indicator calculations.

## Extending the Backtester

To add new functionality to the backtester:

1. Add new configuration parameters to the YAML file
2. Create methods in the `Backtester` class to handle the new functionality
3. Integrate the new methods into the workflow

This extensible design makes it straightforward to add new features or modify existing behavior.
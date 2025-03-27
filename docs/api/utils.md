# Utilities

The backtesting framework includes utility functions that handle common tasks such as configuration management. These utilities help maintain clean, modular code and provide consistent behavior across the project.

## Configuration Management

The `utils` module provides functions for loading and saving configuration files.

### Loading Configuration

The `load_config()` function loads configuration settings from YAML files:

```python
def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
```

#### Example Usage

```python
from utils.utils import load_config

# Load strategy configuration
config = load_config("config.yaml")

# Access specific configuration sections
signal_params = config["signals"]
indicator_params = config["indicators"]
backtest_params = config["backtest"]
```

#### Error Handling

The function includes robust error handling:

- If the file doesn't exist, it logs a warning and continues with default settings
- If there's an error parsing the YAML, it logs the error details
- All exceptions are caught and logged to prevent application crashes

### Saving Optimized Configuration

The `save_config_yaml()` function updates specific sections of a YAML configuration file with optimized parameters:

```python
def save_config_yaml(df, config_path):
    """
    Updates specific sections of a YAML configuration file with optimized parameters
    while preserving all other sections and values.
    
    Parameters:
    -----------
    df : pd.Series or pd.DataFrame
        DataFrame or Series containing the optimization parameters
    config_path : str
        Path to the YAML file to update
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
```

#### Example Usage

```python
from utils.utils import save_config_yaml
import pandas as pd

# Optimization results
best_params = pd.Series({
    'tp_level': 2.5,
    'sl_level': 1.2,
    'max_positions': 5,
    'cci_period': 20,
    'bollinger_period': 18,
    'std_multiplier': 2.1,
    'CCI_up_threshold': 75,
    'CCI_low_threshold': -80,
    'Bollinger_Keltner_alignment': 0.005,
    'window_size': 4
})

# Save optimized parameters
success = save_config_yaml(best_params, "optimized_config.yaml")
```

#### Parameter Mapping

The function intelligently maps parameters to the appropriate configuration sections:

**Signals Section:**
- `CCI_up_threshold`
- `CCI_low_threshold`
- `Bollinger_Keltner_alignment`
- `window_size`

**Indicators Section:**
- `cci_period`
- `bollinger_period`
- `std_multiplier`

**Backtest Section:**
- `max_positions`
- `tp_level`
- `sl_level`

#### Error Handling

The function includes comprehensive error handling:

- Checks for empty DataFrames before processing
- Creates new configuration files if none exist
- Initializes missing sections if needed
- Preserves all other configuration settings not being updated
- Provides detailed error messages for troubleshooting

## Configuration Structure

The configuration system supports a hierarchical structure, allowing different components of the system to access their relevant settings. The standard configuration includes:

- Signal generation parameters
- Technical indicator parameters
- Backtesting parameters
- Optimization settings
- File paths for inputs and outputs
- Execution mode flags

### Example Configuration File

```yaml
# Input/output paths
load_data_path: "data/historical_prices.csv"
save_data_paths:
  prefix: "output/"
  optimization_analysis: "optimization_analysis"
  update_config: "optimized_config.yaml"

# Signal generation parameters
signals:
  CCI_up_threshold: 75
  CCI_low_threshold: -80
  Bollinger_Keltner_alignment: 0.005
  window_size: 4

# Technical indicator parameters
indicators:
  cci_period: 20
  bollinger_period: 18
  std_multiplier: 2.1
  keltner_period: 20
  atr_multiplier: 2.5

# Backtesting parameters
backtest:
  initial_budget: 10000
  fee_rate: 0.001
  max_positions: 5
  tp_level: 2.5
  sl_level: 1.2

# Optimization settings
optimization:
  init_points: 5
  n_iter: 20

# Parameter bounds for optimization
pbounds:
  tp_level: [1.0, 3.0]
  sl_level: [0.5, 2.0]
  max_positions: [1, 10]
  # Additional parameter bounds...

# Execution modes
modes:
  analise_optimization_mode: true
```

## Best Practices

When working with the configuration utilities:

1. **Keep configuration separate from code**: Store all customizable parameters in the YAML files
2. **Use default values**: Always provide sensible defaults in case configuration loading fails
3. **Validate configuration**: After loading, verify that required parameters exist
4. **Document configuration options**: Keep the configuration documentation up to date
5. **Type conversion**: Remember that values from YAML need appropriate type conversion (the `save_config_yaml` function handles this automatically)
6. **Preserve structure**: Use the utility functions to maintain the hierarchical structure of your configuration

## Integration with Optimization Workflow

The configuration utilities integrate seamlessly with the optimization workflow:

1. **Initial Configuration**: Load the initial configuration with parameter bounds
2. **Optimization Process**: Run the optimization using the bounds
3. **Parameter Extraction**: Extract the best parameters from optimization results
4. **Configuration Update**: Save the optimized parameters back to a configuration file
5. **Production Use**: Use the optimized configuration for production trading

This workflow allows for iterative improvement of strategy parameters while maintaining a clean separation between code and configuration.

By centralizing configuration management through these utilities, the backtesting framework maintains flexibility while ensuring consistent behavior across development, optimization, and production environments.
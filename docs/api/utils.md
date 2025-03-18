# Utilities

The backtesting framework includes utility functions that handle common tasks such as configuration management. These utilities help maintain clean, modular code and provide consistent behavior across the project.

## Configuration Management

The `utils` module provides functions for loading and handling configuration files.

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

#### Configuration Structure

The configuration system supports a hierarchical structure, allowing different components of the system to access their relevant settings. The standard configuration includes:

- Signal generation parameters
- Technical indicator parameters
- Backtesting parameters
- Optimization settings
- File paths for inputs and outputs
- Execution mode flags

## Best Practices

When working with the configuration utilities:

1. **Keep configuration separate from code**: Store all customizable parameters in the YAML files
2. **Use default values**: Always provide sensible defaults in case configuration loading fails
3. **Validate configuration**: After loading, verify that required parameters exist
4. **Document configuration options**: Keep the configuration documentation up to date

By centralizing configuration management through these utilities, the backtesting framework maintains flexibility while ensuring consistent behavior.
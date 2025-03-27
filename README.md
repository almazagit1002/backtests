# Backtest

## Strategies Overview

The Mean Reversion Volatility Bands strategy identifies potential mean reversion opportunities by combining multiple technical indicators:

- **Bollinger Bands**: Measures price volatility using standard deviations
- **Keltner Channels**: Measures price volatility using Average True Range (ATR)
- **Commodity Channel Index (CCI)**: Identifies potential overbought/oversold conditions

The strategy enters positions when price action indicates a potential reversion to the mean, with additional confirmation from oscillator indicators.

## Strategy Logic

1. **Entry Signals**:
   - CCI moves below the oversold threshold for long positions
   - CCI moves above the overbought threshold for short positions
   - Price is near or beyond the Bollinger Bands
   - Bollinger Bands and Keltner Channels have proper alignment

2. **Exit Signals**:
   - Take profit: When price reaches the target profit level 
   - Stop loss: When price moves against the position by a defined threshold 
   - Time-based exit: After a specified maximum holding period

## Performance Characteristics

The strategy is designed to:

- Capitalize on short-term price overextensions
- Profit from price normalization after volatility events
- Maintain controlled risk through predefined stop loss and take profit levels
- Diversify risk across multiple positions (configurable maximum concurrent positions)

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/backtests.git
   cd backtests
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Running a Backtest

To run a backtest with the default configuration:

```bash
python mean_reversion_volatility_bands.py
```

### Running Optimization

To optimize the strategy parameters:

```bash
python mean_reversion_volatility_bands_optimization.py
```

### Analyzing Optimization Results

After running optimization, analyze the results:

```bash
python mean_reversion_volatility_bands/optimization_analysis.py
```

### Configuration

The strategy parameters are configured in YAML files:

- `mean_reversion_volatility_bands/config.yaml`: Main configuration for backtesting
- `mean_reversion_volatility_bands/optimization_config.yaml`: Configuration for parameter optimization

Example configuration parameters:

```yaml
backtest:
  initial_budget: 1000
  fee_rate: 0.005

pbounds:
  tp_level: 2.0          # Take profit level (%)
  sl_level: 1.5          # Stop loss level (%)
  max_positions: 4       # Maximum number of concurrent positions
  keltner_period: 20     # Period for Keltner Channels
  atr_multiplier: 2.5    # ATR multiplier for Keltner Channels
  bollinger_period: 20   # Period for Bollinger Bands
  std_multiplier: 2.0    # Standard deviation multiplier for Bollinger Bands
  cci_period: 20         # Period for CCI
  CCI_up_threshold: 100  # CCI overbought threshold
  CCI_low_threshold: -100 # CCI oversold threshold
  Bollinger_Keltner_alignment: 0.005 # Alignment threshold
  window_size: 5         # Signal confirmation window
```

## Visualization

The strategy generates various visualization files in the `mean_reversion_volatility_bands/images/` directory:

- `back_test.pdf`: Backtest performance and equity curve
- `signals.pdf`: Entry and exit signals visualization
- `profit_distribution.pdf`: Distribution of trade profits
- `mean_reversion_volatility_bands_all.pdf`: Combined indicators visualization
- `mean_reversion_volatility_bands_splited.pdf`: Individual indicators visualization

To generate visualizations after a backtest:

```bash
python mean_reversion_volatility_bands/visualization.py
```

## Documentation

The project includes comprehensive documentation using MkDocs.

### Installing MkDocs

```bash
pip install mkdocs mkdocs-material
```

### Building and Viewing Documentation

To build and serve the documentation locally:

```bash
mkdocs serve
```

Then open your browser and go to `http://127.0.0.1:8000/`

To build the documentation site:

```bash
mkdocs build
```

This will create a `site` directory with the HTML documentation.

### Documentation Structure

The documentation covers:

- Strategy explanation and theory
- API reference
- Backtesting methodology
- Configuration options
- Visualization tools
- Project structure

## Project Structure

```
backtests/
├── docs/                           # Documentation files
│   ├── api/                        # API documentation
│   ├── strategies/                 # Strategy documentation
│   │   ├── visualization.md
│   │   ├── backtesting.md
│   │   └── configuration.md
│   └── project_structure.md
├── mean_reversion_volatility_bands/ # Strategy implementation
│   ├── data/                       # Data output directory
│   ├── images/                     # Visualization outputs
│   ├── __init__.py
│   ├── back_test.py                # Backtesting implementation
│   ├── config.yaml                 # Strategy configuration
│   ├── optimization_config.yaml    # Optimization settings
│   ├── signal_generator.py         # Trading signal generation
│   ├── trade_indicators.py         # Technical indicators
│   ├── visualization.py            # Visualization tools
│   ├── optimization.py             # Parameter optimization
│   └── optimization_analysis.py    # Analysis of optimization results
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── utils.py
├── .env                            # Environment variables
├── get_data_s3.py                  # Data retrieval script
├── mean_reversion_volatility_bands.py      # Main strategy runner
├── mean_reversion_volatility_bands_optimization.py  # Optimization runner
├── README.md                       # This file
├── requirements.txt                # Package dependencies
├── setup.py                        # Package installation
└── sol_price.csv                   # Sample price data
```

## Data Requirements

The strategy requires historical price data in CSV format with at least these columns:
- timestamp
- open
- high
- low
- close
- volume

Sample data is included in `sol_price.csv`.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
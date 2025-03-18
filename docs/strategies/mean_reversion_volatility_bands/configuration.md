# Strategy Configuration

The Mean Reversion Volatility Bands strategy is configured using a YAML file (`config.yaml`), which allows for easy adjustment of parameters without modifying the code.

## Configuration Parameters

### Signal Generation Parameters

```yaml
signals:
  CCI_up_threshold: 70
  CCI_low_threshold: -30
  Bollinger_Keltner_alignment: 0.005
  window_size: 3
```

| Parameter | Description |
|-----------|-------------|
| `CCI_up_threshold` | The threshold above which CCI indicates an overbought condition (for short entries) |
| `CCI_low_threshold` | The threshold below which CCI indicates an oversold condition (for long entries) |
| `Bollinger_Keltner_alignment` | The maximum allowed difference between Bollinger Bands and Keltner Channels for signal confirmation |
| `window_size` | The number of consecutive periods required for signal confirmation |

### Technical Indicators Parameters

```yaml
indicators:
  keltner_period: 20
  atr_multiplier: 2.0
  cci_period: 20
  bollinger_period: 20
  std_multiplier: 2.0
```

| Parameter | Description |
|-----------|-------------|
| `keltner_period` | The period used for calculating the Keltner Channel's moving average |
| `atr_multiplier` | The multiplier applied to the ATR for Keltner Channel width |
| `cci_period` | The period used for calculating the Commodity Channel Index |
| `bollinger_period` | The period used for calculating Bollinger Bands' moving average |
| `std_multiplier` | The standard deviation multiplier for Bollinger Bands width |

### Backtesting Parameters

```yaml
backtest:
  initial_budget: 10000
  fee_rate: 0.005
  max_positions: 4
  tp_level: 2
  sl_level: 1
```

| Parameter | Description |
|-----------|-------------|
| `initial_budget` | The starting capital for the backtest |
| `fee_rate` | The transaction fee percentage for each trade |
| `max_positions` | The maximum number of concurrent open positions |
| `tp_level` | Take profit level as a multiplier of the entry distance |
| `sl_level` | Stop loss level as a multiplier of the entry distance |

### Optimization Parameters

```yaml
optimization:
  tp_levels: [1.5, 2.0, 2.5, 3.0]
  sl_levels: [0.5, 1.0, 1.5]
```

| Parameter | Description |
|-----------|-------------|
| `tp_levels` | Array of take profit levels to test during optimization |
| `sl_levels` | Array of stop loss levels to test during optimization |

### Plot Paths Configuration

```yaml
plot_paths:
  prefix: mean_reversion_volatility_bands/images/
  indicators: mean_reversion_volatility_bands_all.pdf
  indicators_splited: mean_reversion_volatility_bands_splited.pdf
  back_test: back_test.pdf
  profit_distribution: profit_distribution.pdf
  signals: signals.pdf
  metrics: metrics.pdf
```

| Parameter | Description |
|-----------|-------------|
| `prefix` | Base directory for saving visualization outputs |
| `indicators` | Filename for the combined indicators plot |
| `indicators_splited` | Filename for the individual indicators plot |
| `back_test` | Filename for the backtest results plot |
| `profit_distribution` | Filename for the profit distribution plot |
| `signals` | Filename for the trading signals plot |
| `metrics` | Filename for the performance metrics plot |

### Data Paths Configuration

```yaml
data_paths:
  prefix: mean_reversion_volatility_bands/data/
  indicators: indicators.csv
  signals: signals.csv
  trades: trades.csv
  portafolio: portafolio.csv
```

| Parameter | Description |
|-----------|-------------|
| `prefix` | Base directory for saving data outputs |
| `indicators` | Filename for the calculated indicators data |
| `signals` | Filename for the generated trading signals |
| `trades` | Filename for the executed trades data |
| `portafolio` | Filename for the portfolio performance data |

### Execution Modes

```yaml
modes:
  csv_export: True
  plot_mode: True
  debug_mode: True
  optimize_mode: True
```

| Parameter | Description |
|-----------|-------------|
| `csv_export` | Whether to export data to CSV files |
| `plot_mode` | Whether to generate visualization plots |
| `debug_mode` | Whether to print additional debug information |
| `optimize_mode` | Whether to run parameter optimization |

### Input Data

```yaml
data_path: sol_price.csv
```

| Parameter | Description |
|-----------|-------------|
| `data_path` | Path to the input price data file |

## Customizing Configuration

To modify the strategy behavior:

1. Open the `config.yaml` file
2. Adjust the parameters according to your preferences
3. Save the file and run the backtest again

The configuration changes will be applied without needing to modify the code.
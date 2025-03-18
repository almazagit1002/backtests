# Trading Strategy Backtests

Welcome to the documentation for the Trading Strategy Backtesting framework.

## Overview

This project provides tools for backtesting various trading strategies, with a focus on algorithmic and systematic approaches. Currently, the framework implements:

- **Mean Reversion with Volatility Bands**: A strategy that combines Bollinger Bands, Keltner Channels, and the Commodity Channel Index (CCI) to identify potential mean reversion opportunities.

## Getting Started

To use this backtesting framework:

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Run a backtest:
   ```bash
   python mean_reversion_volatility_bands.py
   ```

3. Analyze the results in the generated reports and visualizations

## Project Structure

The project is organized into modules for different strategies, with shared utilities in the `utils` package. See the [Project Structure](project_structure.md) page for a complete overview.

## Features

- Configurable strategy parameters via YAML files
- Comprehensive backtesting with transaction costs
- Position sizing and risk management
- Detailed reporting and visualization of results
- Extensible framework for adding new strategies
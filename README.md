# Options Strategies and Market Analysis

A Python project for studying options strategies, simulating stock-price
movements, measuring portfolio risk, and generating technical trading signals.
It combines financial mathematics with historical market data and practical
Python analysis.

## Features

- **Options strategies:** Calculate expiration profit/loss, net premiums,
  breakeven points, and risk limits for butterfly spreads and protective puts.
- **Monte Carlo simulation:** Generate possible future stock-price paths from
  historical log-return estimates.
- **Volatility analysis:** Measure daily return fluctuations and annualize
  volatility using trading-day conventions.
- **Portfolio risk:** Estimate historical Value-at-Risk (VaR) and Expected
  Shortfall for an equal-weighted, multi-stock portfolio.
- **Technical indicators:** Generate MACD and EMA crossover signals and
  examine their behavior through a basic backtesting framework.

## Project structure

| File | Purpose |
| --- | --- |
| `Volatility_check.py` | Daily and annualized volatility calculations |
| `montecarlo.py` | Stock-price path simulation and visualization |
| `var.py` | Historical portfolio VaR and Expected Shortfall |
| `butterfly_spread.py` | Long call butterfly payoff analysis |
| `q5.py` | Protective put cost and expiration profit/loss |
| `technical_indicator.py` | MACD/EMA signals, backtesting, and performance measures |
| `validate_results.py` | Historical data preparation and numerical validation |
| `tests/` | Automated tests for the financial calculations |
| `requirements.txt` | Python dependencies |

## Installation

Use Python 3.10 or newer. On Windows, open PowerShell in the project folder.

```powershell
py -m pip install -r requirements.txt
```

Prepare the historical price dataset:

```powershell
py validate_results.py --rebuild
```

The data preparation script downloads a pinned public historical-price
snapshot, checks its file hash, and creates `historical_prices.csv` locally.
The analysis uses adjusted closing prices for AAPL, MSFT, GOOGL, AMZN, and
NVDA, with a default study period from January 2022 through July 2024.

## Run the analysis

Each module can be executed independently:

```powershell
py Volatility_check.py
py montecarlo.py
py var.py
py butterfly_spread.py
py q5.py
py technical_indicator.py
```

For a different historical period or configuration, use command-line options:

```powershell
py Volatility_check.py --start 2023-08-01 --end 2024-07-31
py montecarlo.py --paths 1000 --days 252 --seed 2024
py var.py --confidence 0.99
py technical_indicator.py --indicator EMA --fee-bps 10
```

The scripts save numerical outputs as CSV/JSON files and visualizations as
PNG files. Portfolio-risk, options, and backtesting outputs are placed in
`results/`.

## Methodology

### Volatility and Monte Carlo

Daily returns are calculated from historical prices. The volatility module
annualizes their sample standard deviation using the square root of 252.
Monte Carlo uses estimated daily log-return parameters and normally
distributed random shocks to generate possible future stock-price paths.
A fixed random seed makes the simulation reproducible.

### Portfolio risk

The portfolio contains five stocks with 20% weight each. Daily portfolio
returns are computed from aligned asset returns. Historical VaR uses the
lower-tail percentile of these returns, while Expected Shortfall measures
the average loss beyond that threshold.

### Options strategies

The butterfly spread module combines long and short calls at different
strike prices. The protective put module combines stock ownership with a
long put. Both evaluate profit/loss at expiration after accounting for
option premiums.

### Technical indicators

The MACD implementation uses 12-period and 26-period exponential moving
averages with a 9-period signal line. Buy/sell events occur at crossovers.
The backtest applies signals to the following trading day's closing price,
tracks cash and shares, and reports portfolio performance with transaction
costs.

## Testing

Run the automated test suite:

```powershell
py -m unittest discover -s tests -p "test_*.py" -v
```

The tests cover return calculations, simulated price paths, portfolio-risk
metrics, options payoffs, crossover detection, and trade execution timing.

## Data and limitations

Historical prices are obtained from a [public archived dataset](https://github.com/brytemydata-pixel/DecisionTree_multi_stock_5yr_data_BuySellHold_BA2_Assignment/blob/2cf72df460c31ff6d83aa23677f576904e6ccfb9/multi_stock_5yr.csv)
attributed by its publisher to Yahoo Finance / `yfinance`. Market-data CSVs
and generated outputs are kept locally rather than tracked in Git.

The simulations assume a particular return distribution and do not predict
actual future stock prices. Options modules calculate expiration payoffs,
not live option premiums. The technical backtest uses simplified execution
assumptions and does not account for every real-market trading cost.

This project is intended for educational analysis, not investment advice.

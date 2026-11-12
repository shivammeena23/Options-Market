# Options Strategies and Market-Risk Analysis

Python exercises covering option payoffs, historical portfolio risk,
Monte Carlo stock-price scenarios, and technical-indicator signals.

The main goal is to understand **what each calculation measures** and its
assumptions, rather than to suggest a trading strategy.

## Included modules

| File | What it does |
| --- | --- |
| `Volatility_check.py` | Estimates daily and annualized historical volatility |
| `montecarlo.py` | Simulates 100 AAPL paths over 252 future trading days by default |
| `var.py` | Calculates one-day 95% historical VaR and Expected Shortfall for five equal-weight stocks |
| `butterfly_spread.py` | Calculates expiration payoffs, risk, reward, and breakeven for call butterflies |
| `q5.py` | Calculates the cost and expiration P/L of a protective put |
| `technical_indicator.py` | Generates MACD or EMA crossover events and runs a simplified backtest |
| `validate_results.py` | Reconstructs the dated historical dataset from a pinned public snapshot and checks results |
| `tests/` | Unit tests for all six calculations |

## Quick start (Windows)

Install Python 3.10+ and open PowerShell in this folder.

```powershell
py -m pip install -r requirements.txt
py validate_results.py --rebuild
py -m unittest discover -s tests -p "test_*.py" -v
```

The rebuild step creates the local `historical_prices.csv` using a frozen,
hash-checked public source snapshot. If you already have the CSV from the
separate data-reconstruction package, copy it into this folder and run
`py validate_results.py` without `--rebuild`.

Run each experiment:

```powershell
py Volatility_check.py
py montecarlo.py
py var.py
py butterfly_spread.py
py q5.py
py technical_indicator.py
```

Outputs are saved as CSV, JSON, and PNG files locally. The `results/` folder
and source market-data CSV are intentionally not committed to Git.

## Fixed sample and example results

The default historical window is **3 January 2022 to 31 July 2024** and
uses adjusted closes for AAPL, MSFT, GOOGL, AMZN, and NVDA.

| Calculation | Verified result |
| --- | ---: |
| AAPL annualized volatility, simple returns | 28.12% |
| AAPL annualized volatility, log returns | 28.05% |
| One-day 95% VaR, 20% in each stock | 3.13% loss |
| One-day 95% Expected Shortfall | 4.23% loss |
| Standard long call butterfly (strikes 50, 60, 70; net debit 2) | Max gain 8, max loss 2 |
| Protective put (stock 100, strike 95, premium 2.5) | Max loss 7.5 |

The original project quoted 27% volatility and 3.44% VaR, but its original
VaR input CSV and historical estimation window were not available. Those
original figures are **unverified**; the table reports newly reconstructed
results from a disclosed window and source. They should not be presented as
the original May–July 2024 output.

## Data source and reproducibility

The historical dataset is reconstructed from a [public archived CSV](https://github.com/brytemydata-pixel/DecisionTree_multi_stock_5yr_data_BuySellHold_BA2_Assignment/blob/2cf72df460c31ff6d83aa23677f576904e6ccfb9/multi_stock_5yr.csv)
that attributes its adjusted closing prices to Yahoo Finance / `yfinance`.
Its exact source commit and SHA-256 are checked by `validate_results.py`.
This is a **third-party reconstruction**, not the project's missing original
input file or an exchange-certified market-data feed.

Because the source repository does not provide an explicit redistribution
license, `historical_prices.csv` is excluded from this public repository.
Keep your reconstructed copy locally, or rebuild it for personal analysis
after reviewing the source's applicable terms.

## Important modeling choices

**Monte Carlo.** The model fits normal daily log-return shocks using
historical mean and standard deviation, then compounds them to generate
possible positive prices. It uses a fixed random seed. It does not directly
price option contracts or predict the actual future path. Normality,
independence, and constant volatility are simplifying assumptions.

**Portfolio VaR.** All five stocks have 20% weights applied to same-date
daily simple returns (a daily-rebalanced-weight approximation). Historical
VaR is the negative 5th-percentile return, reported as a positive loss
threshold. It is **not** a guaranteed worst-case loss.

**Options.** Butterfly and protective-put calculations evaluate P/L **at
expiration**, after premiums; they do not estimate option market prices
with Black–Scholes or another valuation engine. A standard call butterfly
has equally spaced strikes. Unequal wings are labelled as broken-wing.

**MACD and backtesting.** The standard 12/26/9 MACD crosses generate discrete
buy/sell events rather than repeating a signal every day. The example
backtest trades on the **following day's close** after a signal, holds
whole shares, applies a 10-basis-point cost per transaction, and marks the
portfolio to market daily. It uses AAPL adjusted prices from the included
dataset to be reproducible. This is a simplified research proxy—actual fills
would need unadjusted tradable prices, slippage, corporate actions, and
proper out-of-sample testing. The original demonstration used the Sensex
index; this reconstruction does **not** claim to reproduce that backtest.

To supply your own `Date,Close` file for another instrument, run for example:

```powershell
py technical_indicator.py --csv my_prices.csv --ticker OTHER --currency USD
```

The calculations are educational and not investment advice.

To replace the older files in your existing GitHub repository, follow
[`PUSH_INSTRUCTIONS.md`](PUSH_INSTRUCTIONS.md), including the correction of
the misspelled butterfly script filename.

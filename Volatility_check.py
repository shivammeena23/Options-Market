"""Historical daily and annualized volatility from a fixed price dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("historical_prices.csv")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def stockdata(ticker, path=DATA_PATH, start="2022-01-01", end="2024-07-31"):
    data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if ticker not in data:
        raise ValueError(f"Ticker {ticker} is missing from {path}")
    prices = data.loc[start:end, ticker].dropna()
    if len(prices) < 3 or (prices <= 0).any():
        raise ValueError(f"Insufficient valid prices for {ticker}")
    return prices.to_frame(name="Close")


def volatility(data):
    """Return sample daily volatility and annualized volatility (simple returns)."""
    close = pd.to_numeric(data["Close"], errors="raise")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("Closing prices must be positive and non-missing")
    daily_returns = close.pct_change().dropna()
    if len(daily_returns) < 2:
        raise ValueError("At least three prices are required")
    daily_volatility = float(daily_returns.std(ddof=1))
    annualized_volatility = daily_volatility * np.sqrt(252)
    return daily_volatility, annualized_volatility


def main():
    parser = argparse.ArgumentParser(description="Compute volatility for five portfolio stocks")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-07-31")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    print(f"Historical volatility | {args.start} to {args.end} | 252 trading days/year")
    print(f"{'Ticker':<8} {'Daily SD':>12} {'Annualized SD':>16}")
    for ticker in TICKERS:
        daily, annualized = volatility(stockdata(ticker, args.csv, args.start, args.end))
        print(f"{ticker:<8} {daily * 100:>11.3f}% {annualized * 100:>15.3f}%")


if __name__ == "__main__":
    main()

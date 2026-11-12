"""Simulate AAPL prices with an empirical normal-log-return Monte Carlo model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("historical_prices.csv")


def load_prices(path=DATA_PATH, ticker="AAPL", start="2022-01-01", end="2024-07-31"):
    data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if ticker not in data:
        raise ValueError(f"Ticker {ticker} is missing from {path}")
    prices = data.loc[start:end, ticker].dropna().astype(float)
    if len(prices) < 3 or (prices <= 0).any():
        raise ValueError("Need at least three positive historical prices")
    return prices


def simulate_price_paths(prices, num_simulations=100, num_days=252, seed=2024):
    """Return simulated prices and model statistics.

    Day zero is the last observed price. The following num_days entries are
    future trading days. Each shock is a sampled *daily log return*.
    """
    if num_simulations < 1 or num_days < 1:
        raise ValueError("Number of paths and future trading days must be positive")

    log_returns = np.log(prices).diff().dropna()
    if len(log_returns) < 2:
        raise ValueError("At least two daily log returns are required")

    daily_mean = float(log_returns.mean())
    daily_std = float(log_returns.std(ddof=1))
    start_price = float(prices.iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.normal(daily_mean, daily_std, size=(num_days, num_simulations))
    future_prices = start_price * np.exp(np.cumsum(shocks, axis=0))
    paths = np.vstack([np.full((1, num_simulations), start_price), future_prices])

    details = {
        "last_historical_date": str(prices.index[-1].date()),
        "start_price": start_price,
        "daily_mean_log_return": daily_mean,
        "daily_standard_deviation_log_return": daily_std,
        "annualized_historical_log_return_volatility_pct": daily_std * np.sqrt(252) * 100,
        "num_simulations": num_simulations,
        "num_future_trading_days": num_days,
        "num_price_points_per_path": num_days + 1,
        "seed": seed,
        "median_final_simulated_price": float(np.median(paths[-1])),
        "mean_realized_simulated_annual_volatility_pct": float(shocks.std(axis=0, ddof=1).mean() * np.sqrt(252) * 100) if num_days > 1 else None,
    }
    return paths, details


def main():
    parser = argparse.ArgumentParser(description="Simulate 100 AAPL price paths")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-07-31")
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--days", type=int, default=252)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    prices = load_prices(args.csv, start=args.start, end=args.end)
    paths, details = simulate_price_paths(prices, args.paths, args.days, args.seed)

    out_dir = Path(__file__).resolve().parent
    pd.DataFrame(paths, columns=[f"path_{i+1:03d}" for i in range(args.paths)])\
        .rename_axis("forward_day").to_csv(out_dir / "monte_carlo_simulations.csv", float_format="%.8f")
    (out_dir / "monte_carlo_metrics.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(prices.index, prices, color="#163E70")
    axes[0].set(title="Historical adjusted AAPL price", xlabel="Date", ylabel="USD")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].plot(paths, alpha=0.22, linewidth=0.7, color="#2C75B7")
    axes[1].plot(np.median(paths, axis=1), color="#092C58", linewidth=2, label="Median path")
    axes[1].set(title=f"{args.paths} Monte Carlo paths", xlabel="Future trading days", ylabel="USD")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "monte_carlo_plot.png", dpi=170)
    plt.close(fig)

    print("Monte Carlo simulation completed")
    print(f"Historic AAPL log-return volatility (annualized): {details['annualized_historical_log_return_volatility_pct']:.2f}%")
    print(f"Paths: {args.paths} | Forward days: {args.days} | Seed: {args.seed}")
    print(f"Median simulated final price: USD {details['median_final_simulated_price']:.2f}")
    print("These are hypothetical price paths, not option premiums or guaranteed forecasts.")


if __name__ == "__main__":
    main()

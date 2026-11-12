"""One-day historical Value-at-Risk for five equal-weighted stocks."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DATA_PATH = Path(__file__).with_name("historical_prices.csv")


def load_prices(path=DATA_PATH, start="2022-01-01", end="2024-07-31"):
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate historical dates found")
    missing = set(TICKERS) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    prices = prices.loc[start:end, TICKERS]
    if len(prices) < 31 or prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Need at least 31 aligned days of positive, complete prices")
    return prices


def historical_var(prices, confidence_level=0.95, weights=None):
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if weights is None:
        weights = np.full(prices.shape[1], 1 / prices.shape[1])
    weights = np.asarray(weights, dtype=float)
    if len(weights) != prices.shape[1] or not np.isfinite(weights).all():
        raise ValueError("There must be one finite weight per asset")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1")

    asset_returns = prices.pct_change(fill_method=None).dropna()
    portfolio_returns = asset_returns.dot(weights)
    tail_quantile = float(np.quantile(portfolio_returns, 1 - confidence_level, method="linear"))
    var_loss = max(0.0, -tail_quantile)
    expected_shortfall_loss = max(0.0, -float(portfolio_returns[portfolio_returns <= tail_quantile].mean()))
    return portfolio_returns, var_loss, expected_shortfall_loss


def main():
    parser = argparse.ArgumentParser(description="Historical portfolio risk")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-07-31")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    prices = load_prices(args.csv, args.start, args.end)
    returns, var_loss, expected_shortfall = historical_var(prices, args.confidence)
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    returns.to_frame("Portfolio Return").to_csv(output / "var_daily_returns.csv", index_label="Date")

    metrics = {
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "assets": TICKERS,
        "weights": {ticker: 0.2 for ticker in TICKERS},
        "daily_return_observations": len(returns),
        "confidence_level": args.confidence,
        "one_day_var_loss_pct": var_loss * 100,
        "expected_shortfall_loss_pct": expected_shortfall * 100,
    }
    (output / "var_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(returns * 100, bins=40, color="#78A3CC", edgecolor="white")
    ax.axvline(-var_loss * 100, color="#BA3434", linewidth=2,
               label=f"{int(args.confidence * 100)}% VaR threshold: {-var_loss * 100:.2f}%")
    ax.set(title="Equal-weight portfolio daily historical returns", xlabel="Daily return (%)", ylabel="Trading days")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "var_distribution.png", dpi=170)
    plt.close(fig)

    print(f"Historical data: {metrics['start']} to {metrics['end']} ({len(returns)} daily returns)")
    print("Portfolio: " + ", ".join(f"{ticker} 20%" for ticker in TICKERS))
    print(f"One-day {args.confidence:.0%} historical VaR: {var_loss * 100:.2f}% loss")
    print(f"Expected shortfall at {args.confidence:.0%}: {expected_shortfall * 100:.2f}% loss")
    print("VaR is a loss threshold, not a guaranteed maximum loss.")


if __name__ == "__main__":
    main()

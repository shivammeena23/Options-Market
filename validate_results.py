"""Reconstruct and validate historical volatility, VaR, and Monte Carlo inputs.

This is a separate audit script. It does not modify the original GitHub project.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
SOURCE_COMMIT = "2cf72df460c31ff6d83aa23677f576904e6ccfb9"
SOURCE_URL = (
    "https://raw.githubusercontent.com/brytemydata-pixel/"
    "DecisionTree_multi_stock_5yr_data_BuySellHold_BA2_Assignment/"
    f"{SOURCE_COMMIT}/multi_stock_5yr.csv"
)
SOURCE_SHA256 = "a3a5a22d42b689db184ab7ee372a17d107c82b2e134b57d195b200163ed584d3"
CUTOFF = "2024-07-31"
START = "2022-01-01"
WINDOWS = {
    "2022_to_July_2024": ("2022-01-01", CUTOFF),
    "trailing_1y_to_July_2024": ("2023-08-01", CUTOFF),
    "2024_year_to_date": ("2024-01-01", CUTOFF),
    "calendar_2023": ("2023-01-01", "2023-12-31"),
    "calendar_2022": ("2022-01-01", "2022-12-31"),
}


def rebuild_prices(destination: Path) -> None:
    with urlopen(SOURCE_URL, timeout=40) as response:
        raw = response.read()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Source bytes changed: {actual_hash}")

    data = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
    data = data.set_index("Date").sort_index()
    data = data.loc[START:CUTOFF, TICKERS]
    ensure_valid_prices(data)
    data.to_csv(destination, index_label="Date", float_format="%.12f")


def ensure_valid_prices(data: pd.DataFrame) -> None:
    if list(data.columns) != TICKERS:
        raise ValueError(f"Expected columns {TICKERS}, got {list(data.columns)}")
    if len(data) < 100 or data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Dates are missing, duplicated, or unsorted")
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Prices contain missing or non-finite values")
    if (data <= 0).any().any():
        raise ValueError("Prices must be positive")
    if data.index.max() > pd.Timestamp(CUTOFF):
        raise ValueError("Found data after the declared project cutoff")


def measure_window(prices: pd.DataFrame, start: str, end: str) -> dict:
    window = prices.loc[start:end]
    returns = window.pct_change().dropna()
    log_returns = np.log(window).diff().dropna()
    if len(returns) < 30:
        raise ValueError(f"Not enough returns from {start} to {end}")

    portfolio_returns = returns.dot(np.full(5, 0.2))
    lower_tail = float(np.quantile(portfolio_returns, 0.05, method="linear"))
    expected_shortfall = -float(portfolio_returns[portfolio_returns <= lower_tail].mean())
    return {
        "first_price_date": window.index.min().date().isoformat(),
        "last_price_date": window.index.max().date().isoformat(),
        "price_rows": len(window),
        "daily_return_rows": len(returns),
        "aapl_annualized_simple_return_vol_pct": float(returns["AAPL"].std(ddof=1) * np.sqrt(252) * 100),
        "aapl_annualized_log_return_vol_pct": float(log_returns["AAPL"].std(ddof=1) * np.sqrt(252) * 100),
        "portfolio_fifth_percentile_return_pct": lower_tail * 100,
        "portfolio_one_day_95pct_var_loss_pct": -lower_tail * 100,
        "portfolio_95pct_expected_shortfall_loss_pct": expected_shortfall * 100,
    }


def monte_carlo(prices: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Empirical log-return model, 100 paths, 252 forward daily shocks."""
    r = np.log(prices["AAPL"]).diff().dropna()
    rng = np.random.default_rng(2024)
    shocks = rng.normal(r.mean(), r.std(ddof=1), size=(252, 100))
    last_price = float(prices["AAPL"].iloc[-1])
    paths = np.vstack((np.full((1, 100), last_price), last_price * np.exp(np.cumsum(shocks, axis=0))))
    path_vol = shocks.std(axis=0, ddof=1) * np.sqrt(252) * 100
    result = {
        "model": "independent normally distributed daily log returns",
        "seed": 2024,
        "paths": 100,
        "forward_trading_days": 252,
        "price_points_per_path": 253,
        "input_aapl_daily_mean_log_return": float(r.mean()),
        "input_aapl_daily_sd_log_return": float(r.std(ddof=1)),
        "mean_realized_path_vol_pct": float(path_vol.mean()),
        "minimum_realized_path_vol_pct": float(path_vol.min()),
        "maximum_realized_path_vol_pct": float(path_vol.max()),
        "median_final_simulated_price": float(np.median(paths[-1, :])),
    }
    return result, paths


def draw_diagnostics(prices: pd.DataFrame, paths: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    portfolio_returns = prices.pct_change().dropna().dot(np.full(5, 0.2)) * 100
    var = -np.quantile(portfolio_returns, 0.05)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for p in paths.T:
        axes[0].plot(p, alpha=0.11, color="#2153B3", linewidth=0.75)
    axes[0].plot(np.median(paths, axis=1), color="#0C2854", linewidth=2, label="Median path")
    axes[0].set(title="AAPL: 100 simulated paths", xlabel="Forward trading days", ylabel="Adjusted price (USD)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.12)
    axes[1].hist(portfolio_returns, bins=45, color="#7AA5CE", edgecolor="white")
    axes[1].axvline(-var, color="#B53333", linewidth=2, label=f"5th percentile: {-var:.2f}%")
    axes[1].set(title="Equal-weight portfolio: daily returns", xlabel="Daily return (%)", ylabel="Number of days")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.12)
    fig.suptitle("Reconstructed market-risk analysis | data through 31 Jul 2024", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Download the exact source snapshot and rebuild the dataset")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    csv_path = root / "historical_prices.csv"
    if args.rebuild or not csv_path.exists():
        rebuild_prices(csv_path)
    prices = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    ensure_valid_prices(prices)

    metrics = {name: measure_window(prices, *period) for name, period in WINDOWS.items()}
    mc, paths = monte_carlo(prices)
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "source_type": "third-party archived yfinance adjusted-close data, not the original project CSV",
        "reconstruction_cutoff": CUTOFF,
        "tickers": TICKERS,
        "portfolio_weights": {ticker: 0.2 for ticker in TICKERS},
        "confidence_level": 0.95,
        "windows": metrics,
        "monte_carlo": mc,
        "reported_resume_figures_are_not_verified": True,
    }
    (root / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw_diagnostics(prices, paths, root / "diagnostics.png")
    for name, item in metrics.items():
        print(f"{name:26s} | AAPL vol {item['aapl_annualized_simple_return_vol_pct']:6.2f}% | VaR {item['portfolio_one_day_95pct_var_loss_pct']:6.2f}%")
    print("\nNote: reconstructed results; original 27% volatility and 3.44% VaR are unverified.")


if __name__ == "__main__":
    main()

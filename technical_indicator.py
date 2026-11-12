"""MACD/EMA crossover signals and a simplified next-close backtest."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("historical_prices.csv")


def load_ticker_data(path=DATA_PATH, ticker="AAPL", start="2022-01-01", end="2024-07-31"):
    data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if ticker in data.columns:
        prices = data[ticker]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise ValueError(f"Neither {ticker} nor a Close column exists in {path}")
    prices = prices.loc[start:end].dropna().astype(float)
    if len(prices) < 40 or (prices <= 0).any():
        raise ValueError("Need at least 40 positive historical closing prices")
    return prices.to_frame(name="Close")


def MACD(df, short_window=12, long_window=26, signal_window=9):
    if not (0 < short_window < long_window and signal_window > 0):
        raise ValueError("Expected 0 < short window < long window and positive signal window")
    data = df.copy()
    data["EMA12"] = data["Close"].ewm(span=short_window, adjust=False, min_periods=short_window).mean()
    data["EMA26"] = data["Close"].ewm(span=long_window, adjust=False, min_periods=long_window).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["Signal Line"] = data["MACD"].ewm(span=signal_window, adjust=False, min_periods=signal_window).mean()
    return data


def generate_signals(df, short_window=12):
    """1 on bullish MACD crossovers, -1 on bearish, 0 otherwise."""
    data = df.copy()
    spread = data["MACD"] - data["Signal Line"]
    buy = (spread > 0) & (spread.shift(1) <= 0)
    sell = (spread < 0) & (spread.shift(1) >= 0)
    data["Signal"] = np.select([buy, sell], [1, -1], default=0).astype(int)
    return data


def generate_ema_signals(df, span=20):
    data = df.copy()
    data["EMA"] = data["Close"].ewm(span=span, adjust=False, min_periods=span).mean()
    gap = data["Close"] - data["EMA"]
    buy = (gap > 0) & (gap.shift(1) <= 0)
    sell = (gap < 0) & (gap.shift(1) >= 0)
    data["EMA Signal"] = np.select([buy, sell], [1, -1], default=0).astype(int)
    return data


def backtest_strategy(df, indicator="MACD", initial_balance=10000.0, fee_bps=10.0):
    """Observe the prior day's signal, then trade at today's adjusted close.

    This is a research proxy, not executable broker-level performance.
    """
    if indicator not in ("MACD", "EMA"):
        raise ValueError("indicator must be MACD or EMA")
    if initial_balance <= 0 or fee_bps < 0 or fee_bps >= 10000:
        raise ValueError("Invalid initial capital or transaction cost")
    column = "Signal" if indicator == "MACD" else "EMA Signal"
    if column not in df:
        raise ValueError(f"Call the relevant signal generator before backtesting: {column}")

    close = df["Close"].to_numpy(dtype=float)
    signals = df[column].to_numpy(dtype=int)
    cash, shares = float(initial_balance), 0
    rows = []
    fee_fraction = fee_bps / 10000

    for day, price in enumerate(close):
        action = 0
        previous_signal = int(signals[day - 1]) if day > 0 else 0
        if previous_signal == 1 and shares == 0:
            quantity = int(np.floor(cash / (price * (1 + fee_fraction))))
            if quantity > 0:
                cash -= quantity * price * (1 + fee_fraction)
                shares = quantity
                action = 1
        elif previous_signal == -1 and shares > 0:
            cash += shares * price * (1 - fee_fraction)
            shares = 0
            action = -1
        rows.append((cash + shares * price, cash, shares, action))

    portfolio = pd.DataFrame(rows, index=df.index,
                             columns=["Portfolio Value", "Cash", "Shares", "Executed Trade"])
    return portfolio, float(portfolio["Portfolio Value"].iloc[-1])


def calculate_performance(portfolio_value, annual_risk_free_rate=0.0):
    returns = pd.Series(portfolio_value).pct_change(fill_method=None).dropna()
    daily_risk_free = (1 + annual_risk_free_rate) ** (1 / 252) - 1
    standard_deviation = float(returns.std(ddof=1))
    sharpe = ((returns.mean() - daily_risk_free) / standard_deviation * np.sqrt(252)
              if standard_deviation > 0 else float("nan"))
    return returns, float(sharpe)


def main():
    parser = argparse.ArgumentParser(description="MACD and EMA crossover backtesting")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-07-31")
    parser.add_argument("--indicator", choices=["MACD", "EMA"], default="MACD")
    parser.add_argument("--initial-balance", type=float, default=10000)
    parser.add_argument("--fee-bps", type=float, default=10)
    parser.add_argument("--currency", default="USD")
    args = parser.parse_args()

    data = MACD(load_ticker_data(args.csv, args.ticker, args.start, args.end))
    data = generate_signals(data)
    if args.indicator == "EMA":
        data = generate_ema_signals(data)
    portfolio, final_value = backtest_strategy(data, args.indicator, args.initial_balance, args.fee_bps)
    returns, sharpe = calculate_performance(portfolio["Portfolio Value"])
    benchmark = args.initial_balance * data["Close"] / data["Close"].iloc[0]
    comparison = data.join(portfolio)
    comparison["Normalized Buy and Hold"] = benchmark

    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    safe_ticker = "".join(c for c in args.ticker.lower() if c.isalnum() or c in "_-")
    prefix = f"{safe_ticker}_{args.indicator.lower()}"
    comparison.to_csv(output / f"{prefix}_backtest.csv", index_label="Date")
    trades = int((portfolio["Executed Trade"] != 0).sum())
    drawdown = (portfolio["Portfolio Value"] / portfolio["Portfolio Value"].cummax() - 1).min()
    metrics = {
        "ticker": args.ticker,
        "indicator": args.indicator,
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "initial_capital": args.initial_balance,
        "currency": args.currency,
        "fee_bps_per_transaction": args.fee_bps,
        "next_day_close_execution": True,
        "final_portfolio_value": final_value,
        "total_return_pct": (final_value / args.initial_balance - 1) * 100,
        "normalized_buy_and_hold_return_pct": (benchmark.iloc[-1] / args.initial_balance - 1) * 100,
        "max_drawdown_pct": float(drawdown * 100),
        "sharpe_zero_risk_free_rate": sharpe,
        "executed_trades": trades,
    }
    (output / f"{prefix}_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(data.index, data["Close"], color="#1E497B", label=args.ticker)
    buys = comparison["Executed Trade"] == 1
    sells = comparison["Executed Trade"] == -1
    axes[0].scatter(data.index[buys], data.loc[buys, "Close"], marker="^", color="#12966C", label="Buy", s=25)
    axes[0].scatter(data.index[sells], data.loc[sells, "Close"], marker="v", color="#C34E54", label="Sell", s=25)
    axes[0].set(ylabel="Adjusted close", title=f"{args.ticker}: {args.indicator} signals (next-close execution)")
    axes[0].legend(frameon=False)
    axes[1].plot(data.index, data["MACD"], label="MACD", color="#376BA5")
    axes[1].plot(data.index, data["Signal Line"], label="Signal line", color="#D68A2D")
    axes[1].axhline(0, linewidth=0.6, color="#777")
    axes[1].set(ylabel="MACD")
    axes[1].legend(frameon=False)
    axes[2].plot(data.index, portfolio["Portfolio Value"], label="Strategy", color="#176C73")
    axes[2].plot(data.index, benchmark, label="Normalized buy and hold", linestyle="--", color="#A56B38")
    axes[2].set(ylabel=f"Portfolio value ({args.currency} proxy)", xlabel="Date")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{prefix}_backtest.png", dpi=170)
    plt.close(fig)

    print(f"{args.ticker} {args.indicator} | {len(returns)} daily portfolio returns")
    print(f"Final capital: {final_value:.2f} (initial {args.initial_balance:.2f})")
    print(f"Total return: {metrics['total_return_pct']:.2f}% | Buy-and-hold proxy: {metrics['normalized_buy_and_hold_return_pct']:.2f}%")
    print(f"Executed trades: {trades} | Max drawdown: {metrics['max_drawdown_pct']:.2f}% | Sharpe: {sharpe:.3f}")
    print("Illustrative adjusted-close backtest; next-session close fills and fees are assumptions.")


if __name__ == "__main__":
    main()

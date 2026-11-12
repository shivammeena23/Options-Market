"""Initial cost and expiration profit/loss of stock plus a protective put."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def protective_put_cost(stock_price, strike_price, put_premium):
    if stock_price <= 0 or strike_price <= 0 or put_premium < 0:
        raise ValueError("Stock and strike prices must be positive; premium cannot be negative")
    return stock_price + put_premium


def protective_put_profit_loss(stock_price, strike_price, put_premium, expiration_price):
    if expiration_price < 0:
        raise ValueError("Expiration stock price cannot be negative")
    cost = protective_put_cost(stock_price, strike_price, put_premium)
    expiration_payoff = expiration_price + max(strike_price - expiration_price, 0)
    return expiration_payoff - cost


def main():
    stock_price, strike_price, put_premium = 100.0, 95.0, 2.5
    cost = protective_put_cost(stock_price, strike_price, put_premium)
    max_loss = max(0.0, cost - strike_price)
    breakeven = cost if strike_price <= cost else None
    spots = np.linspace(50, 150, 201)
    pnl = [protective_put_profit_loss(stock_price, strike_price, put_premium, spot) for spot in spots]
    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"Expiration Spot": spots, "Net Profit Loss": pnl}).to_csv(
        output / "protective_put_payoff.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(spots, pnl, color="#176B72", linewidth=2.5, label="Stock + protective put")
    ax.plot(spots, spots - stock_price, color="#A0A7B3", linestyle="--", label="Unprotected stock")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.axvline(strike_price, color="#A0A7B3", linestyle=":", label=f"Strike {strike_price:g}")
    ax.set(title="Protective put: expiration P/L", xlabel="Spot price at expiration", ylabel="Net P/L per share")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "protective_put_payoff.png", dpi=170)
    plt.close(fig)

    print(f"Initial stock + put cost: {cost:.2f}")
    print(f"Maximum possible loss: {max_loss:.2f}")
    print(f"Breakeven spot: {breakeven if breakeven is not None else 'No positive break-even spot'}")
    for spot in (70, 95, 100, 120):
        print(f"Expiration spot {spot}: net P/L {protective_put_profit_loss(stock_price, strike_price, put_premium, spot):+.2f}")


if __name__ == "__main__":
    main()

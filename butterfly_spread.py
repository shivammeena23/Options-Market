"""Expiration profit/loss of a long call butterfly (or broken-wing butterfly)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def butterfly_spread_profit_loss(K1, K2, K3, P1, P2, P3, S):
    """Long K1 call, short two K2 calls, long K3 call, after net premium."""
    if not (0 < K1 < K2 < K3):
        raise ValueError("Strikes must be positive and ordered K1 < K2 < K3")
    if min(P1, P2, P3) < 0 or S < 0:
        raise ValueError("Premiums and expiration spot must be non-negative")
    net_debit = P1 - 2 * P2 + P3
    call_payoff = max(S - K1, 0) - 2 * max(S - K2, 0) + max(S - K3, 0)
    return float(call_payoff - net_debit)


def describe_butterfly(K1, K2, K3, P1, P2, P3):
    net_debit = P1 - 2 * P2 + P3
    symmetric = np.isclose(K2 - K1, K3 - K2)
    candidates = [butterfly_spread_profit_loss(K1, K2, K3, P1, P2, P3, spot)
                  for spot in (0, K1, K2, K3, 2 * K3)]
    details = {
        "strategy": "Standard long call butterfly" if symmetric else "Broken-wing long call butterfly",
        "net_premium_paid": float(net_debit),
        "maximum_profit": float(max(candidates)),
        "maximum_loss": float(max(0, -min(candidates))),
    }
    if symmetric and 0 < net_debit < K2 - K1:
        details["breakeven_low"] = float(K1 + net_debit)
        details["breakeven_high"] = float(K3 - net_debit)
    return details


def main():
    K1, K2, K3 = 50, 60, 70
    P1, P2, P3 = 12.0, 6.0, 2.0
    details = describe_butterfly(K1, K2, K3, P1, P2, P3)
    spots = np.linspace(35, 85, 201)
    pnl = [butterfly_spread_profit_loss(K1, K2, K3, P1, P2, P3, s) for s in spots]

    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"Expiration Spot": spots, "Net Profit Loss": pnl}).to_csv(
        output / "butterfly_payoff.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(spots, pnl, linewidth=2.5, color="#174E90")
    ax.axhline(0, color="#555", linewidth=0.8)
    for strike in [K1, K2, K3]:
        ax.axvline(strike, color="#A9B2C1", linestyle="--", linewidth=0.8)
    ax.set(title="Long call butterfly: expiration P/L", xlabel="Spot price at expiration", ylabel="Net P/L per share")
    fig.tight_layout()
    fig.savefig(output / "butterfly_payoff.png", dpi=170)
    plt.close(fig)

    print(details["strategy"])
    print(f"Strikes: {K1}, {K2}, {K3} | Net premium paid: {details['net_premium_paid']:.2f}")
    print(f"Maximum profit: {details['maximum_profit']:.2f} | Maximum loss: {details['maximum_loss']:.2f}")
    print(f"Breakeven spots: {details.get('breakeven_low', 'N/A')}, {details.get('breakeven_high', 'N/A')}")
    print(f"Profit/loss at spot 60: {butterfly_spread_profit_loss(K1, K2, K3, P1, P2, P3, 60):.2f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

# Step 1: Load historical price data for a set of assets from a CSV file
csv_file = 'historical_prices.csv'
data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

# Step 2: Calculate the daily returns for each asset
returns = data.pct_change().dropna()

# Step 3: Calculate the portfolio returns based on given weights
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Example weights for each asset

# Ensure the weights sum to 1
weights /= weights.sum()

# Calculate portfolio returns
portfolio_returns = returns.dot(weights)

# Step 4: Calculate the VaR at a 95% confidence level
confidence_level = 0.95
VaR = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

# Step 5: Output the VaR value
print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level: {VaR:.4f}")

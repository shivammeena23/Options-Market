import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Download historical stock data
ticker = "AAPL"  
stock_data = yf.download(ticker, start="2022-01-01", end="2024-12-31")
stock_data = stock_data['Adj Close']
stock_data.dropna(inplace=True)

# Calculate log returns
log_returns = np.log(1 + stock_data.pct_change().dropna())

# Parameters for simulation
num_simulations = 100
num_days = 252  # Simulate for 1 year

last_price = stock_data[-1]
mu = log_returns.mean()
sigma = log_returns.std()

# Monte Carlo simulation
simulations = np.zeros((num_days, num_simulations))
for i in range(num_simulations):
    prices = [last_price]
    for _ in range(1, num_days):
        shock = np.random.normal(loc=mu, scale=sigma)
        price = prices[-1] * np.exp(shock)
        prices.append(price)
    simulations[:, i] = prices

# Plotting actual historical stock price
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(stock_data, label='Historical Price')
plt.title(f"Historical Stock Price of {ticker}")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()

#Ploting Monte Carlo Simulated Prices
plt.subplot(1, 2, 2)
plt.plot(simulations)
plt.title(f"{num_simulations} Monte Carlo Simulations for {ticker}")
plt.xlabel("Days into the Future")
plt.ylabel("Simulated Price")
plt.tight_layout()
plt.show()


future_df = pd.DataFrame(simulations)
future_df.to_excel("monte_carlo_simulations.xlsx", index=False)

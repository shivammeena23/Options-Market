# -*- coding: utf-8 -*-

def protective_put_cost(stock_price, strike_price, put_premium):

    stock_cost = stock_price

    # cost of buying the put option
    put_cost = put_premium

    # total cost of the protective put strategy
    total_cost = stock_cost + put_cost

    return total_cost


stock_price = 100
strike_price = 95
put_premium = 2.5

total_cost = protective_put_cost(stock_price, strike_price, put_premium)
print(f"Total Cost of Protective Put Strategy: {total_cost}")

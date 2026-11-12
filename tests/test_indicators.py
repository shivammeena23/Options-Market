import unittest

import numpy as np
import pandas as pd

from technical_indicator import (
    MACD,
    backtest_strategy,
    calculate_performance,
    generate_ema_signals,
    generate_signals,
    load_ticker_data,
)


class TechnicalIndicatorTests(unittest.TestCase):
    def test_signal_only_on_crossovers(self):
        frame = pd.DataFrame({
            "MACD": [0, 0.2, 0.5, -0.2, -0.5, 0.3],
            "Signal Line": [0, 0.3, 0.4, 0.0, -0.3, 0.1],
        })
        actual = generate_signals(frame)["Signal"].tolist()
        self.assertEqual(actual, [0, -1, 1, -1, 0, 1])

    def test_backtest_trades_on_next_day_not_signal_day(self):
        frame = pd.DataFrame({
            "Close": [100, 100, 110, 120, 90],
            "Signal": [0, 1, 0, -1, 0],
        }, index=pd.date_range("2024-01-01", periods=5))
        portfolio, final = backtest_strategy(frame, initial_balance=1000, fee_bps=0)
        self.assertEqual(portfolio["Executed Trade"].tolist(), [0, 0, 1, 0, -1])
        self.assertEqual(portfolio["Shares"].tolist(), [0, 0, 9, 9, 0])
        self.assertAlmostEqual(final, 820)

    def test_transaction_fee_does_not_improve_profit(self):
        frame = pd.DataFrame({
            "Close": [100, 100, 110, 120, 90],
            "Signal": [0, 1, 0, -1, 0],
        }, index=pd.date_range("2024-01-01", periods=5))
        _, without_fees = backtest_strategy(frame, initial_balance=1000, fee_bps=0)
        _, with_fees = backtest_strategy(frame, initial_balance=1000, fee_bps=10)
        self.assertLess(with_fees, without_fees)

    def test_real_aapl_macd_and_ema(self):
        data = MACD(load_ticker_data())
        data = generate_signals(data)
        self.assertGreater(data["MACD"].notna().sum(), 500)
        self.assertTrue(set(data["Signal"].unique()).issubset({-1, 0, 1}))
        portfolio, final = backtest_strategy(data)
        self.assertTrue(np.isfinite(final))
        returns, sharpe = calculate_performance(portfolio["Portfolio Value"])
        self.assertGreater(len(returns), 500)
        self.assertTrue(np.isfinite(sharpe))

        data = generate_ema_signals(data)
        ema_portfolio, ema_final = backtest_strategy(data, indicator="EMA")
        self.assertTrue(np.isfinite(ema_final))
        self.assertEqual(len(ema_portfolio), len(data))


if __name__ == "__main__":
    unittest.main()

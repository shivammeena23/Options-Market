import unittest

import numpy as np
import pandas as pd

from Volatility_check import stockdata, volatility
from montecarlo import load_prices as load_monte_carlo_prices, simulate_price_paths
from var import historical_var, load_prices as load_portfolio_prices
from butterfly_spread import butterfly_spread_profit_loss, describe_butterfly
from q5 import protective_put_cost, protective_put_profit_loss


class VolatilityTests(unittest.TestCase):
    def test_does_not_invent_a_return_before_first_price(self):
        prices = pd.DataFrame({"Close": [100.0, 110.0, 100.0]})
        daily, annual = volatility(prices)
        expected = pd.Series([0.1, -1 / 11]).std(ddof=1)
        self.assertAlmostEqual(daily, expected)
        self.assertAlmostEqual(annual, expected * np.sqrt(252))

    def test_reconstructed_volatility(self):
        _, annual = volatility(stockdata("AAPL"))
        self.assertAlmostEqual(annual * 100, 28.11545060072747, places=6)


class MonteCarloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = load_monte_carlo_prices()

    def test_shape_seed_positive_prices(self):
        first, meta = simulate_price_paths(self.prices, 100, 252, seed=2024)
        second, _ = simulate_price_paths(self.prices, 100, 252, seed=2024)
        self.assertEqual(first.shape, (253, 100))
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first > 0).all())
        self.assertTrue((first[0] == self.prices.iloc[-1]).all())
        self.assertAlmostEqual(meta["annualized_historical_log_return_volatility_pct"], 28.05379501054282, places=6)


class HistoricalVarTests(unittest.TestCase):
    def test_reproduces_historical_var_and_expected_shortfall(self):
        prices = load_portfolio_prices()
        returns, var, es = historical_var(prices, 0.95)
        self.assertEqual(len(returns), 646)
        self.assertAlmostEqual(var * 100, 3.133349949569575, places=6)
        self.assertAlmostEqual(es * 100, 4.228469137201212, places=6)
        self.assertGreater(es, var)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            historical_var(load_portfolio_prices(), weights=np.ones(5))

    def test_confidence_is_checked(self):
        with self.assertRaises(ValueError):
            historical_var(load_portfolio_prices(), confidence_level=1.0)


class ButterflyTests(unittest.TestCase):
    def test_symmetric_butterfly_profit_loss(self):
        args = (50, 60, 70, 12, 6, 2)
        self.assertEqual(butterfly_spread_profit_loss(*args, 40), -2)
        self.assertEqual(butterfly_spread_profit_loss(*args, 60), 8)
        self.assertEqual(butterfly_spread_profit_loss(*args, 80), -2)
        details = describe_butterfly(*args)
        self.assertEqual(details["maximum_profit"], 8)
        self.assertEqual(details["maximum_loss"], 2)
        self.assertEqual(details["breakeven_low"], 52)
        self.assertEqual(details["breakeven_high"], 68)

    def test_asymmetric_wings_are_not_called_symmetric(self):
        details = describe_butterfly(50, 60, 65, 12, 6, 2)
        self.assertIn("Broken-wing", details["strategy"])

    def test_invalid_strikes_rejected(self):
        with self.assertRaises(ValueError):
            butterfly_spread_profit_loss(60, 50, 70, 12, 6, 2, 60)


class ProtectivePutTests(unittest.TestCase):
    def test_cost_and_downside_floor(self):
        self.assertEqual(protective_put_cost(100, 95, 2.5), 102.5)
        self.assertEqual(protective_put_profit_loss(100, 95, 2.5, 70), -7.5)
        self.assertEqual(protective_put_profit_loss(100, 95, 2.5, 0), -7.5)
        self.assertEqual(protective_put_profit_loss(100, 95, 2.5, 120), 17.5)

    def test_negative_premium_rejected(self):
        with self.assertRaises(ValueError):
            protective_put_cost(100, 95, -1)


if __name__ == "__main__":
    unittest.main()

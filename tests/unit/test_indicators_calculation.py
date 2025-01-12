import unittest

import numpy as np
import pandas as pd

from src.utils.utils import find_pivots, calculate_sma, calculate_rsi, calculate_fib_levels_pivots


class TestFibonacciCalculations(unittest.TestCase):
    def setUp(self):
        # Setup a simple dataset with clear pivot points
        self.df = pd.DataFrame({
            'H': [140, 105, 115, 95, 110],  # 115 should be a high pivot
            'L': [95, 90, 100, 85, 105],  # 85 should be a low pivot
            'C': [97, 92, 112, 90, 107]
        })
        self.depth = 3  # Smaller window to easily capture single pivot highs/lows
        self.deviation = 2  # Deviation threshold

    def test_find_pivots(self):
        pivot_high, pivot_low = find_pivots(self.df, self.depth, self.deviation)
        expected_pivot_high = (115, 2)  # High pivot at index 2
        expected_pivot_low = (85, 3)  # Low pivot at index 3

        # Assert that the pivots are correctly identified
        self.assertEqual(pivot_high, expected_pivot_high)
        self.assertEqual(pivot_low, expected_pivot_low)

    def test_calculate_fib_levels_pivots(self):
        fib_levels = calculate_fib_levels_pivots(self.df, depth=self.depth, deviation=self.deviation)
        # Expected Fibonacci levels for a range from 85 (low) to 115 (high)
        expected_fib_levels = {
            'fib_0.000': 85, 'fib_0.236': 92.08, 'fib_0.382': 96.46000000000001, 'fib_0.500': 100.0,
            'fib_0.618': 103.53999999999999, 'fib_0.786': 108.58, 'fib_1.000': 115, 'fib_1.272': 123.16,
            'fib_1.618': 133.54000000000002
        }

        self.assertEqual(fib_levels, expected_fib_levels)


class TestTradingIndicators(unittest.TestCase):
    def setUp(self):
        # Setup sample data for testing
        self.df = pd.DataFrame({
            'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'H': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'L': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'O': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'V': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        })

    def test_calculate_sma(self):
        sma_5 = calculate_sma(self.df, period=5)
        expected_sma_5 = [np.nan, np.nan, np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        np.testing.assert_equal(sma_5, expected_sma_5)

    def test_calculate_rsi(self):
        rsi_14 = calculate_rsi(self.df, period=14)
        expected_rsi_14 = [np.nan] * 10
        np.testing.assert_equal(rsi_14, expected_rsi_14)


if __name__ == '__main__':
    unittest.main()

import unittest
from decimal import Decimal, ROUND_HALF_UP

from src.utils.utils import get_adjusted_amount


# Unit Test Class
class TestGetAdjustedAmount(unittest.TestCase):
    def test_valid_precision(self):
        # Test rounding with valid precision
        self.assertAlmostEqual(get_adjusted_amount(0.54, 0.1), 0.5)
        self.assertAlmostEqual(get_adjusted_amount(0.55, 0.1), 0.6)
        self.assertAlmostEqual(get_adjusted_amount(123.456, 0.01), 123.46)
        self.assertAlmostEqual(get_adjusted_amount(123.456, 0.25), 123.5)

    def test_small_numbers(self):
        # Test small numbers with small precision
        self.assertAlmostEqual(get_adjusted_amount(0.000133, 0.0001), 0.0001)
        self.assertAlmostEqual(get_adjusted_amount(0.000177, 0.0001), 0.0002)
        self.assertAlmostEqual(get_adjusted_amount(0.000133, 0.00001), 0.00013)

    def test_large_numbers(self):
        # Test large numbers with larger precision
        self.assertAlmostEqual(get_adjusted_amount(1234567.89, 10), 1234570.0)
        self.assertAlmostEqual(get_adjusted_amount(987654321.123, 1000), 987654000.0)

    def test_precision_zero_or_negative(self):
        # Test invalid precision values
        self.assertEqual(get_adjusted_amount(123.456, 0), 123)
        self.assertEqual(get_adjusted_amount(0.54, 0), 1)
        self.assertEqual(get_adjusted_amount(123.456, -0.1), 123)

    def test_edge_cases(self):
        # Test edge cases like exactly halfway points
        self.assertAlmostEqual(get_adjusted_amount(0.25, 0.1), 0.3)
        self.assertAlmostEqual(get_adjusted_amount(1.05, 1.0), 1.0)
        self.assertAlmostEqual(get_adjusted_amount(1.5, 1.0), 2.0)
        self.assertAlmostEqual(get_adjusted_amount(123.500, 0.01), 123.5)

    def test_non_standard_precision(self):
        # Test non-standard precision like 0.2, 0.25
        self.assertAlmostEqual(get_adjusted_amount(0.54, 0.2), 0.6)
        self.assertAlmostEqual(get_adjusted_amount(0.48, 0.2), 0.4)
        self.assertAlmostEqual(get_adjusted_amount(123.456, 0.5), 123.5)
        self.assertAlmostEqual(get_adjusted_amount(123.456, 0.25), 123.5)

    def test_rounding_up(self):
        # Test rounding up at halfway points
        self.assertAlmostEqual(get_adjusted_amount(0.55, 0.1), 0.6)
        self.assertAlmostEqual(get_adjusted_amount(0.75, 0.5), 1.0)
        self.assertAlmostEqual(get_adjusted_amount(1.5, 1.0), 2.0)

    def test_rounding_down(self):
        # Test rounding down at below-halfway points
        self.assertAlmostEqual(get_adjusted_amount(0.54, 0.1), 0.5)
        self.assertAlmostEqual(get_adjusted_amount(0.74, 0.5), 0.5)
        self.assertAlmostEqual(get_adjusted_amount(1.4, 1.0), 1.0)


    def test_amount_calculation(self):
        free = 1000
        risk_cap = 1000 * 0.01
        atr = 0.05

        contract_size = 100

        bla = risk_cap / atr
        bla_cont = bla / contract_size

        bla_test_cont = (risk_cap / atr) / contract_size

        print(bla, bla_cont, bla_test_cont)

"""Tests for the deterministic weighted-average calculation."""

import json
import unittest
from pathlib import Path

from src.calculator import weighted_average_expense_ratio
from src.generate_synthetic_data import FUNDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WeightedAverageExpenseRatioTests(unittest.TestCase):
    def test_hand_computed_two_fund_example(self) -> None:
        # Fund A: expense_ratio 1.0, aum 100 -> contributes 1.0 * 100 = 100
        # Fund B: expense_ratio 2.0, aum 300 -> contributes 2.0 * 300 = 600
        # weighted average = (100 + 600) / (100 + 300) = 700 / 400 = 1.75
        funds = [
            {"expense_ratio": 1.0, "aum": 100},
            {"expense_ratio": 2.0, "aum": 300},
        ]

        self.assertAlmostEqual(weighted_average_expense_ratio(funds), 1.75)

    def test_equal_weights_reduces_to_plain_average(self) -> None:
        # Three equal-aum funds -> weighting has no effect, so this is just
        # the arithmetic mean of the expense ratios: (0.2 + 0.4 + 0.6) / 3 = 0.4
        funds = [
            {"expense_ratio": 0.2, "aum": 50},
            {"expense_ratio": 0.4, "aum": 50},
            {"expense_ratio": 0.6, "aum": 50},
        ]

        self.assertAlmostEqual(weighted_average_expense_ratio(funds), 0.4)

    def test_matches_project_ground_truth(self) -> None:
        # Reuses the real Phase 1 synthetic-data fixtures (not a re-typed
        # copy of the numbers) so this test fails if the fixture data ever
        # drifts from data/ground_truth.json, the same way test_fund_filter.py
        # checks filtering against that file.
        ground_truth = json.loads(
            (PROJECT_ROOT / "data" / "ground_truth.json").read_text()
        )
        qualifying_names = set(ground_truth["qualifying_funds"])
        funds = [
            {"expense_ratio": fund.expense_ratio_percent, "aum": fund.aum_millions_usd}
            for fund in FUNDS
            if fund.fund_name in qualifying_names
        ]

        self.assertAlmostEqual(
            weighted_average_expense_ratio(funds), ground_truth["correct_answer"]
        )

    def test_empty_fund_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_average_expense_ratio([])

    def test_zero_total_aum_raises(self) -> None:
        funds = [{"expense_ratio": 0.5, "aum": 0}, {"expense_ratio": 0.3, "aum": 0}]

        with self.assertRaises(ValueError):
            weighted_average_expense_ratio(funds)

    def test_missing_field_raises(self) -> None:
        with self.assertRaises(ValueError):
            weighted_average_expense_ratio([{"expense_ratio": 0.5}])


if __name__ == "__main__":
    unittest.main()

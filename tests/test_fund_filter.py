"""Tests for deterministic structured fund filtering."""

import json
import unittest
from pathlib import Path

from src.fund_filter import FilterSpec, filter_funds
from src.pdf_extraction import extract_pdf_content


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FundFilterTests(unittest.TestCase):
    def test_esg_active_funds_match_ground_truth(self) -> None:
        extracted_funds = [
            extract_pdf_content(pdf_path)
            for pdf_path in sorted((PROJECT_ROOT / "data" / "raw_pdfs").glob("*.pdf"))
        ]
        ground_truth = json.loads(
            (PROJECT_ROOT / "data" / "ground_truth.json").read_text()
        )

        actual = filter_funds(
            extracted_funds, FilterSpec(is_esg=True, status="active")
        )

        self.assertEqual(actual, ground_truth["qualifying_funds"])

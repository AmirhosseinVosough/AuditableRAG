"""Tests for the three standalone cascade tiers: BM25, semantic, and OCR.

None of these need a GROQ_API_KEY - that is the point of splitting the tiers
out of `extraction_cascade.py`. Each one is a pure "given pages, which pages
matter" (or "given a scan, what text") decision, testable without spending an
LLM call.

What these tests deliberately pin down is the *abstain* behavior, not just the
happy path. Every tier's contract is "return an answer, or return None so the
next tier gets a turn", and the None cases are the ones that keep a wrong
guess out of the pipeline - a ranking tier that confidently narrows to the
wrong page is worse than one that admits it doesn't know, because the LLM then
never sees the text containing the answer. Those are the regressions worth
catching.
"""

import unittest
from pathlib import Path

from src import bm25_search, ocr, semantic_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = PROJECT_ROOT / "data" / "user_uploads"

# One page that plainly states the target fields, padded out with filler pages
# of unrelated-but-plausible fund boilerplate. Written inline rather than read
# from a fixture so the "which page is correct" answer is visible right here.
_TARGET_PAGE = (
    "Fund Facts. Net Expense Ratio 0.15%. Total Net Assets $17,746.93 million. "
    "The fund screens holdings on environmental, social and governance (ESG) "
    "criteria and is currently open to new investors."
)
_FILLER_PAGE_A = (
    "Past performance does not guarantee future results. Index returns are for "
    "illustrative purposes only and do not represent actual fund performance."
)
_FILLER_PAGE_B = (
    "Carefully consider the investment objectives, risk factors, and charges "
    "before investing. This and other information can be found in the prospectus."
)
_FILLER_PAGE_C = (
    "Shares are bought and sold at market price, not net asset value, and are "
    "not individually redeemed from the fund. Brokerage commissions will reduce returns."
)

# rank_pages() now requires a query argument (no more module-level default) -
# these are the tests' own stand-ins for a real user's question.
_BM25_TEST_QUERY = (
    "ESG environmental social governance expense ratio net assets total "
    "assets under management active closed fund status"
)
_SEMANTIC_TEST_QUERY = (
    "Does this page state the fund's ESG or sustainability screening, its "
    "expense ratio, its net assets or assets under management, or whether "
    "the fund is currently open or closed?"
)


class BM25RankPagesTests(unittest.TestCase):
    def test_finds_the_page_stating_the_fields(self) -> None:
        pages = [_FILLER_PAGE_A, _FILLER_PAGE_B, _TARGET_PAGE, _FILLER_PAGE_C]

        ranked = bm25_search.rank_pages(pages, _BM25_TEST_QUERY)

        self.assertIsNotNone(ranked)
        assert ranked is not None  # narrowing for the type checker
        self.assertEqual(ranked[0], 2, "the page stating expense ratio/net assets should rank first")

    def test_abstains_when_document_is_too_small_to_narrow(self) -> None:
        """A document no longer than the top-K cut is already as narrow as it gets."""
        self.assertIsNone(bm25_search.rank_pages([_TARGET_PAGE, _FILLER_PAGE_A], _BM25_TEST_QUERY))

    def test_abstains_when_no_page_contains_any_query_term(self) -> None:
        """Zero term overlap must abstain, never pick an arbitrary page."""
        pages = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima"]

        self.assertIsNone(bm25_search.rank_pages(pages, _BM25_TEST_QUERY))


class SemanticRankPagesTests(unittest.TestCase):
    """Exercises the real model. Skipped (not failed) if it can't be loaded.

    A machine with no network on first run, or no sentence-transformers
    installed, is a legitimate environment for this project - the cascade is
    built to degrade there - so it must not turn the suite red. That degrading
    is itself asserted by `test_abstains_when_model_is_unavailable` below,
    which fakes the failure rather than depending on the environment.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if semantic_search._get_model() is None:
            raise unittest.SkipTest("sentence-transformers model unavailable (no network / not installed)")

    def test_finds_the_page_stating_the_fields(self) -> None:
        pages = [_FILLER_PAGE_A, _FILLER_PAGE_B, _TARGET_PAGE, _FILLER_PAGE_C]

        ranked = semantic_search.rank_pages(pages, _SEMANTIC_TEST_QUERY)

        self.assertIsNotNone(ranked, "a clearly-relevant page among clear filler should clear the gate")
        assert ranked is not None  # narrowing for the type checker
        self.assertEqual(ranked[0], 2)

    def test_abstains_when_every_page_is_equally_irrelevant(self) -> None:
        """The MIN_MARGIN gate's whole job: no real winner means no narrowing.

        This is the case that made whole-page embedding dangerous - cosine
        similarities never reach zero, so *something* always ranks first. The
        tier must decline rather than promote a meaningless winner.
        """
        pages = [_FILLER_PAGE_A, _FILLER_PAGE_B, _FILLER_PAGE_C, _FILLER_PAGE_A]

        self.assertIsNone(semantic_search.rank_pages(pages, _SEMANTIC_TEST_QUERY))

    def test_abstains_when_document_is_too_small_to_narrow(self) -> None:
        self.assertIsNone(semantic_search.rank_pages([_TARGET_PAGE, _FILLER_PAGE_A], _SEMANTIC_TEST_QUERY))

    def test_real_fact_sheets_are_never_narrowed_to_a_page_missing_every_field(self) -> None:
        """On the real fixture set, a confident pick must land on a page that actually states something.

        Not asserting a specific page index per document: several fact sheets
        state the fields across more than one page, and the tier is free to
        prefer any of them. What must never happen is confidently narrowing to
        a page carrying none of the target fields, which would hide the answer
        from the LLM entirely.
        """
        from src.pdf_extraction import extract_pdf_content

        # Pages (1-indexed) that state at least one of expense ratio, net
        # assets, or an ESG policy - verified by hand against each PDF.
        relevant_pages = {
            "esgu_fact_sheet": {1, 2, 3, 4, 5},
            "esgv_fact_sheet": {1, 2, 3, 4},
            "ussg_fact_sheet": {1, 2, 3, 4},
            "spy_fact_sheet": {1, 2},
            "ivv_fact_sheet": {1},
        }

        for stem, relevant in relevant_pages.items():
            pdf_path = REAL_DATA_DIR / f"{stem}.pdf"
            if not pdf_path.is_file():
                self.skipTest(f"real-data fixture missing: {pdf_path}")

            pages = extract_pdf_content(pdf_path)["pages"]
            assert isinstance(pages, list)
            ranked = semantic_search.rank_pages(pages, _SEMANTIC_TEST_QUERY)

            if ranked is None:
                continue  # abstaining is always an acceptable outcome
            self.assertIn(
                ranked[0] + 1,
                relevant,
                f"{stem}: semantic tier confidently narrowed to page {ranked[0] + 1}, which states "
                f"none of the target fields - narrowing there hides the answer from the LLM",
            )

    def test_abstains_when_model_is_unavailable(self) -> None:
        """A failed/missing model must degrade to "next tier", never raise.

        Fakes the unavailable state rather than uninstalling anything, so this
        runs identically on a machine where the model loads fine.
        """
        saved_model, saved_flag = semantic_search._model, semantic_search._model_unavailable
        semantic_search._model, semantic_search._model_unavailable = None, True
        try:
            pages = [_FILLER_PAGE_A, _FILLER_PAGE_B, _TARGET_PAGE, _FILLER_PAGE_C]
            self.assertIsNone(semantic_search.rank_pages(pages, _SEMANTIC_TEST_QUERY))
        finally:
            semantic_search._model, semantic_search._model_unavailable = saved_model, saved_flag


def _tesseract_available() -> bool:
    """True if both pytesseract and the system Tesseract binary are usable."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001 - missing package or missing binary, same answer either way
        return False
    return True


class OCRRecoverTextTests(unittest.TestCase):
    def test_recovers_text_from_a_pdf_with_no_text_layer(self) -> None:
        """The tier's actual job: read a scan that `extract_text()` returns nothing for.

        Builds the scan on the fly by rendering a real fact sheet to images and
        rewrapping them as an image-only PDF - so the fixture is guaranteed to
        have zero text layer (asserted below before OCR is even attempted)
        rather than trusting a checked-in file to stay that way.

        Skipped where Tesseract isn't installed; `ocr.py`'s contract is that
        the cascade degrades there, and the two None-returning tests below
        cover that direction.
        """
        if not _tesseract_available():
            raise unittest.SkipTest("Tesseract binary not installed (brew install tesseract)")

        source = REAL_DATA_DIR / "esgu_fact_sheet.pdf"
        if not source.is_file():
            self.skipTest(f"real-data fixture missing: {source}")

        import tempfile

        import pdfplumber

        with pdfplumber.open(source) as pdf:
            images = [page.to_image(resolution=150).original.convert("RGB") for page in pdf.pages]

        with tempfile.TemporaryDirectory() as tmp_dir:
            scanned = Path(tmp_dir) / "scanned.pdf"
            images[0].save(scanned, save_all=True, append_images=images[1:])

            with pdfplumber.open(scanned) as pdf:
                extractable = "".join(page.extract_text() or "" for page in pdf.pages).strip()
            self.assertEqual(extractable, "", "fixture bug: the generated scan still has a text layer")

            recovered = ocr.recover_text(scanned)

        self.assertIsNotNone(recovered, "OCR should recover text from a legible scan")
        assert recovered is not None  # narrowing for the type checker
        joined = "\n".join(recovered).lower()
        # Asserting on content the document plainly shows, not on exact
        # character counts - OCR output varies slightly with render resolution
        # and Tesseract version, and pinning that would make this brittle.
        self.assertIn("ishares", joined)
        self.assertIn("expense ratio", joined)

    def test_returns_none_for_a_file_that_cannot_be_opened(self) -> None:
        """The corrupted fixture must stay "could not be read" - OCR cannot rescue a broken file.

        This is the boundary that keeps
        test_real_data_pipeline.py's unreadable-document test meaningful: if
        OCR ever started returning text here, that test's premise would be
        gone. Nothing can render a file with no /Root, so None is correct.
        """
        corrupted = REAL_DATA_DIR / "corrupted_fact_sheet.pdf"
        if not corrupted.is_file():
            self.skipTest(f"corrupted fixture missing: {corrupted}")

        self.assertIsNone(ocr.recover_text(corrupted))

    def test_returns_none_for_a_missing_file(self) -> None:
        """A nonexistent path is a failure to recover, not an exception to propagate."""
        self.assertIsNone(ocr.recover_text(REAL_DATA_DIR / "does_not_exist_at_all.pdf"))


if __name__ == "__main__":
    unittest.main()

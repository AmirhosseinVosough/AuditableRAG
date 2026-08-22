"""Cascade tier 7: OCR recovery for PDFs that open fine but carry no text layer.

Split out of `extraction_cascade.py` so each retrieval tier lives in its own
file - see that module's docstring for where this sits in the overall
regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> human-review order.

What this is for, precisely: a *scanned* PDF. The file itself is valid and
opens without complaint, but its pages are images of text rather than text,
so `pdfplumber.extract_text()` returns empty strings for every page and every
downstream tier has nothing to work with. That document is not unreadable -
it's readable by eye, just not by a text parser - and OCR is what closes that
gap. This is the "messy/heterogeneous source docs" row of the README's risk
table.

What this is NOT for: a genuinely broken file (truncated, no /Root, not
really a PDF). Those fail earlier, inside `extract_pdf_content`, and nothing
can render them to an image either - so they still end up flagged "could not
be read", which is the honest answer. `extract_with_cascade` only reaches
this module when the PDF opened successfully and *then* turned out to have no
extractable text.

Approach: render each page to a bitmap via pdfplumber's `to_image` (already a
dependency; it renders through pypdfium2 under the hood) and run Tesseract
over it via pytesseract. Chosen over a vision-capable LLM call because it
runs locally - no per-page API cost, no document text leaving the machine,
and no dependence on the Groq account's rate limits mid-batch. The tradeoff
is a system-level dependency: pytesseract is only a thin wrapper, and the
actual `tesseract` binary must be installed separately (`brew install
tesseract` on macOS, `apt install tesseract-ocr` on Debian/Ubuntu).

Failure policy, matching every other tier here: this module never raises. A
missing `tesseract` binary, a missing package, a render failure, or output too
sparse to be real text all come back as None, so the document ends up flagged
"could not be read" exactly as it did before this tier existed. OCR is
strictly an attempt to *recover* a document that was already lost - if the
attempt fails, nothing is worse off than before.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)

# Tesseract's accuracy is very sensitive to input resolution. 300 DPI is the
# usual recommendation for document OCR - meaningfully better than the ~72 DPI
# a PDF page renders at by default, without the memory cost (and diminishing
# returns) of going higher on a full page.
RENDER_DPI = 300

# Below this many characters across the whole document, treat the OCR result
# as failed rather than usable. Tesseract does not report "I found nothing" -
# it happily returns a handful of stray punctuation marks picked out of page
# noise, and passing that downstream would turn "could not be read" into a
# confident-looking extraction over garbage, which is exactly the failure mode
# this project exists to avoid. A real fund document page carries well over
# 1000 characters; this floor is deliberately far below that, since its job is
# only to separate "genuine text" from "noise", not to judge completeness.
MIN_USABLE_CHARS = 100


def recover_text(pdf_path: Path) -> list[str] | None:
    """OCR every page of *pdf_path* and return the recovered per-page text, or None if unusable.

    Args:
        pdf_path: The PDF to OCR. Expected to be a file that opens correctly
            but has no extractable text layer - see the module docstring.

    Returns:
        One string per page, in page order (same shape as
        `pdf_extraction.extract_pdf_content`'s "pages", so callers can swap it
        in directly). None if OCR is unavailable, failed, or produced less
        than `MIN_USABLE_CHARS` of text in total - all of which mean "this
        document still could not be read".
    """
    try:
        import pdfplumber
        import pytesseract
    except ImportError as exc:
        logger.warning("OCR unavailable (%s) - document cannot be recovered", exc)
        return None

    pages: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                image = page.to_image(resolution=RENDER_DPI).original
                pages.append(pytesseract.image_to_string(image))
                logger.debug("OCR'd page %d of %s", page_number, pdf_path.name)
    except Exception as exc:  # noqa: BLE001 - a missing tesseract binary, a render failure, anything: all mean "not recovered"
        logger.warning(
            "OCR failed for %s: %s - if this is a missing Tesseract binary, install it "
            "(brew install tesseract / apt install tesseract-ocr)",
            pdf_path.name,
            exc,
        )
        return None

    total_chars = sum(len(page.strip()) for page in pages)
    if total_chars < MIN_USABLE_CHARS:
        logger.warning(
            "OCR of %s produced only %d characters (below the %d-character usability floor) - "
            "treating as not recovered rather than passing noise downstream",
            pdf_path.name,
            total_chars,
            MIN_USABLE_CHARS,
        )
        return None

    logger.info("OCR recovered %d characters across %d page(s) of %s", total_chars, len(pages), pdf_path.name)
    return pages

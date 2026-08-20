"""Extract text and detected table structures from PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pdfplumber


Table: TypeAlias = list[list[str | None]]
PDFExtraction: TypeAlias = dict[str, str | list[Table]]


def extract_pdf_content(pdf_path: str | Path) -> PDFExtraction:
    """Return all extracted text and` any tables detected in *pdf_path*.

    Text is joined with a blank line between pages. ``tables`` preserves the
    page-by-page table order as returned by pdfplumber; it is empty when no
    tables are detected.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    page_text: list[str] = []
    tables: list[Table] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text.append(page.extract_text() or "")
            tables.extend(page.extract_tables())

    return {"text": "\n\n".join(page_text), "tables": tables}


extract_pdf = extract_pdf_content

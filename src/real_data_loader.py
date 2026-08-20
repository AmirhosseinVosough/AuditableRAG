"""Phase 9a: document discovery for real-data mode.

Real-data mode has no fixture list the way synthetic mode has
`generate_synthetic_data.FUNDS` - the whole point is that the pipeline can
be pointed at an arbitrary folder of PDFs it has never seen before. This
module's only job is finding them; it makes no assumptions about how many
there are or what's inside any of them - those questions belong to Phase 2
(extraction) and the real-data field extraction in field_extraction.py, not
here.
"""

from __future__ import annotations

from pathlib import Path


def load_real_pdfs(directory: Path) -> list[Path]:
    """Return every PDF found in *directory*, sorted for a deterministic run order.

    Args:
        directory: Folder to search. Not searched recursively - only files
            directly inside it - so a caller who organizes uploads into
            subfolders gets an explicit empty result rather than a silent
            partial scan.

    Returns:
        Every file in *directory* whose extension is ``.pdf`` (matched
        case-insensitively, since real uploads won't reliably be
        lowercase), sorted by filename. An empty list if the directory
        exists but contains no PDFs is a legitimate outcome for the caller
        to handle, not an error condition.

    Raises:
        FileNotFoundError: If *directory* does not exist.
        NotADirectoryError: If *directory* exists but is a file, not a folder.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Real-data directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Real-data path is not a directory: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

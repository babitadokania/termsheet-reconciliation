"""Turn term sheets (PDF, Word, text) into plain text for the LLM.

PDF: pdfplumber (layout-aware), falling back to pypdf if pdfplumber fails or returns no text.
Word: python-docx, including tables (term sheets are often two-column tables).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class DocumentParseError(Exception):
    """Raised when a document cannot be read or contains no extractable text."""


@dataclass
class ParsedDocument:
    name: str
    path: Path
    format: str
    text: str
    pages: int

    @property
    def char_count(self) -> int:
        return len(self.text)


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: Path) -> tuple[str, int]:
    errors = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
        text = "\n\n".join(f"[Page {i}]\n{t}" for i, t in enumerate(pages, 1) if t.strip())
        if text.strip():
            return text, len(pages)
        errors.append("pdfplumber returned no text")
    except Exception as exc:  # noqa: BLE001 - try the fallback parser
        errors.append(f"pdfplumber: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n\n".join(f"[Page {i}]\n{t}" for i, t in enumerate(pages, 1) if t.strip())
        if text.strip():
            log.warning("Used pypdf fallback for %s (%s)", path.name, "; ".join(errors))
            return text, len(pages)
        errors.append("pypdf returned no text")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pypdf: {exc}")

    raise DocumentParseError(
        f"No text could be extracted from {path.name} ({'; '.join(errors)}). "
        "If this is a scanned PDF, OCR it first (e.g. ocrmypdf)."
    )


def _parse_docx(path: Path) -> tuple[str, int]:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise DocumentParseError("python-docx is required for .docx files") from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentParseError(f"Could not open Word document {path.name}: {exc}") from exc

    parts: list[str] = []
    # Walk the body in order so paragraphs and tables keep their relative position.
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            txt = Paragraph(child, document).text.strip()
            if txt:
                parts.append(txt)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells: list[str] = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and (not cells or cells[-1] != t):  # merged cells repeat their text
                        cells.append(t)
                if cells:
                    parts.append(": ".join(cells) if len(cells) == 2 else " | ".join(cells))
    return "\n".join(parts), 1


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    if not path.exists():
        raise DocumentParseError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(f"Unsupported term sheet format '{ext}' for {path.name}")

    if ext == ".pdf":
        text, pages = _parse_pdf(path)
    elif ext == ".docx":
        text, pages = _parse_docx(path)
    else:
        text, pages = path.read_text(encoding="utf-8", errors="replace"), 1

    text = _clean(text)
    if len(text) < 50:
        raise DocumentParseError(f"{path.name} contains too little text ({len(text)} chars) to be a term sheet")
    log.info("Parsed %s: %d page(s), %d chars", path.name, pages, len(text))
    return ParsedDocument(name=path.stem, path=path, format=ext.lstrip("."), text=text, pages=pages)


def discover_term_sheets(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Term sheet folder not found: {folder}")
    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No term sheets (*.pdf, *.docx) found in {folder}")
    return files

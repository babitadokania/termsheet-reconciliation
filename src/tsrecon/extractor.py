"""Step 1: term sheet -> validated, evidence-backed fields (LLM pass #1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .document_parser import DocumentParseError, ParsedDocument, parse_document
from .heuristics import heuristic_extract
from .llm.base import LLMClient, LLMError
from .llm.prompts import EXTRACTION_SYSTEM, build_extraction_prompt
from .schema import TermSheetExtraction, extraction_json_schema, validate_extraction

log = logging.getLogger(__name__)

# Gemini Flash has a 1M-token window; this cap only protects against pathological inputs.
MAX_PROMPT_CHARS = 400_000


class ExtractionService:
    def __init__(self, llm: LLMClient, *, fallback_to_heuristics: bool = True, output_dir: Path | None = None):
        self.llm = llm
        self.fallback = fallback_to_heuristics
        self.output_dir = output_dir

    def extract_document(self, doc: ParsedDocument) -> TermSheetExtraction:
        text = doc.text
        if len(text) > MAX_PROMPT_CHARS:
            log.warning("%s is %d chars; truncating to %d for the LLM", doc.name, len(text), MAX_PROMPT_CHARS)
            text = text[:MAX_PROMPT_CHARS]

        method = self.llm.name
        warnings: list[str] = []
        try:
            raw = self.llm.generate_json(
                system=EXTRACTION_SYSTEM,
                prompt=build_extraction_prompt(doc.name, text),
                task="extract_fields",
                context={"text": text, "document": doc.name},
                schema=extraction_json_schema(),
            )
            if isinstance(raw, dict) and "fields" in raw and isinstance(raw["fields"], dict):
                raw = raw["fields"]  # tolerate a wrapper object
            if not isinstance(raw, dict):
                raise LLMError(f"expected a JSON object, got {type(raw).__name__}")
        except LLMError as exc:
            if not self.fallback:
                raise
            log.error("LLM extraction failed for %s (%s) - falling back to heuristic extractor", doc.name, exc)
            warnings.append(f"LLM extraction failed ({exc}); heuristic fallback used")
            raw, method = heuristic_extract(text), "heuristic-fallback"

        extraction = validate_extraction(
            raw, document=doc.name, source_path=str(doc.path), source_text=doc.text, method=method
        )
        extraction.warnings = warnings + extraction.warnings
        for w in extraction.warnings:
            log.warning("[%s] %s", doc.name, w)
        if extraction.parsing_errors:
            log.warning("[%s] parsing errors in fields: %s", doc.name, extraction.parsing_errors)
        ok = sum(1 for f in extraction.fields.values() if f.status.value == "OK")
        log.info("[%s] extracted %d/%d fields via %s", doc.name, ok, len(extraction.fields), method)
        self._persist(extraction)
        return extraction

    def extract_path(self, path: Path) -> TermSheetExtraction | None:
        try:
            doc = parse_document(path)
        except DocumentParseError as exc:
            log.error("Skipping %s: %s", path.name, exc)
            return None
        return self.extract_document(doc)

    def _persist(self, extraction: TermSheetExtraction) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"{extraction.document}.json"
        out.write_text(json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        log.debug("Wrote %s", out)

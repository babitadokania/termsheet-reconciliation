"""End-to-end orchestration + command-line entry point.

python run_pipeline.py                       # uses .env / defaults
python run_pipeline.py --provider mock       # offline, no API key needed
python run_pipeline.py --provider gemini --record   # live Gemini run, saved to recordings/
python run_pipeline.py --provider replay            # recorded Gemini output, no key needed
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .adjudicator import MismatchAdjudicator
from .booking_ingest import load_bookings
from .config import Settings, load_settings
from .document_parser import discover_term_sheets
from .evaluation import accuracy_summary, evaluate_all
from .excel_report import write_workbook
from .extractor import ExtractionService
from .llm import LLMError, build_llm_client
from .llm.cache import CachedLLMClient
from .llm.prompts import PROMPT_VERSION
from .logging_setup import setup_logging
from .reconcile import compare
from .results import FALSE_ALERT, MATCH, PARSING_ERROR, TRUE_MISMATCH, classify, trade_summary
from .spark_session import SparkSetupError, get_spark

log = logging.getLogger("tsrecon")


def run(settings: Settings, report_name: str | None = None) -> Path:
    started = time.monotonic()
    llm = build_llm_client(settings)

    # 1. Term sheets -> validated extraction (LLM pass 1)
    term_sheets = discover_term_sheets(settings.term_sheet_dir)
    log.info("Found %d term sheet(s): %s", len(term_sheets), [p.name for p in term_sheets])
    extractor = ExtractionService(
        llm, fallback_to_heuristics=settings.extraction_fallback, output_dir=settings.extracted_dir
    )
    extractions = [e for e in (extractor.extract_path(p) for p in term_sheets) if e is not None]
    if not extractions:
        raise RuntimeError("No term sheet could be parsed - nothing to reconcile")

    # 2. Bookings -> PySpark
    spark = get_spark(settings.spark_master)
    try:
        ingest = load_bookings(spark, settings.booking_dir)

        # 3. Comparison in Spark, then LLM pass 2 on mismatches
        detail = compare(spark, ingest.bookings, extractions, settings.numeric_tolerance).toPandas()
        bookings_pd = ingest.bookings.orderBy("ISIN", "TradeID").toPandas()
    finally:
        spark.stop()

    breaks = detail[detail["initial_result"] == "MISMATCH"].to_dict("records")
    verdicts = MismatchAdjudicator(llm).adjudicate(breaks)
    final = classify(detail, verdicts)
    trades = trade_summary(final)
    accuracy = evaluate_all(extractions, settings.expected_dir)

    # 4. Output
    counts = final["final_status"].value_counts().to_dict()
    meta = {
        "Run timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pipeline version": f"{__version__} (prompts {PROMPT_VERSION})",
        "LLM provider": llm.name
        + (
            "  [OFFLINE MOCK - set GEMINI_API_KEY for Gemini]"
            if llm.name.startswith("mock")
            else "  [recorded Gemini responses, no API calls]"
            if llm.name.startswith("replay")
            else ""
        ),
        "Term sheets": ", ".join(f"{e.document} ({e.extraction_method})" for e in extractions),
        "Booking extracts": ", ".join(ingest.files),
        "Trades reconciled": trades["booking_ref"].nunique() if not trades.empty else 0,
        "Fields compared": len(final),
        "Result": " | ".join(f"{k}: {counts.get(k, 0)}" for k in (MATCH, FALSE_ALERT, TRUE_MISMATCH, PARSING_ERROR)),
        "Data-quality issues": len(ingest.dq_issues),
        "Extraction accuracy vs golden values": accuracy_summary(accuracy),
    }
    out = settings.output_dir / (report_name or f"reconciliation_report_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    write_workbook(
        out,
        detail=final,
        trades=trades,
        bookings=bookings_pd,
        extractions=extractions,
        dq_issues=ingest.dq_issues,
        meta=meta,
        accuracy=accuracy,
    )
    latest = settings.output_dir / "reconciliation_report_latest.xlsx"
    try:
        shutil.copyfile(out, latest)
    except OSError as exc:  # e.g. the file is open in Excel on Windows
        log.warning("Could not refresh %s (%s) - is it open in Excel?", latest.name, exc)

    if isinstance(llm, CachedLLMClient):
        log.info("LLM cache: %d hit(s), %d new call(s)", llm.hits, llm.misses)
    log.info("Done in %.1fs. %s", time.monotonic() - started, meta["Result"])
    if not accuracy.empty:
        print("\nExtraction accuracy: " + accuracy_summary(accuracy))
    print("\n" + trades[["document", "booking_ref", "overall_status", "true_mismatch_fields"]].to_string(index=False))
    print(f"\nReport: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LLM-powered term sheet vs booking reconciliation")
    p.add_argument(
        "--provider",
        choices=["auto", "gemini", "groq", "replay", "mock"],
        help="LLM provider (default: env LLM_PROVIDER or auto)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="with --provider gemini/groq: save responses to recordings/ for keyless replay",
    )
    p.add_argument("--model", help="Gemini model name (default: env GEMINI_MODEL or gemini-3.8-flash)")
    p.add_argument("--term-sheets", type=Path, help="Folder with term sheets (PDF/DOCX)")
    p.add_argument("--bookings", type=Path, help="Folder with booking extracts (CSV/JSON)")
    p.add_argument("--output", type=Path, help="Output folder")
    p.add_argument("--report-name", help="Workbook file name (default: timestamped)")
    p.add_argument("--no-cache", action="store_true", help="Ignore cached LLM responses")
    p.add_argument(
        "--no-fallback", action="store_true", help="Fail instead of using heuristic extraction if the LLM fails"
    )
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    overrides = dict(
        llm_provider=args.provider,
        gemini_model=args.model,
        term_sheet_dir=args.term_sheets,
        booking_dir=args.bookings,
        output_dir=args.output,
        log_level=args.log_level,
    )
    if args.no_cache:
        overrides["use_cache"] = False
    if args.record:
        overrides["record"] = True
    if args.no_fallback:
        overrides["extraction_fallback"] = False
    try:
        settings = load_settings(**overrides)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    log_file = setup_logging(settings.log_dir, settings.log_level)
    log.info("Term sheet reconciliation v%s - log file: %s", __version__, log_file)
    try:
        run(settings, args.report_name)
        return 0
    except (SparkSetupError, LLMError, FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130
    except Exception:  # noqa: BLE001
        log.exception("Unexpected failure - see the log file for the full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())

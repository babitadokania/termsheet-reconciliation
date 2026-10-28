"""Score the latest extractions against the golden values.

    python scripts/evaluate_extraction.py                 # uses output/extracted and data/expected
    python scripts/evaluate_extraction.py --extracted sample_output/extracted

Exits with code 1 if accuracy is below --min-accuracy, so it can gate prompt/model changes in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrecon.evaluation import accuracy_summary, evaluate_all  # noqa: E402
from tsrecon.schema import TermSheetExtraction  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted", type=Path, default=ROOT / "output" / "extracted")
    p.add_argument("--expected", type=Path, default=ROOT / "data" / "expected")
    p.add_argument("--min-accuracy", type=float, default=0.0, help="fail if overall accuracy is below this (0-1)")
    args = p.parse_args()

    files = sorted(args.extracted.glob("*.json"))
    if not files:
        print(f"No extraction files in {args.extracted} - run the pipeline first.")
        return 1
    extractions = [TermSheetExtraction.model_validate(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    df = evaluate_all(extractions, args.expected)
    if df.empty:
        print("No golden values matched the extracted documents.")
        return 1
    with_cols = ["document", "field", "expected", "extracted", "outcome", "extraction_method"]
    print(df[with_cols].to_string(index=False))
    print("\n" + accuracy_summary(df))
    return 0 if df["correct"].mean() >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())

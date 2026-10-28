"""Convenience entry point: `python run_pipeline.py [options]` (no install step needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tsrecon.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

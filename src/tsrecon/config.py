"""Central configuration. Values come from environment variables (or a local .env file).

Nothing secret is hard-coded: the Gemini key is read from GEMINI_API_KEY (or GOOGLE_API_KEY).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_PROVIDERS = {"auto", "gemini", "groq", "mock", "replay"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _envd(name: str, default: str) -> str:
    """Like _env but never None (a default is always supplied)."""
    return _env(name) or default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str = "auto"  # auto | gemini | mock | replay
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.8-flash"
    groq_api_key: str | None = field(default=None, repr=False)
    groq_model: str | None = None  # None = best available free-tier model
    llm_temperature: float = 0.0
    llm_max_retries: int = 3  # per model; Gemini then moves to another model if still overloaded
    llm_timeout_s: int = 90
    llm_min_interval_s: float = 4.0  # throttle between calls (free tier ~15 RPM)
    use_cache: bool = True
    gemini_structured_output: bool = True  # pass the JSON schema to Gemini (response_json_schema)
    gemini_use_vertex: bool = False  # True = Vertex AI endpoint (needs a billing-enabled GCP project)
    record: bool = False  # gemini runs write their responses to recordings_dir
    recordings_dir: Path = PROJECT_ROOT / "recordings"
    expected_dir: Path = PROJECT_ROOT / "data" / "expected"  # golden values for accuracy evaluation
    extraction_fallback: bool = True  # fall back to heuristic extractor if the LLM fails
    # Paths
    term_sheet_dir: Path = PROJECT_ROOT / "data" / "term_sheets"
    booking_dir: Path = PROJECT_ROOT / "data" / "bookings"
    output_dir: Path = PROJECT_ROOT / "output"
    # Reconciliation
    numeric_tolerance: float = 1e-6
    spark_master: str = "local[*]"
    log_level: str = "INFO"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "llm_cache"

    @property
    def extracted_dir(self) -> Path:
        return self.output_dir / "extracted"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"


def load_settings(env_file: Path | None = None, **overrides) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    provider = _envd("LLM_PROVIDER", "auto").lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got '{provider}'")

    settings = Settings(
        llm_provider=provider,
        gemini_api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL"),
        gemini_model=_envd("GEMINI_MODEL", Settings.gemini_model),
        llm_temperature=float(_envd("LLM_TEMPERATURE", "0.0")),
        llm_max_retries=int(_envd("LLM_MAX_RETRIES", "3")),
        llm_timeout_s=int(_envd("LLM_TIMEOUT_S", "90")),
        llm_min_interval_s=float(_envd("LLM_MIN_INTERVAL_S", "4.0")),
        use_cache=_env_bool("LLM_USE_CACHE", True),
        gemini_structured_output=_env_bool("GEMINI_STRUCTURED_OUTPUT", True),
        gemini_use_vertex=_env_bool("GEMINI_USE_VERTEX", False),
        extraction_fallback=_env_bool("EXTRACTION_FALLBACK", True),
        term_sheet_dir=Path(_envd("TERM_SHEET_DIR", str(Settings.term_sheet_dir))),
        booking_dir=Path(_envd("BOOKING_DIR", str(Settings.booking_dir))),
        output_dir=Path(_envd("OUTPUT_DIR", str(Settings.output_dir))),
        numeric_tolerance=float(_envd("NUMERIC_TOLERANCE", "1e-6")),
        spark_master=_envd("SPARK_MASTER", "local[*]"),
        log_level=_envd("LOG_LEVEL", "INFO").upper(),
    )
    clean = {k: v for k, v in overrides.items() if v is not None}
    if "llm_provider" in clean and clean["llm_provider"] not in VALID_PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(VALID_PROVIDERS)}")
    for key in ("term_sheet_dir", "booking_dir", "output_dir", "recordings_dir", "expected_dir"):
        if key in clean:
            clean[key] = Path(clean[key])
    return replace(settings, **clean)

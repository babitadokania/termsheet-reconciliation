"""Step 2: booking system extracts (CSV / JSON) -> one normalised Spark DataFrame.

- Any number of CSV and JSON files in the bookings folder; format detected from the extension.
- JSON may be {"trades": [...]}, a top-level array, or JSON-lines.
- Column names are matched case/space-insensitively to the canonical booking model.
- Dates in mixed formats (ISO, US M/d/yyyy, 14-Oct-2020, 14.10.2020) -> ISO yyyy-MM-dd.
- Numbers tolerate thousands separators and '%'; lists (JSON arrays or 'A|B') -> 'A|B'.
- Malformed rows and un-parseable values are logged as data-quality issues, never silently dropped.
- The same trade present in several extracts (e.g. CSV and JSON copies) is de-duplicated when the
  normalised content is identical; conflicting copies are kept and flagged.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

log = logging.getLogger(__name__)

BOOKING_COLUMNS = [
    "TradeID",
    "ISIN",
    "Issuer",
    "Maturity",
    "Notional",
    "Coupon",
    "Currency",
    "SettlementDate",
    "DayCountFraction",
    "InterestPaymentDate",
    "IssueDate",
    "IssueAmount",
    "IssuePrice",
    "NominalAmountPerBond",
    "InterestPaymentFrequency",
    "BusinessDayConvention",
    "BusinessDayLocation",
    "AmortizationType",
    "MinimumSubscription",
    "Parent",
]
DATE_COLUMNS = ["Maturity", "SettlementDate", "InterestPaymentDate", "IssueDate"]
NUMERIC_COLUMNS = ["Notional", "Coupon", "IssueAmount", "IssuePrice", "NominalAmountPerBond", "MinimumSubscription"]
LIST_COLUMNS = ["BusinessDayLocation"]
REQUIRED_COLUMNS = ["TradeID", "ISIN", "Issuer"]
# Date formats tried in order. Slash dates are interpreted as US M/d/yyyy (assumption, see REPORT.md).
DATE_FORMATS = ["yyyy-MM-dd", "M/d/yyyy", "d-MMM-yyyy", "dd-MMM-yy", "d.M.yyyy", "yyyyMMdd", "d MMMM yyyy"]

SUPPORTED = {".csv": "csv", ".json": "json", ".jsonl": "json"}


@dataclass
class IngestResult:
    bookings: DataFrame
    files: list[str]
    dq_issues: list[dict] = field(default_factory=list)


def _canon(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_CANON_MAP = {_canon(c): c for c in BOOKING_COLUMNS}


def _rename_to_canonical(df: DataFrame, source: str, issues: list[dict]) -> DataFrame:
    for col in df.columns:
        target = _CANON_MAP.get(_canon(col))
        if target and target != col:
            df = df.withColumnRenamed(col, target)
    extra = [c for c in df.columns if c not in BOOKING_COLUMNS and not c.startswith("_")]
    if extra:
        log.info("%s: ignoring non-reconciled columns %s", source, extra)
    for col in BOOKING_COLUMNS:
        if col not in df.columns:
            sev = "ERROR" if col in REQUIRED_COLUMNS else "WARNING"
            issues.append(
                {
                    "source_file": source,
                    "row": None,
                    "column": col,
                    "severity": sev,
                    "issue": "Column missing from extract",
                }
            )
            df = df.withColumn(col, F.lit(None).cast(StringType()))
    return df


def _read_csv(spark: SparkSession, path: Path, issues: list[dict]) -> DataFrame:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh), None)
    if not header:
        raise ValueError("empty CSV file")
    schema = StructType(
        [StructField(h.strip(), StringType(), True) for h in header]
        + [StructField("_corrupt_record", StringType(), True)]
    )
    df = (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("encoding", "UTF-8")
        .schema(schema)
        .csv(str(path))
    ).cache()
    bad = df.filter(F.col("_corrupt_record").isNotNull()).select("_corrupt_record").collect()
    for r in bad:
        issues.append(
            {
                "source_file": path.name,
                "row": r["_corrupt_record"][:120],
                "column": None,
                "severity": "ERROR",
                "issue": "Malformed CSV row skipped",
            }
        )
    return df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")


def _read_json(spark: SparkSession, path: Path, issues: list[dict]) -> DataFrame:
    multiline = path.suffix.lower() == ".json"
    df = spark.read.option("multiLine", multiline).option("mode", "PERMISSIVE").json(str(path))
    if df.columns == ["_corrupt_record"]:
        raise ValueError("file is not valid JSON")
    if "_corrupt_record" in df.columns:
        df = df.cache()
        n_bad = df.filter(F.col("_corrupt_record").isNotNull()).count()
        if n_bad:
            issues.append(
                {
                    "source_file": path.name,
                    "row": None,
                    "column": None,
                    "severity": "ERROR",
                    "issue": f"{n_bad} malformed JSON record(s) skipped",
                }
            )
        df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")
    # Unwrap {"trades": [...]} (or any single array-of-struct wrapper)
    array_cols = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, ArrayType) and isinstance(f.dataType.elementType, StructType)
    ]
    if len(df.columns) == 1 and array_cols:
        df = df.select(F.explode(array_cols[0]).alias("t")).select("t.*")
    # Flatten arrays of scalars (e.g. BusinessDayLocation) and cast everything to string
    cols = []
    for f in df.schema.fields:
        if isinstance(f.dataType, ArrayType):
            cols.append(F.concat_ws("|", F.col(f.name).cast("array<string>")).alias(f.name))
        else:
            cols.append(F.col(f.name).cast(StringType()).alias(f.name))
    return df.select(*cols)


def _normalise(df: DataFrame) -> DataFrame:
    for c in BOOKING_COLUMNS:
        df = df.withColumn(c, F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))))
    for c in DATE_COLUMNS:
        parsed = F.coalesce(*[F.to_date(F.col(c), fmt) for fmt in DATE_FORMATS])
        df = df.withColumn(f"_raw_{c}", F.col(c)).withColumn(c, F.date_format(parsed, "yyyy-MM-dd"))
    for c in NUMERIC_COLUMNS:
        cleaned = F.regexp_replace(F.col(c), r"[,%\s]", "").cast("double")
        df = df.withColumn(f"_raw_{c}", F.col(c)).withColumn(c, cleaned)
    for c in LIST_COLUMNS:
        parts = F.transform(F.split(F.col(c), r"\s*[|;,]\s*"), lambda x: F.trim(x))
        df = df.withColumn(c, F.array_join(F.filter(parts, lambda x: x != ""), "|"))
    return df.withColumn("ISIN", F.upper(F.col("ISIN"))).withColumn("Currency", F.upper(F.col("Currency")))


def _collect_value_issues(df: DataFrame) -> list[dict]:
    checks = [(c, F.col(f"_raw_{c}").isNotNull() & F.col(c).isNull(), "Unparseable date") for c in DATE_COLUMNS]
    checks += [(c, F.col(f"_raw_{c}").isNotNull() & F.col(c).isNull(), "Non-numeric value") for c in NUMERIC_COLUMNS]
    checks += [(c, F.col(c).isNull(), "Required value is empty") for c in REQUIRED_COLUMNS]
    issues = []
    for col, cond, label in checks:
        raw_col = f"_raw_{col}" if f"_raw_{col}" in df.columns else col
        for r in df.filter(cond).select("source_file", "TradeID", F.col(raw_col).alias("v")).collect():
            issues.append(
                {
                    "source_file": r["source_file"],
                    "row": f"TradeID={r['TradeID']}",
                    "column": col,
                    "severity": "ERROR" if col in REQUIRED_COLUMNS else "WARNING",
                    "issue": f"{label}: {r['v']!r}",
                }
            )
    return issues


def load_bookings(spark: SparkSession, folder: Path) -> IngestResult:
    if not folder.exists():
        raise FileNotFoundError(f"Booking folder not found: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"No booking extracts (*.csv, *.json) found in {folder}")

    issues: list[dict] = []
    frames: list[DataFrame] = []
    for path in files:
        fmt = SUPPORTED[path.suffix.lower()]
        try:
            df = _read_csv(spark, path, issues) if fmt == "csv" else _read_json(spark, path, issues)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            log.error("Could not read booking file %s: %s", path.name, exc)
            issues.append(
                {
                    "source_file": path.name,
                    "row": None,
                    "column": None,
                    "severity": "ERROR",
                    "issue": f"File skipped: {exc}",
                }
            )
            continue
        df = _rename_to_canonical(df, path.name, issues)
        df = (
            df.select(*BOOKING_COLUMNS)
            .withColumn("source_file", F.lit(path.name))
            .withColumn("source_format", F.lit(fmt))
        )
        frames.append(df)
        log.info("Loaded booking extract %s (%s)", path.name, fmt)

    if not frames:
        raise RuntimeError("No booking files could be read - see data-quality issues in the log")

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.unionByName(df)
    combined = _normalise(combined).cache()
    issues.extend(_collect_value_issues(combined))

    # De-duplicate identical trades that arrive via several extracts (CSV + JSON copies)
    deduped = (
        combined.groupBy(*BOOKING_COLUMNS)
        .agg(
            F.array_join(F.array_sort(F.collect_set("source_file")), ", ").alias("source_files"),
            F.array_join(F.array_sort(F.collect_set("source_format")), ", ").alias("source_formats"),
        )
        .withColumn("booking_ref", F.concat_ws("-", F.coalesce("ISIN", F.lit("NOISIN")), F.col("TradeID")))
    )

    conflicts = deduped.groupBy("booking_ref").count().filter("count > 1").collect()
    for r in conflicts:
        issues.append(
            {
                "source_file": "*",
                "row": r["booking_ref"],
                "column": None,
                "severity": "WARNING",
                "issue": f"Trade appears with {r['count']} different versions across extracts - each reconciled separately",
            }
        )
    if conflicts:
        deduped = deduped.withColumn(
            "booking_ref",
            F.when(
                F.col("booking_ref").isin([r["booking_ref"] for r in conflicts]),
                F.concat_ws("@", "booking_ref", "source_files"),
            ).otherwise(F.col("booking_ref")),
        )

    n_raw, n_dedup = combined.count(), deduped.count()
    log.info("Bookings: %d rows from %d file(s) -> %d unique trades after de-duplication", n_raw, len(frames), n_dedup)
    for i in issues:
        log.warning(
            "Booking DQ [%s] %s %s %s: %s",
            i["severity"],
            i["source_file"],
            i["row"] or "",
            i["column"] or "",
            i["issue"],
        )
    return IngestResult(bookings=deduped.cache(), files=[p.name for p in files], dq_issues=issues)

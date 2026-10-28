"""Step 3a: field-by-field comparison of term sheet extractions against bookings, in PySpark.

Flow:
  extractions (driver, small)  --> long DF  (document, booking_column, ts_value, ts_status, quote...)
  bookings (Spark)             --> linked to a term sheet by ISIN (or issuer when the term sheet has
                                   no ISIN) --> unpivoted to long DF (booking_ref, booking_column, value)
  join on (document, booking_column) --> type-aware comparison --> initial_result per field

Initial results: MATCH | MISMATCH | PARSING_ERROR | NOT_IN_TERM_SHEET | MISSING_IN_BOOKING |
NOT_COMPARED | RULE_BREACH | NO_TERM_SHEET. MISMATCH rows go to the LLM second pass (adjudicator.py).

The comparison is deliberately strict (exact after trimming/case-folding, numeric equality within a
tolerance, ISO date equality). Formatting/unit/synonym differences therefore surface as MISMATCH and
are resolved by the LLM into FALSE ALERT vs TRUE MISMATCH, with a reason.
"""

from __future__ import annotations

import logging

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, StringType, StructField, StructType

from .booking_ingest import NUMERIC_COLUMNS
from .schema import FIELD_SPECS, FieldStatus, Kind, TermSheetExtraction

log = logging.getLogger(__name__)

EXTRACT_SCHEMA = StructType(
    [
        StructField("document", StringType()),
        StructField("extraction_method", StringType()),
        StructField("field", StringType()),
        StructField("field_label", StringType()),
        StructField("booking_column", StringType()),
        StructField("kind", StringType()),
        StructField("ts_status", StringType()),
        StructField("ts_value", StringType()),
        StructField("ts_raw_value", StringType()),
        StructField("ts_quote", StringType()),
        StructField("quote_verified", BooleanType()),
        StructField("ts_note", StringType()),
        StructField("ts_error", StringType()),
        StructField("field_order", DoubleType()),
    ]
)
DOC_KEY_SCHEMA = StructType(
    [
        StructField("document", StringType()),
        StructField("doc_key_isin", StringType()),
        StructField("doc_key_issuer", StringType()),
        StructField("doc_nominal", DoubleType()),
        StructField("doc_nominal_quote", StringType()),
        StructField("doc_min_subscription", DoubleType()),
    ]
)

NOTIONAL_FIELD = "notional_lot_check"
PEER_FIELD = "notional_peer_check"
MIN_PEERS = 3  # need at least this many trades in the same bond for a peer comparison


def _key_isin(col: Column) -> Column:
    return F.upper(F.regexp_replace(col, r"\s", ""))


def _key_issuer(col: Column) -> Column:
    return F.lower(F.regexp_replace(col, r"[^A-Za-z0-9]", ""))


def _py_key_isin(v: str | None) -> str | None:
    return "".join(v.split()).upper() if v else None


def _py_key_issuer(v: str | None) -> str | None:
    return "".join(ch for ch in v.lower() if ch.isalnum()) if v else None


def _num_to_str(col: Column) -> Column:
    """Render doubles without scientific notation or trailing zeros (15000000000, 9.25)."""
    return F.when(col.isNotNull(), F.regexp_replace(F.format_string("%.10f", col), r"\.?0+$", ""))


def extractions_to_frames(spark: SparkSession, extractions: list[TermSheetExtraction]) -> tuple[DataFrame, DataFrame]:
    rows, keys = [], []
    for ex in extractions:
        for order, spec in enumerate(FIELD_SPECS):
            f = ex.fields[spec.name]
            raw = None if f.raw_value is None else str(f.raw_value)
            rows.append(
                (
                    ex.document,
                    ex.extraction_method,
                    spec.name,
                    spec.label,
                    spec.booking_column,
                    spec.kind.value,
                    f.status.value,
                    f.canonical(),
                    raw,
                    f.quote,
                    f.quote_verified,
                    f.note,
                    f.error,
                    float(order),
                )
            )
        isin, issuer = ex.fields["isin"], ex.fields["issuer"]
        nominal, min_sub = ex.fields["nominal_amount_per_bond"], ex.fields["minimum_subscription"]
        keys.append(
            (
                ex.document,
                _py_key_isin(isin.value) if isin.status == FieldStatus.OK else None,
                _py_key_issuer(issuer.value) if issuer.status == FieldStatus.OK else None,
                float(nominal.value) if nominal.status == FieldStatus.OK else None,
                nominal.quote,
                float(min_sub.value) if min_sub.status == FieldStatus.OK else None,
            )
        )
    return spark.createDataFrame(rows, EXTRACT_SCHEMA), spark.createDataFrame(keys, DOC_KEY_SCHEMA)


def link_bookings(bookings: DataFrame, doc_keys: DataFrame) -> DataFrame:
    """Attach each booking to its term sheet: by ISIN when the term sheet has one, else by issuer."""
    b = (
        bookings.withColumn("trade_isin", F.col("ISIN"))
        .withColumn("b_key_isin", _key_isin(F.col("ISIN")))
        .withColumn("b_key_issuer", _key_issuer(F.col("Issuer")))
    )
    cond = F.when(F.col("doc_key_isin").isNotNull(), F.col("b_key_isin") == F.col("doc_key_isin")).otherwise(
        F.col("b_key_issuer") == F.col("doc_key_issuer")
    )
    linked = b.join(doc_keys, cond, "left").withColumn(
        "link_method",
        F.when(F.col("document").isNull(), F.lit("unlinked"))
        .when(F.col("doc_key_isin").isNotNull(), F.lit("ISIN"))
        .otherwise(F.lit("Issuer name")),
    )
    return linked


def _equal_expr(ts: Column, bk: Column, kind: Column, tol: float) -> Column:
    def norm_s(c: Column) -> Column:
        return F.lower(F.regexp_replace(F.trim(c), r"\s+", " "))

    def norm_list(c: Column) -> Column:
        return F.array_sort(F.transform(F.split(F.lower(c), r"\|"), lambda x: F.trim(x)))

    ts_d, bk_d = ts.cast("double"), bk.cast("double")
    numeric_eq = (
        ts_d.isNotNull() & bk_d.isNotNull() & (F.abs(ts_d - bk_d) <= F.lit(tol) * F.greatest(F.lit(1.0), F.abs(bk_d)))
    )
    return (
        F.when(kind.isin(Kind.PERCENT.value, Kind.AMOUNT.value), numeric_eq)
        .when(kind == Kind.LIST.value, norm_list(ts) == norm_list(bk))
        .when(kind.isin(Kind.DATE.value, Kind.DATE_OR_PERPETUAL.value), F.upper(ts) == F.upper(bk))
        .otherwise(norm_s(ts) == norm_s(bk))
    )


def compare(
    spark: SparkSession, bookings: DataFrame, extractions: list[TermSheetExtraction], tolerance: float = 1e-6
) -> DataFrame:
    extract_df, doc_keys = extractions_to_frames(spark, extractions)
    linked = link_bookings(bookings, doc_keys).cache()

    unlinked = linked.filter(F.col("document").isNull())
    n_unlinked = unlinked.count()
    if n_unlinked:
        log.warning("%d booking(s) could not be linked to any term sheet (no ISIN / issuer match)", n_unlinked)

    compare_cols = [s.booking_column for s in FIELD_SPECS]
    id_cols = ["booking_ref", "TradeID", "trade_isin", "source_files", "document", "link_method"]
    as_str = linked.select(
        *id_cols,
        *[(_num_to_str(F.col(c)) if c in NUMERIC_COLUMNS else F.col(c).cast("string")).alias(c) for c in compare_cols],
    )
    long_bk = as_str.filter(F.col("document").isNotNull()).unpivot(
        id_cols, compare_cols, "booking_column", "booking_value"
    )

    joined = long_bk.join(extract_df, ["document", "booking_column"], "inner")
    eq = _equal_expr(F.col("ts_value"), F.col("booking_value"), F.col("kind"), tolerance)
    detail = joined.withColumn(
        "initial_result",
        F.when(F.col("ts_status") == FieldStatus.PARSING_ERROR.value, "PARSING_ERROR")
        .when((F.col("ts_status") == FieldStatus.NOT_FOUND.value) & F.col("booking_value").isNull(), "NOT_COMPARED")
        .when(F.col("ts_status") == FieldStatus.NOT_FOUND.value, "NOT_IN_TERM_SHEET")
        .when(F.col("booking_value").isNull(), "MISSING_IN_BOOKING")
        .when(eq, "MATCH")
        .otherwise("MISMATCH"),
    )

    # Rule check: trade notional must be a positive whole multiple of the bond's denomination.
    notional = (
        linked.filter(F.col("document").isNotNull())
        .select(*id_cols, "Notional", "doc_nominal", "doc_nominal_quote", "doc_min_subscription")
        .withColumn("booking_column", F.lit("Notional"))
        .withColumn("booking_value", _num_to_str(F.col("Notional")))
        .withColumn("field", F.lit(NOTIONAL_FIELD))
        .withColumn("field_label", F.lit("Notional (lot-size rule)"))
        .withColumn("kind", F.lit("rule"))
        .withColumn(
            "ts_status",
            F.when(F.col("doc_nominal").isNull(), FieldStatus.NOT_FOUND.value).otherwise(FieldStatus.OK.value),
        )
        .withColumn(
            "ts_value",
            F.when(
                F.col("doc_nominal").isNotNull(), F.concat(F.lit("Multiple of "), _num_to_str(F.col("doc_nominal")))
            ),
        )
        .withColumn("ts_raw_value", F.col("ts_value"))
        .withColumn("ts_quote", F.col("doc_nominal_quote"))
        .withColumn("quote_verified", F.lit(None).cast("boolean"))
        .withColumn(
            "ts_note",
            F.lit(
                "Term sheets carry no trade-level notional; the booked notional is "
                "validated against the bond denomination instead."
            ),
        )
        .withColumn("ts_error", F.lit(None).cast("string"))
        .withColumn("field_order", F.lit(-1.0))
        .withColumn(
            "initial_result",
            F.when(F.col("Notional").isNull(), "MISSING_IN_BOOKING")
            .when(F.col("doc_nominal").isNull() | (F.col("doc_nominal") <= 0), "NOT_COMPARED")
            .when((F.col("Notional") > 0) & (F.pmod(F.col("Notional"), F.col("doc_nominal")) == 0), "MATCH")
            .otherwise("RULE_BREACH"),
        )
        .withColumn(
            "rule_detail",
            F.when(
                F.col("initial_result") == "RULE_BREACH",
                F.format_string(
                    "Notional %s is not a whole multiple of the %s denomination (%.4f bonds).",
                    _num_to_str(F.col("Notional")),
                    _num_to_str(F.col("doc_nominal")),
                    F.col("Notional") / F.col("doc_nominal"),
                ),
            ),
        )
        .withColumn("extraction_method", F.lit("rule"))
        .drop("Notional", "doc_nominal", "doc_nominal_quote", "doc_min_subscription")
    )

    peer = _peer_notional_check(linked, id_cols)

    no_ts = (
        unlinked.select(*id_cols)
        .withColumn("booking_column", F.lit(None).cast("string"))
        .withColumn("booking_value", F.lit(None).cast("string"))
        .withColumn("field", F.lit("term_sheet"))
        .withColumn("field_label", F.lit("Term sheet"))
        .withColumn("kind", F.lit("link"))
        .withColumn("initial_result", F.lit("NO_TERM_SHEET"))
    )

    result = (
        detail.withColumn("rule_detail", F.lit(None).cast("string"))
        .unionByName(notional, allowMissingColumns=True)
        .unionByName(peer, allowMissingColumns=True)
        .unionByName(no_ts, allowMissingColumns=True)
    )
    return result.orderBy("document", "booking_ref", "field_order")


def _peer_notional_check(linked: DataFrame, id_cols: list[str]) -> DataFrame:
    """Flag trades whose notional differs from the dominant notional booked for the same bond.

    The term sheet cannot confirm a trade size, so this is a REVIEW signal (possible fat-finger), not a
    break: it fires only when one notional is clearly dominant (a unique mode across >= MIN_PEERS trades).
    """
    base = linked.filter(F.col("document").isNotNull()).select(*id_cols, "Notional")
    counts = (
        base.filter(F.col("Notional").isNotNull()).groupBy("document", "Notional").agg(F.count("*").alias("n_same"))
    )
    w = Window.partitionBy("document").orderBy(F.desc("n_same"), F.asc("Notional"))
    mode = (
        counts.withColumn("rk", F.row_number().over(w))
        .withColumn("runner_up", F.lead("n_same").over(w))
        .filter("rk = 1")
        .select("document", F.col("Notional").alias("mode_notional"), F.col("n_same").alias("mode_count"), "runner_up")
    )
    totals = base.groupBy("document").agg(F.count("Notional").alias("peer_n"))
    unique_mode = F.col("runner_up").isNull() | (F.col("runner_up") < F.col("mode_count"))
    peer_ok = (F.col("peer_n") >= MIN_PEERS) & unique_mode

    return (
        base.join(mode, "document", "left")
        .join(totals, "document", "left")
        .withColumn("booking_column", F.lit("Notional"))
        .withColumn("booking_value", _num_to_str(F.col("Notional")))
        .withColumn("field", F.lit(PEER_FIELD))
        .withColumn("field_label", F.lit("Notional (peer consistency)"))
        .withColumn("kind", F.lit("rule"))
        .withColumn("ts_status", F.lit(FieldStatus.OK.value))
        .withColumn(
            "ts_value",
            F.when(
                peer_ok,
                F.format_string(
                    "Dominant notional %s (%d of %d trades)",
                    _num_to_str(F.col("mode_notional")),
                    F.col("mode_count"),
                    F.col("peer_n"),
                ),
            ),
        )
        .withColumn("ts_raw_value", F.col("ts_value"))
        .withColumn("ts_note", F.lit("Peer check across bookings of the same bond; not a term sheet value."))
        .withColumn("field_order", F.lit(-0.5))
        .withColumn("extraction_method", F.lit("rule"))
        .withColumn(
            "initial_result",
            F.when(F.col("Notional").isNull(), "MISSING_IN_BOOKING")
            .when(~peer_ok, "NOT_COMPARED")
            .when(F.abs(F.col("Notional") - F.col("mode_notional")) < 1e-6, "MATCH")
            .otherwise("PEER_OUTLIER"),
        )
        .withColumn(
            "rule_detail",
            F.when(
                F.col("initial_result") == "PEER_OUTLIER",
                F.format_string(
                    "Notional %s differs from the %s booked on %d of %d trades in this bond - verify against the trade ticket.",
                    _num_to_str(F.col("Notional")),
                    _num_to_str(F.col("mode_notional")),
                    F.col("mode_count"),
                    F.col("peer_n"),
                ),
            ),
        )
        .drop("Notional", "mode_notional", "mode_count", "runner_up", "peer_n")
    )

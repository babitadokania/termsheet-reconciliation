"""Step 3c: turn initial comparison + LLM verdicts into the final, business-facing classification."""

from __future__ import annotations

import pandas as pd

from .adjudicator import Verdict

MATCH, FALSE_ALERT, TRUE_MISMATCH = "Match", "False Alert", "True Mismatch"
PARSING_ERROR, NOT_IN_TS, MISSING_BK = "Parsing Error", "Not in Term Sheet", "Missing in Booking"
NOT_COMPARED, NO_TS, OUTLIER = "Not Compared", "No Term Sheet", "Outlier (Review)"

STATUS_ORDER = [TRUE_MISMATCH, PARSING_ERROR, MISSING_BK, NO_TS, OUTLIER, FALSE_ALERT, NOT_IN_TS, MATCH, NOT_COMPARED]

STATUS_HELP = {
    MATCH: "Term sheet and booking agree.",
    FALSE_ALERT: "Values differ on the surface (format, units, synonyms) but the LLM second pass confirmed they are economically identical.",
    TRUE_MISMATCH: "Genuine economic difference between the term sheet and the booking - action required.",
    PARSING_ERROR: "The value extracted from the term sheet failed strict schema validation, so it was not compared.",
    NOT_IN_TS: "The booking holds a value that the term sheet does not state; it cannot be verified from the document.",
    MISSING_BK: "The term sheet states a value but the booking field is empty.",
    NOT_COMPARED: "Neither the term sheet nor the booking holds a value.",
    NO_TS: "No term sheet could be linked to this booking (by ISIN or issuer).",
    OUTLIER: "Booking is unusual versus other trades in the same bond (e.g. notional); not provable from the term sheet - verify.",
}


def classify(detail: pd.DataFrame, verdicts: dict[tuple, Verdict]) -> pd.DataFrame:
    df = detail.copy()
    statuses, reasons, confidences, by, review = [], [], [], [], []
    for row in df.itertuples(index=False):
        r = row.initial_result
        conf, adj, needs = None, "", False
        if r == "MATCH":
            status, reason = MATCH, "Values agree."
        elif r == "MISMATCH":
            v = verdicts.get((row.document, row.field, str(row.ts_value), str(row.booking_value)))
            if v is None:
                status, reason, needs = TRUE_MISMATCH, "Not adjudicated; treated as true mismatch.", True
            else:
                status = FALSE_ALERT if v.verdict == "FALSE_ALERT" else TRUE_MISMATCH
                reason, conf, adj, needs = v.reason, v.confidence, v.adjudicated_by, v.needs_review
        elif r == "PEER_OUTLIER":
            status, reason, adj, needs = OUTLIER, row.rule_detail, "rule", True
        elif r == "RULE_BREACH":
            status, reason, adj = TRUE_MISMATCH, row.rule_detail, "rule"
        elif r == "PARSING_ERROR":
            status = PARSING_ERROR
            reason = f"Extracted value {row.ts_raw_value!r} failed schema validation: {row.ts_error}"
            needs = True
        elif r == "NOT_IN_TERM_SHEET":
            status = NOT_IN_TS
            reason = "Booking value cannot be verified - field not stated in the term sheet." + (
                f" ({row.ts_note})" if isinstance(row.ts_note, str) and row.ts_note else ""
            )
        elif r == "MISSING_IN_BOOKING":
            status, reason = MISSING_BK, "Booking field is empty but the term sheet states a value."
        elif r == "NO_TERM_SHEET":
            status, reason, needs = NO_TS, STATUS_HELP[NO_TS], True
        else:
            status, reason = NOT_COMPARED, STATUS_HELP[NOT_COMPARED]
        statuses.append(status)
        reasons.append(reason)
        confidences.append(conf)
        by.append(adj)
        review.append(needs)
    df["final_status"] = statuses
    df["reason"] = reasons
    df["llm_confidence"] = confidences
    df["adjudicated_by"] = by
    df["needs_review"] = review
    return df


def trade_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (doc, ref), g in df.groupby(["document", "booking_ref"], dropna=False, sort=True):
        counts = g["final_status"].value_counts()
        breaks = g.loc[g["final_status"] == TRUE_MISMATCH, "field_label"].tolist()
        perr = g.loc[g["final_status"] == PARSING_ERROR, "field_label"].tolist()
        if counts.get(NO_TS, 0):
            overall = "UNLINKED"
        elif breaks:
            overall = "BREAK"
        elif perr or counts.get(MISSING_BK, 0) or g["needs_review"].any():
            overall = "REVIEW"
        else:
            overall = "CLEAN"
        first = g.iloc[0]
        rows.append(
            {
                "document": doc if isinstance(doc, str) else "-",
                "booking_ref": ref,
                "TradeID": first["TradeID"],
                "ISIN": first["trade_isin"],
                "source_files": first["source_files"],
                "link_method": first["link_method"],
                "overall_status": overall,
                **{
                    s: int(counts.get(s, 0))
                    for s in [MATCH, FALSE_ALERT, TRUE_MISMATCH, PARSING_ERROR, NOT_IN_TS, MISSING_BK]
                },
                "true_mismatch_fields": ", ".join(breaks),
                "parsing_error_fields": ", ".join(perr),
            }
        )
    order = {"BREAK": 0, "UNLINKED": 1, "REVIEW": 2, "CLEAN": 3}
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["overall_status", "document", "booking_ref"],
            key=lambda s: s.map(order) if s.name == "overall_status" else s,
        )
    return out.reset_index(drop=True)

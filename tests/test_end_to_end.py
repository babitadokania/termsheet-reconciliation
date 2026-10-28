"""Full workflow on the supplied mock data with the offline provider."""

import pandas as pd

from tsrecon.config import load_settings
from tsrecon.pipeline import run


def test_full_pipeline_offline(root, tmp_path):
    settings = load_settings(llm_provider="mock", output_dir=tmp_path, use_cache=False)
    out = run(settings, "e2e.xlsx")
    assert out.exists()
    xl = pd.ExcelFile(out)
    assert {
        "Summary",
        "Trade Summary",
        "Exceptions",
        "Reconciliation Detail",
        "Term Sheet Extraction",
        "Booking Data",
        "Data Quality",
    } <= set(xl.sheet_names)

    detail = pd.read_excel(out, sheet_name="Reconciliation Detail")

    def status(ref, field):
        rows = detail[(detail["Booking Ref"] == ref) & (detail["Field"] == field)]
        return rows["Final Status"].iloc[0]

    # Planted breaks
    assert status("NO0010894330-2", "Maturity") == "True Mismatch"
    assert status("NO0010894330-4", "Coupon (%)") == "True Mismatch"
    assert status("INE008A08U84-2", "Coupon (%)") == "True Mismatch"
    assert status("INE008A08U84-4", "Issue Date") == "True Mismatch"
    assert status("INE008A08U84-3", "Notional (lot-size rule)") == "True Mismatch"
    # Formatting-only differences resolved by the second pass
    assert status("NO0010894330-1", "ISIN") == "False Alert"
    # Clean baseline trade
    trades = pd.read_excel(out, sheet_name="Trade Summary")
    trades = trades.set_index("booking_ref")
    assert trades.loc["NO0010894330-1", "overall_status"] == "CLEAN"
    # 3.1m vs 3.0m on the other four Genel trades -> flagged for review by the peer check
    assert trades.loc["NO0010894330-3", "overall_status"] == "REVIEW"
    assert "Extraction Accuracy" in xl.sheet_names
    # Every compared row carries term-sheet evidence
    compared = detail[
        detail["Final Status"].isin(["Match", "False Alert", "True Mismatch"])
        & ~detail["Field"].str.startswith("Notional (")
    ]
    assert compared["Term Sheet Evidence (verbatim)"].notna().all()

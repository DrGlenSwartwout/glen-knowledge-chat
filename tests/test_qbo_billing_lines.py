"""Tests for dashboard.qbo_billing._build_invoice_lines — the pure QBO Line
array builder, incl. the optional Wellness Credit discount line (Phase 3b)."""

import pytest


def _items():
    return [
        {"name": "A", "amount": 25.00, "qty": 40, "item_id": "11", "description": "A wholesale"},
        {"name": "B", "amount": 25.00, "qty": 10, "item_id": "22"},
    ]


def test_builds_one_sales_line_per_item_no_discount():
    from dashboard.qbo_billing import _build_invoice_lines
    lines = _build_invoice_lines(_items(), 0)
    assert len(lines) == 2
    assert all(l["DetailType"] == "SalesItemLineDetail" for l in lines)
    first = lines[0]
    assert first["Amount"] == 1000.00          # 25.00 * 40
    assert first["SalesItemLineDetail"]["ItemRef"]["value"] == "11"
    assert first["SalesItemLineDetail"]["Qty"] == 40
    assert first["SalesItemLineDetail"]["UnitPrice"] == 25.00
    assert first["Description"] == "A wholesale"
    # second line falls back to name for description
    assert lines[1]["Description"] == "B"


def test_appends_one_discount_line_when_credit_applied():
    from dashboard.qbo_billing import _build_invoice_lines
    lines = _build_invoice_lines(_items(), 14850)   # $148.50 credit
    assert len(lines) == 3
    disc = lines[-1]
    assert disc["DetailType"] == "DiscountLineDetail"
    assert disc["Amount"] == 148.50
    assert disc["DiscountLineDetail"]["PercentBased"] is False


def test_no_discount_line_when_zero():
    from dashboard.qbo_billing import _build_invoice_lines
    assert all(l["DetailType"] == "SalesItemLineDetail"
               for l in _build_invoice_lines(_items(), 0))

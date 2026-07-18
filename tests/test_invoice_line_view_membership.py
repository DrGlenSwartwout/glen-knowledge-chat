# tests/test_invoice_line_view_membership.py
import pytest
app_mod = pytest.importorskip("app")


def test_membership_line_view_passthrough():
    line = {"slug": "membership:month", "name": "Monthly Membership",
            "qty": 1, "unit_cents": 9900, "line_cents": 9900,
            "kind": "membership", "tier": "month"}
    out = app_mod._invoice_line_view(line)
    assert out["kind"] == "membership"
    assert out["tier"] == "month"
    assert out["name"] == "Monthly Membership"
    assert out["line_cents"] == 9900


def test_regular_product_line_view_has_no_membership_marker():
    # A normal product line must not sprout a membership marker.
    line = {"slug": "paracleanse", "name": "ParaCleanse", "qty": 1,
            "unit_cents": 6997, "line_cents": 6997}
    out = app_mod._invoice_line_view(line)
    assert out.get("kind") != "membership"

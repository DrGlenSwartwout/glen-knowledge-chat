"""Caregiver-pay portal UI slice.

Pins the two things this slice adds:
(a) the data the member's consent toggle reads — household.caregivers_for now
    carries pay_consent + pay_share_scope (with correct defaults);
(b) the client-portal.html wiring (member consent control, payer surface, and
    the beneficiary badge), asserted statically per the repo's render-verify
    convention for this page; plus get_portal_view exposing the gating flag.
"""

import sqlite3
from pathlib import Path


def test_caregivers_for_carries_pay_fields():
    from dashboard import household as hh
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    hh.init_household_tables(cx)
    hh.add_member(cx, "steve@ex.com", "michael@ex.com", relationship="partner")
    hh.add_member(cx, "anna@ex.com", "michael@ex.com", relationship="child")
    hh.set_pay_consent(cx, "steve@ex.com", "michael@ex.com", 1, share_scope="line_items")

    cgs = {c["primary_email"]: c for c in hh.caregivers_for(cx, "michael@ex.com")}
    # consented caregiver reflects state
    assert cgs["steve@ex.com"]["pay_consent"] == 1
    assert cgs["steve@ex.com"]["pay_share_scope"] == "line_items"
    # un-consented caregiver defaults (drives an unchecked box + amount_only)
    assert cgs["anna@ex.com"]["pay_consent"] == 0
    assert cgs["anna@ex.com"]["pay_share_scope"] == "amount_only"


def test_get_portal_view_exposes_caregiver_pay_enabled_flag():
    # The card gates on view.caregiver_pay_enabled; assert get_portal_view emits it.
    src = (Path(__file__).resolve().parent.parent / "dashboard" / "portal_view.py").read_text()
    assert '"caregiver_pay_enabled": bool(caregiver_pay_enabled)' in src


def test_client_portal_html_wires_caregiver_pay_ui():
    html = (Path(__file__).resolve().parent.parent / "static" / "client-portal.html").read_text()
    # gating flag from the /view payload
    assert "caregiver_pay_enabled" in html
    # member consent control -> POST /pay-consent
    assert "hh-pay" in html and "hh-payscope" in html
    assert "/pay-consent" in html
    assert "Caregiver payments" in html
    # payer surface -> POST /caregiver-pay
    assert "cg-pay" in html and "/caregiver-pay" in html
    assert "Orders you're paying for" in html
    # beneficiary badge (fields from _orders_block via the /view payload)
    assert "paid_by_caregiver" in html and "Paid by caregiver" in html

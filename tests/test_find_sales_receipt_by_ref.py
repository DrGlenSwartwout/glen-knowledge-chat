"""Tests for dashboard.qbo_billing.find_sales_receipt_by_ref — the exact-match
PrivateNote token lookup the heal sweep uses to decide Case B (receipt exists
-> stamp) vs Case A (no receipt -> rebook). Two paths: primary LIKE query on
PrivateNote, and a customer-scan fallback when the primary raises or comes up
empty and an email is given. Never trusts a match unless the PrivateNote
actually contains "order:<token>" client-side -- QBO's LIKE may not be
supported on PrivateNote (raises) or may return false-positive rows."""

import pytest
from dashboard import qbo_billing as qb


def _qr(receipts):
    """Wrap a list of SalesReceipt dicts in the QBO QueryResponse envelope."""
    return {"QueryResponse": {"SalesReceipt": receipts}} if receipts else {"QueryResponse": {}}


def test_primary_match_returned(monkeypatch):
    receipt = {"Id": "SR1", "PrivateNote": "order:tok1 — checkout ref"}

    def fake_query(q):
        assert "PrivateNote LIKE" in q
        assert "tok1" in q
        return _qr([receipt])

    monkeypatch.setattr(qb, "_query", fake_query)
    out = qb.find_sales_receipt_by_ref("tok1")
    assert out == receipt


def test_primary_false_hit_not_returned_and_no_email_is_none(monkeypatch):
    # LIKE query "succeeds" but returns a receipt whose PrivateNote does NOT
    # actually contain the token (e.g. QBO ignored LIKE and returned everything).
    unrelated = {"Id": "SR2", "PrivateNote": "order:othertoken"}

    def fake_query(q):
        return _qr([unrelated])

    monkeypatch.setattr(qb, "_query", fake_query)
    out = qb.find_sales_receipt_by_ref("tok1")
    assert out is None


def test_primary_raises_fallback_via_email_finds_match(monkeypatch):
    match = {"Id": "SR3", "PrivateNote": "order:tok1 stamped"}
    calls = {"query_n": 0}

    def fake_query(q):
        calls["query_n"] += 1
        if calls["query_n"] == 1:
            raise RuntimeError("QBO: LIKE not supported on PrivateNote")
        assert "CustomerRef" in q
        assert "CUST9" in q
        return _qr([match])

    def fake_find_or_create_customer(email, name=""):
        assert email == "buyer@example.com"
        return {"Id": "CUST9"}

    monkeypatch.setattr(qb, "_query", fake_query)
    monkeypatch.setattr(qb, "find_or_create_customer", fake_find_or_create_customer)
    out = qb.find_sales_receipt_by_ref("tok1", email="buyer@example.com")
    assert out == match
    assert calls["query_n"] == 2


def test_primary_empty_fallback_via_email_finds_match(monkeypatch):
    match = {"Id": "SR4", "PrivateNote": "order:tok2 stamped"}
    calls = {"query_n": 0}

    def fake_query(q):
        calls["query_n"] += 1
        if calls["query_n"] == 1:
            return _qr([])
        return _qr([match])

    def fake_find_or_create_customer(email, name=""):
        return {"Id": "CUST9"}

    monkeypatch.setattr(qb, "_query", fake_query)
    monkeypatch.setattr(qb, "find_or_create_customer", fake_find_or_create_customer)
    out = qb.find_sales_receipt_by_ref("tok2", email="buyer@example.com")
    assert out == match


def test_since_date_included_in_fallback_query(monkeypatch):
    match = {"Id": "SR5", "PrivateNote": "order:tok3"}
    calls = {"query_n": 0, "second_q": None}

    def fake_query(q):
        calls["query_n"] += 1
        if calls["query_n"] == 1:
            return _qr([])
        calls["second_q"] = q
        return _qr([match])

    monkeypatch.setattr(qb, "_query", fake_query)
    monkeypatch.setattr(qb, "find_or_create_customer", lambda email, name="": {"Id": "CUST9"})
    out = qb.find_sales_receipt_by_ref("tok3", email="buyer@example.com",
                                       since_date="2026-06-01")
    assert out == match
    assert "TxnDate >= '2026-06-01'" in calls["second_q"]
    assert "ORDERBY TxnDate DESC MAXRESULTS 50" in calls["second_q"]


def test_no_match_anywhere_returns_none(monkeypatch):
    def fake_query(q):
        return _qr([])

    monkeypatch.setattr(qb, "_query", fake_query)
    out = qb.find_sales_receipt_by_ref("tok4")
    assert out is None


def test_no_email_and_primary_empty_returns_none_without_fallback(monkeypatch):
    def fake_query(q):
        return _qr([])

    def boom_customer(*a, **k):
        raise AssertionError("find_or_create_customer must not be called without an email")

    monkeypatch.setattr(qb, "_query", fake_query)
    monkeypatch.setattr(qb, "find_or_create_customer", boom_customer)
    out = qb.find_sales_receipt_by_ref("tok5")
    assert out is None


def test_exact_match_discipline_same_amount_wrong_privatenote_never_returned(monkeypatch):
    # Same TotalAmt as a "real" match would have, but PrivateNote is absent/wrong --
    # must never be returned on amount alone.
    wrong_note = {"Id": "SR6", "TotalAmt": 42.00, "PrivateNote": "no ref here"}
    no_note = {"Id": "SR7", "TotalAmt": 42.00}

    def fake_query_primary(q):
        return _qr([wrong_note, no_note])

    monkeypatch.setattr(qb, "_query", fake_query_primary)
    out = qb.find_sales_receipt_by_ref("tok6")
    assert out is None

    # Same discipline applies on the fallback path.
    calls = {"query_n": 0}

    def fake_query_fallback(q):
        calls["query_n"] += 1
        if calls["query_n"] == 1:
            raise RuntimeError("no LIKE support")
        return _qr([wrong_note, no_note])

    monkeypatch.setattr(qb, "_query", fake_query_fallback)
    monkeypatch.setattr(qb, "find_or_create_customer", lambda email, name="": {"Id": "CUST1"})
    out2 = qb.find_sales_receipt_by_ref("tok6", email="buyer@example.com")
    assert out2 is None

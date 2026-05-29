"""Tests for dashboard.ghl.find_contact_by_name — recipient name -> contact email."""

import types

import dashboard.ghl as ghl


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fake_requests(payload):
    return types.SimpleNamespace(get=lambda *a, **k: _Resp(payload))


def test_exact_single_match_is_high_confidence(monkeypatch):
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": [
        {"id": "c1", "contactName": "Cyndi O'Brien", "email": "cyndi@example.com"},
    ]}))
    m = ghl.find_contact_by_name("Cyndi O'Brien")
    assert m["email"] == "cyndi@example.com"
    assert m["contact_id"] == "c1"
    assert m["confidence"] == "high"


def test_single_nonexact_match_is_medium(monkeypatch):
    # GHL returned one contact but the stored name differs (nickname/maiden).
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": [
        {"id": "c2", "firstName": "Cynthia", "lastName": "OBrien",
         "email": "cynthia@example.com"},
    ]}))
    m = ghl.find_contact_by_name("Cyndi O'Brien")
    assert m["confidence"] == "medium"
    assert m["email"] == "cynthia@example.com"


def test_unique_exact_among_fuzzy_results_is_high(monkeypatch):
    # Search returns fuzzy extras, but exactly one is an exact full-name match.
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": [
        {"id": "c3", "contactName": "Frank Other", "email": "other@example.com"},
        {"id": "c4", "contactName": "Frank Swartwout", "email": "frank@example.com"},
    ]}))
    m = ghl.find_contact_by_name("Frank Swartwout")
    assert m["email"] == "frank@example.com"
    assert m["confidence"] == "high"


def test_two_people_same_name_is_low(monkeypatch):
    # Genuine ambiguity: two exact-name matches -> flag for human review.
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": [
        {"id": "c6", "contactName": "Lotika Savant", "email": "lotika1@example.com"},
        {"id": "c7", "contactName": "Lotika Savant", "email": "lotika2@example.com"},
    ]}))
    m = ghl.find_contact_by_name("Lotika Savant")
    assert m["confidence"] == "low"


def test_no_email_candidates_returns_none(monkeypatch):
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": [
        {"id": "c5", "contactName": "Ghost Lead"},  # no email
    ]}))
    assert ghl.find_contact_by_name("Ghost Lead") is None


def test_empty_name_returns_none(monkeypatch):
    monkeypatch.setattr(ghl, "requests", _fake_requests({"contacts": []}))
    assert ghl.find_contact_by_name("") is None
    assert ghl.find_contact_by_name("   ") is None


def test_api_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(ghl, "requests", types.SimpleNamespace(get=boom))
    assert ghl.find_contact_by_name("Anyone") is None

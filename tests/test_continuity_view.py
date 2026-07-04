"""Task 6: continuity_view.patient_view — gate-first per-patient continuity view.

The security keystone: patient_view() must call the authorization gate FIRST and
return None for a patient who is NOT the doctor's, having read NO patient data
(no scan trajectory, no biofield narrative, no reorder/portal build). The
authorized path assembles trajectory + narrative + suggested_step by REUSING the
existing scan_analysis / biofield narrative / portal-publish engines.
"""
import sqlite3

import pytest

from dashboard import continuity_view as cv, subscriptions as subs
from dashboard import scan_analysis as scan_mod
from dashboard import biofield_narrative as narr_mod
from dashboard import biofield_portal_publish as portal_mod
from dashboard import biofield_authoring as auth_mod


def _cx():
    """Real subscriptions init + migrate chain (mirrors the sibling continuity
    tests) so attribution + consent columns exist; plus the scan_analyses table."""
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx)
    scan_mod.init_table(cx)
    return cx


def _consented_member(cx, email, pid):
    sid = subs.create_membership(
        cx, email=email, stripe_customer_id="c", stripe_payment_method_id="p",
        amount_cents=9900, next_charge_date="2026-08-01", attributed_practitioner_id=pid,
    )
    cx.execute("UPDATE subscriptions SET practitioner_share_consent=1 WHERE id=?", (sid,))
    cx.commit()
    return sid


# ---------------------------------------------------------------------------
# THE CRITICAL SECURITY TEST: gate-first, no patient data read when unauthorized
# ---------------------------------------------------------------------------
def test_patient_view_denies_and_reads_no_patient_data_when_unauthorized(monkeypatch):
    """pat@x.com is NOT prac-99's patient. patient_view must return None AND must
    NOT read any patient data: scan_analysis.get, the biofield narrative read, the
    latest-test lookup, and the portal build are all spied and asserted UNCALLED."""
    cx = _cx()
    # pat@x.com is a consented patient of a DIFFERENT doctor (prac-1), not prac-99.
    _consented_member(cx, "pat@x.com", "prac-1")

    calls = []

    def _spy(name, ret=None):
        def f(*a, **k):
            calls.append(name)
            return ret
        return f

    monkeypatch.setattr(scan_mod, "get", _spy("scan.get"))
    monkeypatch.setattr(narr_mod, "get_narrative", _spy("narrative", ""))
    monkeypatch.setattr(portal_mod, "build_portal_content", _spy("portal", {}))
    monkeypatch.setattr(cv, "latest_biofield_test_id", _spy("latest_test", None))

    result = cv.patient_view(cx, "prac-99", "pat@x.com")

    assert result is None
    assert calls == [], f"unauthorized path read patient data: {calls}"


# ---------------------------------------------------------------------------
# Authorized assembly: trajectory + narrative + suggested_step all present
# ---------------------------------------------------------------------------
def test_patient_view_assembles_trajectory_narrative_and_suggested_step(monkeypatch):
    cx = _cx()
    _consented_member(cx, "pat@x.com", "prac-7")

    # Real trajectory artifact via the scan_analysis engine.
    scan_mod.upsert(cx, "pat@x.com", {
        "scan_count": 3, "date_range": ["2026-01-01", "2026-06-01"],
        "generated_at": "2026-06-02", "summary": "improving",
    })
    # Real biofield test for this email (maps by the auth-tests email column).
    tid = auth_mod.create_test(cx, "Pat Example", "pat@x.com", "2026-06-01")
    auth_mod.add_chain_row(cx, tid, 1, "Surface", "field", "Neuro-Magnesium",
                           dosage="1 cap", frequency="daily", timing="am")

    # Deterministic narrative + portal deps (their internals are covered elsewhere).
    monkeypatch.setattr(narr_mod, "get_narrative",
                        lambda cx_, t: "Aloha Pat, here is what changed.")
    seen = {}

    def _fake_portal(cx_, test_id, *, special_price_cents, catalog=None, **k):
        seen["test_id"] = test_id
        seen["price"] = special_price_cents
        return {"content": {"reorder_items": [
            {"slug": "neuro-magnesium", "qty": 1, "price_cents": 7000}]}}

    monkeypatch.setattr(portal_mod, "build_portal_content", _fake_portal)

    view = cv.patient_view(cx, "prac-7", "pat@x.com")

    assert view is not None
    # trajectory came from the real scan_analysis engine
    assert view["trajectory"]["scan_count"] == 3
    assert view["trajectory"]["analysis"]["summary"] == "improving"
    # narrative = the latest-vs-prior "what changed" read
    assert "what changed" in view["narrative"]
    # suggested_step = reorder_items from the portal builder
    assert view["suggested_step"] == [
        {"slug": "neuro-magnesium", "qty": 1, "price_cents": 7000}]
    # wired the REAL latest test id into the portal builder
    assert seen["test_id"] == tid


def test_patient_view_degrades_when_patient_has_no_scans_or_biofield(monkeypatch):
    """An authorized patient with NO scan analysis and NO biofield test must not
    crash: empty trajectory, empty narrative, empty suggested_step."""
    cx = _cx()
    _consented_member(cx, "fresh@x.com", "prac-7")

    # No portal build should ever happen with no test; guard it.
    monkeypatch.setattr(portal_mod, "build_portal_content",
                        lambda *a, **k: pytest.fail("portal built with no biofield test"))

    view = cv.patient_view(cx, "prac-7", "fresh@x.com")

    assert view is not None
    assert view["trajectory"] in ({}, None) or view["trajectory"] == {}
    assert view["narrative"] == ""
    assert view["suggested_step"] == []


def test_latest_biofield_test_id_returns_most_recent_then_none(monkeypatch):
    cx = _cx()
    assert cv.latest_biofield_test_id(cx, "nobody@x.com") is None
    t1 = auth_mod.create_test(cx, "P", "who@x.com", "2026-01-01")
    t2 = auth_mod.create_test(cx, "P", "who@x.com", "2026-05-01")
    # most recent == highest id (creation order, matching list_authored)
    assert cv.latest_biofield_test_id(cx, "who@x.com") == t2
    assert t2 != t1
    # case-insensitive email match
    assert cv.latest_biofield_test_id(cx, "WHO@x.com") == t2

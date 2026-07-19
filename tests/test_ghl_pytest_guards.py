"""The GHL write paths must be inert under pytest.

Both of these reach out to Glen's LIVE CRM: dashboard.ghl_email.send_via_ghl upserts a real
contact and then sends a real email, and app.ghl_upsert_contact creates/mutates a real
contact (five callers). Until these guards existed they were safe only by accident -- GHL
credentials happen to be absent from the dev config, so both short-circuited on their
"not configured" branch. Adding creds there would have made a full-suite run write to the
CRM and email real people.

So each test below supplies fake credentials, which is the ONLY state in which the guard is
load-bearing, and asserts no HTTP call is made. Without the guard these fail (they attempt a
live request); with it they pass.
"""
import os

import pytest


def _explode(*a, **kw):  # any HTTP call here means the guard did not hold
    raise AssertionError("a live HTTP request was attempted under pytest")


def test_send_via_ghl_is_a_noop_under_pytest(monkeypatch):
    from dashboard import ghl_email

    # Fake creds so is_configured() is True -- without this the function raises on the
    # not-configured branch and never exercises the guard at all.
    monkeypatch.setenv("GHL_PIT", "fake-pit")
    monkeypatch.setenv("GHL_LOCATION_ID", "fake-location")
    monkeypatch.setattr(ghl_email.requests, "post", _explode)
    assert ghl_email.is_configured() is True
    assert os.environ.get("PYTEST_CURRENT_TEST")  # sanity: pytest really does set this

    res = ghl_email.send_via_ghl("someone@example.com", "Subject", text="body")

    assert res["skipped"] == "pytest"
    assert res["id"] is None
    assert res["via"] == "ghl"  # keeps the shape callers destructure


def test_ghl_upsert_contact_is_a_noop_under_pytest(monkeypatch):
    # Skipped automatically if app fails to import (repo convention -- app.py makes a live
    # Pinecone call at import, so this runs locally under real secrets, not in secretless
    # CI). The send_via_ghl guard above needs no such skip: it imports only dashboard.
    try:
        import app as appmod
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"app not importable: {e}")

    # A truthy key is what makes this dangerous; without it the function returns early.
    monkeypatch.setattr(appmod, "GHL_API_KEY", "fake-key")
    # app.py reaches GHL through these curl/subprocess helpers, not `requests` (the
    # LeadConnector edge 403s python-urllib), so these are the network doors to block.
    monkeypatch.setattr(appmod, "_ghl_get", _explode)
    monkeypatch.setattr(appmod, "_ghl_post", _explode)
    monkeypatch.setattr(appmod, "_ghl_put", _explode)

    contact_id, created, err = appmod.ghl_upsert_contact("someone@example.com")

    assert contact_id is None
    assert created is False
    assert err == "skipped: pytest"


def test_upsert_contact_stays_private_and_guarded():
    # _upsert_contact is a live CRM write with no guard of its own; it is safe only because
    # send_via_ghl (which is guarded) is its sole caller. If someone adds another caller,
    # this is the reminder to guard it there.
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "dashboard" / "ghl_email.py"
    text = src.read_text()
    assert text.count("_upsert_contact(") == 2  # the def, plus the one call in send_via_ghl

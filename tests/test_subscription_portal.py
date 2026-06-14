"""Task 5 — Manage-plan portal: /subscription page + /api/subscription/* routes.

Three tests:
1. 401 without cookie — all endpoints return 401.
2. Details + skip + pause + cancel mutate correctly (real LOG_DB subscription row).
3. Cross-account guard — a user cannot act on another email's subscription (403/404).
"""
import sqlite3
import app as appmod
from dashboard import subscriptions as subs

_TEST_EMAILS = ("a@x.com", "b@x.com")


def _cleanup(cx):
    """Remove test rows so the real DB is not polluted."""
    for em in _TEST_EMAILS:
        cx.execute("DELETE FROM subscriptions WHERE email = ?", (em,))
    cx.commit()


def _init(cx):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    _cleanup(cx)


# ── Test 1: 401 without cookie ─────────────────────────────────────────────────

def test_portal_requires_auth_cookie(monkeypatch):
    """Every portal endpoint returns 401 when no cookie is present."""
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "")
    c = appmod.app.test_client()

    assert c.get("/api/subscription/details").status_code == 401
    assert c.post("/api/subscription/skip", json={"id": 1}).status_code == 401
    assert c.post("/api/subscription/resume-skip", json={"id": 1}).status_code == 401
    assert c.post("/api/subscription/pause", json={"id": 1}).status_code == 401
    assert c.post("/api/subscription/resume", json={"id": 1}).status_code == 401
    assert c.post("/api/subscription/cancel", json={"id": 1}).status_code == 401
    assert c.post("/api/subscription/cadence", json={"id": 1, "cadence_months": 1}).status_code == 401


# ── Test 2: details + skip + pause + cancel mutate correctly ───────────────────

def test_portal_actions(monkeypatch):
    """Details returns the subscription; skip/pause/cancel mutate the row."""
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    _init(cx)

    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c",
                      stripe_payment_method_id="p",
                      items=[{"slug": "x", "qty": 1}], cadence_months=1,
                      ship_address={}, next_charge_date="2030-01-01")
    cx.commit()

    try:
        c = appmod.app.test_client()

        # details
        d = c.get("/api/subscription/details").get_json()
        assert d["subscriptions"][0]["id"] == sid

        # skip
        r = c.post("/api/subscription/skip", json={"id": sid})
        assert r.status_code == 200
        assert subs.get(cx, sid)["skip_next"] == 1

        # resume-skip
        r = c.post("/api/subscription/resume-skip", json={"id": sid})
        assert r.status_code == 200
        assert subs.get(cx, sid)["skip_next"] == 0

        # cadence change
        r = c.post("/api/subscription/cadence", json={"id": sid, "cadence_months": 3})
        assert r.status_code == 200
        assert subs.get(cx, sid)["cadence_months"] == 3

        # pause
        r = c.post("/api/subscription/pause", json={"id": sid})
        assert r.status_code == 200
        assert subs.get(cx, sid)["status"] == "paused"

        # resume
        r = c.post("/api/subscription/resume", json={"id": sid})
        assert r.status_code == 200
        assert subs.get(cx, sid)["status"] == "active"

        # cancel
        r = c.post("/api/subscription/cancel", json={"id": sid})
        assert r.status_code == 200
        assert subs.get(cx, sid)["status"] == "cancelled"

    finally:
        _cleanup(cx)
        cx.close()


# ── Test 3: cross-account guard ────────────────────────────────────────────────

def test_cross_account_guard(monkeypatch):
    """User b@x.com cannot mutate a@x.com's subscription — must get 403 or 404,
    and the target subscription must be unchanged."""
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "b@x.com")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    _init(cx)

    # create subscription owned by a@x.com
    sid = subs.create(cx, email="a@x.com", stripe_customer_id="c",
                      stripe_payment_method_id="p",
                      items=[{"slug": "x", "qty": 1}], cadence_months=1,
                      ship_address={}, next_charge_date="2030-01-01")
    cx.commit()

    try:
        c = appmod.app.test_client()

        # details for b@x.com should not expose a@x.com's subscription
        d = c.get("/api/subscription/details").get_json()
        assert all(s["email"] != "a@x.com" for s in d.get("subscriptions", []))

        # every mutating action should be rejected with 403 or 404
        for endpoint, payload in [
            ("/api/subscription/skip", {"id": sid}),
            ("/api/subscription/resume-skip", {"id": sid}),
            ("/api/subscription/pause", {"id": sid}),
            ("/api/subscription/resume", {"id": sid}),
            ("/api/subscription/cancel", {"id": sid}),
            ("/api/subscription/cadence", {"id": sid, "cadence_months": 2}),
        ]:
            r = c.post(endpoint, json=payload)
            assert r.status_code in (403, 404), (
                f"{endpoint} returned {r.status_code} — expected 403 or 404"
            )

        # target subscription must be unchanged
        s = subs.get(cx, sid)
        assert s["status"] == "active"
        assert s["skip_next"] == 0
        assert s["cadence_months"] == 1

    finally:
        _cleanup(cx)
        cx.close()

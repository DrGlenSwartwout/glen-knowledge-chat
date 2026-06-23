"""Route-level gating for /member/scan-analysis (SP2).

Seeds the real LOG_DB (mirrors test_membership_reminder_cancel). The pure
tier logic is covered in test_scan_analysis_view; here we verify the route
wires cookie->email->tier->payload and emits no-store + the right access level.
"""
import sqlite3

import app as appmod
from dashboard import scan_analysis as sa

EMAIL = "scan-page-test@example.com"

ART = {
    "email": EMAIL, "scan_count": 7, "date_range": ["2024-01-01", "2026-06-01"],
    "narrative": "Over time, your Calm Mind pattern recurs.",
    "top_patterns": [{"code": "MR2", "name": "Calm Mind", "pct": 0.4, "color": "purple"}],
    "clusters": [{"structure": "Nervous System", "codes": ["MR2", "ED4"],
                  "color": "purple", "code_count": 2}],
    "functional_relation": [{"structure": "Detoxification", "pct": 0.2,
                             "color": "green", "is_functional": True}],
}


def _seed(paid: bool):
    cx = sqlite3.connect(appmod.LOG_DB)
    sa.init_table(cx)
    cx.execute("DELETE FROM scan_analyses WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM memberships WHERE email=?", (EMAIL,))
    sa.upsert(cx, EMAIL, ART)
    if paid:
        appmod._grant_membership(cx, EMAIL, 31, "test")
    cx.commit()
    cx.close()


def _cleanup():
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.execute("DELETE FROM scan_analyses WHERE email=?", (EMAIL,))
    cx.execute("DELETE FROM memberships WHERE email=?", (EMAIL,))
    cx.commit()
    cx.close()


def test_paid_member_sees_full_analysis(monkeypatch):
    monkeypatch.delenv("SCAN_ANALYSIS_FREE", raising=False)
    _seed(paid=True)
    try:
        c = appmod.app.test_client()
        c.set_cookie("rm_member_email", EMAIL)
        r = c.get("/member/scan-analysis")
        assert r.status_code == 200
        assert r.headers.get("Cache-Control", "").startswith("no-cache")
        body = r.get_data(as_text=True)
        assert '"access": "full"' in body
        assert "Nervous System" in body          # cluster reached the payload
        assert "Calm Mind pattern recurs" in body  # narrative present
    finally:
        _cleanup()


class _FakeMsg:
    def __init__(self, text):
        self.content = [type("B", (), {"text": text})()]


def _capture_cl(monkeypatch):
    """Stub _cl.messages.create; capture the system prompt it was called with."""
    seen = {}

    def fake_create(*a, **k):
        seen["system"] = k.get("system", "")
        return _FakeMsg("Here is what we see. This is education, not a promise.")

    monkeypatch.setattr(appmod._cl, "messages",
                        type("M", (), {"create": staticmethod(fake_create)})())
    return seen


def test_chat_grounded_for_paid_member(monkeypatch):
    monkeypatch.delenv("SCAN_ANALYSIS_FREE", raising=False)
    seen = _capture_cl(monkeypatch)
    _seed(paid=True)
    try:
        c = appmod.app.test_client()
        c.set_cookie("rm_member_email", EMAIL)
        r = c.post("/member/scan-analysis/chat", json={"query": "What stands out over time?"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["access"] == "full" and body["upsell"] is False
        assert "education" in body["answer"]
        # the member's own facts were injected into the system prompt
        assert "THE MEMBER'S ANALYSIS FACTS" in seen["system"]
        assert "Nervous System" in seen["system"]
    finally:
        _cleanup()


def test_chat_is_educate_only_and_upsell_for_anonymous(monkeypatch):
    seen = _capture_cl(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/member/scan-analysis/chat", json={"query": "What should I take?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["access"] == "locked" and body["upsell"] is True
    # no personal facts; the educate-only consent gate is applied instead
    assert "THE MEMBER'S ANALYSIS FACTS" not in seen["system"]
    assert "CONSENT GATE" in seen["system"]


def test_chat_requires_query(monkeypatch):
    _capture_cl(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/member/scan-analysis/chat", json={})
    assert r.status_code == 400


def test_anonymous_visitor_is_locked():
    c = appmod.app.test_client()
    r = c.get("/member/scan-analysis")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert '"access": "locked"' in body
    # nothing sensitive leaks into a locked page
    assert "Calm Mind pattern recurs" not in body


def test_free_flag_promotes_anonymous_nothing_but_paid_still_full(monkeypatch):
    """Flag on does NOT unlock a no-ToS visitor (still locked), but a paid member
    stays full. (Free/ToS->full transition is covered by the unit test.)"""
    monkeypatch.setenv("SCAN_ANALYSIS_FREE", "true")
    _seed(paid=True)
    try:
        c = appmod.app.test_client()
        c.set_cookie("rm_member_email", EMAIL)
        assert '"access": "full"' in c.get("/member/scan-analysis").get_data(as_text=True)
        c2 = appmod.app.test_client()
        assert '"access": "locked"' in c2.get("/member/scan-analysis").get_data(as_text=True)
    finally:
        _cleanup()

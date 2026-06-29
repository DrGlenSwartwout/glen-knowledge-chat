# tests/test_begin_doorway_routes.py
import json, time
import pytest


@pytest.fixture
def appmod(monkeypatch):
    import app as appmod
    calls = {"unlocks": [], "ghl": []}

    def fake_record_unlock(cx, **kw):
        calls["unlocks"].append(kw)
        return {"current_rung": "assess", "email": kw.get("email", "")}

    def fake_ghl(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
        calls["ghl"].append({"email": email, "source_tag": source_tag, "tags": extra_tags or []})
        return {"contact_id": "c1"}

    monkeypatch.setattr(appmod.begin_funnel, "record_unlock", fake_record_unlock)
    monkeypatch.setattr(appmod, "ghl_onboard_contact", fake_ghl)
    monkeypatch.setattr(appmod, "_log_inbound_lead", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_capture_concierge_referral", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_mint_lead_magnet_guide_link", lambda email: "tok123")
    appmod._test_calls = calls
    return appmod


def test_doorway_optin_requires_email(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={"tos": True})
    assert r.status_code == 400


def test_doorway_optin_requires_tos(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={"email": "a@b.com", "tos": False})
    assert r.status_code == 400


def test_doorway_optin_captures_and_records_gates(appmod):
    c = appmod.app.test_client()
    r = c.post("/begin/doorway/opt-in", json={
        "name": "Jane Doe", "email": "Jane@B.com", "tos": True,
        "signals": {"dominant_element": "Water", "top_themes": ["grounding"]},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["guide_token"] == "tok123"
    triggers = [u["trigger"] for u in appmod._test_calls["unlocks"]]
    assert "tos" in triggers and "quiz" in triggers
    # the GHL onboard runs in a daemon thread; allow it to flush
    for _ in range(50):
        if appmod._test_calls["ghl"]:
            break
        time.sleep(0.05)
    assert appmod._test_calls["ghl"], "ghl_onboard_contact should be called"
    g = appmod._test_calls["ghl"][0]
    assert g["source_tag"] == "source:voice"
    assert "voice-doorway" in g["tags"] and "element:water" in g["tags"]


def test_no_scoreapp_url_outside_dormant_webhook():
    import re, pathlib
    src = pathlib.Path("app.py").read_text()
    # allow the route name/handler to mention scoreapp, but no built URLs
    bad = [ln for ln in src.splitlines()
           if "healing.scoreapp.com" in ln]
    assert bad == [], f"scoreapp URLs still present: {bad}"

import json
import importlib


def _client(tmp_path, monkeypatch):
    # Point DATA_DIR at a fresh tmp dir and reload so _PRICING_SETTINGS_PATH (which honors
    # the env DATA_DIR) resolves under tmp_path -- each test is isolated and never touches
    # the real persistent settings file.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app


def test_pricing_settings_accessor_live_reloads(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    assert _app._pricing_settings() == {}
    assert _app._rewards_settings() == {}
    path = _app._PRICING_SETTINGS_PATH
    path.write_text(json.dumps({"discount_floor_pct": 0.50,
                                "rewards": {"referral_reward_pct": 0.08}}))
    import os, time
    os.utime(path, (time.time() + 1, time.time() + 1))
    assert _app._pricing_settings()["discount_floor_pct"] == 0.50
    assert _app._rewards_settings()["referral_reward_pct"] == 0.08


def _key(_app):
    return _app.CONSOLE_SECRET or ""


def test_get_requires_console_key(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    if not _app.CONSOLE_SECRET:
        return  # auth is a no-op when unset in this env; nothing to assert
    c = _app.app.test_client()
    assert c.get("/api/console/pricing-settings").status_code == 401


def test_get_returns_defaults_when_no_file(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/api/console/pricing-settings", headers={"X-Console-Key": _key(_app)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["saved"] == {}
    assert body["effective"]["discount_floor_pct"] == 0.57
    assert body["defaults"]["rewards"]["cash_out_threshold_cents"] == 10000


def test_post_persists_and_live_applies(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    payload = {"discount_floor_pct": 0.55,
               "volume_anchors": [[1, 0], [3, 15], [6, 30], [12, 45]],
               "rewards": {"referral_reward_pct": 0.07, "cash_out_threshold_cents": 12000,
                           "cash_out_face_pct": 0.70}}
    r = c.post("/api/console/pricing-settings",
               headers={"X-Console-Key": _key(_app), "Content-Type": "application/json"},
               data=json.dumps(payload))
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["saved"]["discount_floor_pct"] == 0.55
    assert _app._PRICING_SETTINGS_PATH.exists()
    assert _app._pricing_settings()["discount_floor_pct"] == 0.55
    assert _app._rewards_settings()["referral_reward_pct"] == 0.07


def test_post_rejects_invalid(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.post("/api/console/pricing-settings",
               headers={"X-Console-Key": _key(_app), "Content-Type": "application/json"},
               data=json.dumps({"discount_floor_pct": 9.9}))
    assert r.status_code == 400
    assert any("discount_floor_pct" in e for e in r.get_json()["errors"])
    assert not _app._PRICING_SETTINGS_PATH.exists()


def test_console_page_served_no_store(tmp_path, monkeypatch):
    _app = _client(tmp_path, monkeypatch)
    c = _app.app.test_client()
    r = c.get("/console/pricing-settings")
    assert r.status_code == 200
    assert b"Pricing" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")

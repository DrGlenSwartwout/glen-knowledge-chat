import json
import importlib


def _client(tmp_path, monkeypatch):
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

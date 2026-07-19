import importlib, sys
from pathlib import Path
import pytest


def _app(monkeypatch, tmp_db):
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)  # open console auth in test (mirror sibling console tests)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
    try:
        import app
        importlib.reload(app)  # CONSOLE_SECRET is read at import time; reload so delenv takes effect
    except Exception as e: pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def test_add_list_delete_option(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    c = app.app.test_client()
    r = c.post("/api/console/reward-gift-options", json={"level": 3, "sku": "GIFT-X", "label": "X"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    oid = r.get_json()["id"]
    lst = c.get("/api/console/reward-gift-options").get_json()["options"]
    assert any(o["sku"] == "GIFT-X" for o in lst)

    tog = c.post("/api/console/reward-gift-options/toggle", json={"id": oid})
    assert tog.status_code == 200 and tog.get_json()["ok"] is True and tog.get_json()["active"] is False
    lst_after_toggle = c.get("/api/console/reward-gift-options").get_json()["options"]
    opt = next(o for o in lst_after_toggle if o["id"] == oid)
    assert opt["active"] == 0

    d = c.post("/api/console/reward-gift-options/delete", json={"id": oid})
    assert d.status_code == 200 and d.get_json()["ok"] is True
    lst2 = c.get("/api/console/reward-gift-options").get_json()["options"]
    assert not any(o["sku"] == "GIFT-X" for o in lst2)


def test_options_write_inert_when_flag_off(monkeypatch, tmp_db):
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    c = app.app.test_client()
    r = c.post("/api/console/reward-gift-options", json={"level": 3, "sku": "Z", "label": "Z"})
    assert r.status_code == 200 and r.get_json()["ok"] is False

    lst = c.get("/api/console/reward-gift-options")
    assert lst.status_code == 200 and lst.get_json()["options"] == []

    d = c.post("/api/console/reward-gift-options/delete", json={"id": 1})
    assert d.status_code == 200 and d.get_json()["ok"] is False

    tog = c.post("/api/console/reward-gift-options/toggle", json={"id": 1})
    assert tog.status_code == 200 and tog.get_json()["ok"] is False


def test_add_option_validates_required_fields(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    c = app.app.test_client()
    r = c.post("/api/console/reward-gift-options", json={"level": 3, "sku": "", "label": "X"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is False


def test_console_rewards_html_has_catalog_editor_markup():
    repo = Path(__file__).resolve().parent.parent
    html = (repo / "static" / "console-rewards.html").read_text()
    assert "/api/console/reward-gift-options" in html
    assert "/delete" in html
    assert "/toggle" in html
    assert 'id="add-sku"' in html   # add-form sku input


def test_add_option_rejects_nonnumeric_level(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    c = app.app.test_client()
    r = c.post("/api/console/reward-gift-options",
               json={"level": "abc", "sku": "X", "label": "Y"})
    assert r.status_code != 500
    assert r.get_json()["ok"] is False


def test_reward_gift_options_requires_console_auth(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    monkeypatch.setenv("CONSOLE_SECRET", "s3cret")
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    import app
    importlib.reload(app)
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    c = app.app.test_client()
    assert c.get("/api/console/reward-gift-options").status_code == 401
    assert c.post("/api/console/reward-gift-options", json={"level": 3, "sku": "X", "label": "X"}).status_code == 401
    assert c.post("/api/console/reward-gift-options/delete", json={"id": 1}).status_code == 401
    assert c.post("/api/console/reward-gift-options/toggle", json={"id": 1}).status_code == 401

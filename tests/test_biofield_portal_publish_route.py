import sqlite3
import biofield_local_app
from dashboard import biofield_portal_publish as bpp
from dashboard.biofield_authoring import create_test, add_chain_row


def _client(tmp_path):
    db = str(tmp_path / "t.db")
    cx = sqlite3.connect(db)
    tid = create_test(cx, "Karin", "k@example.com", "2026-06-25")
    aid = f"a{tid}"
    add_chain_row(cx, aid, layer=1, head="ED3", most_affected="Circ",
                  remedy="Vitality", dosage="1 cap", frequency="daily", timing="")
    cx.commit(); cx.close()
    app = biofield_local_app.create_app(db_path=db)
    return app.test_client(), aid


def test_publish_route_success(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {"vitality": {"name": "Vitality"}})
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: {"ok": True,
                                               "url": "https://illtowell.com/portal/xyz"})
    monkeypatch.setenv("PORTAL_PUBLISH_BASE_URL", "https://illtowell.com")
    monkeypatch.setenv("CONSOLE_SECRET", "")     # gate open in tests
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["url"] == "https://illtowell.com/portal/xyz"


def test_publish_route_409_on_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(bpp, "load_catalog", lambda: {})   # nothing resolves
    called = {"n": 0}
    monkeypatch.setattr(bpp, "publish_to_portal",
                        lambda payload, **kw: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setenv("CONSOLE_SECRET", "")
    client, aid = _client(tmp_path)
    r = client.post(f"/test/{aid}/publish-portal", json={"special_price_cents": 5000})
    assert r.status_code == 409
    assert r.get_json()["unresolved"] == ["Vitality"]
    assert called["n"] == 0      # no publish attempted

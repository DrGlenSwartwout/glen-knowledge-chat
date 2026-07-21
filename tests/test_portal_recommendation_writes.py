import sqlite3
import app as app_module
from dashboard import recommendation_events as re, recommendation_prefs as rp, client_portal as cp


def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx)
    token, _pid = cp.upsert_portal(cx, "a@b.com", "Al", {})
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="2026-07-10", origin_ref="1")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    return db, token


def test_hide_client_note_and_section_writes(tmp_path, monkeypatch):
    db, token = _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    assert c.post(f"/api/portal/{token}/recommendation/hide",
                  json={"product_key": "neuro-magnesium", "hidden": True}).get_json()["ok"]
    assert c.post(f"/api/portal/{token}/recommendation/client-note",
                  json={"product_key": "neuro-magnesium", "note": "great"}).get_json()["ok"]
    assert c.post(f"/api/portal/{token}/recommendation/section",
                  json={"section_key": "purchased", "collapsed": True}).get_json()["ok"]
    cx = sqlite3.connect(db)
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["client_note"] == "great"
    assert rp.get_section_state(cx, "a@b.com")["purchased"] is True
    # operator_note must remain untouched by the client-note endpoint
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["operator_note"] == ""
    cx.close()
    # hidden product no longer appears in the portal sections
    r = c.get(f"/api/portal/{token}/recommendations").get_json()
    assert all(p["product_key"] != "neuro-magnesium"
               for s in r["sections"] for p in s["products"])


def test_cross_client_isolation(tmp_path, monkeypatch):
    """Top-risk PII surface: A's token must never read or write B's recommendations."""
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db)
    cp.init_client_portal_table(cx); re.init_recommendation_events(cx); rp.init_recommendation_prefs(cx)
    token_a, _ = cp.upsert_portal(cx, "a@b.com", "Al", {})
    token_b, _ = cp.upsert_portal(cx, "b@b.com", "Bea", {})
    re.record_event(cx, "a@b.com", "neuro-magnesium", "purchased", occurred_at="2026-07-10", origin_ref="1")
    re.record_event(cx, "b@b.com", "iron-syntropy", "purchased", occurred_at="2026-07-11", origin_ref="2")
    cx.commit(); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()

    # A's read must return only A's product, never B's.
    r = c.get(f"/api/portal/{token_a}/recommendations").get_json()
    a_keys = {p["product_key"] for s in r["sections"] for p in s["products"]}
    assert a_keys == {"neuro-magnesium"}
    assert "iron-syntropy" not in a_keys

    # B's read must return only B's product, never A's.
    r = c.get(f"/api/portal/{token_b}/recommendations").get_json()
    b_keys = {p["product_key"] for s in r["sections"] for p in s["products"]}
    assert b_keys == {"iron-syntropy"}
    assert "neuro-magnesium" not in b_keys

    # A write under A's token lands on A's email and never touches B.
    assert c.post(f"/api/portal/{token_a}/recommendation/hide",
                  json={"product_key": "neuro-magnesium", "hidden": True}).get_json()["ok"]
    assert c.post(f"/api/portal/{token_a}/recommendation/client-note",
                  json={"product_key": "neuro-magnesium", "note": "A's private note"}).get_json()["ok"]

    cx = sqlite3.connect(db)
    assert rp.get_notes(cx, "a@b.com")["neuro-magnesium"]["client_note"] == "A's private note"
    # B's notes/hidden-flags are entirely untouched by A's writes.
    assert rp.get_notes(cx, "b@b.com") == {}
    b_hidden = cx.execute(
        "SELECT product_key FROM recommendation_hidden WHERE client_email='b@b.com'"
    ).fetchall()
    assert b_hidden == []
    cx.close()

    # B's read is still unaffected after A's writes.
    r = c.get(f"/api/portal/{token_b}/recommendations").get_json()
    b_keys = {p["product_key"] for s in r["sections"] for p in s["products"]}
    assert b_keys == {"iron-syntropy"}


def test_unknown_token_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    c = app_module.app.test_client()
    r = c.post("/api/portal/not-a-real-token/recommendation/hide", json={"product_key": "x", "hidden": True})
    assert r.status_code == 404
    r = c.post("/api/portal/not-a-real-token/recommendation/client-note", json={"product_key": "x", "note": "n"})
    assert r.status_code == 404
    r = c.post("/api/portal/not-a-real-token/recommendation/section", json={"section_key": "x", "collapsed": True})
    assert r.status_code == 404

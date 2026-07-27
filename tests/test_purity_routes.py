"""Phase-2a purity routes: request (portal+console), screen, tier2, confirm.
Console writes are console-secret gated; client identity comes from the portal
token only."""
import sqlite3
import pytest
import app as app_mod
from dashboard import product_ratings as pr


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx); cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    # portal token -> email; console ok on a secret header value only
    monkeypatch.setattr(app_mod, "_portal_record_for",
                        lambda cx, tok: {"email": "a@b.com"} if tok == "TOKA" else None)
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: True)
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "full", raising=False)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _get(db, key):
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    r = cx.execute("SELECT * FROM product_ratings WHERE product_key=?", (key,)).fetchone()
    cx.close(); return dict(r) if r else None


def test_console_request_creates_row(client):
    r = client.post("/api/console/purity/request",
                    json={"product_key": "jarrow-mag", "brand": "Jarrow", "product_name": "Mag Taurate"})
    assert r.status_code == 200
    assert _get(app_mod.LOG_DB, "jarrow-mag")["status"] == "requested"


def test_portal_request_uses_token_email_not_a_field(client):
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N",
                          "email": "attacker@evil.com"})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k")
    assert row["status"] == "requested"
    # requested_by came from the token (a@b.com), never the body's email field
    assert "attacker@evil.com" not in (row.get("requested_by") or "")


def test_portal_request_unknown_token_401(client):
    r = client.post("/api/portal/NOPE/purity/request",
                    json={"product_key": "k", "brand": "B", "product_name": "N"})
    assert r.status_code in (401, 404)
    assert _get(app_mod.LOG_DB, "k") is None


def test_portal_request_not_entitled_403(client, monkeypatch):
    # A non-paid client with no explicit grant is refused, and no row is created.
    monkeypatch.setattr(app_mod, "membership_category", lambda email: "none", raising=False)
    r = client.post("/api/portal/TOKA/purity/request",
                    json={"product_key": "blocked", "brand": "B", "product_name": "N"})
    assert r.status_code == 403
    assert _get(app_mod.LOG_DB, "blocked") is None


def test_console_screen_runs_the_real_screen(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k",
                          "other_ingredients": ["Hypromellose", "Magnesium Stearate"]})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k")
    assert row["status"] == "screened" and row["color"] == "red"


def test_console_screen_none_data_is_unrated_not_green(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k2", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k2", "other_ingredients": None})
    assert r.status_code == 200
    assert _get(app_mod.LOG_DB, "k2")["status"] == "unrated"
    assert _get(app_mod.LOG_DB, "k2")["color"] is None


def test_console_screen_text_comma_split_reaches_screen(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k3", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k3",
                          "other_ingredients_text": "Cellulose, Magnesium Stearate, Silica"})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k3")
    assert row["status"] == "screened" and row["color"] == "red"


def test_console_screen_text_newline_split_reaches_screen(client):
    # "Magnesium" and "Stearate" on separate label lines are two distinct,
    # unlisted items -- neither is the red alias "magnesium stearate" on its
    # own. If newline splitting were dropped, the two lines would collapse
    # into one normalized string "magnesium stearate" and false-positive red,
    # so this also proves the newline (not just comma) split actually runs.
    client.post("/api/console/purity/request",
                json={"product_key": "k4", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k4",
                          "other_ingredients_text": "Magnesium\nStearate"})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k4")
    assert row["status"] == "screened" and row["color"] == "green"


def test_console_screen_blank_text_is_unrated_not_green(client):
    client.post("/api/console/purity/request",
                json={"product_key": "k5", "brand": "B", "product_name": "N"})
    r = client.post("/api/console/purity/screen",
                    json={"product_key": "k5", "other_ingredients_text": "   "})
    assert r.status_code == 200
    row = _get(app_mod.LOG_DB, "k5")
    assert row["status"] == "unrated"
    assert row["color"] is None


def _screen(client, key, oi):
    client.post("/api/console/purity/request", json={"product_key": key, "brand": "B", "product_name": "N"})
    client.post("/api/console/purity/screen", json={"product_key": key, "other_ingredients": oi})


def test_green_tier2_then_confirm(client):
    _screen(client, "g", ["Hypromellose"])   # green
    r = client.post("/api/console/purity/tier2", json={"product_key": "g", "score": 8.5, "detail_json": "{}"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "g")["status"] == "ai_draft"
    r = client.post("/api/console/purity/confirm", json={"product_key": "g"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "g")["status"] == "confirmed"


def test_red_confirms_without_tier2(client):
    _screen(client, "r", ["Magnesium Stearate"])   # red
    r = client.post("/api/console/purity/confirm", json={"product_key": "r"})
    assert r.status_code == 200 and _get(app_mod.LOG_DB, "r")["status"] == "confirmed"


def test_tier2_on_a_red_is_rejected(client):
    _screen(client, "r2", ["Gelatin"])   # red
    r = client.post("/api/console/purity/tier2", json={"product_key": "r2", "score": 9, "detail_json": "{}"})
    assert r.status_code == 400   # engine raises; route returns 400, does not 500


def test_console_list_returns_rows(client):
    _screen(client, "g2", ["Hypromellose"])
    r = client.get("/api/console/purity-ratings")
    assert r.status_code == 200
    keys = [row["product_key"] for row in r.get_json()["ratings"]]
    assert "g2" in keys


def test_console_routes_require_secret(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_portal_console_ok", lambda: False)
    for path, body in [("/api/console/purity/request", {"product_key": "x"}),
                       ("/api/console/purity/screen", {"product_key": "x"}),
                       ("/api/console/purity/tier2", {"product_key": "x"}),
                       ("/api/console/purity/confirm", {"product_key": "x"})]:
        assert client.post(path, json=body).status_code == 401
    assert client.get("/api/console/purity-ratings").status_code == 401

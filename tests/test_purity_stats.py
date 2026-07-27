import json, sqlite3, pytest, app as app_mod
from dashboard import product_ratings as pr


def _confirm_color(cx, key, color):
    pr.record_screen(cx, key, brand="B", product_name="P"+key,
                     other_ingredients_raw="x", other_ingredients_parsed=["x"],
                     screen={"color": color, "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    if color == "red":
        pr.confirm(cx, key)                     # red: screened -> confirmed
    else:
        pr.set_tier2(cx, key, None, "{}"); pr.confirm(cx, key)   # green/yellow via ai_draft


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "s.db")
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    _confirm_color(cx, "fullscript::r1", "red")
    _confirm_color(cx, "fullscript::g1", "green")
    _confirm_color(cx, "fullscript::g2", "green")
    _confirm_color(cx, "fullscript::y1", "yellow")
    # a screened-but-unconfirmed row must NOT count
    pr.record_screen(cx, "fullscript::r2", brand="B", product_name="R2",
                     other_ingredients_raw="magnesium stearate", other_ingredients_parsed=["magnesium stearate"],
                     screen={"color": "red", "red_hits": ["stearate"], "yellow_hits": [], "avoidlist_version": "v1"})
    cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def test_aggregate_confirmed_counts_only_confirmed():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    pr.init_tables(cx)
    _confirm_color(cx, "fullscript::a", "green")
    pr.record_screen(cx, "fullscript::b", brand="B", product_name="B",   # screened, not confirmed
                     other_ingredients_raw="x", other_ingredients_parsed=["x"],
                     screen={"color": "red", "red_hits": [], "yellow_hits": [], "avoidlist_version": "v1"})
    agg = pr.aggregate_confirmed(cx)
    assert agg == {"screened": 1, "red": 0, "yellow": 0, "green": 1}


def test_stats_endpoint_public_aggregate(client):
    r = client.get("/api/purity/stats")
    assert r.status_code == 200
    b = r.get_json()
    assert b["screened"] == 4                       # 1 red + 2 green + 1 yellow (r2 unconfirmed excluded)
    assert b["red"] == 1 and b["green"] == 2 and b["yellow"] == 1
    assert b["fail_pct"] == 25                       # 1/4
    assert b["headline"] == "25% of professional-quality products we've screened fail our purity standard"
    # aggregate-only: no product identity leaks into the public payload
    blob = json.dumps(b).lower()
    assert "product_key" not in blob and "fullscript::" not in blob and "brand" not in blob


def test_stats_endpoint_no_auth_required(client):
    # no console key header -> still 200 (public)
    assert client.get("/api/purity/stats").status_code == 200


def test_stats_endpoint_empty_no_confirmed_rows(monkeypatch, tmp_path):
    # Real prod state before anything is confirmed: 200, zeroed, graceful headline,
    # no ZeroDivisionError.
    import sqlite3, app as app_mod
    from dashboard import product_ratings as pr
    db = str(tmp_path / "empty.db")
    cx = sqlite3.connect(db); pr.init_tables(cx); cx.commit(); cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    app_mod.app.config["TESTING"] = True
    r = app_mod.app.test_client().get("/api/purity/stats")
    assert r.status_code == 200
    b = r.get_json()
    assert b["screened"] == 0 and b["fail_pct"] == 0
    assert b["headline"] == "No products screened yet."

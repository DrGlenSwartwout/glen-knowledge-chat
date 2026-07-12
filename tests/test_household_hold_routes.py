import os, sqlite3, importlib
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    monkeypatch.setenv("LOG_DB", str(tmp_path / "log.db"))
    import app as _app
    importlib.reload(_app)
    _app.app.config["TESTING"] = True
    return _app


def _seed_hold(app):
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        o1 = O.upsert_order(cx, source="t", external_ref="cg@x.com", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        o2 = O.upsert_order(cx, source="t", external_ref="kid@x.com", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com")
        raw = H.set_release_token(cx, g)
    return g, raw


def test_get_renders_confirm_post_releases(client):
    g, raw = _seed_hold(client)
    c = client.app.test_client()
    # GET must NOT mutate (scanner safety)
    r_get = c.get(f"/hold/{raw}/ship")
    assert r_get.status_code == 200 and b"<form" in r_get.data
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "open"   # still open after GET
    # POST releases
    r_post = c.post(f"/hold/{raw}/ship")
    assert r_post.status_code in (200, 302)
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "released"

import os, sqlite3, importlib
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "1")
    monkeypatch.setenv("HOUSEHOLD_SHIPMENTS_ENABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
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
    return g, raw, o1, o2


def test_get_renders_confirm_post_releases(client):
    g, raw, o1, o2 = _seed_hold(client)
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
        from dashboard import orders as O
        from dashboard import combined_shipments as CS
        assert H.get_hold(cx, g)["status"] == "released"
        # The combined shipment must actually have been created, not just
        # silently swallowed by the route's bare try/except.
        o1_after = O.get_order(cx, o1)
        o2_after = O.get_order(cx, o2)
        assert o1_after["group_shipment_id"] is not None
        assert o2_after["group_shipment_id"] is not None
        assert o1_after["group_shipment_id"] == o2_after["group_shipment_id"]
        open_shipments = CS.list_open_shipments(cx)
        matching = [s for s in open_shipments if s["id"] == o1_after["group_shipment_id"]]
        assert len(matching) == 1
        assert len(matching[0]["members"]) == 2


def test_post_is_idempotent_on_already_released_hold(client):
    g, raw, o1, o2 = _seed_hold(client)
    c = client.app.test_client()
    r_post1 = c.post(f"/hold/{raw}/ship")
    assert r_post1.status_code in (200, 302)
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "released"

    # Second POST to the same token must not crash or double-release, and
    # must not attempt a second create_shipment on already-grouped orders.
    r_post2 = c.post(f"/hold/{raw}/ship")
    assert r_post2.status_code in (200, 302, 404)
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        from dashboard import household_holds as H
        assert H.get_hold(cx, g)["status"] == "released"

    # A GET on the already-released token shows the "already on its way"
    # page, not the confirm form (scanner safety — no re-triggering).
    r_get = c.get(f"/hold/{raw}/ship")
    assert r_get.status_code in (200, 404)
    assert b"<form" not in r_get.data


def test_release_recomputes_combined_shipping(client, monkeypatch):
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    from dashboard import combined_shipments as CS
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        CS.init_combined_shipments_table(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        # NOTE: dashboard/orders.py has upsert_order (requires external_ref), not create_order.
        o1 = O.upsert_order(cx, source="t", external_ref="cg@x.com", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000,
                            shipping_cents=800, channel="ship")
        o2 = O.upsert_order(cx, source="t", external_ref="kid@x.com", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000,
                            shipping_cents=800, channel="ship")
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com")["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com")
        raw = H.set_release_token(cx, g)

    # Spy on the real recompute so the assertion is meaningful even if the
    # geometric packer happens to price the fake "x" slug's combined parcel
    # at exactly 1600 (which would make a bare "<= 1600" check pass whether
    # or not recompute actually ran).
    calls = []
    orig = client._recompute_combined_shipping

    def _spy(cx, sid):
        calls.append(sid)
        return orig(cx, sid)

    monkeypatch.setattr(client, "_recompute_combined_shipping", _spy)

    client.app.test_client().post(f"/hold/{raw}/ship")

    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        sid = O.get_order(cx, o1)["group_shipment_id"]
        assert sid is not None
        # _recompute_combined_shipping actually ran for this new shipment,
        # not just create_shipment in isolation.
        assert calls == [sid]
        members = O.orders_in_group(cx, sid)
        # combined parcel shipping is split, so the pair's total shipping is <= 2x single
        assert sum(int(m["shipping_cents"] or 0) for m in members) <= 1600


def test_flag_off_no_hold_end_to_end(monkeypatch):
    # Deliberately does NOT use the app-reloading `client` fixture: the flag
    # is read at call time (os.environ.get inside household_holds._enabled()),
    # so no app/module reload is needed to prove inertness. This runs bare.
    monkeypatch.setenv("HOUSEHOLD_AUTO_BATCH_ENABLED", "")
    from dashboard import orders as O, family_plan as FP, household_holds as H
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    FP.init_family_plan_table(cx)
    H.init_hold_tables(cx)
    FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
    oid = O.upsert_order(cx, source="t", external_ref="cg@x.com", email="cg@x.com", name="cg",
                         items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
    assert H.maybe_hold_new_order(cx, oid) is None
    assert O.get_order(cx, oid)["hold_group_id"] is None


def test_sweep_releases_due_holds(client, monkeypatch):
    import sqlite3, datetime as _dt
    from dashboard import orders as O, family_plan as FP, household as HH, household_holds as H
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        FP.init_family_plan_table(cx); HH.init_household_tables(cx); H.init_hold_tables(cx)
        FP.activate(cx, "cg@x.com", next_charge_at="2999-01-01")
        HH.add_member(cx, "cg@x.com", "kid@x.com", relationship="child")
        o1 = O.upsert_order(cx, source="t", external_ref="cg@x.com", email="cg@x.com", name="cg",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        o2 = O.upsert_order(cx, source="t", external_ref="kid@x.com", email="kid@x.com", name="kid",
                            items=[{"slug": "x", "qty": 1}], total_cents=1000, channel="ship")
        past = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
        g = H.open_or_join_hold(cx, o1, caregiver_email="cg@x.com", household_key="cg@x.com", now=past)["group_id"]
        H.open_or_join_hold(cx, o2, caregiver_email="cg@x.com", household_key="cg@x.com", now=past)
    c = client.app.test_client()
    r = c.post("/api/cron/household-holds/sweep",
               headers={"X-Console-Key": client.CONSOLE_SECRET})
    assert r.status_code == 200
    with sqlite3.connect(client.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        assert H.get_hold(cx, g)["status"] == "released"

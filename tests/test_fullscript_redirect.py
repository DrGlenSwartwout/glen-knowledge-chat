"""GET /fs/<token>/<product_slug>: records a click then 302s OUTBOUND to
Fullscript. Identity comes from the portal token only. The destination is built
from a hardcoded base + config + the DB row, never from the request, so the
route is structurally incapable of becoming an open redirect."""
import sqlite3
import pytest

import app as app_mod
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": None, "relation": None, "focus_tags": [],
     "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
    {"name": "Retired Thing", "brand": "X", "product_slug": "retired-thing",
     "external_id": "P9", "best_ff": None, "relation": None, "focus_tags": [],
     "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 0},
    # Hostile-but-active row: this slug has no slash, so it genuinely matches
    # Flask's <product_slug> segment and the lookup SUCCEEDS -- the
    # destination-assignment line in fullscript_click_redirect is reached with
    # attacker-controlled data sitting in both product_slug and url. If the
    # route ever assigned the destination from the row instead of from the
    # hardcoded dispensary builder, this is what would leak.
    {"name": "Poisoned Row", "brand": "X", "product_slug": "javascript:alert(1)",
     "external_id": "P66", "best_ff": None, "relation": None, "focus_tags": [],
     "ff_alts": [], "product_type": "supplement", "url": "https://evil.example.com/pwn",
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [], "focus_area_items": [],
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.commit()
    cx.close()
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")
    monkeypatch.setattr(app_mod, "_portal_record_for",
                        lambda cx, token: {"email": "a@b.com"} if token == "TOKA"
                        else ({"email": "b@b.com"} if token == "TOKB" else None))
    app_mod.app.config["TESTING"] = True
    return app_mod.app.test_client()


def _clicks(db):
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    rows = [dict(r) for r in cx.execute("SELECT * FROM fullscript_clicks").fetchall()]
    cx.close()
    return rows


def test_redirects_to_dispensary_and_records(client, tmp_path):
    r = client.get("/fs/TOKA/mag-taurate")
    assert r.status_code == 302
    assert r.headers["Location"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    rows = _clicks(app_mod.LOG_DB)
    assert len(rows) == 1
    assert rows[0]["email"] == "a@b.com"
    assert rows[0]["fs_product_name"] == "Mag Taurate"


def test_unknown_token_records_nothing_and_goes_home(client):
    r = client.get("/fs/NOPE/mag-taurate")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_unknown_slug_goes_home(client):
    r = client.get("/fs/TOKA/not-a-real-slug")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_inactive_product_goes_home(client):
    r = client.get("/fs/TOKA/retired-thing")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_poisoned_row_does_not_steer_the_redirect(client):
    """A row that is hostile in every field a redirect could plausibly read from
    -- product_slug containing a javascript: payload, url pointing at an
    attacker host -- must still send the client to the real dispensary. This
    row is active and its slug matches the route cleanly (no slash), so the
    lookup SUCCEEDS and the destination-assignment line in
    fullscript_click_redirect is genuinely reached with poisoned data in hand.
    That's what makes this test able to fail: unlike a slug that never matches
    any row, this one does, so a mutation that assigns `dest` from the row's
    product_slug or url is exercised, not skipped."""
    r = client.get("/fs/TOKA/javascript:alert(1)")
    assert r.status_code == 302
    assert r.headers["Location"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    assert "evil.example.com" not in r.headers["Location"]
    assert "javascript:" not in r.headers["Location"]
    rows = _clicks(app_mod.LOG_DB)
    assert len(rows) == 1
    assert rows[0]["fs_product_name"] == "Poisoned Row"


def test_destination_never_comes_from_the_request(client):
    """An attacker-supplied absolute URL in the slug position must not be honored.

    Flask's default <string> converter does not match a path segment containing
    an encoded slash, so this specific request may 404 rather than reach the
    view at all. A 404 is an equally safe outcome here -- the attacker-supplied
    host is never used as a destination either way. What must be true
    regardless of status code is the actual security property: no response to
    this request ever carries a Location header pointing at the attacker's host.
    """
    r = client.get("/fs/TOKA/https:%2F%2Fevil.example.com")
    assert r.status_code in (302, 404)
    location = r.headers.get("Location")
    if location is not None:
        assert "evil.example.com" not in location
    if r.status_code == 302:
        assert location == "/"
    assert _clicks(app_mod.LOG_DB) == []


def test_cross_client_isolation(client):
    """Client B's token records against B, never against A."""
    client.get("/fs/TOKB/mag-taurate")
    rows = _clicks(app_mod.LOG_DB)
    assert [r["email"] for r in rows] == ["b@b.com"]


def test_recording_failure_never_blocks_the_click(client, monkeypatch):
    """A broken click recorder must not cost the client their click. Analytics is
    strictly less important than the person trying to reach the dispensary, so
    the redirect has to survive record_click blowing up. Held only by code
    inspection until this test existed."""
    def _boom(*a, **kw):
        raise RuntimeError("click recorder is down")
    monkeypatch.setattr(fs, "record_click", _boom)

    r = client.get("/fs/TOKA/mag-taurate")

    assert r.status_code == 302
    assert r.headers["Location"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    assert _clicks(app_mod.LOG_DB) == [], "recorder raised, so nothing is stored"

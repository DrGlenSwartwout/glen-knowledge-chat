"""_fullscript_for: flag-gated Fullscript channel card builder. Mirrors
_prl_supplement_for. Covers derive-from-scan, pin priority, the default-OFF
flag, and the byte-identical guarantee that the payload key is absent when off.
"""
import sqlite3

import app as app_mod
from dashboard import client_portal as cp
from dashboard import fullscript as fs

SEED = {
  "products": [
    {"name": "Mag Taurate", "brand": "Jarrow", "product_slug": "mag-taurate",
     "external_id": "P1", "best_ff": "Neuro Magnesium", "relation": "substitute",
     "focus_tags": [], "ff_alts": [], "product_type": "supplement", "url": None,
     "source": "seed", "active": 1},
  ],
  "focus_area_products": [
    {"focus_area_id": 9, "focus_area_name": "Nervous System",
     "fs_product_name": "Mag Taurate", "rank": 0},
  ],
  "focus_area_items": [{"focus_area_id": 9, "item_code": "ED4"}],
}


def _seed(cx):
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT,
         priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations "
               "VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.commit()


def _db(tmp_path):
    db = str(tmp_path / "c.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    _seed(cx)
    cx.close()
    return db


def _db_two_scans(tmp_path):
    """A client with TWO scans: the newer one (2026-07-10) carries an item code
    that matches no focus area, the OLDER one (2026-07-01) carries ED4, which
    maps to the seeded 'Nervous System' focus area / Mag Taurate. Mirrors the
    stale-scan-bleed scenario the Blocking-1 fix must close: with no explicit
    scan_date, only the newest scan's item codes may surface a product."""
    db = str(tmp_path / "two_scans.db")
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    fs.sync_from_seed(cx, SEED)
    cx.execute("""CREATE TABLE IF NOT EXISTS scan_recommendations
        (email TEXT, scan_id TEXT, scan_date TEXT, item_code TEXT,
         priority_rank INTEGER, label TEXT)""")
    cx.execute("INSERT INTO scan_recommendations "
               "VALUES ('a@b.com','s1','2026-07-01','ED4',1,'ED4 - Nerve')")
    cx.execute("INSERT INTO scan_recommendations "
               "VALUES ('a@b.com','s2','2026-07-10','ZZ9',1,'ZZ9 - Unrelated')")
    cx.commit()
    cx.close()
    return db


def test_stale_scan_does_not_bleed_into_default_card(monkeypatch, tmp_path):
    """Blocking 1: with no explicit scan_date, the card must resolve to the
    client's MOST RECENT scan only. The older scan's ED4 -> Mag Taurate match
    must not surface just because the client had it in some scan, ever."""
    monkeypatch.setattr(app_mod, "LOG_DB", _db_two_scans(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")

    out_default = app_mod._fullscript_for("a@b.com", None)
    assert out_default is None, \
        "the newest scan (ZZ9) matches no focus area, so no card should build"

    out_older = app_mod._fullscript_for("a@b.com", "2026-07-01")
    assert out_older is not None
    names = [p["name"] for g in out_older["groups"] for p in g["products"]]
    assert "Mag Taurate" in names, \
        "explicitly requesting the older scan's date must still surface its match"


def test_flag_off_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    assert app_mod._fullscript_for("a@b.com", "2026-07-01") is None


def test_derive_builds_card(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    monkeypatch.setenv("FULLSCRIPT_DISPENSARY_SLUG", "remedymatch")
    out = app_mod._fullscript_for("a@b.com", "2026-07-01")
    assert out["dispensary_url"] == \
        "https://us.fullscript.com/welcome/remedymatch/store-start"
    g = out["groups"][0]
    assert g["origin"] == "scan" and g["heading"] == "Matched from your scan"
    p = g["products"][0]
    assert p["name"] == "Mag Taurate" and p["brand"] == "Jarrow"
    assert p["ff"]["name"] == "Neuro Magnesium"
    assert p["ff"]["relation"] == "substitute"


def test_no_candidates_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("nobody@nowhere.com", None) is None


def test_never_raises_on_bad_db(monkeypatch):
    monkeypatch.setattr(app_mod, "LOG_DB", "/nonexistent/dir/nope.db")
    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    assert app_mod._fullscript_for("a@b.com", None) is None


def test_flag_off_payload_has_no_fullscript_key(monkeypatch, tmp_path):
    """The byte-identical guarantee: with the flag unset, the portal payload must
    not gain a `fullscript` key at all. A present-but-null key is a regression."""
    monkeypatch.setattr(app_mod, "LOG_DB", _db(tmp_path))
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    payload = {}
    payload["fullscript_enabled"] = app_mod._fullscript_enabled()
    if app_mod._fullscript_enabled():
        blk = app_mod._fullscript_for("a@b.com", None)
        if blk:
            payload["fullscript"] = blk
    assert payload["fullscript_enabled"] is False
    assert "fullscript" not in payload


def test_flag_off_real_endpoint_payload_has_no_fullscript_key(monkeypatch, tmp_path):
    """Stronger sibling of the guard above: the guard above calls
    `_fullscript_enabled`/`_fullscript_for` directly, mirroring the endpoint's
    logic rather than exercising it, so a regression written only into the
    endpoint's own payload-assembly code (e.g. `payload["fullscript"] = None`
    added unconditionally, ahead of the flag check) would slip past it. This
    test drives the real `GET /api/portal/<token>` route end-to-end and would
    catch that regression.
    """
    db_path = str(tmp_path / "endpoint.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db_path)
    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, "a@b.com", "Test Client", {})
    app_mod.app.config["TESTING"] = True
    client = app_mod.app.test_client()
    j = client.get(f"/api/portal/{token}").get_json()
    assert j.get("fullscript_enabled") is False
    assert "fullscript" not in j


def test_real_endpoint_payload_never_echoes_the_portal_token(monkeypatch, tmp_path):
    """Blocking 2: `payload["token"] = token` was added OUTSIDE the flag gate,
    so every portal response -- even with the channel dark -- carried the
    portal bearer token. That widens every response and is redundant: the
    frontend already derives its own token from the URL. Checked with the flag
    BOTH off and on, since the bug lived outside the `if _fullscript_enabled()`
    branch either way."""
    db_path = str(tmp_path / "endpoint2.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db_path)
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, "a@b.com", "Test Client", {})
    app_mod.app.config["TESTING"] = True
    client = app_mod.app.test_client()

    monkeypatch.delenv("FULLSCRIPT_ENABLED", raising=False)
    j_off = client.get(f"/api/portal/{token}").get_json()
    assert "token" not in j_off

    monkeypatch.setenv("FULLSCRIPT_ENABLED", "1")
    j_on = client.get(f"/api/portal/{token}").get_json()
    assert "token" not in j_on

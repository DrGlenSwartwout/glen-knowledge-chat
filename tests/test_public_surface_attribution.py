import os
import sqlite3
import pytest

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)

import app as appmod


def _seed(db):
    cx = sqlite3.connect(db)
    cx.executescript("""
      CREATE TABLE IF NOT EXISTS affiliate_signups (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, name TEXT,
        email TEXT, organization TEXT DEFAULT '', website TEXT DEFAULT '',
        promo_method TEXT DEFAULT '', slug TEXT, token TEXT,
        status TEXT DEFAULT 'approved', notes TEXT DEFAULT '',
        referred_by TEXT DEFAULT '', short_url TEXT DEFAULT '');
    """)
    cx.execute(
        "INSERT INTO affiliate_signups (created_at,name,email,organization,slug,token,status)"
        " VALUES ('2026-01-01','Jane Doe','jane@example.com','Doe Wellness',"
        "'prof-jane-doe','tok','approved')")
    cx.commit()
    cx.close()


@pytest.fixture
def client(monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    _seed(db)
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def _views(slug):
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        rows = cx.execute(
            "SELECT * FROM public_surface_views WHERE slug=?", (slug,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    cx.close()
    return rows


def test_storefront_visit_records_a_view(client):
    client.get("/p/prof-jane-doe")
    rows = _views("prof-jane-doe")
    assert len(rows) == 1
    assert rows[0]["surface"] == "storefront"


def test_sample_slug_visit_records_a_view(client):
    client.get("/sample/prof-jane-doe")
    rows = _views("prof-jane-doe")
    assert len(rows) == 1
    assert rows[0]["surface"] == "sample"


def test_bare_sample_records_nothing(client):
    client.get("/sample")
    assert _views("") == []


def test_unknown_slug_records_nothing(client):
    client.get("/sample/no-such-person")
    assert _views("no-such-person") == []


def test_repeat_visits_are_all_recorded(client):
    """Per-slug view counts are the instrumentation. Do not dedupe here."""
    client.get("/p/prof-jane-doe")
    client.get("/p/prof-jane-doe")
    assert len(_views("prof-jane-doe")) == 2


def test_recording_failure_never_breaks_the_page(client, monkeypatch):
    """Instrumentation is not load-bearing. If it throws, the page still serves."""
    from dashboard import public_surface as ps

    def _boom(*a, **k):
        raise RuntimeError("db exploded")

    monkeypatch.setattr(ps, "record_view", _boom)
    assert client.get("/p/prof-jane-doe").status_code == 200
    assert client.get("/sample/prof-jane-doe").status_code == 200


@pytest.fixture
def client_no_affiliate_table(monkeypatch, tmp_path):
    """A LOG_DB that has no affiliate_signups table at all — e.g. a fresh
    or otherwise-divergent database. Unlike test_recording_failure_never_
    breaks_the_page above (which only monkeypatches record_view), this
    exercises the real sqlite3 OperationalError path: the slug-resolution
    query itself must be inside the guard, not just the record_view call."""
    db = str(tmp_path / "chat_log_no_affiliate.db")
    sqlite3.connect(db).close()  # create an empty db file, no tables at all
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setenv("PUBLIC_SURFACE_ENABLED", "1")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_sample_slug_survives_missing_affiliate_table(client_no_affiliate_table):
    """Regression: /sample/<slug> made zero DB calls before Task 10 added
    view recording, and could not fail. A LOG_DB missing affiliate_signups
    (the slug-resolution query record_view's gate depends on) must degrade
    to "don't record a view", never to a 500 on this static demo page."""
    resp = client_no_affiliate_table.get("/sample/prof-jane-doe")
    assert resp.status_code == 200


def test_api_sample_slug_survives_missing_affiliate_table(client_no_affiliate_table):
    """Same guarantee for /api/sample/<slug>: a missing affiliate_signups
    table must not 500 the JSON endpoint either."""
    resp = client_no_affiliate_table.get("/api/sample/prof-jane-doe")
    assert resp.status_code == 200

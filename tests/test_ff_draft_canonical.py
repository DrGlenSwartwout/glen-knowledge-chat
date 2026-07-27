"""Backend: the console FF-draft review list enriches each draft with a
`canonical` block (canonical_tags.get_person minus tags). Env-gated like the
other api tests."""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires app env (use doppler run / CI)", allow_module_level=True)

import app  # noqa: E402
from dashboard import canonical_tags as ct  # noqa: E402
from dashboard import ff_match_drafts as ffd  # noqa: E402


def _cx_mem():
    cx = sqlite3.connect(":memory:")
    ct.init_tables(cx)
    return cx


def _seed_canon(cx, email, **fields):
    for f, vals in fields.items():
        for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
            ct.set_attr(cx, email, f, v, source="test")
    cx.commit()


# --- helper unit ----------------------------------------------------------

def test_helper_returns_fields_minus_tags():
    cx = _cx_mem()
    _seed_canon(cx, "c@x.com", conditions="glaucoma", terrain_concerns="oxidative",
                body_systems="liver", tags="vip")
    out = app._ff_draft_canonical(cx, "c@x.com")
    assert out.get("conditions") == ["glaucoma"]
    assert out.get("terrain_concerns") == ["oxidative"]
    assert out.get("body_systems") == ["liver"]
    assert "tags" not in out                       # CRM bucket dropped


def test_helper_includes_scalars():
    cx = _cx_mem()
    _seed_canon(cx, "c@x.com", challenges="fatigue", goals="more energy")
    out = app._ff_draft_canonical(cx, "c@x.com")
    assert out.get("challenges") == "fatigue" and out.get("goals") == "more energy"


def test_helper_best_effort_on_raise(monkeypatch):
    cx = _cx_mem()
    monkeypatch.setattr(ct, "get_person",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert app._ff_draft_canonical(cx, "c@x.com") == {}     # no raise


def test_helper_blank_email():
    assert app._ff_draft_canonical(_cx_mem(), "") == {}


# --- endpoint integration -------------------------------------------------

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    p = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", p)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return p


def _seed_draft(db, email, scan_date, items):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        ffd.init_table(cx)
        ffd.get_or_create(cx, email, scan_date, lambda: items)
        cx.commit()


def _seed_canon_db(db, email, **fields):
    with sqlite3.connect(db) as cx:
        ct.init_tables(cx)
        for f, vals in fields.items():
            for v in (vals if isinstance(vals, (list, tuple)) else [vals]):
                ct.set_attr(cx, email, f, v, source="test")
        cx.commit()


def test_list_endpoint_attaches_canonical_and_leaves_items(app_env):
    items = [{"name": "Neuro-Mag", "slug": "neuro-mag", "url": "/x", "meaning": "m"}]
    _seed_draft(app_env, "c@x.com", "2026-07-01", items)
    _seed_canon_db(app_env, "c@x.com", conditions="glaucoma", tags="vip")
    r = app.app.test_client().get("/api/console/ff-match-drafts?status=draft",
                                  headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    draft = r.get_json()["drafts"][0]
    assert draft["canonical"]["conditions"] == ["glaucoma"]
    assert "tags" not in draft["canonical"]
    assert draft["items"] == items                 # items byte-identical


def test_list_endpoint_no_canonical_row(app_env):
    _seed_draft(app_env, "c@x.com", "2026-07-01",
                [{"name": "X", "slug": "x", "url": "/x", "meaning": ""}])
    r = app.app.test_client().get("/api/console/ff-match-drafts?status=draft",
                                  headers={"X-Console-Key": "testkey"})
    draft = r.get_json()["drafts"][0]
    assert draft["canonical"] == {} or all(not v for v in draft["canonical"].values())


def test_list_endpoint_requires_console_key(app_env):
    assert app.app.test_client().get("/api/console/ff-match-drafts").status_code == 401

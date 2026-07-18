"""Tests for the structured household ship-to: a household can store a
street/city/state/zip ship-to, and a member with no address of their own
(and no prior order) auto-inherits it at order time via _resolve_ship_address.

Follows the `_app(tmp_path, monkeypatch)` + real-CONSOLE_SECRET convention
from tests/test_household_connect_api.py.
"""
import importlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        import app as appmod; importlib.reload(appmod)
        import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)
        monkeypatch.setattr(appmod, "CONSOLE_SECRET", "testkey")
        monkeypatch.setattr(appmod, "_push_household_tags_to_ghl",
                            lambda *a, **k: (True, None), raising=False)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


def _person(cx, email, first, last):
    cx.execute("INSERT INTO people (email, first_name, last_name) VALUES (?,?,?)",
               (email, first, last))
    return cx.execute("SELECT id FROM people WHERE email=?", (email,)).fetchone()[0]


def _make_household(appmod, name, head_id, member_ids):
    """Insert a household row directly and tag members (bypassing the
    create_household HTTP route, which isn't the thing under test here)."""
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(appmod.LOG_DB) as cx:
        slug = appmod._household_slug(name, existing=appmod._existing_household_slugs(cx))
        cx.execute("""
            INSERT INTO households (slug, name, head_person_id, address, notes,
                                    created_at, updated_at, created_by)
            VALUES (?, ?, ?, '', '', ?, ?, 'test')
        """, (slug, name, head_id, ts, ts))
        for pid in member_ids:
            adds = {f"household:{slug}"}
            if pid == head_id:
                adds.add(f"household-head:{slug}")
            appmod._mutate_person_tags(cx, pid, add=adds)
        cx.commit()
    return slug


def test_patch_and_get_household_ship_to_roundtrips(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        head = _person(cx, "head@x.com", "Head", "Person")
        cx.commit()
    slug = _make_household(appmod, "Roundtrip Household", head, [head])

    c = appmod.app.test_client()
    r = c.patch(f"/api/households/{slug}", json={
        "ship_to": {"name": "The Persons", "street": "123 Main St", "street2": "Apt 4",
                    "city": "Honolulu", "state": "HI", "zip": "96815"},
    }, headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200, r.get_json()

    r = c.get(f"/api/households/{slug}", headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    ship_to = r.get_json()["ship_to"]
    assert ship_to["street"] == "123 Main St"
    assert ship_to["city"] == "Honolulu"
    assert ship_to["state"] == "HI"
    assert ship_to["zip"] == "96815"
    assert ship_to["name"] == "The Persons"
    assert ship_to["street2"] == "Apt 4"


def test_resolve_ship_address_falls_back_to_household_shipto(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        head = _person(cx, "head2@x.com", "Head", "Two")
        member = _person(cx, "member2@x.com", "Mem", "Ber")
        cx.commit()
    slug = _make_household(appmod, "Fallback Household", head, [head, member])

    c = appmod.app.test_client()
    r = c.patch(f"/api/households/{slug}", json={
        "ship_to": {"street": "456 Oak Ave", "city": "Kailua", "state": "HI", "zip": "96734"},
    }, headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200

    # Member has no address of their own and no prior order -> inherits household ship-to.
    ship = appmod._resolve_ship_address("member2@x.com", {})
    assert ship["street"] == "456 Oak Ave"
    assert ship["city"] == "Kailua"
    assert ship["state"] == "HI"
    assert ship["zip"] == "96734"


def test_resolve_ship_address_prefers_body_address_over_household(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        head = _person(cx, "head3@x.com", "Head", "Three")
        member = _person(cx, "member3@x.com", "Mem", "Ber3")
        cx.commit()
    slug = _make_household(appmod, "Override Household", head, [head, member])
    c = appmod.app.test_client()
    c.patch(f"/api/households/{slug}", json={
        "ship_to": {"street": "789 Household Way", "city": "Hilo", "state": "HI", "zip": "96720"},
    }, headers={"X-Console-Key": "testkey"})

    body_addr = {"street": "1 My Own St", "city": "Kona", "state": "HI", "zip": "96740"}
    ship = appmod._resolve_ship_address("member3@x.com", body_addr)
    assert ship == body_addr


def test_resolve_ship_address_no_household_returns_empty(tmp_path, monkeypatch):
    appmod = _app(tmp_path, monkeypatch)
    with sqlite3.connect(appmod.LOG_DB) as cx:
        _person(cx, "lone@x.com", "Lone", "Wolf")
        cx.commit()
    ship = appmod._resolve_ship_address("lone@x.com", {})
    assert ship == {}

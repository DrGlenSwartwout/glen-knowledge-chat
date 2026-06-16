# tests/test_portal_view.py
"""The role-aware view assembler: one payload composed from the unified person
row + orders + points + the existing biofield portal content. Visibility is
driven by roles; absent data hides its block (never errors)."""
import json
import sqlite3

import pytest


def _conn(tmp_path):
    from dashboard import portal_identity as pi
    from dashboard import orders as o
    from dashboard import points as pts
    from dashboard import client_portal as cp
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    pi._ensure_people_table(cx)
    o.init_orders_table(cx)
    pts.init_points_table(cx)
    cp.init_client_portal_table(cx)
    return cx


def _add_person(cx, email, name="C", roles='["client"]'):
    cur = cx.execute(
        "INSERT INTO people (email, name, roles, created_at, updated_at) VALUES (?,?,?,?,?)",
        (email, name, roles, "t", "t"))
    cx.commit()
    return cur.lastrowid


def test_view_composes_account_orders_points_and_stub(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import points as pts
    cx = _conn(tmp_path)
    pid = _add_person(cx, "c@example.com", "Client One")
    cx.execute(
        "INSERT INTO orders (source, external_ref, email, name, items_json, total_cents, status, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("test", "ext1", "c@example.com", "Client One", "[]", 6997, "paid", "2026-06-01", "2026-06-01"))
    cx.commit()
    pts.earn(cx, "c@example.com", full_price_cents=6997, earn_pct=0.05, order_ref="o1")  # 350c

    view = pv.get_portal_view(cx, pid)

    assert view["person_id"] == pid
    assert view["roles"] == ["client"]
    assert view["account"]["email"] == "c@example.com"
    assert view["account"]["name"] == "Client One"
    assert view["account"]["points_cents"] == 350
    assert "Client" in view["account"]["role_badges"]
    # orders block visible for a client, with the one order summarized
    assert view["orders"]["visible"] is True
    assert len(view["orders"]["items"]) == 1
    assert view["orders"]["items"][0]["total_cents"] == 6997
    assert view["orders"]["items"][0]["status"] == "paid"
    # no biofield content seeded → block hidden, not errored
    assert view["biofield"]["visible"] is False
    # sales/upgrade is a reserved stub seam (feature #2)
    assert view["upgrade"] == {"enabled": False, "placeholder": True}


def test_view_shows_biofield_when_portal_content_present(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "bf@example.com", "Biofield Client")
    cp.upsert_portal(cx, "bf@example.com", "Biofield Client", {
        "greeting": "Aloha.",
        "video": {"url": "https://app.heygen.com/share/x", "label": "Watch"},
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "r", "dosing": "d"}],
    })

    view = pv.get_portal_view(cx, pid)

    assert view["biofield"]["visible"] is True
    assert view["biofield"]["greeting"] == "Aloha."
    assert view["biofield"]["layers"][0]["title"] == "Calm"
    assert view["biofield"]["video"]["url"].endswith("/x")


def test_view_unknown_person_is_none(tmp_path):
    from dashboard import portal_view as pv
    cx = _conn(tmp_path)
    assert pv.get_portal_view(cx, 99999) is None

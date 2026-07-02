# tests/test_portal_view.py
"""The role-aware view assembler: one payload composed from the unified person
row + orders + points + the existing biofield portal content. Visibility is
driven by roles; absent data hides its block (never errors)."""
import datetime
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
    # no offer flags passed -> upgrade disabled (block hidden)
    assert view["upgrade"] == {"enabled": False}


def test_view_surfaces_first_eligible_offer(tmp_path):
    from dashboard import portal_view as pv
    cx = _conn(tmp_path)
    pid = _add_person(cx, "off@example.com", "Offer Client")
    view = pv.get_portal_view(cx, pid, offers_enabled_keys={"live_group", "biofield"})
    assert view["upgrade"]["enabled"] is True
    assert view["upgrade"]["offer"]["key"] == "live_group"
    assert view["upgrade"]["offer"]["price_cents"] == 9900


def test_orders_block_excludes_cancelled(tmp_path):
    from dashboard import portal_view as pv
    cx = _conn(tmp_path)
    pid = _add_person(cx, "co@example.com", "C")
    for ext, status in (("a", "paid"), ("b", "cancelled")):
        cx.execute(
            "INSERT INTO orders (source, external_ref, email, items_json, total_cents, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("test", ext, "co@example.com", "[]", 1000, status, "2026-06-01", "2026-06-01"))
    cx.commit()
    view = pv.get_portal_view(cx, pid)
    statuses = [o["status"] for o in view["orders"]["items"]]
    assert "cancelled" not in statuses
    assert statuses == ["paid"]


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


def test_biofield_block_blurs_remedies_until_confirmed(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "bf@example.com", "BF")
    cp.upsert_portal(cx, "bf@example.com", "BF", {"biofield_status": "interested", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["status"] == "interested" and bf["blurred"] is True
    assert bf["layers"][0]["title"] == "Calm" and bf["layers"][0]["meaning"] == "m"   # shown
    assert "remedy" not in bf["layers"][0] and "dosing" not in bf["layers"][0]        # withheld


def test_biofield_block_reveals_remedies_when_confirmed(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "cf@example.com", "CF")
    cp.upsert_portal(cx, "cf@example.com", "CF", {"biofield_status": "confirmed", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["status"] == "confirmed" and bf["blurred"] is False
    assert bf["layers"][0]["remedy"] == "Nous Energy"


def test_biofield_confirmed_stays_blurred_when_not_unlocked(tmp_path):
    # A confirmed report published for a client who hasn't PAID (free E4L reveal,
    # no membership / no paid Biofield Analysis) must stay blurred in the portal —
    # same gate as the funnel. Paid clients (unlocked=True) still see it.
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "unpaid@example.com", "UP")
    cp.upsert_portal(cx, "unpaid@example.com", "UP", {"biofield_status": "confirmed", "greeting": "hi",
        "pricing_note": "buy now", "layers": [{"n": 1, "title": "Calm", "meaning": "m",
        "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid, biofield_unlocked=False)["biofield"]
    assert bf["status"] == "confirmed"        # the report exists / status unchanged
    assert bf["blurred"] is True              # but content is gated behind payment
    assert "remedy" not in bf["layers"][0] and "dosing" not in bf["layers"][0]
    assert bf["pricing_note"] == ""
    # A paid/unlocked client still sees everything.
    bf2 = pv.get_portal_view(cx, pid, biofield_unlocked=True)["biofield"]
    assert bf2["blurred"] is False and bf2["layers"][0]["remedy"] == "Nous Energy"


def test_biofield_unlocked_defaults_true_backcompat(tmp_path):
    # Default (no flag) keeps the old behavior: confirmed → shown.
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "d@example.com", "D")
    cp.upsert_portal(cx, "d@example.com", "D", {"biofield_status": "confirmed", "greeting": "hi",
        "layers": [{"n": 1, "title": "Calm", "meaning": "m", "remedy": "Nous Energy", "dosing": "1/day"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["blurred"] is False and bf["layers"][0]["remedy"] == "Nous Energy"


def test_view_unknown_person_is_none(tmp_path):
    from dashboard import portal_view as pv
    cx = _conn(tmp_path)
    assert pv.get_portal_view(cx, 99999) is None


def test_biofield_uses_reports_newest_default_with_tabs(tmp_path):
    from dashboard import portal_view as pv, portal_biofield_reports as R
    import datetime
    cx = _conn(tmp_path); R.init_table(cx)
    pid = _add_person(cx, "m@example.com", "M")
    today = datetime.date.today()
    new_d = today.isoformat()
    old_d = (today - datetime.timedelta(days=60)).isoformat()
    R.upsert_report(cx, "m@example.com", old_d, "s0",
                    {"layers": [{"n": 1, "title": "Old", "meaning": "o", "remedy": "X", "dosing": "1"}]}, "ai_draft")
    R.upsert_report(cx, "m@example.com", new_d, "s1",
                    {"layers": [{"n": 1, "title": "New", "meaning": "n", "remedy": "Y", "dosing": "2"}]}, "interested")
    bf = pv.get_portal_view(cx, pid)["biofield"]            # default newest
    assert bf["scan_date"] == new_d and bf["scan_dates"] == [new_d, old_d]
    assert bf["status"] == "interested" and bf["blurred"] is True and bf["actionable"] is True
    assert "remedy" not in bf["layers"][0]
    bf_old = pv.get_portal_view(cx, pid, scan_date=old_d)["biofield"]
    assert bf_old["scan_date"] == old_d and bf_old["actionable"] is False
    assert "remedy" not in bf_old["layers"][0]              # old + unconfirmed -> blurred, no CTA


def test_biofield_legacy_fallback_when_no_reports(tmp_path):
    from dashboard import portal_view as pv
    from dashboard import client_portal as cp
    cx = _conn(tmp_path)
    pid = _add_person(cx, "leg@example.com", "Leg")
    cp.upsert_portal(cx, "leg@example.com", "Leg",
                     {"layers": [{"n": 1, "title": "C", "meaning": "m", "remedy": "R", "dosing": "d"}]})
    bf = pv.get_portal_view(cx, pid)["biofield"]
    assert bf["scan_dates"] == [] and bf["blurred"] is False
    assert bf["layers"][0]["remedy"] == "R"

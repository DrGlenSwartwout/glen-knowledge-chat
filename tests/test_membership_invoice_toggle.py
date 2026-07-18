"""Task 7: the token-authed customer control to add/remove the membership line on
an unpaid invoice, repricing the order live. No owner auth on this path."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

FFS = ["paracleanse", "nerve-repair", "neuroceramides",
       "microbiome", "oxygen-cleanse", "macular-wellness-lycopene"]


def _client(tmp_path, monkeypatch, *, email="toggle-nonmember@example.com"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    items = [{"slug": s, "qty": 1, "name": s, "unit_cents": 6997, "line_cents": 6997}
             for s in FFS]
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        O.upsert_order(cx, source="inhouse", external_ref="INH-TOGGLE-1",
                       email=email, name="Toggle Tester", items=items,
                       total_cents=6 * 6997, status="proposed")
        cx.commit()
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    # Point the invoice-token module at the same db the fixture seeded (order id=1).
    from dashboard import practitioner_portal as PP
    monkeypatch.setattr(PP, "_LOG_DB", Path(db))
    token = PP.create_order_invoice_token(1)
    appmod.app.config["TESTING"] = True
    return appmod, appmod.app.test_client(), token


def test_add_then_remove_membership_reprices(tmp_path, monkeypatch):
    appmod, client, token = _client(tmp_path, monkeypatch)

    add = client.post(f"/api/invoice/{token}/membership",
                      json={"action": "add", "tier": "month"})
    assert add.status_code == 200, add.data
    j = add.get_json()["order"]
    assert any(l["slug"] == "membership:month" for l in j["lines"])
    ff = [l for l in j["lines"] if l["slug"] != "membership:month"]
    assert all(l["unit_cents"] < 6997 for l in ff)      # products now member-priced

    rem = client.post(f"/api/invoice/{token}/membership", json={"action": "remove"})
    assert rem.status_code == 200, rem.data
    j2 = rem.get_json()["order"]
    assert not any(l["slug"] == "membership:month" for l in j2["lines"])
    ff2 = [l for l in j2["lines"] if l["slug"] != "membership:month"]
    assert all(l["unit_cents"] == 6997 for l in ff2)    # reverted to list


def test_add_is_idempotent(tmp_path, monkeypatch):
    """Adding twice keeps exactly one membership line (strip-then-append)."""
    appmod, client, token = _client(tmp_path, monkeypatch)
    client.post(f"/api/invoice/{token}/membership", json={"action": "add", "tier": "month"})
    r = client.post(f"/api/invoice/{token}/membership", json={"action": "add", "tier": "month"})
    lines = r.get_json()["order"]["lines"]
    assert sum(1 for l in lines if l["slug"] == "membership:month") == 1


def test_unoffered_tier_rejected(tmp_path, monkeypatch):
    appmod, client, token = _client(tmp_path, monkeypatch)
    r = client.post(f"/api/invoice/{token}/membership",
                    json={"action": "add", "tier": "year_prepay"})
    assert r.status_code == 400


def test_bad_action_rejected(tmp_path, monkeypatch):
    appmod, client, token = _client(tmp_path, monkeypatch)
    r = client.post(f"/api/invoice/{token}/membership", json={"action": "frobnicate"})
    assert r.status_code == 400


def test_paid_invoice_blocked(tmp_path, monkeypatch):
    appmod, client, token = _client(tmp_path, monkeypatch)
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        cx.execute("UPDATE orders SET pay_status='paid' WHERE id=1")
        cx.commit()
    r = client.post(f"/api/invoice/{token}/membership",
                    json={"action": "add", "tier": "month"})
    assert r.status_code == 409


def test_cancelled_invoice_blocked(tmp_path, monkeypatch):
    """A cancelled (still unpaid) order can't be repriced by the customer."""
    appmod, client, token = _client(tmp_path, monkeypatch)
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        cx.execute("UPDATE orders SET status='cancelled' WHERE id=1")
        cx.commit()
    r = client.post(f"/api/invoice/{token}/membership",
                    json={"action": "add", "tier": "month"})
    assert r.status_code == 409
    assert "no longer be changed" in (r.get_json().get("error") or "")


def test_already_member_blocked(tmp_path, monkeypatch):
    """A buyer who already owns a paid membership can't add another."""
    appmod, client, token = _client(tmp_path, monkeypatch, email="already-member@example.com")
    from datetime import datetime, timedelta
    # App boot (reload) created the real memberships table; insert an active tier grant.
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        cx.execute(
            "INSERT INTO memberships (id, email, granted_at, expires_at, source) "
            "VALUES (?,?,?,?,?)",
            ("m-toggle-1", "already-member@example.com", datetime.utcnow().isoformat(),
             (datetime.utcnow() + timedelta(days=30)).isoformat(), "membership_month"))
        cx.commit()
    r = client.post(f"/api/invoice/{token}/membership",
                    json={"action": "add", "tier": "month"})
    assert r.status_code == 409


def test_offer_injected_for_unpaid_nonmember(tmp_path, monkeypatch):
    """api_invoice_get surfaces membership_offer for an unpaid non-member."""
    appmod, client, token = _client(tmp_path, monkeypatch)
    r = client.get(f"/api/invoice/{token}")
    assert r.status_code == 200
    offer = r.get_json()["order"].get("membership_offer")
    assert offer is not None
    assert offer["tier"] == "month"
    assert offer["gross_cents"] == 9900
    assert offer["savings_cents"] > 0
    assert "month" in offer["offered_tiers"]

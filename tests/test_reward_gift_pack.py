"""Task 4: pack-board surfacing (all order sources) + attach-on-pack for reward gifts.

A member's earned reward gift (review_gifts row, source='reward', status='approved',
fulfilled_order_id IS NULL) must show on the pack board for their next order across
ALL sources, and attach to that order when it's marked packed. Gated by
REWARD_GIFTS_ENABLED (default OFF)."""
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import review_gifts as rg
from dashboard import orders as _ord


def _app(monkeypatch, tmp_db):
    """/api/orders is gated by _bos_actor() (owner master key or per-user token) —
    unlike the console-secret-optional endpoints, it never vacuously authorizes when
    CONSOLE_SECRET is unset. So (mirroring tests/test_bos_routes.py) we import app as-is
    and authenticate requests with its real dashboard.CONSOLE_SECRET, rather than
    deleting the env var."""
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    return app


def _seed_order(tmp_db, email="a@ex.com", with_gift=True):
    with sqlite3.connect(str(tmp_db)) as cx:
        _ord.init_orders_table(cx)
        oid = _ord.upsert_order(cx, source="test", external_ref="t1", email=email, name="A",
                                items=[{"slug": "x", "name": "X", "qty": 1}], total_cents=1000,
                                status="new")
        if with_gift:
            rg.init_table(cx)
            rg.migrate_reward_columns(cx)
            rg.add_reward_gift(cx, email, "GIFT-SAMPLE-3", "Placeholder", 1)
    return oid


# --- Direct helper / action-executor tests (in-memory sqlite, no app import) ---

def test_mark_packed_attaches_pending_reward_gift(monkeypatch):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    cx = sqlite3.connect(":memory:")
    _ord.init_orders_table(cx)
    oid = _ord.upsert_order(cx, source="test", external_ref="t1", email="a@ex.com", name="A",
                            items=[{"slug": "x", "name": "X", "qty": 1}], total_cents=1000,
                            status="new")
    rg.init_table(cx)
    rg.migrate_reward_columns(cx)
    gift_id = rg.add_reward_gift(cx, "a@ex.com", "GIFT-SAMPLE-3", "Placeholder", 1)

    from dashboard import actions
    result = actions.get_action("orders.mark_packed").executor({"order_id": oid}, {"cx": cx})

    assert result["status"] == "packed"
    assert rg.pending_reward_for(cx, "a@ex.com") == []
    gift = rg._row(cx, "id=?", (gift_id,))
    assert gift["status"] == "fulfilled"
    assert gift["fulfilled_order_id"] == oid


def test_mark_packed_idempotent_repack_does_not_error_or_reattach(monkeypatch):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    cx = sqlite3.connect(":memory:")
    _ord.init_orders_table(cx)
    oid = _ord.upsert_order(cx, source="test", external_ref="t1", email="a@ex.com", name="A",
                            items=[{"slug": "x", "name": "X", "qty": 1}], total_cents=1000,
                            status="new")
    rg.init_table(cx)
    rg.migrate_reward_columns(cx)
    gift_id = rg.add_reward_gift(cx, "a@ex.com", "GIFT-SAMPLE-3", "Placeholder", 1)

    from dashboard import actions
    action = actions.get_action("orders.mark_packed")
    action.executor({"order_id": oid}, {"cx": cx})  # first pack: attaches
    action.executor({"order_id": oid}, {"cx": cx})  # re-pack: no error, no re-attach

    gift = rg._row(cx, "id=?", (gift_id,))
    assert gift["status"] == "fulfilled"
    assert gift["fulfilled_order_id"] == oid  # unchanged, still this order


def test_mark_packed_flag_off_leaves_gift_pending(monkeypatch):
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    cx = sqlite3.connect(":memory:")
    _ord.init_orders_table(cx)
    oid = _ord.upsert_order(cx, source="test", external_ref="t1", email="a@ex.com", name="A",
                            items=[{"slug": "x", "name": "X", "qty": 1}], total_cents=1000,
                            status="new")
    rg.init_table(cx)
    rg.migrate_reward_columns(cx)
    rg.add_reward_gift(cx, "a@ex.com", "GIFT-SAMPLE-3", "Placeholder", 1)

    from dashboard import actions
    result = actions.get_action("orders.mark_packed").executor({"order_id": oid}, {"cx": cx})

    assert result["status"] == "packed"
    assert len(rg.pending_reward_for(cx, "a@ex.com")) == 1  # still pending, untouched


def test_mark_packed_never_touches_review_source_gifts(monkeypatch):
    """pending_reward_for (and the attach loop built on it) is reward-only —
    a review-source suggested gift must never be swept up by a pack."""
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    cx = sqlite3.connect(":memory:")
    _ord.init_orders_table(cx)
    oid = _ord.upsert_order(cx, source="test", external_ref="t1", email="a@ex.com", name="A",
                            items=[{"slug": "x", "name": "X", "qty": 1}], total_cents=1000,
                            status="new")
    rg.init_table(cx)
    rg.migrate_reward_columns(cx)
    review_gift_id = rg.add_suggestion(cx, review_id=99, email="a@ex.com", sku="GIFT-REVIEW",
                                       label="Review pick", reason="great review")
    rg.set_status(cx, review_gift_id, "approved", by="rae")

    from dashboard import actions
    actions.get_action("orders.mark_packed").executor({"order_id": oid}, {"cx": cx})

    review_gift = rg._row(cx, "id=?", (review_gift_id,))
    assert review_gift["status"] == "approved"  # untouched — not fulfilled by the reward attach path
    assert review_gift["fulfilled_order_id"] is None


# --- Board annotation (GET /api/orders), via the app test_client ---

def test_pack_board_surfaces_reward_gift_when_flag_on(monkeypatch, tmp_db):
    monkeypatch.setenv("REWARD_GIFTS_ENABLED", "1")
    app = _app(monkeypatch, tmp_db)
    _seed_order(tmp_db)
    key = app.dashboard.CONSOLE_SECRET or ""
    resp = app.app.test_client().get("/api/orders", headers={"X-Console-Key": key})
    assert resp.status_code == 200
    orders = resp.get_json()["data"]
    matches = [o for o in orders if (o.get("email") or "").strip().lower() == "a@ex.com"]
    assert matches, f"no order found for a@ex.com in {orders!r}"
    o = matches[0]
    assert any(g["gift_label"] == "Placeholder" for g in o.get("reward_gifts", []))


def test_pack_board_omits_reward_gifts_key_when_flag_off(monkeypatch, tmp_db):
    monkeypatch.delenv("REWARD_GIFTS_ENABLED", raising=False)
    app = _app(monkeypatch, tmp_db)
    _seed_order(tmp_db)
    key = app.dashboard.CONSOLE_SECRET or ""
    resp = app.app.test_client().get("/api/orders", headers={"X-Console-Key": key})
    assert resp.status_code == 200
    orders = resp.get_json()["data"]
    matches = [o for o in orders if (o.get("email") or "").strip().lower() == "a@ex.com"]
    assert matches
    o = matches[0]
    assert "reward_gifts" not in o

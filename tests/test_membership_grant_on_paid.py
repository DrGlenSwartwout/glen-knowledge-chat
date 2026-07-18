import sqlite3, pytest

app_mod = pytest.importorskip("app")
from dashboard import orders


@pytest.fixture
def cx(tmp_path, monkeypatch):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    app_mod.init_membership_tables(c)
    return c


def _order_with_membership(cx, email="grant-test@example.com", ref="INH-TESTMEM"):
    oid = orders.upsert_order(
        cx, source="inhouse", external_ref=ref,
        email=email, total_cents=9900,
        items=[{"slug": "membership:month", "name": "Monthly Membership",
                "qty": 1, "unit_cents": 9900, "line_cents": 9900,
                "kind": "membership", "tier": "month"}])
    return orders.get_order(cx, oid)


def test_grants_once_and_is_idempotent(cx):
    o = _order_with_membership(cx)
    assert app_mod._grant_membership_line_on_paid(cx, o) == "granted"
    assert app_mod._is_paid_member("grant-test@example.com") is True
    # second call on the same order does not double-grant
    assert app_mod._grant_membership_line_on_paid(cx, o) == "already"


def test_no_membership_line_is_noop(cx):
    oid = orders.upsert_order(
        cx, source="inhouse", external_ref="INH-NOMEM",
        email="x@example.com", total_cents=6997,
        items=[{"slug": "paracleanse", "name": "ParaCleanse",
                "qty": 1, "unit_cents": 6997, "line_cents": 6997}])
    o = orders.get_order(cx, oid)
    assert app_mod._grant_membership_line_on_paid(cx, o) == "none"


def test_none_order_is_noop(cx):
    assert app_mod._grant_membership_line_on_paid(cx, None) == "none"


def test_already_member_does_not_regrant(cx, monkeypatch):
    monkeypatch.setattr(app_mod._mp, "owns_group", lambda _cx, _e: True)
    o = _order_with_membership(cx, email="already@example.com", ref="INH-ALREADY")
    assert app_mod._grant_membership_line_on_paid(cx, o) == "member"

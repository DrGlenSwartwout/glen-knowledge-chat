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


def test_altpay_record_payment_delivers_membership(tmp_path, monkeypatch):
    """Alt-pay path (Zelle/check/owner-recorded) grants the membership too: the app
    registers _grant_membership_line_on_paid as orders' injection hook on import, and
    _record_payment_exec invokes it on the full-payment transition."""
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    app_mod.init_membership_tables(c)
    # Keep the exec focused on the grant wiring (points settler needs unrelated tables).
    monkeypatch.setattr(orders, "settle_order_points", lambda *a, **k: None)
    # The app wires the hook at import -- prove that seam is live. The hook is a thin
    # lambda that IGNORES the request cx and delegates to the own-connection wrapper
    # (_grant_membership_line_dep), so the grant never writes on the request cx.
    assert orders._membership_grant_hook is not None

    oid = orders.upsert_order(
        c, source="inhouse", external_ref="INH-ALTPAY",
        email="altpay@example.com", total_cents=9900,
        items=[{"slug": "membership:month", "name": "Monthly Membership",
                "qty": 1, "unit_cents": 9900, "line_cents": 9900,
                "kind": "membership", "tier": "month"}])
    res = orders._record_payment_exec(
        {"order_id": oid, "method": "Zelle"}, {"cx": c})
    assert res["pay_status"] == "paid"
    assert app_mod._is_paid_member("altpay@example.com") is True
    # Idempotent: re-recording is rejected before any re-grant (single grant row).
    assert c.execute(
        "SELECT COUNT(*) FROM order_membership_grants WHERE order_ref=?",
        ("INH-ALTPAY",)).fetchone()[0] == 1


def test_altpay_failed_grant_rolls_back_claim_and_is_retryable(tmp_path, monkeypatch):
    """The whole point of the isolation fix: if the grant RAISES, the claim row must
    roll back with it (atomicity), the payment must still be recorded (grant failure
    is swallowed, never fails the payment), and a later RETRY must still deliver the
    membership. Under the old bug the claim was written on the request cx, left
    pending when _grant_membership raised, then flushed by a downstream commit --
    orphaning the claim so every retry saw it and returned "already", permanently
    starving the membership. This drives the REAL alt-pay path (_record_payment_exec
    -> the registered own-connection hook)."""
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_mod, "LOG_DB", db)
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    app_mod.init_membership_tables(c)
    monkeypatch.setattr(orders, "settle_order_points", lambda *a, **k: None)

    email, ref = "rollback@example.com", "INH-ROLLBACK"
    oid = orders.upsert_order(
        c, source="inhouse", external_ref=ref,
        email=email, total_cents=9900,
        items=[{"slug": "membership:month", "name": "Monthly Membership",
                "qty": 1, "unit_cents": 9900, "line_cents": 9900,
                "kind": "membership", "tier": "month"}])

    # 1. Make the grant explode. It runs inside the hook's OWN connection's `with`
    #    block, so the exception must roll the claim INSERT back on that connection.
    def _boom(*a, **k):
        raise RuntimeError("grant write failed")
    monkeypatch.setattr(app_mod, "_grant_membership", _boom)

    # 2. Drive the real alt-pay path. The grant failure is swallowed inside
    #    _record_payment_exec, so this must NOT raise and the payment IS recorded.
    res = orders._record_payment_exec({"order_id": oid, "method": "Zelle"}, {"cx": c})
    assert res["pay_status"] == "paid"                      # (b) payment not lost
    assert orders.get_order(c, oid)["pay_status"] == "paid"  # durable on the order row

    # (a) NO claim row survived -- it rolled back atomically with the failed grant.
    assert c.execute(
        "SELECT COUNT(*) FROM order_membership_grants WHERE order_ref=?",
        (ref,)).fetchone()[0] == 0
    # ...and of course no membership was granted.
    assert app_mod._is_paid_member(email) is False

    # 3. Retry once the grant works again -- because the claim rolled back, the retry
    #    is NOT falsely short-circuited as "already"; it grants for real. (The payment
    #    already recorded, so we re-invoke the grant hook directly, as a heal/retry
    #    would.) This proves the retryability the old bug destroyed.
    monkeypatch.undo()  # restore the real _grant_membership (undoes _boom + settle stub)
    monkeypatch.setattr(app_mod, "LOG_DB", db)  # re-pin LOG_DB for the own-conn dep
    orders._membership_grant_hook(c, orders.get_order(c, oid))  # heal/retry the grant
    assert c.execute(
        "SELECT COUNT(*) FROM order_membership_grants WHERE order_ref=?",
        (ref,)).fetchone()[0] == 1
    assert app_mod._is_paid_member(email) is True

import importlib
import sqlite3
from dashboard import points

KEY = "test-console-secret"


def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    return cx


def test_add_is_atomically_idempotent_bypassing_has_entry():
    # Simulate the cross-process race: two _add calls with the SAME
    # (order_ref, reason, scope) that both bypass the has_entry fast-path.
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tok1")
    points._add(cx, "a@b.com", 500, "earn", "tok1")   # OR IGNORE -> no second row
    n = cx.execute("SELECT COUNT(*) FROM points_ledger "
                   "WHERE order_ref='tok1' AND reason='earn' AND scope='rm'").fetchone()[0]
    assert n == 1
    assert points.balance(cx, "a@b.com") == 500        # NOT 1000


def test_redeem_does_not_double_debit_on_same_ref():
    cx = _mk()
    points._add(cx, "a@b.com", 1000, "earn", "seed")
    points.redeem(cx, "a@b.com", value_cents=400, order_ref="tok1")
    # a racing duplicate redeem for the same order_ref is an atomic no-op
    points._add(cx, "a@b.com", -400, "redeem", "tok1")
    assert points.balance(cx, "a@b.com") == 600        # debited once, not twice


def test_distinct_keys_still_insert():
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tokA")
    points._add(cx, "a@b.com", 500, "earn", "tokB")     # different order_ref
    points._add(cx, "a@b.com", 500, "referral", "tokA")  # different reason
    assert cx.execute("SELECT COUNT(*) FROM points_ledger").fetchone()[0] == 3


def test_scope_discriminates_ship_credit_from_rm():
    cx = _mk()
    points._add(cx, "a@b.com", 500, "earn", "tok1", scope="rm")
    points._add(cx, "a@b.com", 500, "ship_overpay", "tok1", scope="ship_credit")
    assert points.balance(cx, "a@b.com", scope="rm") == 500
    assert points.balance(cx, "a@b.com", scope="ship_credit") == 500


def test_add_returns_authoritative_balance_when_ignored():
    cx = _mk()
    assert points._add(cx, "a@b.com", 500, "earn", "tok1") == 500
    # the ignored duplicate must return the TRUE balance (500), not an optimistic 1000
    assert points._add(cx, "a@b.com", 500, "earn", "tok1") == 500


def _reload(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", KEY)
    import app as appmod
    importlib.reload(appmod)
    appmod.app.config["TESTING"] = True
    return appmod


def _seed_duplicate(appmod):
    """Seed a duplicate (order_ref, reason, scope) row, bypassing the UNIQUE index
    (which init_points_table creates) by dropping it first, matching the brief's
    Step 6 guidance."""
    with sqlite3.connect(appmod.LOG_DB) as cx:
        points.init_points_table(cx)
        cx.execute("DROP INDEX IF EXISTS ux_points_order_ref_reason_scope")
        cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after,scope)
                      VALUES (?,?,?,?,?,?)""", ("a@b.com", 500, "earn", "tok1", 500, "rm"))
        cx.execute("""INSERT INTO points_ledger(email,delta_cents,reason,order_ref,balance_after,scope)
                      VALUES (?,?,?,?,?,?)""", ("a@b.com", 500, "earn", "tok1", 1000, "rm"))
        cx.commit()


def test_points_dedup_route_dry_run_then_apply(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    _seed_duplicate(appmod)
    c = appmod.app.test_client()

    dry = c.post("/api/console/points-dedup", headers={"X-Console-Key": KEY}).get_json()
    assert dry["ok"] is True
    assert dry["applied"] is False
    assert dry["duplicate_groups"] == 1
    assert dry["rows_removed"] == 0
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cx.execute(
            "SELECT COUNT(*) FROM points_ledger WHERE order_ref='tok1'").fetchone()[0] == 2

    applied = c.post("/api/console/points-dedup?apply=1", headers={"X-Console-Key": KEY}).get_json()
    assert applied["applied"] is True
    assert applied["rows_removed"] == 1
    assert applied["index_exists"] is True
    with sqlite3.connect(appmod.LOG_DB) as cx:
        rows = cx.execute(
            "SELECT id FROM points_ledger WHERE order_ref='tok1'").fetchall()
    assert len(rows) == 1

    # Re-running apply after dedup is a no-op (idempotent/re-runnable).
    again = c.post("/api/console/points-dedup?apply=1", headers={"X-Console-Key": KEY}).get_json()
    assert again["duplicate_groups"] == 0
    assert again["rows_removed"] == 0
    assert again["index_exists"] is True


def test_points_dedup_route_requires_key(monkeypatch, tmp_path):
    appmod = _reload(monkeypatch, tmp_path)
    assert appmod.app.test_client().post("/api/console/points-dedup").status_code == 401

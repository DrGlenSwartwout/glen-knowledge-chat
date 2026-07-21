import os, pytest
from dashboard import db, points

pg = bool(os.environ.get("PG_DSN"))

def _exercise(cx):
    points.init_points_table(cx)
    cx.execute("DELETE FROM points_ledger WHERE email=?", ("u@x.com",))
    cx.commit()
    assert points.balance(cx, "u@x.com") == 0
    points.earn(cx, "u@x.com", full_price_cents=10000, earn_pct=0.05, order_ref="o1")
    assert points.balance(cx, "u@x.com") == 500
    # idempotent: same (order_ref, reason, scope) inserts once
    points.earn(cx, "u@x.com", full_price_cents=10000, earn_pct=0.05, order_ref="o1")
    assert points.balance(cx, "u@x.com") == 500
    points.redeem(cx, "u@x.com", value_cents=200, order_ref="r1")
    assert points.balance(cx, "u@x.com") == 300
    assert points.has_entry(cx, order_ref="o1", reason="earn") is True

def test_points_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    cx = db.connect(str(tmp_path / "p.db"))
    _exercise(cx)
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_points_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS points_ledger")
    cx.commit()
    _exercise(cx)
    cx.close()

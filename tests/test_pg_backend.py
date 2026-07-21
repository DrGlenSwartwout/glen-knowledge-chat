import os, pytest, threading
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_parity_roundtrip(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS parity_t")
    cx.execute("CREATE TABLE parity_t (id BIGINT PRIMARY KEY, v TEXT)")
    cx.execute("INSERT INTO parity_t (id, v) VALUES (?, ?)", (1, "hi"))
    cx.commit()
    row = cx.execute("SELECT id, v FROM parity_t WHERE id=?", (1,)).fetchone()
    assert row[0] == 1 and row["v"] == "hi"
    cx.close()

@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_pg_rotate_race_no_lost_update(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    cx = db.connect("ignored")
    cx.execute("DROP TABLE IF EXISTS rota")
    cx.execute("CREATE TABLE rota (id BIGINT PRIMARY KEY, n BIGINT)")
    cx.execute("INSERT INTO rota (id, n) VALUES (1, 0)")
    cx.commit()
    WRITERS, ITERS = 6, 20
    errors = []
    def worker():
        try:
            c = db.connect("ignored")
            for _ in range(ITERS):
                row = c.execute("SELECT n FROM rota WHERE id=? FOR UPDATE", (1,)).fetchone()
                c.execute("UPDATE rota SET n=? WHERE id=?", (row[0] + 1, 1))
                c.commit()
            c.close()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=worker) for _ in range(WRITERS)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    final = cx.execute("SELECT n FROM rota WHERE id=?", (1,)).fetchone()[0]
    assert not errors, errors[:2]
    assert final == WRITERS * ITERS, f"lost updates: {final} != {WRITERS*ITERS}"
    cx.close()

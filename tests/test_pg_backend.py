import os, pytest
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

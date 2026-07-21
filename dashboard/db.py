"""Single DB access point so the backend (sqlite | postgres) is swappable.
Default is local SQLite — identical behavior to calling sqlite3.connect directly."""
import os
import sqlite3
from typing import Optional

def backend() -> str:
    return (os.environ.get("DB_BACKEND") or "sqlite").strip().lower()

def connect(db_path: str, *, timeout: float = 5.0):
    b = backend()
    if b == "sqlite":
        return sqlite3.connect(db_path, timeout=timeout)
    if b == "postgres":
        return _connect_postgres(db_path, timeout=timeout)
    raise ValueError("unknown DB_BACKEND: %r" % b)

def _connect_postgres(db_path: str, *, timeout: float):
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("DB_BACKEND=postgres but PG_DSN is unset")
    import psycopg  # optional dep
    from dashboard.pgcompat import translate_sql, HybridRow

    class _Cur:
        def __init__(self, cur):
            self._cur = cur
        def execute(self, sql, params=()):
            self._cur.execute(translate_sql(sql), tuple(params))
            return self
        def fetchone(self):
            row = self._cur.fetchone()
            if row is None:
                return None
            cols = [d.name for d in self._cur.description]
            return HybridRow(cols, row)
        def fetchall(self):
            rows = self._cur.fetchall()
            cols = [d.name for d in self._cur.description]
            return [HybridRow(cols, r) for r in rows]

    class _Conn:
        def __init__(self, conn):
            self._conn = conn
        def execute(self, sql, params=()):
            cur = self._conn.cursor()
            return _Cur(cur).execute(sql, params)
        def commit(self):
            self._conn.commit()
        def close(self):
            self._conn.close()

    return _Conn(psycopg.connect(dsn, connect_timeout=int(timeout)))

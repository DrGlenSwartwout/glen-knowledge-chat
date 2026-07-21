"""Single DB access point so the backend (sqlite | postgres) is swappable.
Default is local SQLite — identical behavior to calling sqlite3.connect directly."""
import os
import sqlite3

from dashboard.pgcompat import translate_sql, HybridRow

def backend() -> str:
    return (os.environ.get("DB_BACKEND") or "sqlite").strip().lower()

def connect(db_path: str, *, timeout: float = 5.0):
    b = backend()
    if b == "sqlite":
        return sqlite3.connect(db_path, timeout=timeout)
    if b == "postgres":
        return _connect_postgres(db_path, timeout=timeout)
    raise ValueError("unknown DB_BACKEND: %r" % b)

class _PgCursor:
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

class _PgConn:
    def __init__(self, conn):
        self._conn = conn
    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        return _PgCursor(cur).execute(sql, params)
    def commit(self):
        self._conn.commit()
    def rollback(self):
        self._conn.rollback()
    def close(self):
        self._conn.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        return False

def _connect_postgres(db_path: str, *, timeout: float):
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("DB_BACKEND=postgres but PG_DSN is unset")
    import psycopg  # optional dep
    _connect_timeout = max(1, int(round(timeout)))
    return _PgConn(psycopg.connect(dsn, connect_timeout=_connect_timeout))

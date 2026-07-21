"""Single DB access point so the backend (sqlite | postgres) is swappable.
Default is local SQLite — identical behavior to calling sqlite3.connect directly."""
import os
import sqlite3

from dashboard.pgcompat import translate_sql, HybridRow

def backend() -> str:
    return (os.environ.get("DB_BACKEND") or "sqlite").strip().lower()

def backend_of(cx) -> str:
    """The backend a given connection object belongs to: a _PgConn is 'postgres';
    a plain sqlite3.Connection (or anything without a .backend tag) is 'sqlite'."""
    return getattr(cx, "backend", "sqlite")

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
    def __iter__(self):
        # Match sqlite3.Cursor: `for row in cx.execute(...)` yields rows directly.
        desc = self._cur.description
        if desc is None:
            return
        cols = [d.name for d in desc]
        for row in self._cur:
            yield HybridRow(cols, row)

class _PgConn:
    backend = "postgres"
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool
        self._released = False
    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        return _PgCursor(cur).execute(sql, params)
    def commit(self):
        self._conn.commit()
    def rollback(self):
        self._conn.rollback()
    def _release(self):
        if not self._released:
            self._released = True
            self._pool.putconn(self._conn)
    def close(self):
        self._release()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._release()   # pooled resource: return on context exit
        return False
    def __del__(self):
        try:
            self._release()
        except Exception:
            pass

_PG_POOLS = {}          # dsn -> ConnectionPool
_PG_ENSURED = set()     # (dsn, schema) already CREATE SCHEMA'd
import threading as _threading
_PG_LOCK = _threading.Lock()

def _get_pg_pool(dsn, timeout):
    with _PG_LOCK:
        pool = _PG_POOLS.get(dsn)
        if pool is None:
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(dsn, min_size=2, max_size=10, open=True,
                                  kwargs={"connect_timeout": max(1, int(round(timeout)))})
            _PG_POOLS[dsn] = pool
        return pool

def _ensure_pg_schema(raw, dsn, schema):
    key = (dsn, schema)
    with _PG_LOCK:
        if key in _PG_ENSURED:
            return
    with raw.cursor() as c:
        c.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    raw.commit()
    with _PG_LOCK:
        _PG_ENSURED.add(key)

def _connect_postgres(db_path: str, *, timeout: float):
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        raise RuntimeError("DB_BACKEND=postgres but PG_DSN is unset")
    from dashboard.dbschema import schema_for_path
    schema = schema_for_path(db_path)  # already sanitized to [a-z0-9_] -> safe to quote-interpolate
    pool = _get_pg_pool(dsn, timeout)
    raw = pool.getconn()
    try:
        _ensure_pg_schema(raw, dsn, schema)
        with raw.cursor() as c:
            c.execute(f'SET search_path TO "{schema}"')
        raw.commit()
    except Exception:
        pool.putconn(raw)
        raise
    return _PgConn(raw, pool)

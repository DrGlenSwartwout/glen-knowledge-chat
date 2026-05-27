"""Supabase Postgres connection helper for practitioners table.

SQLite (chat_log.db) handles existing app data. This module adds a separate
connection to the Supabase Postgres instance for the practitioners table,
shared between the Practitioner Finder and the future Approved Portal.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras


def _conn_str() -> str:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL env var is not set")
    return url


@contextmanager
def supabase_conn():
    """Yield a psycopg2 connection with autocommit off. Caller commits or rollbacks."""
    conn = psycopg2.connect(_conn_str())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def supabase_cursor():
    """Yield a RealDictCursor (rows accessed by column name) with auto-commit on exit."""
    with supabase_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

"""PAYOFF validation: the pgcompat DDL-idiom translator alone (AUTOINCREMENT ->
IDENTITY, datetime('now') -> now()::text) makes UNPORTED modules' schema-init
functions work on Postgres, with ZERO source edits to those modules.

Candidates (verified by reading source, not modified):
  - dashboard/client_scans.py::init_client_scans_table
        CREATE TABLE ... id INTEGER PRIMARY KEY AUTOINCREMENT ...
        CREATE INDEX ...
        ALTER TABLE ... ADD COLUMN ... (valid unchanged on both backends)
        No PRAGMA / strftime / json_extract / cross-module FK.
  - dashboard/email_suppression.py::init_table
        CREATE TABLE ... created_at TEXT DEFAULT (datetime('now')) ...
        No AUTOINCREMENT, no PRAGMA / strftime / json_extract / FK.

Together the two modules exercise both idioms independently. Skip-guarded on
PG_DSN so this is a no-op in secretless CI.
"""
import os
import pytest

from dashboard import db

pg = bool(os.environ.get("PG_DSN"))


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_client_scans_unported_module_schema_creates_on_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    from dashboard import client_scans  # unported: no backend_of() branching inside

    cx = db.connect("/data/chat_log.db")
    try:
        cx.execute("DROP TABLE IF EXISTS client_scans")
        cx.commit()

        client_scans.init_client_scans_table(cx)  # source untouched

        row = cx.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name=?",
            ("client_scans",),
        ).fetchone()
        assert row is not None, "client_scans table was not created on Postgres"
    finally:
        cx.close()


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_email_suppression_unported_module_schema_creates_on_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    from dashboard import email_suppression  # unported: no backend_of() branching inside

    cx = db.connect("/data/chat_log.db")
    try:
        cx.execute("DROP TABLE IF EXISTS email_suppression")
        cx.commit()

        email_suppression.init_table(cx)  # source untouched

        row = cx.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name=?",
            ("email_suppression",),
        ).fetchone()
        assert row is not None, "email_suppression table was not created on Postgres"

        # Prove datetime('now') -> now()::text actually took effect end-to-end:
        # the DEFAULT fires and yields a non-null created_at on insert.
        email_suppression.add(cx, "person@example.com", "hard", "test", "unittest")
        got = cx.execute(
            "SELECT created_at FROM email_suppression WHERE email=?",
            ("person@example.com",),
        ).fetchone()
        assert got is not None and got["created_at"] is not None
    finally:
        cx.close()

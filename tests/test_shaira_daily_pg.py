import os, pytest
from dashboard import shaira_daily as sd
from dashboard import db

pg = bool(os.environ.get("PG_DSN"))

_DDL = ("CREATE TABLE IF NOT EXISTS daily_reports ("
        "owner TEXT, report_date TEXT, report_md TEXT, metrics_json TEXT, "
        "created_at TEXT DEFAULT (datetime('now')), "
        "PRIMARY KEY(owner, report_date))")

# Minimal workspace tables gather_metrics reads (todos / todo_time_sessions /
# todo_messages). Real schema (app.py _init_workspace_schema) has more columns;
# these carry only what gather_metrics's queries touch, enough for an empty-DB
# smoke of its COALESCE/CASE/UNION ALL SQL on both backends.
_TODOS_DDL = ("CREATE TABLE IF NOT EXISTS todos ("
              "id INTEGER PRIMARY KEY, owner TEXT, phase TEXT, title TEXT, "
              "done_at TEXT, first_started_at TEXT, action_note TEXT)")
_SESSIONS_DDL = ("CREATE TABLE IF NOT EXISTS todo_time_sessions ("
                 "id INTEGER PRIMARY KEY, todo_id INTEGER, owner TEXT, "
                 "started_at TEXT, ended_at TEXT, duration_seconds INTEGER)")
_MESSAGES_DDL = ("CREATE TABLE IF NOT EXISTS todo_messages ("
                 "id INTEGER PRIMARY KEY, todo_id INTEGER, role TEXT, "
                 "content TEXT, created_at TEXT)")


def _ensure_reports_table(dbp):
    # Production always has this table pre-created by app.py's schema init
    # (app.py:29577) before shaira_daily is ever called; tests standing up an
    # isolated DB have to do that setup step themselves.
    with db.connect(dbp) as cx:
        cx.execute(_DDL)
        cx.commit()


def _seed(dbp):
    with db.connect(dbp) as cx:
        cx.execute(_DDL)
        cx.execute("INSERT INTO daily_reports (owner, report_date, report_md, metrics_json, created_at) "
                   "VALUES (?,?,?,?,?) ON CONFLICT(owner, report_date) DO UPDATE SET report_md=excluded.report_md",
                   ("shaira", "2026-07-21", "# Shaira\n[HIGH] follow up", "{}", "2026-07-21T00:00:00"))
        cx.commit()


def _ensure_workspace_tables(dbp):
    with db.connect(dbp) as cx:
        cx.execute(_TODOS_DDL)
        cx.execute(_SESSIONS_DDL)
        cx.execute(_MESSAGES_DDL)
        cx.commit()


def test_latest_report_sqlite(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    _ensure_reports_table(dbp)
    assert sd.latest_report(dbp, "shaira")["empty"] is True
    _seed(dbp)
    r = sd.latest_report(dbp, "shaira")
    assert r["empty"] is False and r["markdown"].startswith("# Shaira")
    assert r["report_date"] == "2026-07-21"


def test_gather_metrics_empty_db_sqlite(tmp_path):
    dbp = str(tmp_path / "chat_log.db")
    _ensure_workspace_tables(dbp)
    m = sd.gather_metrics(dbp, "shaira")
    assert m["completed_count"] == 0
    assert m["in_process_count"] == 0
    assert m["plan_count"] == 0
    assert m["pending_ask_count"] == 0
    assert m["time_logged_seconds"] == 0
    assert m["stuck_hits"] == []


@pytest.mark.skipif(not pg, reason="PG_DSN not set")
def test_latest_report_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    with db.connect("/data/chat_log.db") as cx:
        cx.execute("DROP TABLE IF EXISTS daily_reports"); cx.commit()
    dbp = "/data/chat_log.db"
    _ensure_reports_table(dbp)
    assert sd.latest_report(dbp, "shaira")["empty"] is True
    _seed(dbp)
    r = sd.latest_report(dbp, "shaira")
    assert r["empty"] is False and r["markdown"].startswith("# Shaira")

# No Postgres gather_metrics smoke: `migtest` already carries the real
# todos/todo_time_sessions/todo_messages/todo_steps schema (with FK
# dependents) from app.py's own schema init, shared across P04 tasks/
# worktrees. Dropping/recreating those here would either fail on FK
# CASCADE or clobber state other tasks rely on, so gather_metrics's
# empty-DB smoke stays SQLite-only (isolated tmp_path); its dialect-
# neutral SQL is otherwise covered by the brief's own risk assessment
# and by pgimport.sh exercising app.py's full schema init under PG.

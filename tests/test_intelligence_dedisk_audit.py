import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (ROOT / rel).read_text()


def test_intelligence_uses_db_not_files():
    s = _src("dashboard/intelligence.py")
    assert "db.connect(" in s
    assert "intelligence_briefings" in s
    # No filesystem briefing store anymore (DB-backed): no file read/writes and
    # no per-slug .md path helpers. NOTE: DATA_DIR may still appear purely to
    # locate chat_log.db in _default_db_path(), exactly as gmail_token/app.LOG_DB
    # do — that is the DB *locator*, not a briefing file store, so it is allowed.
    assert ".write_text(" not in s
    assert ".read_text(" not in s
    assert "_slug_path" not in s
    assert "_links_path" not in s


def test_shaira_daily_uses_db_connect():
    s = _src("dashboard/shaira_daily.py")
    assert "db.connect(" in s
    assert "sqlite3.connect(" not in s

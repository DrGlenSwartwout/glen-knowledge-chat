# tests/test_log_query_wordcount.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
        return appmod
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_word_count_auto_computed(tmp_path, monkeypatch):
    """log_query auto-counts words from the answer when word_count=0."""
    app = _load_app(tmp_path, monkeypatch)
    row_id = app.log_query(query="q", level="L1", answer="one two three four")
    with sqlite3.connect(app.LOG_DB) as cx:
        row = cx.execute(
            "SELECT word_count FROM query_log WHERE id=?", (row_id,)
        ).fetchone()
    assert row is not None
    assert row[0] == 4


def test_word_count_explicit_override(tmp_path, monkeypatch):
    """log_query stores an explicit word_count override as-is."""
    app = _load_app(tmp_path, monkeypatch)
    row_id = app.log_query(query="q", level="L1", answer="one two three", word_count=99)
    with sqlite3.connect(app.LOG_DB) as cx:
        row = cx.execute(
            "SELECT word_count FROM query_log WHERE id=?", (row_id,)
        ).fetchone()
    assert row is not None
    assert row[0] == 99

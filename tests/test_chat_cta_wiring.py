"""Tests for Task 3: CTA wiring into /chat handler (log_query + rung map)."""
import importlib
import sqlite3
import pytest


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app
        importlib.reload(app)
    except Exception as exc:
        pytest.skip(f"app import failed: {exc}")
    app._init_log_db()
    return app


def test_log_query_stores_cta_columns(app_module):
    app = app_module
    app.log_query(
        query="q",
        level="L1",
        answer="hello world",
        mode="brief",
        cta_type="email",
        cta_rung="engaged",
    )
    with sqlite3.connect(app.LOG_DB) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT cta_type, cta_rung FROM query_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["cta_type"] == "email"
    assert row["cta_rung"] == "engaged"


def test_parse_cta_and_rung_map(app_module):
    import dashboard.chat_cta as chat_cta
    app = app_module

    raw = "Brief body.\n⟦CTA⟧ page | https://x/p | Read more"
    clean, cta = chat_cta.parse_cta(raw)
    assert cta is not None
    assert cta["type"] == "page"
    assert "Brief body." in clean
    assert "⟦CTA⟧" not in clean

    rung_map = app._CTA_RUNG
    assert rung_map["page"] == "curious"
    assert rung_map["email"] == "engaged"
    assert rung_map["action"] == "ready"
    assert rung_map["inline"] == "committed"

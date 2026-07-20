"""Routes + wiring for the per-line invoice note feature:
  - GET/DELETE /api/invoice-snippets (owner-gated shared saved-message library)
  - _harvest_line_note_snippets auto-saves line notes on invoice save
  - _invoice_line_view carries `note` through the customer whitelist gate
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import rbac as _rbac


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        cx.commit()
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    appmod.app.config["TESTING"] = True
    monkeypatch.setattr(appmod, "_bos_actor",
                        lambda: _rbac.Actor(role="owner", name="glen"))
    return appmod, appmod.app.test_client()


def test_snippets_empty_then_delete_owner_gated(tmp_path, monkeypatch):
    appmod, client = _app(tmp_path, monkeypatch)
    r = client.get("/api/invoice-snippets")
    assert r.status_code == 200 and r.get_json() == {"ok": True, "snippets": []}
    # Non-owner is rejected on both read and delete.
    monkeypatch.setattr(appmod, "_bos_actor", lambda: None)
    assert client.get("/api/invoice-snippets").status_code == 401
    assert client.delete("/api/invoice-snippets/1").status_code == 401


def test_harvest_autosaves_then_dropdown_lists_then_delete(tmp_path, monkeypatch):
    appmod, client = _app(tmp_path, monkeypatch)
    # Simulate an invoice save harvesting its line notes (create + edit both call this).
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        appmod._harvest_line_note_snippets(cx, [
            {"slug": "a", "note": "Take with food."},
            {"slug": "b", "note": "  "},              # blank -> ignored
            {"slug": "c"},                            # no note key -> ignored
            {"slug": "d", "note": "Take with food."}, # dup -> not a second row
        ])
    snaps = client.get("/api/invoice-snippets").get_json()["snippets"]
    assert [s["text"] for s in snaps] == ["Take with food."]
    sid = snaps[0]["id"]
    # Prune it from the shared library.
    r = client.delete(f"/api/invoice-snippets/{sid}")
    assert r.status_code == 200 and r.get_json()["removed"] is True
    assert client.get("/api/invoice-snippets").get_json()["snippets"] == []


def test_invoice_line_view_passes_note_through_whitelist(tmp_path, monkeypatch):
    appmod, _ = _app(tmp_path, monkeypatch)
    out = appmod._invoice_line_view(
        {"slug": "ghost", "name": "X", "qty": 1, "unit_cents": 100,
         "line_cents": 100, "note": "  Shake well.  "})
    assert out.get("note") == "Shake well."        # stripped, present
    # A line with no note must not carry an empty `note` key onto the customer payload.
    out2 = appmod._invoice_line_view(
        {"slug": "ghost", "name": "X", "qty": 1, "unit_cents": 100, "line_cents": 100})
    assert "note" not in out2

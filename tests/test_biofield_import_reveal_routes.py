"""POST /author/<id>/e4l/import-reveal imports synthesized reveal layers as
needs-review chain rows. synthesize_reveal_layers is monkeypatched so the test
never runs the real vault pipeline; import_layers_to_test runs for real on a tmp db."""
import sqlite3
import pytest

from biofield_local_app import create_app
import dashboard.biofield_reveal_import as RI


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    # dashboard/__init__.py captures CONSOLE_SECRET at import; reloading
    # app does not reset it, so clear the copy the guard actually reads.
    import dashboard as _d; monkeypatch.setattr(_d, "CONSOLE_SECRET", "", raising=False)


_FRESH = {"found": True, "scan_id": 900, "scan_date": "2026-06-22", "days_ago": 3,
          "fresh": True, "layers": [
              {"n": 1, "title": "Oxidative load", "summary": "",
               "most_affected": "Cell membrane", "remedy_name": "Neuro Magnesium"}]}
_STALE = {"found": True, "scan_id": 900, "scan_date": "2026-06-01", "days_ago": 24,
          "fresh": False, "layers": []}
_NONE = {"found": False, "scan_id": None, "scan_date": None, "days_ago": None,
         "fresh": False, "layers": []}


def _new_test_with_email(client, email):
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    client.post(f"/author/{tid}/header", json={"name": "Jane", "email": email,
                                               "date": "2026-06-25"})
    return tid


def test_import_writes_rows_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is True and j["imported"] == 1
    # row landed, unconfirmed
    cx = sqlite3.connect(db)
    row = cx.execute("SELECT remedy, confirmed FROM biofield_auth_chain").fetchone()
    assert row[0] == "Neuro Magnesium" and row[1] == 0


def test_import_rejects_stale_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _STALE)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is False and "24" in j["reason"]


def test_import_needs_confirm_then_appends_with_force(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    client.post(f"/author/{tid}/e4l/import-reveal", json={})          # first import (1 row)
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j == {"ok": False, "needs_confirm": True, "existing": 1}
    j2 = client.post(f"/author/{tid}/e4l/import-reveal", json={"force": True}).get_json()
    assert j2["ok"] is True and j2["imported"] == 1
    cx = sqlite3.connect(db)
    assert cx.execute("SELECT COUNT(*) FROM biofield_auth_chain").fetchone()[0] == 2


def test_import_no_client_email(tmp_path, monkeypatch):
    monkeypatch.setattr(RI, "synthesize_reveal_layers", lambda *a, **k: _FRESH)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = client.post("/author/new").headers["Location"].rstrip("/").split("/")[-1]
    j = client.post(f"/author/{tid}/e4l/import-reveal", json={}).get_json()
    assert j["ok"] is False and "client" in j["reason"].lower()


def test_import_handles_synthesis_failure(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("pinecone down")
    monkeypatch.setattr(RI, "synthesize_reveal_layers", boom)
    db = str(tmp_path / "chat_log.db")
    client = create_app(db, scan_lookup=lambda e: _NONE).test_client()
    tid = _new_test_with_email(client, "jane@x.com")
    r = client.post(f"/author/{tid}/e4l/import-reveal", json={})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is False and "fail" in j["reason"].lower()

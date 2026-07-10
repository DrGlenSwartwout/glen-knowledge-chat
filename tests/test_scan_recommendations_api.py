"""Console sync for a scan's recommendations. Prod cannot read e4l.db, so the local
pusher POSTs here. Mirrors /api/console/client-scans/sync."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import scan_recommendations as sr

HDRS = {"X-Console-Key": "testkey"}
EMAIL = "caregiver@example.com"

BATCH = [{"email": EMAIL, "scans": [{
    "scan_id": "1037250", "scan_date": "2026-07-02", "items": [
        {"item_code": "BFA", "priority_rank": 1, "protocol_days": 15,
         "section": "Infoceuticals", "category": "BFA", "label": "Big Field Aligner"},
        {"item_code": "ER2", "priority_rank": 2, "protocol_days": 2,
         "section": "miHealth Functions", "category": "ER", "label": "Large Intestine"},
    ]}]}]


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return app.app.test_client()


def _rows(tmp_db, scan_id="1037250"):
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        sr.init_table(cx)
        return sr.for_scan(cx, EMAIL, scan_id)


def test_sync_requires_the_console_key(client):
    assert client.post("/api/console/scan-recommendations/sync", json={"batch": BATCH}).status_code == 401


def test_sync_writes_the_rows(client, tmp_db):
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["rows"] == 2 and body["clients"] == 1 and body["scans"] == 1
    assert [x["item_code"] for x in _rows(tmp_db)] == ["BFA", "ER2"]


def test_section_survives_the_round_trip(client, tmp_db):
    client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    got = {x["item_code"]: x["section"] for x in _rows(tmp_db)}
    assert got == {"BFA": "Infoceuticals", "ER2": "miHealth Functions"}


def test_a_repush_is_idempotent(client, tmp_db):
    for _ in range(2):
        assert client.post("/api/console/scan-recommendations/sync", headers=HDRS,
                           json={"batch": BATCH}).status_code == 200
    assert len(_rows(tmp_db)) == 2


def test_a_missing_batch_is_rejected(client):
    assert client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={}).status_code == 400


def test_a_bad_item_does_not_abort_the_whole_batch(client, tmp_db):
    bad = [{"email": EMAIL, "scans": [{"scan_id": "1037250", "scan_date": "2026-07-02",
                                       "items": ["not-a-dict", BATCH[0]["scans"][0]["items"][0]]}]}]
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": bad})
    assert r.status_code == 200
    assert [x["item_code"] for x in _rows(tmp_db)] == ["BFA"]


def test_a_client_with_a_blank_email_is_skipped_not_fatal(client, tmp_db):
    mixed = [{"email": "", "scans": BATCH[0]["scans"]}, BATCH[0]]
    r = client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": mixed})
    assert r.status_code == 200
    assert r.get_json()["rows"] == 2
    assert len(_rows(tmp_db)) == 2


def test_this_endpoint_sends_no_email(client, tmp_db, monkeypatch):
    """Slice 1 renders nothing and notifies nobody. A future slice reads the table."""
    sent = []
    for name in ("_send_reveal_link", "_notify_client_of_reply"):
        if hasattr(_app(), name):
            monkeypatch.setattr(_app(), name, lambda *a, **k: sent.append(name))
    client.post("/api/console/scan-recommendations/sync", headers=HDRS, json={"batch": BATCH})
    assert sent == []

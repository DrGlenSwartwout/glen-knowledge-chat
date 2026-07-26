import sqlite3
import os
import tempfile

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="eye-review-routes-"))
import app as appmod
from dashboard import eye_vision_report
from dashboard import eye_vision_review_requests


REPORT = {
    "section_key": "eye_vision_report",
    "title": "Eye report",
    "status": "ready",
    "scan_date": "2026-07-25",
    "history_eye_issue": True,
    "findings": [{"name": "Vision", "client_text": "Mapped relationship"}],
    "limitations": [],
}


def _setup(monkeypatch, tmp_path, paid):
    path = str(tmp_path / "portal.db")
    monkeypatch.setattr(appmod, "LOG_DB", path)
    monkeypatch.setattr(
        appmod, "_portal_record_for",
        lambda cx, token: (
            {"email": "client@example.com", "name": "Client", "content": {}}
            if token == "good" else None))
    monkeypatch.setattr(appmod, "_household_view_enabled", lambda: False)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda email: paid)
    monkeypatch.setattr(
        eye_vision_report, "build_portal_block",
        lambda *args, **kwargs: dict(REPORT))
    return appmod.app.test_client(), path


def test_nonmember_gets_upgrade_gate(monkeypatch, tmp_path):
    client, _ = _setup(monkeypatch, tmp_path, paid=False)
    response = client.post(
        "/api/portal/good/eye-vision-report/request-review")
    assert response.status_code == 402
    assert response.get_json()["upgrade_url"] == "/membership"


def test_paid_member_request_is_queued(monkeypatch, tmp_path):
    client, path = _setup(monkeypatch, tmp_path, paid=True)
    response = client.post(
        "/api/portal/good/eye-vision-report/request-review")
    assert response.status_code == 200
    cx = sqlite3.connect(path)
    cx.row_factory = sqlite3.Row
    assert eye_vision_review_requests.pending(cx)[0]["email"] == "client@example.com"


def test_console_queue_is_authorized_and_requires_reviewer(
        monkeypatch, tmp_path):
    client, path = _setup(monkeypatch, tmp_path, paid=True)
    client.post("/api/portal/good/eye-vision-report/request-review")
    monkeypatch.setattr(appmod, "_portal_console_ok", lambda: False)
    assert client.get("/api/console/eye-vision-reviews").status_code == 401
    monkeypatch.setattr(appmod, "_portal_console_ok", lambda: True)
    queue = client.get("/api/console/eye-vision-reviews").get_json()["pending"]
    request_id = queue[0]["id"]
    response = client.post("/api/console/eye-vision-reviews", json={
        "id": request_id, "email": "client@example.com",
        "action": "approve", "report": REPORT,
    })
    assert response.status_code == 400
    response = client.post("/api/console/eye-vision-reviews", json={
        "id": request_id, "email": "client@example.com",
        "action": "approve", "report": REPORT, "reviewer_id": "Dr. Glen",
    })
    assert response.status_code == 200
    assert response.get_json()["request"]["status"] == "approved"

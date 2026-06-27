"""Cert cohort portal notification emails."""
from dashboard import cert_notify as cn
from dashboard import practitioner_portal as pp


def _stub_links(monkeypatch):
    monkeypatch.setattr(pp, "create_magic_link_token", lambda pid, email, **k: "MAGICTOK")
    monkeypatch.setattr(pp, "id_for_email", lambda email: None)


def _capture():
    calls = []
    def _send(to, subject, body, from_name=None, html=None):
        calls.append({"to": to, "subject": subject, "body": body, "from_name": from_name})
        return {"id": "x"}
    return calls, _send


def test_feedback_ready_approved(monkeypatch):
    _stub_links(monkeypatch)
    calls, send = _capture()
    assert cn.send_feedback_ready("stu@x.com", "Stu", "approved", practitioner_id=7, send=send) is True
    assert len(calls) == 1
    c = calls[0]
    assert c["to"] == "stu@x.com" and "approved" in c["subject"].lower()
    assert "/practitioner/login-verify?token=MAGICTOK" in c["body"]
    assert "—" not in c["body"]  # voice: no em dash


def test_feedback_ready_refine_subject(monkeypatch):
    _stub_links(monkeypatch)
    calls, send = _capture()
    cn.send_feedback_ready("a@x.com", "A", "refine", send=send)
    assert "feedback" in calls[0]["subject"].lower()


def test_assignment_notice_has_record_and_portal_links(monkeypatch):
    _stub_links(monkeypatch)
    calls, send = _capture()
    rec = "https://illtowell.com/results?tag=ash-cert-l1&p=TOK"
    assert cn.send_assignment_notice("a@x.com", "A", rec, practitioner_id=9, send=send) is True
    body = calls[0]["body"]
    assert rec in body                                   # direct one-click record link
    assert "/practitioner/login-verify?token=MAGICTOK" in body  # portal link


def test_blank_email_no_send(monkeypatch):
    _stub_links(monkeypatch)
    calls, send = _capture()
    assert cn.send_feedback_ready("", "A", "approved", send=send) is False
    assert cn.send_assignment_notice("  ", "A", "url", send=send) is False
    assert calls == []


def test_send_failure_is_caught(monkeypatch):
    _stub_links(monkeypatch)
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    assert cn.send_feedback_ready("a@x.com", "A", "approved", send=_boom) is False

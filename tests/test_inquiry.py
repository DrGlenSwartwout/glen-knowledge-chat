import io
import pytest
import smtplib

# Reuse the _load_app helper from the route tests
from tests.test_begin_routes import _load_app


class FakeSMTP:
    """Drop-in replacement for smtplib.SMTP that records every send."""
    instances = []
    def __init__(self, host, port=0, *a, **k):
        self.host = host; self.port = port
        self.starttls_called = False
        self.login_args = None
        self.sent = []          # list of (from_addr, to_addrs, msg_bytes)
        self.quit_called = False
        FakeSMTP.instances.append(self)
    def starttls(self, *a, **k): self.starttls_called = True
    def login(self, u, p): self.login_args = (u, p)
    def sendmail(self, frm, to, msg): self.sent.append((frm, to, msg))
    def quit(self): self.quit_called = True
    # support context-manager use just in case the helper uses with-statement
    def __enter__(self): return self
    def __exit__(self, *a): self.quit(); return False


@pytest.fixture(autouse=True)
def reset_fakesmtp():
    FakeSMTP.instances = []
    yield


def test_init_inquiry_tables_creates_six_tables_idempotent(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    import sqlite3
    cx = sqlite3.connect(db)
    # call twice; must not raise; must create the tables
    app_module.init_inquiry_tables(cx)
    app_module.init_inquiry_tables(cx)
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {"inquiries","inquiry_practitioners","inquiry_reply_tokens",
                "inquiry_replies","inquiry_reply_impressions",
                "practitioner_inquiry_opt_outs"}
    assert expected.issubset(names)


def test_send_inquiry_email_honors_reply_to(monkeypatch):
    app_module = _load_app()
    # set minimal SMTP env so the helper does real flow with FakeSMTP
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("SMTP_FROM", "hello@remedymatch.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok = app_module._send_inquiry_email(
        "pract@example.com", "Test subject", "Test body line.",
        reply_to="client@example.com")
    assert ok is True
    assert len(FakeSMTP.instances) == 1
    inst = FakeSMTP.instances[0]
    assert inst.starttls_called and inst.quit_called
    assert inst.sent
    frm, to, raw = inst.sent[0]
    assert "pract@example.com" in to
    # Reply-To header must be set to the client
    # raw may be str (as_string()) or bytes (as_bytes()) depending on the send path
    raw_lower = raw.lower() if isinstance(raw, str) else raw.lower().decode("utf-8", errors="replace")
    raw_norm  = raw        if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
    assert "reply-to: client@example.com" in raw_lower or "Reply-To: client@example.com" in raw_norm


def test_send_inquiry_email_returns_false_on_smtp_failure(monkeypatch):
    app_module = _load_app()
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "pw")
    class BrokenSMTP(FakeSMTP):
        def sendmail(self, *a, **k):
            raise smtplib.SMTPException("boom")
    monkeypatch.setattr(smtplib, "SMTP", BrokenSMTP)
    ok = app_module._send_inquiry_email("x@e.com","s","b",reply_to="c@e.com")
    assert ok is False  # never raises


def test_send_inquiry_email_no_smtp_env_returns_true(monkeypatch):
    """Dev fallback: with no SMTP_HOST/USER/PASS, the helper logs to stdout and returns True."""
    app_module = _load_app()
    for k in ("SMTP_HOST","SMTP_USER","SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    ok = app_module._send_inquiry_email("x@e.com","s","b",reply_to=None)
    assert ok is True

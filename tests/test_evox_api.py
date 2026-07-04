# tests/test_evox_api.py  (needs doppler — imports app)
import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

def test_send_evox_email_builds_mixed_with_ics(monkeypatch):
    captured = {}
    class FakeSMTP:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def sendmail(self, frm, to, msg): captured["msg"] = msg
    monkeypatch.setattr(appmod, "SMTP_HOST", "smtp.test")
    monkeypatch.setattr(appmod, "SMTP_USER", "u"); monkeypatch.setattr(appmod, "SMTP_PASS", "p")
    monkeypatch.setattr(appmod.smtplib, "SMTP", FakeSMTP)
    mode, err = appmod.send_evox_email("c@x.com", "C", "EVOX confirmed",
                                       "<p>hi</p>", "hi", b"BEGIN:VCALENDAR\r\nEND:VCALENDAR\r\n")
    assert mode == "smtp" and err is None
    assert "text/calendar" in captured["msg"] and "multipart/mixed" in captured["msg"]

"""The weekly answer-audit endpoint: cron-gated, emails Glen ONLY on findings.

No network — the audit's ask() is stubbed via the _load_answer_audit seam, so
these run in CI like everything else.
"""
import types

import pytest


@pytest.fixture
def client(monkeypatch):
    import app
    monkeypatch.setenv("CRON_SECRET", "sekret")
    return app, app.app.test_client()


def _stub_audit(app, monkeypatch, answer_for):
    """Replace the loaded audit module with one whose ask() returns canned text
    and capture any email instead of sending it."""
    real = app._load_answer_audit()
    stub = types.SimpleNamespace(
        QUESTIONS=real.QUESTIONS,
        load_catalog=real.load_catalog,
        audit=real.audit,
        ask=lambda base, q, timeout=180: answer_for(q),
    )
    monkeypatch.setattr(app, "_load_answer_audit", lambda: stub)
    sent = []
    monkeypatch.setattr(app, "_send_full_report_email",
                        lambda *a, **k: sent.append(a))
    return sent


def test_requires_cron_secret(client):
    app, c = client
    assert c.post("/api/cron/answer-audit").status_code == 401
    assert c.post("/api/cron/answer-audit",
                  headers={"X-Cron-Secret": "wrong"}).status_code == 401


def test_clean_answers_send_no_email(client, monkeypatch):
    app, c = client
    clean = ("[Terrain Restore](https://illtowell.com/begin/product/terrain-restore)"
             " is $69.97 list price.")
    sent = _stub_audit(app, monkeypatch, lambda q: clean)
    r = c.post("/api/cron/answer-audit", headers={"X-Cron-Secret": "sekret"})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True
    assert body["flagged"] == 0 and body["emailed"] is False
    assert sent == [], "clean run must not email"


def test_a_bad_answer_flags_and_emails_glen(client, monkeypatch):
    app, c = client
    bad = ("[NIR Brain Frequency Helmet]"
           "(https://illtowell.com/begin/product/nir-brain-frequency-helmet) "
           "Price: $754 (includes $132 shipping)")
    sent = _stub_audit(app, monkeypatch, lambda q: bad)
    r = c.post("/api/cron/answer-audit", headers={"X-Cron-Secret": "sekret"})
    body = r.get_json()
    assert body["flagged"] > 0 and body["emailed"] is True
    assert len(sent) == 1
    to, name, subject, text = sent[0]
    assert to == "drglenswartwout@gmail.com"
    assert "754" in text  # the flagged figure is in the email Glen gets

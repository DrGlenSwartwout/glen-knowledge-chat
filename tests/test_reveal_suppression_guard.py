"""Reveal-ready emails (Your Biofield Analysis is ready) route through
_send_inquiry_email, which is EXEMPT from the email suppression guard. These tests
pin the correction: the reveal-ready send sites (_send_reveal_link + the
/api/e4l/reveal-draft ingest) must NOT email a suppressed (hard-bounced) address.
Context: test addresses a@b.com/t@x.com were getting reveal emails ~3x/day."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load(mod):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"{mod} not importable: {e}")


def _app_db(monkeypatch, tmp_path):
    app_module = _load("app")
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals as br, email_suppression as es
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        es.init_table(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, "
                   "created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return app_module, db


def test_send_reveal_link_skips_suppressed(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br, email_suppression as es
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "a@b.com", "2026-06-22", {"body": "x"}, [], "src")
        es.add(cx, "a@b.com", "hard", "NXDOMAIN", "bounce-scan")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append(to) or True)
    ok = app_module._send_reveal_link(rid)
    assert ok is False                       # suppressed -> not sent
    assert sent == []                        # the inquiry sender was never called
    with sqlite3.connect(db) as cx:
        assert not br.get(cx, rid)["notified_at"]
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='a@b.com'").fetchone()[0]
    assert n == 0                            # no token minted for a dead address


def test_send_reveal_link_still_sends_normal(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        rid, _ = br.upsert(cx, "real@gmail.com", "2026-06-22", {"body": "x"}, [], "src")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append(to) or True)
    ok = app_module._send_reveal_link(rid)
    assert ok is True and sent == ["real@gmail.com"]


def test_ingest_does_not_email_suppressed(monkeypatch, tmp_path):
    app_module, db = _app_db(monkeypatch, tmp_path)
    key = app_module.os.environ.get("CRON_SECRET") or app_module.CONSOLE_SECRET or ""
    if not key:
        pytest.skip("no secret")
    from dashboard import email_suppression as es
    with sqlite3.connect(db) as cx:
        es.add(cx, "a@b.com", "hard", "NXDOMAIN", "bounce-scan")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, *a, **k: sent.append(to) or True)
    c = app_module.app.test_client()
    r = c.post("/api/e4l/reveal-draft", headers={"X-Console-Key": key},
               json={"email": "a@b.com", "scan_date": "2026-06-22",
                     "interpretation": {"body": "x"},
                     "layers": [{"n": 1, "title": "L", "summary": "s",
                                 "patterns": [], "remedy": None}]})
    assert r.get_json().get("ok") is True    # draft still created
    assert sent == []                        # but no reveal email to the dead address

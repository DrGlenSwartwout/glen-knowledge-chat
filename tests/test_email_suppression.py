import sqlite3
from dashboard import email_suppression as es


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    es.init_table(cx)
    return cx


def test_add_and_is_suppressed_case_insensitive():
    cx = _cx()
    es.add(cx, "Dead@Domain.com", "hard", "NXDOMAIN", "bounce-scan")
    assert es.is_suppressed(cx, "dead@domain.com") is True
    assert es.is_suppressed(cx, "  DEAD@domain.com ") is True
    assert es.is_suppressed(cx, "other@x.com") is False
    assert es.is_suppressed(cx, "") is False


def test_add_is_idempotent():
    cx = _cx()
    es.add(cx, "a@b.com", "hard", "no such user", "bounce-scan")
    es.add(cx, "a@b.com", "hard", "no such user", "bounce-scan")
    assert len(es.list_recent(cx)) == 1


def test_is_suppressed_no_table_is_false():
    cx = sqlite3.connect(":memory:")  # table never created
    assert es.is_suppressed(cx, "x@y.com") is False


# ── send_email guard ──────────────────────────────────────────────────────────

class _Exec:
    def execute(self):
        return {"id": "123", "threadId": "t1"}


class _Svc:
    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId=None, body=None):
        return _Exec()


def test_send_email_skips_suppressed(tmp_path, monkeypatch):
    from dashboard import inbox
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    es.init_table(cx)
    es.add(cx, "x@dead.com", "hard", "NXDOMAIN", "test")
    cx.close()
    monkeypatch.setattr(inbox, "_db_path", lambda: db)
    called = []
    monkeypatch.setattr(inbox, "_get_gmail_service", lambda: called.append(1) or _Svc())
    r = inbox.send_email("x@dead.com", "subj", "body")
    assert r == {"skipped": "suppressed"}
    assert called == []                       # gmail service never constructed


def test_send_email_sends_to_normal_address(tmp_path, monkeypatch):
    from dashboard import inbox
    db = str(tmp_path / "chat_log.db")
    sqlite3.connect(db).close()              # empty db, no suppressions
    monkeypatch.setattr(inbox, "_db_path", lambda: db)
    monkeypatch.setattr(inbox, "_get_gmail_service", lambda: _Svc())
    r = inbox.send_email("real@gmail.com", "subj", "body")
    assert r.get("id") == "123"              # reached the send path
    assert "skipped" not in r


# ── console actions ───────────────────────────────────────────────────────────

def test_action_add_then_list_and_suppress():
    from dashboard import email_suppression_actions as esa
    cx = _cx()
    res = esa._exec_add({"entries": [
        {"email": "a@b.com", "bounce_type": "hard", "reason": "NXDOMAIN"},
        {"email": "", "bounce_type": "hard", "reason": "skip-blank"},
    ], "source": "bounce-scan"}, {"cx": cx})
    assert res == {"added": 1}
    assert es.is_suppressed(cx, "a@b.com") is True
    rows = esa._exec_list({}, {"cx": cx})["rows"]
    assert any(r["email"] == "a@b.com" and r["source"] == "bounce-scan" for r in rows)

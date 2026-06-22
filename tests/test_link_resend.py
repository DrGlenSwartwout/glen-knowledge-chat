# tests/test_link_resend.py
import importlib, sqlite3, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens "
                   "(token_hash TEXT, email TEXT, purpose TEXT, extra TEXT, "
                   "created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    return db


def _seed_token(app_module, db, purpose, email="u@x.com", extra=None, expired=True):
    import secrets, json
    tok = "tk_" + secrets.token_urlsafe(8)
    now = datetime.now(timezone.utc)
    exp = now - timedelta(days=1) if expired else now + timedelta(days=1)
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
                   "VALUES (?,?,?,?,?,?)",
                   (app_module._hash_token(tok), email, purpose,
                    json.dumps(extra) if extra else None, now.isoformat(), exp.isoformat()))
        cx.commit()
    return tok


def test_resend_reorder_mints_fresh_and_sends(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    sent = []
    monkeypatch.setattr(app_module, "send_magic_link_email",
                        lambda to, name, url: sent.append((to, url)) or ("smtp", None))
    tok = _seed_token(app_module, db, "reorder", email="r@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert len(sent) == 1 and sent[0][0] == "r@x.com" and "/reorder/auth/" in sent[0][1]
    # a fresh, unexpired reorder token now exists for the email
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='r@x.com' AND purpose='reorder'").fetchone()[0]
    assert n == 2  # the expired one + the fresh one


def test_resend_preserves_extra_for_practitioner(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "send_magic_link_email", lambda *a, **k: ("smtp", None))
    tok = _seed_token(app_module, db, "practitioner_claim", email="p@x.com", extra={"practitioner_id": "P9"})
    app_module.app.test_client().post("/link/resend", json={"token": tok})
    import json as _j
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT extra FROM auth_tokens WHERE purpose='practitioner_claim' "
                          "AND expires_at > ?", (datetime.now(timezone.utc).isoformat(),)).fetchall()
    assert rows and _j.loads(rows[0][0]) == {"practitioner_id": "P9"}


def test_resend_bogus_token_ok_no_send(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    sent = []
    monkeypatch.setattr(app_module, "send_magic_link_email", lambda *a, **k: sent.append(1) or ("smtp", None))
    r = app_module.app.test_client().post("/link/resend", json={"token": "not-a-real-token"})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert sent == []


def test_resend_reveal_existing_sends_reveal_email(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
        br.upsert(cx, "rev@x.com", "2026-06-20", {"body": "x"},
                  [{"name": "Top", "slug": "top", "meaning": "m"}], "s")
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email",
                        lambda to, subj, body: sent.append((to, subj, body)) or True)
    tok = _seed_token(app_module, db, "biofield_reveal", email="rev@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert len(sent) == 1 and "/begin/biofield/" in sent[0][2]


def test_resend_reveal_missing_ok_no_send(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_table(cx)
    sent = []
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: sent.append(1) or True)
    tok = _seed_token(app_module, db, "biofield_reveal", email="nobody@x.com")
    r = app_module.app.test_client().post("/link/resend", json={"token": tok})
    assert r.status_code == 200 and r.get_json().get("ok") is True
    assert sent == []

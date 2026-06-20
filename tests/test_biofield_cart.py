# tests/test_biofield_cart.py
import importlib, sqlite3, sys
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
    from dashboard import biofield_reveals, subscriptions
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        subscriptions.init_subscriptions_table(cx)
        subscriptions.migrate_add_membership_columns(cx)
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens "
            "(token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, purpose TEXT NOT NULL, "
            "extra TEXT, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, consumed_at TEXT)"
        )
        cx.commit()
    return db


def _approved_reveal(app_module, db, email="t@x.com"):
    import secrets as _s
    from datetime import datetime, timezone, timedelta
    from dashboard import biofield_reveals as br
    token = "tk_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = br.upsert(cx, email, "2026-06-20", {"greeting": "Hi", "body": "b"},
                           [{"name": "Top", "slug": "top", "meaning": "m"},
                            {"name": "Deep1", "slug": "deep1", "meaning": "m2"},
                            {"name": "Deep2", "slug": "deep2", "meaning": "m3"}], "s")
        br.set_token(cx, rid, th)
        br.approve_first(cx, rid, "glen")
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def _row(app_module, db, token):
    th = app_module._hash_token(token)
    valid, row = app_module._biofield_verify_token(th)
    assert valid and row is not None
    return row


def test_visible_slugs_paid_returns_all(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    row = _row(app_module, db, _approved_reveal(app_module, db))
    assert app_module._biofield_visible_slugs(row, "t@x.com") == ["top", "deep1", "deep2"]


def test_visible_slugs_free_top_only(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    token = _approved_reveal(app_module, db)
    # Claim the one-time free top unlock for this member so top_unlocked is true.
    from dashboard import biofield_reveals as br
    with sqlite3.connect(db) as cx:
        br.init_free_unlocks(cx)
        rid = br.get_by_token_hash(cx, app_module._hash_token(token))["id"]
        br.record_free_unlock(cx, "t@x.com", rid)
        cx.commit()
    row = _row(app_module, db, token)
    assert app_module._biofield_visible_slugs(row, "t@x.com") == ["top"]


def test_visible_slugs_free_locked_returns_empty(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    row = _row(app_module, db, _approved_reveal(app_module, db))
    assert app_module._biofield_visible_slugs(row, "t@x.com") == []


def test_reveal_payload_cart_enabled(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    token = _approved_reveal(app_module, db)
    html = app_module.app.test_client().get(f"/begin/biofield/{token}").get_data(as_text=True)
    assert '"cart_enabled": true' in html
    assert '"slug": "top"' in html  # remedy payload now carries slug

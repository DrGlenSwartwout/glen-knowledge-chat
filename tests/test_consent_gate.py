"""Membership & consent gate (Tiers 0-1).

Covers the three load-bearing pieces:
  - is_member() derives Tier-1 membership from journey_state.tos_agreed_at
    (session + email union), distinct from paid coaching membership.
  - _is_gated_question() maps the classifier verdict and fails safe to gated.
  - POST /begin/checkout enforces membership (403 need_optin for non-members,
    proceeds for members).
"""

import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def _seed_tos(app_module, db, *, session_id="s1", email="ada@example.com"):
    import sqlite3
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        begin_funnel.record_unlock(cx, session_id=session_id, trigger="tos",
                                   email=email, tos=True, tos_version="v-test")


# ── is_member ────────────────────────────────────────────────────────────────

def test_is_member_false_without_tos(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    assert app_module.is_member("nobody") is False
    assert app_module.is_member("", "") is False


def test_is_member_true_after_tos_by_session(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_tos(app_module, db, session_id="s1", email="ada@example.com")
    assert app_module.is_member("s1") is True


def test_is_member_true_cross_email_union(monkeypatch, tmp_path):
    # Agreed on one session; a DIFFERENT session that knows only the email is
    # still a member (cross-device continuity via get_state email union).
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_tos(app_module, db, session_id="s1", email="ada@example.com")
    assert app_module.is_member("other-session", "ada@example.com") is True
    assert app_module.is_member("other-session", "someone@else.com") is False


# ── _is_gated_question ───────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(text=text)]


class _FakeMessages:
    def __init__(self, text=None, raises=False):
        self._text, self._raises = text, raises

    def create(self, **_k):
        if self._raises:
            raise RuntimeError("boom")
        return _FakeResp(self._text)


def _fake_cl(text=None, raises=False):
    return types.SimpleNamespace(messages=_FakeMessages(text, raises))


def test_gated_question_open_verdict(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "_cl", _fake_cl("OPEN"))
    assert app_module._is_gated_question("how do floaters form?") is False


def test_gated_question_gated_verdict(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "_cl", _fake_cl("GATED"))
    assert app_module._is_gated_question("I have floaters, what should I take?") is True


def test_gated_question_fails_safe_to_gated(monkeypatch):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "_cl", _fake_cl(raises=True))
    assert app_module._is_gated_question("anything") is True


def test_gated_question_empty_is_open(monkeypatch):
    app_module = _load_app()
    # No classifier call for an empty query.
    monkeypatch.setattr(app_module, "_cl", _fake_cl(raises=True))
    assert app_module._is_gated_question("   ") is False


# ── checkout gate ────────────────────────────────────────────────────────────

def _stub_checkout_deps(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "_get_product",
                        lambda slug: {"slug": "test-remedy", "name": "Test Remedy",
                                      "info_only": False, "qbo_item_id": None,
                                      "price_cents": 1000})
    monkeypatch.setattr(app_module._shipping, "quote", lambda b: {"shipping_cents": 0})
    monkeypatch.setattr(app_module, "_ingest_order", lambda **k: None)
    fake_qb = types.ModuleType("dashboard.qbo_billing")
    fake_qb.find_or_create_customer = lambda email, name: {"Id": "C1"}
    fake_qb.create_invoice = lambda *a, **k: {
        "Id": "INV1", "SyncToken": "0", "DocNumber": "1001", "TotalAmt": 10.0}
    fake_qb.get_invoice_pay_link = lambda inv: "http://pay/INV1"
    monkeypatch.setitem(sys.modules, "dashboard.qbo_billing", fake_qb)
    # `from dashboard import qbo_billing` reads the attribute off the already
    # imported dashboard package, so patch that too (sys.modules alone is bypassed).
    import dashboard
    monkeypatch.setattr(dashboard, "qbo_billing", fake_qb, raising=False)


def test_checkout_blocks_non_member(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    _stub_checkout_deps(app_module, monkeypatch)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "visitor")
    r = client.post("/begin/checkout/test-remedy",
                    json={"email": "new@person.com", "method": "zelle", "qty": 1})
    assert r.status_code == 403
    assert r.get_json().get("need_optin") is True


def test_checkout_allows_member(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_tos(app_module, db, session_id="member-sess", email="ada@example.com")
    _stub_checkout_deps(app_module, monkeypatch)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "member-sess")
    r = client.post("/begin/checkout/test-remedy",
                    json={"email": "ada@example.com", "method": "zelle", "qty": 1})
    assert r.status_code == 200
    body = r.get_json()
    assert body.get("ok") is True
    assert not body.get("need_optin")

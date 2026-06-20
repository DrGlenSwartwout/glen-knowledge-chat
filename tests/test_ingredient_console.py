"""Tests for dashboard/ingredient_page_actions.py and the ingredient-page console routes.

Covers:
- ingredient_page.edit updates sections, scores, traditional-use, related-forms; stays draft.
- ingredient_page.approve -> state approved AND calls the injected send fn once per requester.
- approve does not fail if send raises.
- RBAC: actions registered with (OWNER, OPS).
- Console list + load routes return expected data.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the modules
# ---------------------------------------------------------------------------

def _repo():
    return Path(__file__).resolve().parent.parent


def _ensure_path():
    r = str(_repo())
    if r not in sys.path:
        sys.path.insert(0, r)


def _mod_ip():
    _ensure_path()
    try:
        from dashboard import ingredient_pages
        return ingredient_pages
    except Exception as e:
        pytest.skip(f"ingredient_pages not importable: {e}")


def _mod_ipa():
    _ensure_path()
    try:
        from dashboard import ingredient_page_actions
        return ingredient_page_actions
    except Exception as e:
        pytest.skip(f"ingredient_page_actions not importable: {e}")


def _load_app():
    _ensure_path()
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_ipa_deps():
    """Clear injected deps between tests so they don't bleed."""
    ipa = _mod_ipa()
    ipa._DEPS.clear()
    yield
    ipa._DEPS.clear()


def _mem_cx():
    """In-memory sqlite connection with ingredient_pages tables initialised."""
    ip = _mod_ip()
    cx = sqlite3.connect(":memory:")
    ip.init_table(cx)
    return cx


class _FakeContent:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _FakeMsg:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeMessages:
    def create(self, **kw):
        return _FakeMsg("Supports wellness and cellular function.")


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _actor(role=None, name="glen"):
    _ensure_path()
    from dashboard.rbac import Actor, OWNER
    return Actor(role=role or OWNER, name=name)


def _get_action(key):
    _ensure_path()
    from dashboard.actions import get_action
    return get_action(key)


# ---------------------------------------------------------------------------
# ingredient_page.edit
# ---------------------------------------------------------------------------

def test_edit_section_stays_draft():
    """Edit a narrative section - state must stay draft after the edit."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "Old text.")
    ip.set_state(cx, "zinc", "approved", by="glen")

    act = _get_action("ingredient_page.edit")
    act.executor(
        {"slug": "zinc", "section": "what_it_is", "text": "New text."},
        {"cx": cx, "actor": _actor()},
    )
    page = ip.get_page(cx, "zinc")
    assert page["content"]["what_it_is"] == "New text."
    assert page["state"] == "draft"


def test_edit_research_section():
    """Edit the research section."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "mag", "research", "Old research.")

    act = _get_action("ingredient_page.edit")
    act.executor(
        {"slug": "mag", "section": "research", "text": "New research."},
        {"cx": cx, "actor": _actor()},
    )
    page = ip.get_page(cx, "mag")
    assert page["content"]["research"] == "New research."
    assert page["state"] == "draft"


def test_edit_scores_updates_and_clamps():
    """Edit both scores; the store should clamp out-of-range values."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "x")  # ensure row exists

    act = _get_action("ingredient_page.edit")
    act.executor(
        {"slug": "zinc", "research_score": 8, "traditional_score": 15},
        {"cx": cx, "actor": _actor()},
    )
    page = ip.get_page(cx, "zinc")
    assert page["research_score"] == 8
    assert page["traditional_score"] == 10   # clamped from 15
    assert page["state"] == "draft"


def test_edit_traditional_use():
    """Edit the traditional-use list."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "x")

    entries = [{"system": "TCM", "formula": "Ba Zhen Tang", "uses": "deficiency", "forms": "decoction"}]
    act = _get_action("ingredient_page.edit")
    act.executor(
        {"slug": "zinc", "traditional_use": entries},
        {"cx": cx, "actor": _actor()},
    )
    page = ip.get_page(cx, "zinc")
    assert page["traditional_use"][0]["system"] == "TCM"
    assert page["state"] == "draft"


def test_edit_related_forms():
    """Edit the related-forms list."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "x")

    forms = [{"name": "Zinc Picolinate", "slug": "zinc-picolinate", "verdict": "superior", "note": "Better absorbed"}]
    act = _get_action("ingredient_page.edit")
    act.executor(
        {"slug": "zinc", "related_forms": forms},
        {"cx": cx, "actor": _actor()},
    )
    page = ip.get_page(cx, "zinc")
    assert page["related_forms"][0]["verdict"] == "superior"
    assert page["state"] == "draft"


def test_edit_missing_slug_raises():
    """edit with no slug must raise."""
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    act = _get_action("ingredient_page.edit")
    with pytest.raises(ValueError):
        act.executor({"slug": "", "section": "what_it_is", "text": "x"},
                     {"cx": cx, "actor": _actor()})


# ---------------------------------------------------------------------------
# ingredient_page.approve
# ---------------------------------------------------------------------------

def test_approve_sets_state():
    """Approve sets state to approved."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "Good text.")
    ip.set_name(cx, "zinc", "Zinc")

    act = _get_action("ingredient_page.approve")
    result = act.executor({"slug": "zinc"}, {"cx": cx, "actor": _actor()})
    assert result["state"] == "approved"
    assert ip.get_page(cx, "zinc")["state"] == "approved"


def test_approve_emails_each_requester_once():
    """Approve calls send once per requester."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "Good text.")
    ip.set_name(cx, "zinc", "Zinc")
    ip.record_request(cx, "zinc", "alice@test.com")
    ip.record_request(cx, "zinc", "bob@test.com")

    sent = []
    ipa.configure(send=lambda to, subj, body: sent.append((to, subj, body)),
                  strip=lambda s: s,
                  base_url="https://test.example")

    act = _get_action("ingredient_page.approve")
    act.executor({"slug": "zinc"}, {"cx": cx, "actor": _actor()})

    assert len(sent) == 2
    recipients = {s[0] for s in sent}
    assert "alice@test.com" in recipients
    assert "bob@test.com" in recipients
    # body should contain the ingredient page URL
    assert all("/begin/ingredient/zinc" in s[2] for s in sent)


def test_approve_does_not_fail_on_send_error():
    """Approve must succeed even when the send function raises."""
    ip = _mod_ip()
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    ip.upsert_section(cx, "zinc", "what_it_is", "Good text.")
    ip.set_name(cx, "zinc", "Zinc")
    ip.record_request(cx, "zinc", "fail@test.com")

    def _bad_send(to, subj, body):
        raise RuntimeError("SMTP down")

    ipa.configure(send=_bad_send, strip=lambda s: s, base_url="https://test.example")

    act = _get_action("ingredient_page.approve")
    # must NOT raise
    result = act.executor({"slug": "zinc"}, {"cx": cx, "actor": _actor()})
    assert result["state"] == "approved"
    assert ip.get_page(cx, "zinc")["state"] == "approved"


def test_approve_missing_slug_raises():
    ipa = _mod_ipa()
    ipa.register()
    cx = _mem_cx()
    act = _get_action("ingredient_page.approve")
    with pytest.raises(ValueError):
        act.executor({"slug": ""}, {"cx": cx, "actor": _actor()})


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

def test_actions_registered_with_owner_ops_rbac():
    """All three actions must have permission=(OWNER, OPS)."""
    _ensure_path()
    ipa = _mod_ipa()
    ipa.register()
    from dashboard.rbac import OWNER, OPS

    for key in ("ingredient_page.approve", "ingredient_page.edit", "ingredient_page.regenerate"):
        act = _get_action(key)
        assert act is not None, f"action {key} not registered"
        assert act.permission == (OWNER, OPS), f"{key} has wrong permission: {act.permission}"


def test_register_idempotent():
    """Calling register() twice must not raise a duplicate-key error."""
    ipa = _mod_ipa()
    ipa.register()
    ipa.register()  # second call must be a no-op


# ---------------------------------------------------------------------------
# Console list + load routes
# ---------------------------------------------------------------------------

def _fresh_app(monkeypatch, tmp_path):
    import importlib as _il
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        app_module = _il.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    from dashboard import ingredient_pages as _ip2
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        _ip2.init_table(cx)
    return app_module


def test_console_list_returns_pages(monkeypatch, tmp_path):
    """GET /api/console/ingredient-pages returns draft pages."""
    app_module = _fresh_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""   # bypass auth

    from dashboard import ingredient_pages as _ip2
    with sqlite3.connect(str(tmp_path / "chat_log.db")) as cx:
        _ip2.upsert_section(cx, "zinc", "what_it_is", "Some text.")
        _ip2.set_name(cx, "zinc", "Zinc")

    c = app_module.app.test_client()
    r = c.get("/api/console/ingredient-pages")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"]
    slugs = [p["slug"] for p in data["pages"]]
    assert "zinc" in slugs


def test_console_list_gated_when_secret_set(monkeypatch, tmp_path):
    """GET /api/console/ingredient-pages returns 401 when key is missing."""
    app_module = _fresh_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = "secret123"
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "secret123", raising=False)

    c = app_module.app.test_client()
    r = c.get("/api/console/ingredient-pages")  # no key
    assert r.status_code == 401


def test_console_page_served(monkeypatch, tmp_path):
    """GET /console/ingredient-pages serves the HTML page."""
    app_module = _fresh_app(monkeypatch, tmp_path)
    import dashboard as _d
    _d.CONSOLE_SECRET = ""

    c = app_module.app.test_client()
    r = c.get("/console/ingredient-pages")
    assert r.status_code == 200
    assert b"Ingredient" in r.data

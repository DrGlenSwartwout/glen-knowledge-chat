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
    from dashboard import ingredient_pages
    with sqlite3.connect(db) as cx:
        ingredient_pages.init_table(cx)
    return db


def _known_slug():
    from dashboard import ingredients
    return next(iter(ingredients._name_index().keys()))


def test_pagedata_locked_for_non_member(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "non@x.com"})
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{_known_slug()}").get_json()
    assert body["state"] == "locked"
    assert "research_score" not in body and "sections" not in body


def test_pagedata_preparing_for_paid_no_approved(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    # neutralize the background build so the test does not call the model
    monkeypatch.setattr(app_module, "_ingredient_kickoff_build", lambda slug, name: None, raising=False)
    slug = _known_slug()
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{slug}").get_json()
    assert body["state"] == "preparing"
    from dashboard import ingredient_pages
    with sqlite3.connect(db) as cx:
        assert any(r["email"] == "paid@x.com" for r in ingredient_pages.requesters_to_email(cx, slug))


def test_pagedata_approved_returns_full(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "INGREDIENT_PAGES_PAID_ONLY", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    slug = _known_slug()
    from dashboard import ingredient_pages as ip
    with sqlite3.connect(db) as cx:
        ip.upsert_section(cx, slug, "what_it_is", "Hello.")
        ip.set_scores(cx, slug, 8, 7)
        ip.set_state(cx, slug, "approved", by="glen")
    body = app_module.app.test_client().get(f"/begin/ingredient-page-data/{slug}").get_json()
    assert body["state"] == "approved" and body["research_score"] == 8
    assert "formulations" in body and any(s["id"] == "what_it_is" for s in body["sections"])


def test_unknown_slug_pagedata(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "paid@x.com"})
    r = app_module.app.test_client().get("/begin/ingredient-page-data/bogus-xyz")
    assert r.status_code == 404 or r.get_json().get("state") == "unknown"

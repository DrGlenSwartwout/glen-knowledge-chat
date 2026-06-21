# tests/test_begin_ascend.py
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_bf():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("begin_funnel")
    except Exception as e:
        pytest.skip(f"begin_funnel not importable: {e}")


def test_recommend_heal_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("heal") == "biofield-analysis"


def test_recommend_learn_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("learn") == "certification"


def test_recommend_learn_certified_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("learn", reached={"certification"}) == "one-to-one"


def test_recommend_build_entry():
    bf = _load_bf()
    assert bf.recommend_ascend("build") == "one-to-one"


def test_recommend_build_practitioner_bumps():
    bf = _load_bf()
    assert bf.recommend_ascend("build", reached={"one-to-one"}) == "healing-oasis-tools"


def test_recommend_unknown_goal_falls_back_to_heal():
    bf = _load_bf()
    assert bf.recommend_ascend("nonsense") == "biofield-analysis"
    assert bf.recommend_ascend("") == "biofield-analysis"
    assert bf.recommend_ascend(None) == "biofield-analysis"


def test_recommend_all_reached_returns_track_top():
    bf = _load_bf()
    allslugs = set(bf.TIER_CATALOG.keys())
    assert bf.recommend_ascend("build", reached=allslugs) == "consultant-package"


def test_recommend_returns_valid_catalog_slug():
    bf = _load_bf()
    for goal in ("heal", "learn", "build", "x"):
        assert bf.recommend_ascend(goal) in bf.TIER_CATALOG


def test_ascend_is_valid_trigger():
    bf = _load_bf()
    assert "ascend" in bf.VALID_TRIGGERS


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
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        cx.commit()
    return db


def test_recommend_endpoint_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", False, raising=False)
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=heal")
    body = r.get_json()
    assert body["ok"] is True and body["enabled"] is False


def test_recommend_endpoint_returns_hero_and_ladder(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: None)
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=learn")
    body = r.get_json()
    assert body["ok"] is True and body["enabled"] is True
    assert body["recommended"]["slug"] == "certification"
    # full ladder, ordered by n
    ns = [t["n"] for t in body["ladder"]]
    assert ns == sorted(ns) and len(body["ladder"]) == len(app_module.begin_funnel.TIER_CATALOG)


def test_recommend_endpoint_member_reaches_biofield(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_active_membership_for_email", lambda e: {"ok": True})
    # A paid member asking the heal track has already reached biofield-analysis ->
    # the only heal rung is reached, so the recommendation is the track top (still biofield-analysis).
    r = app_module.app.test_client().get("/begin/ascend/recommend?goal=heal")
    assert r.get_json()["recommended"]["slug"] == "biofield-analysis"


def _seed_member(app_module, monkeypatch):
    # ToS member (ordering gate) for the inquire path.
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": True)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "t@x.com"})


def test_inquire_flag_off(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", False, raising=False)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.get_json().get("ok") is False


def test_inquire_non_member_403(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "get_authenticated_user", lambda req: {"email": "t@x.com"})
    monkeypatch.setattr(app_module, "is_member", lambda session_id="", email="": False)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True


def test_inquire_unknown_slug_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "not-a-tier", "goal": "heal"})
    assert r.status_code == 400 and r.get_json().get("ok") is False


def test_inquire_member_records_and_notifies(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    calls = {"ghl": 0, "email": 0}
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: calls.__setitem__("ghl", calls["ghl"] + 1))
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: calls.__setitem__("email", calls["email"] + 1) or True)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "certification", "goal": "learn", "note": "ready"})
    assert r.get_json() == {"ok": True}
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT email, slug, goal, note FROM ascend_inquiries").fetchall()
    assert rows == [("t@x.com", "certification", "learn", "ready")]
    assert calls["ghl"] == 1 and calls["email"] == 1


def test_inquire_best_effort_email_failure_still_ok(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    def _boom(*a, **k):
        raise RuntimeError("smtp down")
    monkeypatch.setattr(app_module, "ghl_onboard_contact", _boom)
    monkeypatch.setattr(app_module, "_send_inquiry_email", _boom)
    r = app_module.app.test_client().post("/begin/ascend/inquire", json={"slug": "biofield-analysis", "goal": "heal"})
    assert r.get_json() == {"ok": True}  # row written despite GHL/email failure
    with sqlite3.connect(db) as cx:
        assert cx.execute("SELECT COUNT(*) FROM ascend_inquiries").fetchone()[0] == 1


def test_inquire_idempotent_per_email_slug(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "ASCEND_PERSONALIZED_ENABLED", True, raising=False)
    _seed_member(app_module, monkeypatch)
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    c = app_module.app.test_client()
    c.post("/begin/ascend/inquire", json={"slug": "certification", "goal": "learn", "note": "first"})
    c.post("/begin/ascend/inquire", json={"slug": "certification", "goal": "build", "note": "second"})
    with sqlite3.connect(db) as cx:
        rows = cx.execute("SELECT goal, note FROM ascend_inquiries WHERE email='t@x.com' AND slug='certification'").fetchall()
    assert rows == [("build", "second")]  # single row, updated

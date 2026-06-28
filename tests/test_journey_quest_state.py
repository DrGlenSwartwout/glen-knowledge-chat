"""Tests for Task 7: member progress mirror — /api/journey/quest-state.

Follows the fixture pattern in test_journey_shell_inject.py:
  monkeypatch LOG_DB + JOURNEY_QUEST_ENABLED, use app.test_client().
Pure-helper tests run without Flask.
"""
import json
import sqlite3
import pytest

from dashboard.journey_quest import (
    STAGE_KEYS,
    RAILS,
    empty_state,
    normalize,
    merge_quest,
    init_quest_store,
    load,
    save,
)


# ---------- pure helpers ----------

class TestEmptyState:
    def test_has_all_stages(self):
        s = empty_state()
        for k in STAGE_KEYS:
            assert k in s
            assert s[k] == {"found": False, "done": False}

    def test_has_paths_and_entered(self):
        s = empty_state()
        assert s["paths"] == []
        assert s["entered"] is False

    def test_is_independent(self):
        a = empty_state()
        b = empty_state()
        a["paths"].append("hunt")
        assert b["paths"] == []


class TestNormalize:
    def test_passes_through_valid_state(self):
        s = empty_state()
        s["home"]["found"] = True
        s["paths"] = ["hunt"]
        s["entered"] = True
        n = normalize(s)
        assert n["home"]["found"] is True
        assert n["paths"] == ["hunt"]
        assert n["entered"] is True

    def test_drops_unknown_top_level_keys(self):
        s = empty_state()
        s["junk"] = "whatever"
        n = normalize(s)
        assert "junk" not in n

    def test_drops_unknown_stage_sub_keys(self):
        s = empty_state()
        s["scan"]["extra"] = "bad"
        n = normalize(s)
        assert "extra" not in n["scan"]

    def test_coerces_truthy_strings_to_bool(self):
        s = empty_state()
        s["find"]["found"] = 1        # truthy int
        s["find"]["done"] = "true"    # truthy string
        s["entered"] = 1
        n = normalize(s)
        assert n["find"]["found"] is True
        assert n["find"]["done"] is True
        assert n["entered"] is True

    def test_coerces_falsy_to_false(self):
        s = empty_state()
        s["give"]["found"] = 0
        s["give"]["done"] = None
        n = normalize(s)
        assert n["give"]["found"] is False
        assert n["give"]["done"] is False

    def test_restricts_paths_to_valid_rails(self):
        s = empty_state()
        s["paths"] = ["hunt", "BADTRACK", "video", 42, "chat"]
        n = normalize(s)
        for p in n["paths"]:
            assert p in RAILS

    def test_dedupes_paths(self):
        s = empty_state()
        s["paths"] = ["hunt", "hunt", "video"]
        n = normalize(s)
        assert n["paths"].count("hunt") == 1

    def test_handles_missing_stages_gracefully(self):
        # partial state from client (missing some stages)
        n = normalize({"entered": True, "paths": [], "home": {"found": True, "done": False}})
        for k in STAGE_KEYS:
            assert k in n
        assert n["home"]["found"] is True
        assert n["scan"]["found"] is False

    def test_handles_non_dict_input(self):
        n = normalize(None)
        assert n == empty_state()

    def test_handles_empty_dict(self):
        n = normalize({})
        assert n == empty_state()


class TestMergeQuest:
    def test_or_semantics_for_found(self):
        a = empty_state()
        b = empty_state()
        a["home"]["found"] = True
        m = merge_quest(a, b)
        assert m["home"]["found"] is True

    def test_or_semantics_for_done(self):
        a = empty_state()
        b = empty_state()
        b["scan"]["done"] = True
        m = merge_quest(a, b)
        assert m["scan"]["done"] is True

    def test_paths_union_deduped(self):
        a = empty_state()
        b = empty_state()
        a["paths"] = ["hunt", "video"]
        b["paths"] = ["video", "chat"]
        m = merge_quest(a, b)
        assert sorted(m["paths"]) == ["chat", "hunt", "video"]

    def test_paths_sorted(self):
        a = empty_state()
        b = empty_state()
        a["paths"] = ["video"]
        b["paths"] = ["hunt"]
        m = merge_quest(a, b)
        assert m["paths"] == sorted(m["paths"])

    def test_entered_or(self):
        a = empty_state()
        b = empty_state()
        b["entered"] = True
        m = merge_quest(a, b)
        assert m["entered"] is True

    def test_monotonic_never_unsets_done(self):
        a = empty_state()
        a["heal"]["done"] = True
        a["heal"]["found"] = True
        b = empty_state()   # older/empty state
        m = merge_quest(a, b)
        assert m["heal"]["done"] is True
        assert m["heal"]["found"] is True

    def test_monotonic_never_unsets_found(self):
        advanced = empty_state()
        advanced["give"]["found"] = True
        older = empty_state()
        m = merge_quest(advanced, older)
        assert m["give"]["found"] is True

    def test_all_five_stages_merged(self):
        a = empty_state()
        b = empty_state()
        for i, k in enumerate(STAGE_KEYS):
            if i % 2 == 0:
                a[k]["found"] = True
            else:
                b[k]["found"] = True
        m = merge_quest(a, b)
        for k in STAGE_KEYS:
            assert m[k]["found"] is True

    def test_paths_restricted_to_rails(self):
        a = empty_state()
        b = empty_state()
        a["paths"] = ["hunt", "BADTRACK"]
        b["paths"] = ["video"]
        m = merge_quest(a, b)
        for p in m["paths"]:
            assert p in RAILS


# ---------- storage ----------

class TestQuestStore:
    def test_load_returns_empty_when_no_row(self, tmp_path):
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as cx:
            init_quest_store(cx)
            result = load(cx, "test@example.com")
        assert result == empty_state()

    def test_save_and_load_round_trip(self, tmp_path):
        db = str(tmp_path / "test.db")
        state = empty_state()
        state["home"]["found"] = True
        state["paths"] = ["hunt"]
        state["entered"] = True
        with sqlite3.connect(db) as cx:
            init_quest_store(cx)
            save(cx, "test@example.com", state)
            result = load(cx, "test@example.com")
        assert result["home"]["found"] is True
        assert result["paths"] == ["hunt"]
        assert result["entered"] is True

    def test_save_lowercases_email(self, tmp_path):
        db = str(tmp_path / "test.db")
        state = empty_state()
        state["entered"] = True
        with sqlite3.connect(db) as cx:
            init_quest_store(cx)
            save(cx, "UPPER@EXAMPLE.COM", state)
            result = load(cx, "upper@example.com")
        assert result["entered"] is True

    def test_upsert_replaces_existing(self, tmp_path):
        db = str(tmp_path / "test.db")
        with sqlite3.connect(db) as cx:
            init_quest_store(cx)
            s1 = empty_state()
            s1["scan"]["done"] = True
            save(cx, "a@b.com", s1)
            s2 = empty_state()
            s2["find"]["done"] = True
            save(cx, "a@b.com", s2)
            result = load(cx, "a@b.com")
        assert result["find"]["done"] is True


# ---------- Flask route ----------

@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "JOURNEY_QUEST_ENABLED", True)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _member_cookies(c):
    """Set a dummy amg_session cookie so the request reaches begin_funnel.get_state."""
    return {"amg_session": "testsession123"}


def test_flag_off_returns_404(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "JOURNEY_QUEST_ENABLED", False)
    appmod.app.config["TESTING"] = True
    c = appmod.app.test_client()
    assert c.get("/api/journey/quest-state").status_code == 404
    assert c.post("/api/journey/quest-state", json={}).status_code == 404


def test_get_returns_empty_state_for_anonymous(client):
    c, _ = client
    r = c.get("/api/journey/quest-state")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["state"] == empty_state()


def test_post_anonymous_no_persist(client, tmp_path):
    """Anonymous POST returns the posted state but does NOT persist (no email)."""
    c, appmod = client
    state = empty_state()
    state["home"]["found"] = True
    r = c.post("/api/journey/quest-state", json=state)
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    # state echoed back
    assert data["state"]["home"]["found"] is True
    # nothing in DB (no email to key on)
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        # table may not even exist
        try:
            rows = cx.execute("SELECT * FROM quest_state").fetchall()
            assert rows == []
        except sqlite3.OperationalError:
            pass   # table never created -- also fine


def test_post_then_get_merges_and_persists(client, monkeypatch):
    """Authenticated member: POST persists, GET returns merged state."""
    c, appmod = client

    # Patch begin_funnel.get_state to return a state with a real email
    import begin_funnel
    orig_get_state = begin_funnel.get_state
    def fake_get_state(cx, session_id, email):
        return {"email": "member@test.com", "tos_agreed_at": "2026-01-01"}
    monkeypatch.setattr(begin_funnel, "get_state", fake_get_state)

    state = empty_state()
    state["home"]["found"] = True
    state["paths"] = ["hunt"]
    state["entered"] = True

    r = c.post("/api/journey/quest-state",
               json=state,
               headers={"Cookie": "amg_session=testsession"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True
    assert d["state"]["home"]["found"] is True
    assert d["state"]["paths"] == ["hunt"]

    # GET should return the persisted state
    r2 = c.get("/api/journey/quest-state",
               headers={"Cookie": "amg_session=testsession"})
    assert r2.status_code == 200
    d2 = r2.get_json()
    assert d2["ok"] is True
    assert d2["state"]["home"]["found"] is True
    assert d2["state"]["entered"] is True


def test_post_merges_with_existing(client, monkeypatch):
    """Second POST merges monotonically with stored state."""
    c, appmod = client

    import begin_funnel
    def fake_get_state(cx, session_id, email):
        return {"email": "merge@test.com", "tos_agreed_at": "2026-01-01"}
    monkeypatch.setattr(begin_funnel, "get_state", fake_get_state)

    # First POST: home done
    s1 = empty_state()
    s1["home"]["done"] = True
    c.post("/api/journey/quest-state", json=s1,
           headers={"Cookie": "amg_session=testsession"})

    # Second POST: scan done (home NOT included)
    s2 = empty_state()
    s2["scan"]["done"] = True
    r = c.post("/api/journey/quest-state", json=s2,
               headers={"Cookie": "amg_session=testsession"})
    d = r.get_json()
    # Merged: both home AND scan should be done
    assert d["state"]["home"]["done"] is True
    assert d["state"]["scan"]["done"] is True

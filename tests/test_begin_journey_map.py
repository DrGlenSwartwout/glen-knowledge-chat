"""Begin #2 - journey_map status logic + href threading."""

import importlib
import sys
from pathlib import Path

import pytest


def _bf():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("begin_funnel")
    except Exception as e:
        pytest.skip(f"begin_funnel not importable: {e}")


def _state(gates):
    return {"unlocked_gates": list(gates)}


def test_no_signals_scan_is_next():
    bf = _bf()
    m = bf.journey_map(_state([]), "")
    assert [c["key"] for c in m] == ["scan", "find", "heal", "give"]
    assert m[0]["status"] == "next" and m[0]["fill"] == 0.0
    assert all(c["fill"] == 0.0 for c in m)


def test_scan_gate_half_fills_scan():
    bf = _bf()
    m = bf.journey_map(_state(["scan"]), "")
    by = {c["key"]: c for c in m}
    assert by["scan"]["fill"] == 0.5 and by["scan"]["status"] == "next"


def test_scan_complete_advances_next_to_find():
    bf = _bf()
    m = bf.journey_map(_state(["scan", "course_ww"]), "")
    by = {c["key"]: c for c in m}
    assert by["scan"]["fill"] == 1.0 and by["scan"]["status"] == "done"
    assert by["find"]["status"] == "next"


def test_give_label_and_predicate_ambassador():
    bf = _bf()
    m = bf.journey_map(_state([]), "", {"ambassador": True})
    by = {c["key"]: c for c in m}
    assert by["give"]["label"] == "Give"
    assert by["give"]["fill"] == 0.5
    assert by["give"]["steps"][0]["done"] is True


def test_smart_scan_href():
    bf = _bf()
    sign = bf.journey_map(_state([]), "slug", {"has_e4l": False})
    nos = {c["key"]: c for c in sign}
    assert nos["scan"]["href"].startswith("https://truly.vip/E4L")
    have = bf.journey_map(_state([]), "slug", {"has_e4l": True})
    yes = {c["key"]: c for c in have}
    assert yes["scan"]["href"].startswith("https://portal.e4l.com")


def test_thread_href_external_threads_utm():
    bf = _bf()
    h = bf._thread_href("https://x.example.com", "slug", "begin-journey-scan")
    assert h.startswith("https://x.example.com?utm_source=slug")
    assert "utm_campaign=begin-journey-scan" in h
    # internal pass-through unchanged
    assert bf._thread_href("/begin/voice", "slug", "begin-journey-scan") == "/begin/voice"


def test_card_href_unchanged_after_refactor():
    bf = _bf()
    # internal CARD_CATALOG entry returns bare base_url
    assert bf.card_href("voice_distinctions", "slug") == "/begin/voice"
    # external entry threads the original begin-card-<key> campaign
    h = bf.card_href("quiz", "slug")
    assert h.startswith("https://healing.scoreapp.com?utm_source=slug")
    assert "utm_campaign=begin-card-quiz" in h


import sqlite3


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_state_payload_includes_journey_map(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "j1")
    body = client.get("/begin/state").get_json()
    jm = body["journey_map"]
    assert [c["key"] for c in jm] == ["scan", "find", "heal", "give"]
    assert jm[0]["status"] == "next"            # nothing done yet
    assert all(set(c) >= {"key", "label", "paren", "href", "status"} for c in jm)


def test_begin_serves_journey_strip(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert 'id="journey-strip"' in html
    assert 'id="journey-cards"' in html
    for label in ("Scan", "Find", "Heal", "Give"):
        assert label in html
    assert "function renderJourney" in html


def test_begin_serves_give_and_fill(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "Give" in html
    assert "jc-fill" in html
    assert "jc-count" in html
    assert "JOURNEY_TRIGGER" not in html


def test_begin_serves_unfold_and_framing(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "function unfoldJourney" in html
    assert "the path we'll walk together" in html
    assert "unfoldJourney('chat')" in html
    assert "unfoldJourney('video')" in html

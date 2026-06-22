import sqlite3
import sys
from pathlib import Path

import pytest


def _ensure_path():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)


def _tp():
    _ensure_path()
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _tpa():
    _ensure_path()
    try:
        from dashboard import topic_page_actions
        return topic_page_actions
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_page_actions not importable: {e}")


def _actor():
    _ensure_path()
    from dashboard.rbac import Actor, OWNER
    return Actor(role=OWNER, name="glen")


def _get_action(key):
    _ensure_path()
    from dashboard.actions import get_action
    return get_action(key)


@pytest.fixture(autouse=True)
def _reset_deps():
    tpa = _tpa()
    tpa._DEPS.clear()
    yield
    tpa._DEPS.clear()


def _cx():
    tp = _tp()
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_approve_refused_when_compliance_failed():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "detox", "overview", "It cures cancer.")
    tp.set_compliance(cx, "detox", {"passed": False, "flags": [{"phrase": "cures", "reason": "claim"}]})
    res = _get_action("topic_page.approve").executor({"slug": "detox"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is False
    assert res["error"] == "compliance_failed"
    assert tp.get_page(cx, "detox")["state"] != "approved"


def test_approve_refused_when_no_scan_present():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "detox", "overview", "Supports healthy energy.")
    res = _get_action("topic_page.approve").executor({"slug": "detox"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is False
    assert tp.get_page(cx, "detox")["state"] != "approved"


def test_approve_succeeds_when_compliance_passed_and_notifies():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "low-energy", "overview", "Supports healthy energy.")
    tp.set_name(cx, "low-energy", "Low Energy")
    tp.set_compliance(cx, "low-energy", {"passed": True, "flags": []})
    tp.record_request(cx, "low-energy", "a@example.com")
    sent = []
    tpa.configure(send=lambda to, s, b: sent.append((to, s, b)), strip=lambda s: s, base_url="https://x.test")
    res = _get_action("topic_page.approve").executor({"slug": "low-energy"}, {"cx": cx, "actor": _actor()})
    assert res["ok"] is True
    assert tp.get_page(cx, "low-energy")["state"] == "approved"
    assert len(sent) == 1 and "/learn/low-energy" in sent[0][2]


def test_edit_clears_compliance_and_resets_draft():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.upsert_section(cx, "low-energy", "overview", "old")
    tp.set_compliance(cx, "low-energy", {"passed": True, "flags": []})
    tp.set_state(cx, "low-energy", "approved", by="glen")
    _get_action("topic_page.edit").executor(
        {"slug": "low-energy", "section": "overview", "text": "new"}, {"cx": cx, "actor": _actor()})
    page = tp.get_page(cx, "low-energy")
    assert page["content"]["overview"] == "new"
    assert page["state"] == "draft"
    assert page["compliance"] == {}


def test_actions_registered_owner_ops():
    tpa = _tpa()
    tpa.register()
    from dashboard.rbac import OWNER, OPS
    for key in ("topic_page.approve", "topic_page.edit", "topic_page.regenerate"):
        act = _get_action(key)
        assert act is not None
        assert act.permission == (OWNER, OPS)


def test_dismiss_sets_state_dismissed_and_drops_from_suggestions():
    tp, tpa = _tp(), _tpa()
    tpa.register()
    cx = _cx()
    tp.record_suggestion(cx, "junk-topic", "Junk Topic", "symptom", "a@x.com")
    assert any(r["slug"] == "junk-topic" for r in tp.list_suggestions(cx))
    res = _get_action("topic_page.dismiss").executor({"slug": "junk-topic"}, {"cx": cx, "actor": _actor()})
    assert res["state"] == "dismissed"
    assert tp.get_page(cx, "junk-topic")["state"] == "dismissed"
    assert not any(r["slug"] == "junk-topic" for r in tp.list_suggestions(cx))

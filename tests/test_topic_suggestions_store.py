import sqlite3, sys
from pathlib import Path
import pytest


def _mod():
    r = str(Path(__file__).resolve().parent.parent)
    if r not in sys.path:
        sys.path.insert(0, r)
    try:
        from dashboard import topic_pages
        return topic_pages
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"topic_pages not importable: {e}")


def _cx(tp):
    cx = sqlite3.connect(":memory:")
    tp.init_table(cx)
    return cx


def test_record_new_suggestion_sets_suggested_and_records_request():
    tp = _mod(); cx = _cx(tp)
    st = tp.record_suggestion(cx, "magnesium-deficiency", "Magnesium Deficiency", "condition", "a@x.com")
    assert st == "suggested"
    page = tp.get_page(cx, "magnesium-deficiency")
    assert page["state"] == "suggested" and page["kind"] == "condition" and page["name"] == "Magnesium Deficiency"
    assert tp.requesters_to_email(cx, "magnesium-deficiency")  # one asker recorded


def test_second_asker_increments_demand_keeps_suggested():
    tp = _mod(); cx = _cx(tp)
    tp.record_suggestion(cx, "dry-skin", "Dry Skin", "symptom", "a@x.com")
    st = tp.record_suggestion(cx, "dry-skin", "Dry Skin", "symptom", "b@x.com")
    assert st == "suggested"
    rows = [r for r in tp.list_suggestions(cx) if r["slug"] == "dry-skin"]
    assert rows and rows[0]["demand"] == 2


def test_record_does_not_downgrade_existing_pipeline_row():
    tp = _mod(); cx = _cx(tp)
    tp.upsert_section(cx, "low-energy", "overview", "x")
    tp.set_state(cx, "low-energy", "approved", by="glen")
    st = tp.record_suggestion(cx, "low-energy", "Low Energy", "symptom", "c@x.com")
    assert st == "approved"
    assert tp.get_page(cx, "low-energy")["state"] == "approved"  # not downgraded
    assert tp.requesters_to_email(cx, "low-energy")             # request still recorded


def test_list_suggestions_only_suggested_ordered_by_demand():
    tp = _mod(); cx = _cx(tp)
    tp.record_suggestion(cx, "a-topic", "A Topic", "function", "1@x.com")
    tp.record_suggestion(cx, "b-topic", "B Topic", "function", "1@x.com")
    tp.record_suggestion(cx, "b-topic", "B Topic", "function", "2@x.com")
    tp.upsert_section(cx, "c-topic", "overview", "x")  # a non-suggested row must not appear
    out = tp.list_suggestions(cx)
    slugs = [r["slug"] for r in out]
    assert slugs[:2] == ["b-topic", "a-topic"]   # b has demand 2, a has 1
    assert "c-topic" not in slugs
